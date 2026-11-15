"""Ordinary finite-state elimination and conditional bounds for V52's target.

The public action/observation domain is unchanged. Independent nuisance
observations and decision-free absorbing ride suffixes are integrated out.
"""
import numpy as np
from time import process_time,perf_counter

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v52.reference import CausalReference, point
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v53.bounds import destination_lower_bound,commitment_upper_bound
from experiments.reconstruction_v53.chain import terminal_ride_value
from experiments.reconstruction_v53.relevance import RemainingReadSets, history_key


class CompressedReference(CausalReference):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,implementation='mass_terminal',**kwargs)
        self.reads=None if self.observations is None else RemainingReadSets(self.model,self.observations)
        self.context_trips=None
        self.lower_cache={}
        self.terminal_cache={}
        self.stats.update(whole_tail_integrations=0,whole_tail_hits=0,
            destination_bounds=0,destination_bound_hits=0,projected_event_states=0,
            partial_cache_hits=0)

    def solve(self,node,row,*,mode='optimal'):
        self.mode=mode
        self.observations=self.observations or ObservationContract.from_menu(node)
        self._prepare(node)
        clock=float(row['clock'])
        specifications={}
        if row['kind']=='onboard':
            position=node.path.index(row['cid'])
            specifications={'alight':(node.exits[row['cid']],None,None,float('-inf')),
                            'continue':(node,position+1,None,float('-inf'))}
        elif row['kind']=='platform':
            ready=float(row.get('ready',row['entry']+self.graph.stops[node.stop].min_transfer_time))
            specifications={'wait':(node,None,ready,clock)}
            specifications.update({str(cid):(node.rides[cid],0,clock,float('-inf'))
                                   for cid in row['visible'] if clock>=ready})
        elif row['kind']=='platform_start':
            specifications={'policy':(node,None,None,float('-inf'))}
        initial_bounds={}
        for action,(target,position,ready,after) in specifications.items():
            lower=self._lower(target,self.snapshot,clock,position)
            initial_bounds[action]=dict(lower=lower,upper=self.deadline,commitment=None)
        if mode=='optimal':
            # This is an explicitly limited feasible-policy search. Its
            # policy is legal but is not the fixed cheap or finite-depth rule.
            ceiling=None if self.max_process_cpu is None else max(0.,self.max_process_cpu*.15)
            for action,(target,position,ready,after) in specifications.items():
                receipt=commitment_upper_bound(self.graph,self.root_law,self.deadline,
                    target,self.snapshot,clock,position=position,ready=ready,after=after,
                    max_routes=16,available=lambda:self._available() and
                    (ceiling is None or process_time()-self.cpu_start<ceiling))
                initial_bounds[action].update(upper=receipt['upper'],commitment=receipt)
        result=super().solve(node,row,mode=mode)
        for action,value in result['actions'].items():
            bound=initial_bounds[action]
            lo=max(value['lower'],bound['lower']-clock)
            hi=min(value['upper'],bound['upper']-clock)
            if lo>hi+1e-7:
                raise AssertionError('Root feasible and relaxation bounds conflict')
            value.update(lower=min(lo,hi),upper=hi,
                         mean=min(lo,hi) if value['complete'] else None)
        actions=result['actions']
        best=min(actions,key=lambda a:actions[a]['upper'])
        best_label=best if best in ('wait','alight','continue','policy') else int(best)
        if mode in ('optimal','one_step_p0','two_step_p0'):
            lo=min(v['lower'] for v in actions.values());hi=min(v['upper'] for v in actions.values())
            complete=all(v['complete'] for v in actions.values())
            result['value']=dict(lower=lo,upper=hi,complete=complete,mean=lo if complete else None)
            result['selected_action']=best_label
        else:
            result['value']=dict(actions[str(result['cheap_action'])])
        competitors=[v['lower'] for a,v in actions.items() if a!=best]
        result['certified_action']=best_label if competitors and actions[best]['upper']<min(competitors)-1e-9 else None
        if {'alight','continue'}.issubset(actions):
            a,c=actions['alight'],actions['continue']
            result['difference_interval']=[a['lower']-c['upper'],a['upper']-c['lower']]
        result.update(implementation='compressed_conditional',root_bounds=initial_bounds,
            cost=dict(process_cpu=process_time()-self.cpu_start,wall=perf_counter()-self.wall_start),
            scope='Same V52 causal target; ordinary independent-state elimination, absorbing-chain integration and valid conditional destination/feasible-route bounds. No method novelty claimed.')
        return result

    def _prepare(self,node):
        # Every retained scorer dependency is included. The independent
        # observation-only trips outside it integrate to one.
        if self.reads is None:
            self.reads=RemainingReadSets(self.model,self.observations)
        for trip in sorted(self.reads.node_trips(node)):
            if trip not in self.root_law.roots:
                self.root_law.roots[trip]=self.snapshot.law(trip)
                self.stats['root_law_preparations']+=1

    def _history_key(self,history):
        return (super()._history_key(history) if self.context_trips is None
                else history_key(history,self.context_trips))

    def _lower(self,node,history,clock,position=None):
        trips=self.reads.node_trips(node,position=position or 0)
        key=(self._structure(node),position,float(clock),history_key(history,trips))
        if key in self.lower_cache:
            self.stats['destination_bound_hits']+=1
            return self.lower_cache[key]
        if not self._available():
            return min(float(clock),self.deadline)
        value=destination_lower_bound(self.graph,self.root_law,self.deadline,
            node,history,clock,position=position)
        self.stats['destination_bounds']+=1
        self.lower_cache[key]=value
        return value

    @staticmethod
    def _intersect(value,lower,previous=None):
        lo,hi,complete=value
        lo=max(lo,lower)
        if previous is not None:
            lo=max(lo,previous[0]);hi=min(hi,previous[1])
            complete=complete or previous[2]
        if lo>hi+1e-7:
            raise AssertionError('Valid conditional bounds became disjoint')
        # Roundoff within the existing numerical tolerance is not an
        # exact-completion claim. Preserve the explicit completion flag.
        return min(lo,hi),hi,complete

    def _node(self,key,clock,build):
        old=self.memo.get(key)
        if old is not None:
            self.stats['cache_hits']+=1
            if old[2] or not self._available():
                return old
            self.stats['partial_cache_hits']+=1
        if not self._available():
            return self._unknown(clock)
        self.stats['states']+=1
        try:
            value=build()
        except UnsupportedHistory as error:
            value=self._unknown(clock,str(error))
        if old is not None:
            value=self._intersect(value,min(float(clock),self.deadline),old)
        self.memo[key]=value
        return value

    def _ride(self,ride,position,history,clock,depth):
        if position>=len(ride.path) or clock>=self.deadline:
            return point(self.deadline)
        prior=self.context_trips
        self.context_trips=self.reads.ride_trips(ride,position)
        try:
            key=(self._structure(ride),position,float(clock),self._history_key(history))
            if key in self.terminal_cache:
                self.stats['whole_tail_hits']+=1
                return point(self.terminal_cache[key])
            if self._available():
                value=terminal_ride_value(self.root_law,history,ride,position,clock,self.deadline)
                if value is not None:
                    self.stats['states']+=1
                    self.stats['whole_tail_integrations']+=1
                    self.terminal_cache[key]=value
                    return point(value)
            lower=self._lower(ride,history,clock,position)
            result=super()._ride(ride,position,history,clock,depth)
            return self._intersect(result,lower)
        finally:
            self.context_trips=prior

    def _onboard(self,ride,position,history,clock,depth):
        prior=self.context_trips
        self.context_trips=self.reads.ride_trips(ride,position)
        try:
            lower=self._lower(ride,history,clock,position)
            return self._intersect(super()._onboard(ride,position,history,clock,depth),lower)
        finally:
            self.context_trips=prior

    def _event_grid(self,cids,entry):
        key=tuple(cids),float(entry)
        if key not in self.grids:
            times=set()
            for cid in cids:
                trip,slot=self.model._positions[cid,'departure']
                law=self.root_law.roots[trip]
                mass=law.marginal(slot)
                keep=(~law.canceled[slot])&(mass>0)&(law.times[slot]>=entry)&(law.times[slot]<self.deadline)
                times.update(float(t) for t in law.times[slot][keep])
            self.grids[key]=np.array(sorted(times))
        return self.grids[key]

    def _platform(self,platform,entry,history,*,depth,after=float('-inf')):
        if entry>=self.deadline:
            return point(self.deadline)
        prior=self.context_trips
        self.context_trips=self.reads.platform_trips(platform)
        ready=entry+self.graph.stops[platform.stop].min_transfer_time
        initial_seen=tuple(cid for cid in platform.rides if any(
            r[:3]==(cid,'departure','observed') and entry<=r[3]<=after
            for r in history._records.get(self.model._positions[cid,'departure'][0],())))
        lower=self._lower(platform,history,entry)

        def event(seen,h,d,previous_clock):
            chain=[]
            while True:
                remaining=tuple(cid for cid in platform.rides if cid not in seen)
                if not remaining:
                    tail=point(self.deadline);break
                relevant=self.reads.platform_trips(platform,remaining)
                cids=self.reads.observation_cids(platform.stop,relevant)
                times=self._event_grid(cids,entry)
                index=int(np.searchsorted(times,previous_clock,side='right'))
                if index==len(times):
                    tail=point(self.deadline);break
                clock=float(times[index])
                key=('projected_event',self._structure(platform),float(entry),clock,
                     remaining,history_key(h,relevant),d)
                old=self.memo.get(key)
                if old is not None:
                    self.stats['cache_hits']+=1
                    if old[2] or not self._available():
                        tail=old;break
                    self.stats['partial_cache_hits']+=1
                if not self._available():
                    tail=self._unknown(clock);break
                self.stats['states']+=1
                self.stats['projected_event_states']+=1
                lo=hi=consumed=no_event=0.
                complete=True
                try:
                    outcomes=self._joint_outcomes(cids,clock,h)
                    if outcomes is None:
                        tail=self._unknown(clock);break
                    for mass,visible in outcomes:
                        if mass<=0:
                            continue
                        if not self._available():
                            remainder=max(0.,1.-consumed)
                            lo+=remainder*clock;hi+=remainder*self.deadline
                            complete=False
                            self.stats['budget_unknowns']+=1
                            self.reasons['budget_chance_remainder']+=1
                            break
                        if visible:
                            observed=h.clone()
                            observed.observe_platform(cids,entry,clock,{cid:clock for cid in visible})
                            next_seen=tuple(sorted(set(seen)|set(visible)))
                            available=tuple(cid for cid in platform.rides if cid in visible)
                            if clock<ready or not available:
                                value=event(next_seen,observed,d,clock)
                            else:
                                options={'wait':lambda dd:event(next_seen,observed,dd,clock)}
                                options.update({cid:(lambda dd,cid=cid:self._ride(
                                    platform.rides[cid],0,observed,clock,dd)) for cid in available})
                                def cheap(score_history):
                                    chosen,_=self.engine.platform(platform,entry,ready,score_history,
                                        available,tuple(cid for cid in next_seen if cid in platform.rides),
                                        clock,choose=self.policy.choose,method='rolling_fixed')
                                    return 'wait' if chosen is None else chosen
                                value=self._choose(options,clock,d,observed,
                                    dict(kind='platform',stop=platform.stop,entry=float(entry),ready=float(ready),
                                         visible=list(available),observed_visible=list(visible)),cheap)
                            lo+=mass*value[0];hi+=mass*value[1];complete &= value[2]
                        else:
                            no_event+=mass
                        consumed+=mass
                        self.stats['chance_outcomes']+=1
                except UnsupportedHistory as error:
                    tail=self._unknown(clock,str(error));break
                chain.append((key,clock,lo,hi,no_event,complete,old))
                if no_event==0:
                    tail=point(self.deadline);break
                h=h.clone();h.observe_platform(cids,entry,clock,{})
                previous_clock=clock
                self.stats['iterative_no_event_steps']+=1
            for key,clock,lo,hi,mass,complete,old in reversed(chain):
                value=(max(clock,lo+mass*tail[0]),min(self.deadline,hi+mass*tail[1]),
                       complete and (mass==0 or tail[2]))
                tail=self._intersect(value,clock,old)
                self.memo[key]=tail
            return tail

        try:
            return self._intersect(event(initial_seen,history,depth,after),lower)
        finally:
            self.context_trips=prior
