"""Ordinary exact event DP retaining shared-current-trip observations.

This extends the causal full-pi reference, not the online policy. The small
interface is constructed from every original scoring tail, including tails
after a physical destination. Root P0 conditioning supports reverse slots.
"""
from time import process_time

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v33.execute import Executor
from experiments.reconstruction_v46.oracle import (
    FullPolicyOracle, physical_path, terminal, structural, point, unknown, receipt,
)


def shared_structure(model,platform):
    roots={cid:model._positions[cid,'departure'][0] for cid in platform.rides}
    reasons=[];interface=set();references=[]
    if len(set(roots.values()))!=len(roots):reasons.append('current_root_trips_not_distinct')
    for cid,ride in platform.rides.items():
        positions=[model._positions[c,kind] for c in ride.path for kind in ('departure','arrival')]
        if (any(trip!=roots[cid] for trip,_ in positions)
                or any(b[1]!=a[1]+1 for a,b in zip(positions,positions[1:]))):
            reasons.append(f'noncontiguous_ride:{cid}')
        reachable=set(physical_path(ride))
        for exit_cid,tail in ride.exits.items():
            if tail is None:continue
            trips=tail.trips(model)
            if roots[cid] in trips:reasons.append(f'own_trip_reappears:{cid}:{exit_cid}')
            for other,trip in roots.items():
                if trip in trips:
                    interface.add(other)
                    references.append(dict(root_cid=other,via_candidate=cid,exit_cid=exit_cid,
                                           physical_reachable=exit_cid in reachable))
            if exit_cid not in reachable:continue
            if any(not terminal(child) for child in tail.rides.values()):
                reasons.append(f'more_than_one_transfer:{cid}:{exit_cid}')
            else:
                leaf=shared_structure(model,tail)
                reasons.extend(leaf['reasons'])
    return dict(eligible=not reasons,reasons=reasons,candidates=len(roots),
                interface_cids=sorted(interface),references=references,
                interface_scope='All original recursive scoring tails; no outcome selection')


class SharedObservationOracle(FullPolicyOracle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.policy=Executor(self.graph,self.planner,self.deadline,method='rolling_fixed',waiting='combined',trace=False)
        self.shared_cache={}
        self.stats.update(shared_platform_solves=0,shared_state_queries=0,
            shared_state_hits=0,shared_states=0,shared_policy_compiles=0,
            shared_interfaces_max=0,shared_budget_states=0,shared_board_calls=0)

    def _available(self):
        return not ((self._compile_limit is not None and self.stats['charged_compiles']>=self._compile_limit)
            or (self._cpu_limit is not None and process_time()-self.cpu_start>=self._cpu_limit))

    def event_history(self,platform,base,entry,clock,remaining,visible,seen_times):
        """Drop only new irrelevant seen clocks; preserve all entry evidence."""
        h=base.clone()
        for cid,when in seen_times:
            h.observe(cid,'departure',when)
        for cid in visible:
            h.observe(cid,'departure',clock)
        unseen=tuple(cid for cid in platform.rides if cid in remaining)
        h.observe_platform(unseen,entry,clock,{})
        return h

    def boarding_history(self,base,event,trip):
        """Represent all current departure values with one conditional table.

        Restore only this visit's new evidence on the selected trip; the row
        d supplies its actual departure. Every other shared observation stays.
        """
        h=event.clone()
        if trip in base._records:h._records[trip]=base._records[trip]
        else:h._records.pop(trip,None)
        h._laws.pop(trip,None)
        for name in ('_physical_bases','_physical_processed','_physical_anchors'):
            target=getattr(h,name);source=getattr(base,name)
            if trip in source:target[trip]=source[trip]
            else:target.pop(trip,None)
        return h

    def boarding_value(self,ride,clock,base,event):
        self.stats['shared_board_calls']+=1
        trip,slot=self.model._positions[ride.first_cid,'departure']
        h=self.boarding_history(base,event,trip)
        clocks,values=self.ride_marks(ride,h)
        law=self.root_law.law(trip,h)
        selected=np.flatnonzero((clocks==clock)&~law.canceled[slot])
        if len(selected)!=1:raise ValueError('No unique operating selected departure')
        return values[selected[0]]

    def _platform(self,platform,entry,history):
        if structural(self.model,platform)['eligible']:
            return super()._platform(platform,entry,history)
        key=self._key(platform,history),float(entry)
        if key in self.platform_cache:
            self.stats['platform_hits']+=1
            return self.platform_cache[key]
        if entry>=self.deadline:return point(self.deadline)
        check=shared_structure(self.model,platform)
        if not check['eligible']:
            self.stats['structure_refusals']+=1
            return unknown(entry,self.deadline)
        if not self._spend('platform_solves'):return unknown(entry,self.deadline)
        self.stats['shared_platform_solves']+=1
        interface=set(check['interface_cids'])
        self.stats['shared_interfaces_max']=max(self.stats['shared_interfaces_max'],len(interface))
        cids=tuple(platform.rides);nc=len(cids);full=(1<<nc)-1
        physical={cid:self.root_law.departure_distribution(cid,history) for cid in cids}
        times=np.array(sorted({float(t) for cid in cids for t,p in zip(*physical[cid][:2],strict=True)
            if p>0 and entry<=t<self.deadline}))
        hazards=np.zeros((len(times),nc))
        for i,cid in enumerate(cids):
            clocks,p,cancel=physical[cid]
            tails=np.r_[np.cumsum(p[::-1])[::-1],0.]
            absent=cancel+p[clocks<entry].sum()
            survival=absent+tails[np.searchsorted(clocks,times,side='left')]
            included=(p>0)&(clocks>=entry)&(clocks<self.deadline)
            np.add.at(hazards[:,i],np.searchsorted(times,clocks[included]),p[included])
            np.divide(hazards[:,i],survival,out=hazards[:,i],where=survival>0)
        ready=entry+self.graph.stops[platform.stop].min_transfer_time
        bits=[[i for i in range(nc) if mask&(1<<i)] for mask in range(1<<nc)]
        subsets=[]
        for active in bits:
            choices=np.array([0],dtype=int)
            for i in active:choices=np.r_[choices,choices|(1<<i)]
            subsets.append(choices)

        def value(j,mask,z):
            self.stats['shared_state_queries']+=1
            state_key=key,j,mask,z
            if state_key in self.shared_cache:
                self.stats['shared_state_hits']+=1
                return self.shared_cache[state_key]
            if j==len(times) or not mask:return point(self.deadline)
            if not self._available():
                self.stats['shared_budget_states']+=1
                return unknown(times[j],self.deadline)
            self.stats['shared_states']+=1
            probability=np.ones(1)
            for i in bits[mask]:
                h=hazards[j,i];probability=np.r_[probability*(1-h),probability*h]
            result=np.zeros(3)
            # Fixed input-only order: largest physical probability first.
            order=sorted(range(len(probability)),key=lambda k:(-probability[k],int(subsets[mask][k])))
            for k in order:
                mass=float(probability[k])
                if mass<=0:continue
                departed=int(subsets[mask][k]);left=mask^departed;clock=float(times[j])
                visible=tuple(cids[i] for i in bits[departed])
                additions=tuple((cid,clock) for cid in visible if cid in interface)
                next_z=tuple(sorted((*z,*additions)))
                if not departed or clock<ready:
                    continuation=value(j+1,left,next_z)
                elif not self._spend('shared_policy_compiles'):
                    continuation=unknown(clock,self.deadline)
                else:
                    remaining={cids[i] for i in bits[left]}
                    h=self.event_history(platform,history,entry,clock,remaining,visible,next_z)
                    seen=tuple(cid for cid in cids if cid not in remaining)
                    try:
                        chosen,_=self.engine.platform(platform,entry,ready,h,visible,seen,clock,
                            choose=self.policy.choose,method='rolling_fixed')
                    except UnsupportedHistory:
                        self.stats['unsupported_scores']+=1
                        continuation=unknown(clock,self.deadline)
                    else:
                        continuation=(value(j+1,left,next_z) if chosen is None else
                            self.boarding_value(platform.rides[chosen],clock,history,h))
                result+=mass*continuation
                self.stats['positive_event_branches']+=1
            if result[2]==0:self.shared_cache[state_key]=result.copy()
            return result

        answer=value(0,full,())
        if answer[2]==0:self.platform_cache[key]=answer.copy()
        return answer

    def platform(self,platform,entry,history=None):
        self._prepare(platform)
        history=self.snapshot if history is None else history
        check=shared_structure(self.model,platform)
        result=receipt(self._platform(platform,float(entry),history),
            None if check['eligible'] else ';'.join(check['reasons']))
        return dict(**result,structure=check,scope='P0 full-pi event DP with shared observation interface',
                    counters=dict(self.stats),cost=self.cost())
