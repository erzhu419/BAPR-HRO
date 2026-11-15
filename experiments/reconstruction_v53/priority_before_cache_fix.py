"""Bounded impact screening before depth-first chance expansion.

A skipped branch keeps its full physical interval and probability. The
cutoff is a scheduling heuristic, not a bound on total error or a change of
the finite law. Reported root intervals remain the actual error bounds.
"""
import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v52.reference import point
from experiments.reconstruction_v53.reference import CompressedReference
from experiments.reconstruction_v53.relevance import history_key


class ImpactReference(CompressedReference):
    def __init__(self,*args,impact_cutoff=.01,**kwargs):
        super().__init__(*args,**kwargs)
        self.impact_cutoff=float(impact_cutoff)
        self.reach=1.
        self.stats.update(impact_pruned_branches=0)

    def _small(self,clock,reach=None):
        return ((self.reach if reach is None else reach)*max(0.,self.deadline-clock)
                <=self.impact_cutoff and self.impact_cutoff>0)

    def _pruned(self,clock):
        self.stats['impact_pruned_branches']+=1
        self.reasons['impact_cutoff']+=1
        return min(float(clock),self.deadline),self.deadline,False

    def _chance(self,call,mass,clock):
        before=self.reach
        self.reach*=mass
        try:
            return self._pruned(clock) if self._small(clock) else call()
        finally:
            self.reach=before

    def _expect(self,outcomes,clock):
        lower=upper=consumed=0.
        complete=True
        for mass,continuation in outcomes:
            if mass<=0:
                continue
            if not self._available():
                remainder=max(0.,1.-consumed)
                self.stats['budget_unknowns']+=1
                self.reasons['budget_chance_remainder']+=1
                return (lower+remainder*min(clock,self.deadline),
                        upper+remainder*self.deadline,False)
            lo,hi,exact=self._chance(continuation,mass,clock)
            lower+=mass*lo;upper+=mass*hi;consumed+=mass;complete &= exact
            self.stats['chance_outcomes']+=1
        return max(min(clock,self.deadline),lower),min(self.deadline,upper),complete

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

        def event(seen,h,d,previous_clock,reach):
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
                key=('impact_event',self._structure(platform),float(entry),clock,
                     remaining,history_key(h,relevant),d)
                old=self.memo.get(key)
                if old is not None:
                    self.stats['cache_hits']+=1
                    if old[2] or not self._available() or self._small(clock,reach):
                        tail=old;break
                    self.stats['partial_cache_hits']+=1
                if self._small(clock,reach):
                    tail=self._pruned(clock);break
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
                            branch_reach=reach*mass
                            if self._small(clock,branch_reach):
                                value=self._pruned(clock)
                            else:
                                before=self.reach;self.reach=branch_reach
                                try:
                                    observed=h.clone()
                                    observed.observe_platform(cids,entry,clock,{cid:clock for cid in visible})
                                    next_seen=tuple(sorted(set(seen)|set(visible)))
                                    available=tuple(cid for cid in platform.rides if cid in visible)
                                    if clock<ready or not available:
                                        value=event(next_seen,observed,d,clock,branch_reach)
                                    else:
                                        options={'wait':lambda dd:event(next_seen,observed,dd,clock,branch_reach)}
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
                                finally:
                                    self.reach=before
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
                previous_clock=clock;reach*=no_event
                self.stats['iterative_no_event_steps']+=1
            for key,clock,lo,hi,mass,complete,old in reversed(chain):
                value=(max(clock,lo+mass*tail[0]),min(self.deadline,hi+mass*tail[1]),
                       complete and (mass==0 or tail[2]))
                tail=self._intersect(value,clock,old)
                self.memo[key]=tail
            return tail

        try:
            return self._intersect(event(initial_seen,history,depth,after,self.reach),lower)
        finally:
            self.context_trips=prior

    def solve(self,*args,**kwargs):
        result=super().solve(*args,**kwargs)
        result.update(implementation='impact_conditional',impact_cutoff=self.impact_cutoff,
            impact_scope='Chance-path influence heuristic only. Every skipped probability retains a physical interval. The cutoff is not the total error, visitation under the optimal policy, or a probability filter.')
        return result
