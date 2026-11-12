"""Cheap public-history propagation for restored root/tail overlap.

Only six already-selected V29 origin disagreements are replayed. Root actions
remain the original full-menu first vehicle set; each root's schedule-best
path regains all catalog tails. Fixed V31 grid/law retained. A seen root
departure is projected between adjacent operating states exactly in the
existing score-projection convention. An unseen root departure excludes only
operating departure states in [entry, current_clock], retaining pre-entry mass
and cancellation. These posterior vectors are propagated along the SAME trip
to any downstream tail departure. Durations remain cheap marginal means;
there is no integration of first-leg arrival risk or multistage optimization.
Public history is updated only at the same observed events as V31. All actual
future endpoints are confined to offers/outcome replay. This is development.
"""
from copy import copy
import gzip,json
from pathlib import Path
from time import perf_counter
from collections import Counter
from statistics import mean
import numpy as np

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from,ranks_from
from experiments.reconstruction_v25.run import public_offers,execute
from experiments.reconstruction_v31.model import SegmentDurationBelief
from src.recourse_itinerary_router import trip_key
from src.transfer_opportunity import TransferOpportunityScores
from src.full_menu_opportunity import FullMenuOpportunity
from src.plan_waiting_controls import SingleTargetOpportunity,CombinedWaitingOpportunity
from src.risk_bounds import CommitmentBounds

OUT=Path(__file__).parent;SOURCE=Path('experiments/reconstruction_v25')
class History:
    def __init__(self,graph,model,params,roots,entry,clock,seen,known,bounds):
        self.graph,self.model,self.params=graph,model,params
        self.roots={trip_key(graph.connections[cid]):cid for cid in roots}
        self.entry,self.clock,self.seen,self.known=entry,clock,set(seen),known
        self.bounds=bounds;self.cache={};self.projected=[]
    def law_at(self,cid):
        trip,slot=self.model._positions[cid,'departure']
        if (trip,slot) in self.cache:return self.cache[trip,slot]
        law=self.bounds._law(trip)
        root=self.roots.get(trip)
        if root is None:prob=law.marginal(slot)
        else:
            _,root_slot=self.model._positions[root,'departure'];prob=law.marginal(root_slot).copy()
            if root in self.known:
                time=self.known[root];active=np.flatnonzero((~law.canceled[root_slot])&(prob>0));times=law.times[root_slot][active]
                idx=np.searchsorted(times,time);prob[:]=0.
                if idx<len(times) and times[idx]==time:prob[active[idx]]=1.
                elif 0<idx<len(times):
                    w=(time-times[idx-1])/(times[idx]-times[idx-1]);prob[active[idx-1]]=1-w;prob[active[idx]]=w
                    self.projected.append(dict(cid=root,clock=time,left=int(times[idx-1]),right=int(times[idx])))
                else:raise ValueError(('unsupported observed clock',root,time))
            elif root in self.seen:
                keep=(~law.canceled[root_slot])&(law.times[root_slot]>=self.entry)&(law.times[root_slot]<=self.clock)
                prob*=keep;prob/=prob.sum()
            else:
                remove=(~law.canceled[root_slot])&(law.times[root_slot]>=self.entry)&(law.times[root_slot]<=self.clock)
                prob[remove]=0.;prob/=prob.sum()
            if slot>=root_slot:
                for k in range(root_slot,slot):prob=prob@self.bounds._posterior_edge(trip,k)
            else:
                for k in range(root_slot-1,slot-1,-1):
                    joint=law.marginal(k)[:,None]*self.bounds._posterior_edge(trip,k)
                    den=law.marginal(k+1)
                    reverse=np.divide(joint,den[None,:],out=np.zeros_like(joint),where=den[None,:]>0).T
                    prob=prob@reverse
        operating=~law.canceled[slot];out=(law.times[slot][operating],prob[operating],float(prob[~operating].sum()))
        if abs(float(prob.sum())-1)>1e-8:raise AssertionError('Posterior propagation lost mass')
        self.cache[trip,slot]=out;return out

class ConditionalTailScores(TransferOpportunityScores):
    def __init__(self,base,history,root):
        self.__dict__.update(base.__dict__)
        self.history,self.root=history,root
        self._departures,self._tail_scores,self._tables={},{},{}
    def departure_distribution(self,cid):
        if cid==self.root:return super().departure_distribution(cid)
        return self.history.law_at(cid)

class ContextualMenu:
    def __init__(self,bases,history):
        self.compilers={cid:ConditionalTailScores(comp,history,cid) for cid,comp in bases.items()}
    def departure_distribution(self,cid):return self.compilers[cid].departure_distribution(cid)
    def clock_nominal_score(self,cid,clock):return self.compilers[cid].clock_nominal_score(cid,clock)

class Rolling:
    def __init__(self,graph,model,params,root,options,ranks,mode):
        self.graph,self.model,self.params,self.root,self.options,self.ranks,self.mode=graph,model,params,root,options,ranks,mode
        self.bases={cid:TransferOpportunityScores(graph,model,params,root['deadline'],{cid:op}) for cid,op in options.items()}
        self.bounds=CommitmentBounds(model,params,graph,None,root['deadline']);self.known={}
        self.diagnostics=[]
    def choose(self,visible,seen,clock):
        self.known.update({cid:clock for cid in visible})
        history=History(self.graph,self.model,self.params,self.options,self.root['start'],clock,seen,self.known,self.bounds)
        menu=ContextualMenu(self.bases,history)
        one=FullMenuOpportunity(menu,self.ranks,self.root['start'],self.root['start']+30,self.root['deadline'])
        single=SingleTargetOpportunity(one);control=single if self.mode=='single' else CombinedWaitingOpportunity(single)
        choice,receipt=control.choose(visible,seen,clock)
        receipt['history_conditioning']='same-trip downstream departure posterior; marginal ride means'
        receipt['conditioned_tail_marginals']=len(history.cache)
        return choice,receipt

def main():
    began=perf_counter();selection=json.loads((SOURCE/'selection.json').read_text());fit=json.loads((SOURCE/'model_fit.json').read_text())
    sf=json.loads(Path('experiments/reconstruction_v31/segment_fit.json').read_text())
    with gzip.open('experiments/reconstruction_v31/results.json.gz','rt') as f:old={r['root_id']:r for r in json.load(f)['policies']}
    data,_=load_city(SOURCE/'city_transfer_endpoints.json.gz');day='2023-07-17'
    graph,_,ids,_,metadata,_=dated_graph(data,day,'gtfs_station_max',True)
    model=SegmentDurationBelief(graph,duration_fit=fit['duration_fit'],segment_fit=sf,stop_identities={v:k for k,v in ids.items()},public_occurrences=False,grid_step_seconds=20,**fit['increments'])
    params={c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections};rows=[]
    for root in selection['by_day'][day]['roots']:
        if root['od']!=0 or root['root_kind']!='origin':continue
        catalog=options_from(root['catalog_options']);options={r['first_cid']:catalog[r['first_cid']] for r in root['full_options']}
        ranks=ranks_from(graph,[r for r in root['catalog_options'] if r['first_cid'] in options])
        offers=public_offers(metadata,tuple(options),root['start'],root['start']+30,root['deadline'])
        outcomes={}
        for mode in ('single','combined'):
            control=Rolling(graph,model,params,root,options,ranks,mode)
            outcomes[mode]=execute(graph,metadata,root,options,ranks,offers,control)
        rows.append(dict(root_id=root['root_id'],base=old[root['root_id']]['outcomes']['single_target_nominal'],outcomes=outcomes))
        print(root['root_id'],{m:(o['loss'],o['action'],o['trace'][0]['receipt']) for m,o in outcomes.items()},flush=True)
    summary={m:dict(n=len(rows),paired=sum(r['outcomes'][m]['loss'] is not None for r in rows),
        mean_base=mean(r['base']['loss'] for r in rows),mean_new=mean(r['outcomes'][m]['loss'] for r in rows),
        wins=sum(r['base']['loss']>r['outcomes'][m]['loss'] for r in rows),harms=sum(r['base']['loss']<r['outcomes'][m]['loss'] for r in rows)) for m in ('single','combined')}
    report=dict(protocol=__doc__,commit='86ba89c327e5230f1290e8678bebcb589b2465a9',wall=perf_counter()-began,rows=rows,summary=summary)
    (OUT/'probe_conditional_overlap.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(wall=report['wall'],summary=summary),indent=2),flush=True)
if __name__=='__main__':main()
