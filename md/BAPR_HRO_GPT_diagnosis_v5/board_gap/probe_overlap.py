"""Causal cheap-controller probe with root/tail overlap restored.

The score is explicitly a marginal mean heuristic, not a certified conditional
value: each possible root ride is scored separately so its tail menu may include
another root vehicle. The policy sees exactly public local receipts; it never
uses actual future endpoint data to select an action. Outcome replay reuses V25.
The extended menu is schedule-only and frozen. All dates are development reuse.
"""
import gzip,json
from pathlib import Path
from statistics import mean
from time import perf_counter
from collections import Counter

from probe_board import alternative_options
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from,ranks_from
from experiments.reconstruction_v25.run import physical_path,public_offers,route_outcome,execute
from experiments.reconstruction_v31.model import SegmentDurationBelief
from src.recourse_itinerary_router import build_trip_blocks,trip_key
from src.transfer_opportunity import TransferOpportunityScores
from src.full_menu_opportunity import FullMenuOpportunity
from src.plan_waiting_controls import SingleTargetOpportunity,CombinedWaitingOpportunity

OUT=Path(__file__).parent;SOURCE=Path('experiments/reconstruction_v25')
class MarginalOptionMenu:
    def __init__(self,graph,model,params,deadline,options):
        self.compilers={cid:[(op,TransferOpportunityScores(graph,model,params,deadline,{cid:op})) for op in variants]
                        for cid,variants in options.items()}
        self.cache={}
    def departure_distribution(self,cid):return self.compilers[cid][0][1].departure_distribution(cid)
    def choose_option(self,cid,clock):
        key=cid,clock
        if key not in self.cache:
            self.cache[key]=min(((comp.clock_nominal_score(cid,clock),op) for op,comp in self.compilers[cid]),key=lambda p:(p[0],p[1].first_path))
        return self.cache[key]
    def clock_nominal_score(self,cid,clock):return self.choose_option(cid,clock)[0]

class ChosenOptions:
    def __init__(self,compiler):self.compiler=compiler;self.clock=None
    def __bool__(self):return bool(self.compiler.compilers)
    def __getitem__(self,cid):return self.compiler.choose_option(cid,self.clock)[1]
class WithClock:
    def __init__(self,control,options):self.control=control;self.options=options
    def choose(self,visible,seen,clock):
        self.options.clock=clock
        return self.control.choose(visible,seen,clock)

def summarize(rows,method,kind):
    rs=[r for r in rows if r['root_kind']==kind]
    ps=[r for r in rs if r['base']['loss'] is not None and r['outcomes'][method]['loss'] is not None]
    gs=[r['base']['loss']-r['outcomes'][method]['loss'] for r in ps]
    return dict(n=len(rs),paired=len(ps),mean_base=mean(r['base']['loss'] for r in ps),mean_new=mean(r['outcomes'][method]['loss'] for r in ps),
        mean_gain=mean(gs),wins=sum(g>0 for g in gs),harms=sum(g<0 for g in gs),ties=sum(g==0 for g in gs),
        known=sum(r['outcomes'][method]['loss'] is not None for r in rs),statuses=dict(Counter(r['outcomes'][method]['status'] for r in rs)),
        changes=[dict(root_id=r['root_id'],gain=r['base']['loss']-r['outcomes'][method]['loss'],base=r['base']['loss'],new=r['outcomes'][method]['loss'],
                      base_action=r['base']['action'],new_action=r['outcomes'][method]['action']) for r in ps if r['base']['loss']!=r['outcomes'][method]['loss']])

def main():
    start=perf_counter(); selection=json.loads((SOURCE/'selection.json').read_text());fit=json.loads((SOURCE/'model_fit.json').read_text())
    sf=json.loads(Path('experiments/reconstruction_v31/segment_fit.json').read_text())
    with gzip.open('experiments/reconstruction_v31/results.json.gz','rt') as f:policies={r['root_id']:r for r in json.load(f)['policies']}
    data,_=load_city(SOURCE/'city_transfer_endpoints.json.gz');rows=[];witnesses=[]
    for day,group in selection['by_day'].items():
        graph,_,ids,_,metadata,_=dated_graph(data,day,'gtfs_station_max',True);blocks=build_trip_blocks(graph)
        model=SegmentDurationBelief(graph,duration_fit=fit['duration_fit'],segment_fit=sf,stop_identities={v:k for k,v in ids.items()},public_occurrences=False,grid_step_seconds=20,**fit['increments'])
        params={c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
        for root in group['roots']:
            entry,status=root['start'],'arrived'
            if root['prefix']:entry,status=physical_path(metadata,root['prefix'],root['start']+30,root['deadline'])
            if status!='arrived':raise RuntimeError('Unexpected unresolved prefix')
            ready=entry+(max(30,graph.stops[root['stop']].min_transfer_time) if root['prefix'] else 30)
            base=policies[root['root_id']]['outcomes']['single_target_nominal'];outcomes={}
            full=options_from(root['full_options']);catalog=options_from(root['catalog_options'])
            menus={'same_roots_all_tails':{cid:[catalog[cid]] for cid in full},'catalog':{cid:[op] for cid,op in catalog.items()},'all_alighting':alternative_options(graph,blocks,root)}
            for label,options in menus.items():
                compiler=MarginalOptionMenu(graph,model,params,root['deadline'],options)
                ranks=ranks_from(graph,[r for r in root['catalog_options'] if r['first_cid'] in options])
                offers=public_offers(metadata,tuple(options),entry,ready,root['deadline'])
                if options:
                    reference=FullMenuOpportunity(compiler,ranks,entry,ready,root['deadline'])
                    single=SingleTargetOpportunity(reference)
                for mode in ('single','combined'):
                    chosen=ChosenOptions(compiler)
                    control=None if not options else WithClock(single if mode=='single' else CombinedWaitingOpportunity(single),chosen)
                    outcome=execute(graph,metadata,root,chosen,ranks,offers,control)
                    if outcome['action'] is not None:outcome['first_path']=chosen[outcome['action']].first_path
                    outcomes[label+'_'+mode]=outcome
            rows.append(dict(root_id=root['root_id'],root_kind=root['root_kind'],day=day,od=root['od'],base=base,outcomes=outcomes))
            if root['root_id'] in ('2023-07-17_0_29400_origin','2023-07-17_0_31200_origin'):
                baseline=policies[root['root_id']]['outcomes'];local=baseline['full_first_offer']['action'];target=base['action'];op=catalog[local]
                clock=baseline['full_first_offer']['trace'][0]['clock'];arr,status,tail=route_outcome(graph,metadata,op,clock,root['deadline'])
                relevant=set(op.first_path);relevant.update(op.tails.get(tail,()));relevant.update(full[target].first_path)
                witnesses.append(dict(root_id=root['root_id'],root=root,local=local,target=target,restored_tail=tail,local_clock=clock,
                    local_tail_arrival=arr,local_tail_loss=arr-root['start'] if arr is not None else None,strong_loss=base['loss'],
                    same_target_tail_trip=trip_key(graph.connections[target])==trip_key(graph.connections[tail]),
                    connections={cid:dict(id=cid,route=graph.connections[cid].route,trip=trip_key(graph.connections[cid]),
                        dep_stop=graph.stops[graph.connections[cid].dep_stop].name,arr_stop=graph.stops[graph.connections[cid].arr_stop].name,
                        scheduled_departure=graph.connections[cid].dep_time,scheduled_arrival=graph.connections[cid].arr_time,actual=metadata[cid]) for cid in sorted(relevant)},
                    base=base,expanded=outcomes))
            print(root['root_id'],{m:o['loss'] for m,o in outcomes.items()},flush=True)
    summary={kind:{m:summarize(rows,m,kind) for m in rows[0]['outcomes']} for kind in ('origin','first_transfer')}
    report=dict(protocol=__doc__,commit='86ba89c327e5230f1290e8678bebcb589b2465a9',wall=perf_counter()-start,summary=summary,rows=rows,witnesses=witnesses)
    (OUT/'probe_overlap.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(wall=report['wall'],summary=summary),indent=2),flush=True)
if __name__=='__main__':main()
