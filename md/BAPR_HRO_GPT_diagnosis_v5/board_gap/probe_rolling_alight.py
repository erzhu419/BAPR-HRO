"""At each actually reached stop, re-evaluate alight vs stay using mean tails.

The selected first vehicle is frozen to either V31 single-target or the
independently run expanded all-alighting marginal single-target heuristic.
Only the vehicle's already-observed arrival is passed to each decision. Later
same-vehicle travel is predicted by marginal mean remaining durations; tails
use frozen V31 departure laws and first-legal-tail recourse. This is a cheap
causal rolling policy. Separately recorded per-vehicle oracle values use
actual FUTURE outcomes and are only retrospective headroom diagnostics.
"""
import gzip,json
from pathlib import Path
from time import perf_counter
from statistics import mean
from collections import Counter
from probe_board import alternative_options
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v25.run import physical_path,route_outcome
from experiments.reconstruction_v31.model import SegmentDurationBelief
from src.recourse_itinerary_router import build_trip_blocks
from src.transfer_opportunity import TransferOpportunityScores

OUT=Path(__file__).parent;SOURCE=Path('experiments/reconstruction_v25')
def rolling(graph,metadata,root,variants,comps,cid,clock):
    longest=max(variants,key=lambda x:len(x.first_path)).first_path
    trace=[]
    for i in range(1,len(longest)+1):
        prefix=longest[:i]
        now,status=physical_path(metadata,prefix,clock,root['deadline'])
        if status!='arrived':return dict(loss=None if now is None else now-root['start'],status=status,tail=None,trace=trace)
        choices=[]
        for op,comp in zip(variants,comps):
            if len(op.first_path)<i:continue
            future=max(0.,comp._duration(op.first_path)-comp._duration(prefix))
            predicted=now+future
            if not op.tails:value=min(predicted,root['deadline'])
            else:
                stop=graph.connections[op.first_path[-1]].arr_stop
                value=float(comp.tail_value(cid,predicted+graph.stops[stop].min_transfer_time,'clock_nominal'))
            choices.append((value,op))
        value,chosen=min(choices,key=lambda x:(x[0],x[1].first_path))
        trace.append(dict(arrived_cid=prefix[-1],observed_arrival=now,chosen_alight_cid=chosen.first_path[-1],predicted_cost=value,
            alternatives=[dict(value=v,alight_cid=op.first_path[-1]) for v,op in choices]))
        if len(chosen.first_path)==i:
            arrival,status,tail=route_outcome(graph,metadata,chosen,clock,root['deadline'])
            return dict(loss=None if arrival is None else arrival-root['start'],status=status,tail=tail,trace=trace,first_path=chosen.first_path)
    raise AssertionError('No terminal candidate')

def main():
    began=perf_counter();selection=json.loads((SOURCE/'selection.json').read_text());fit=json.loads((SOURCE/'model_fit.json').read_text())
    sf=json.loads(Path('experiments/reconstruction_v31/segment_fit.json').read_text())
    with gzip.open('experiments/reconstruction_v31/results.json.gz','rt') as f:old={r['root_id']:r for r in json.load(f)['policies']}
    expanded={r['root_id']:r for r in json.loads((OUT/'probe_overlap.json').read_text())['rows']}
    data,_=load_city(SOURCE/'city_transfer_endpoints.json.gz');rows=[]
    for day,group in selection['by_day'].items():
        graph,_,ids,_,metadata,_=dated_graph(data,day,'gtfs_station_max',True);blocks=build_trip_blocks(graph)
        model=SegmentDurationBelief(graph,duration_fit=fit['duration_fit'],segment_fit=sf,stop_identities={v:k for k,v in ids.items()},public_occurrences=False,grid_step_seconds=20,**fit['increments'])
        params={c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
        for root in group['roots']:
            variants=alternative_options(graph,blocks,root)
            for source in ('v31_single','expanded_alight_single'):
                base=old[root['root_id']]['outcomes']['single_target_nominal'] if source=='v31_single' else expanded[root['root_id']]['outcomes']['all_alighting_single']
                cid=base['action']
                if cid is None:outcome=dict(loss=base['loss'],status=base['status'],trace=[]);oracle=[]
                else:
                    clock=next(e['clock'] for e in base['trace'] if e['choice']==cid)
                    comps=[TransferOpportunityScores(graph,model,params,root['deadline'],{cid:op}) for op in variants[cid]]
                    outcome=rolling(graph,metadata,root,variants[cid],comps,cid,clock)
                    oracle=[]
                    for op in variants[cid]:
                        arr,st,tail=route_outcome(graph,metadata,op,clock,root['deadline'])
                        oracle.append(dict(loss=None if arr is None else arr-root['start'],status=st,tail=tail,first_path=op.first_path))
                known=[o['loss'] for o in oracle if o['loss'] is not None]
                rows.append(dict(root_id=root['root_id'],root_kind=root['root_kind'],day=day,source=source,base=base,outcome=outcome,
                    retrospective_oracle_best_known=min(known) if known else base['loss'],oracle_unknown=sum(o['loss'] is None for o in oracle),oracle_options=oracle))
    summary={}
    for source in ('v31_single','expanded_alight_single'):
        summary[source]={}
        for kind in ('origin','first_transfer'):
            rs=[r for r in rows if r['source']==source and r['root_kind']==kind]
            ps=[r for r in rs if r['base']['loss'] is not None and r['outcome']['loss'] is not None]
            gs=[r['base']['loss']-r['outcome']['loss'] for r in ps]
            summary[source][kind]=dict(n=len(rs),paired=len(ps),mean_base=mean(r['base']['loss'] for r in ps),mean_new=mean(r['outcome']['loss'] for r in ps),
                mean_gain=mean(gs),wins=sum(g>0 for g in gs),harms=sum(g<0 for g in gs),ties=sum(g==0 for g in gs),
                statuses=dict(Counter(r['outcome']['status'] for r in rs)),
                changes=[dict(root_id=r['root_id'],gain=r['base']['loss']-r['outcome']['loss']) for r in ps if r['base']['loss']!=r['outcome']['loss']],
                oracle_known_gain=sum(r['base']['loss']-r['retrospective_oracle_best_known'] for r in rs if r['base']['loss'] is not None and r['retrospective_oracle_best_known'] is not None),
                oracle_unknown_options=sum(r['oracle_unknown'] for r in rs),
                oracle_improving_roots=[dict(root_id=r['root_id'],gain=r['base']['loss']-r['retrospective_oracle_best_known']) for r in rs if r['base']['loss'] is not None and r['retrospective_oracle_best_known'] is not None and r['base']['loss']>r['retrospective_oracle_best_known']])
    report=dict(protocol=__doc__,wall=perf_counter()-began,summary=summary,rows=rows)
    (OUT/'probe_rolling_alight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(wall=report['wall'],summary=summary),indent=2),flush=True)
if __name__=='__main__':main()
