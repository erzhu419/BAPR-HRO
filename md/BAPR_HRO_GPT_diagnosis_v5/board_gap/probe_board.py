"""Read-only, causal BOARD coverage audit at frozen V31 public choices.

Run from the pinned repository with PYTHONPATH=. python /absolute/path/probe_board.py.
No future endpoint enters a decision: metadata is confined to public-offer
construction and realized outcome evaluation. All model-based scores use the
frozen V31 training fit, and all tail controllers pay the same station MCT.
This is exploratory development reuse, not independent confirmation.
"""
from collections import Counter, defaultdict
from functools import cache
import gzip,json
from pathlib import Path
from statistics import mean
from time import perf_counter

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from
from experiments.reconstruction_v25.run import departure,physical_path,public_offers,route_outcome
from experiments.reconstruction_v31.model import SegmentDurationBelief
from src.recourse_itinerary_router import build_trip_blocks,trip_key
from src.transfer_opportunity import RouteOption,TransferOpportunityScores
from src.full_menu_opportunity import FullMenuOpportunity
from src.plan_waiting_controls import SingleTargetOpportunity,CombinedWaitingOpportunity

OUT=Path(__file__).parent
SOURCE=Path('experiments/reconstruction_v25')

def alternative_options(graph,blocks,root):
    """Enumerate same-vehicle alight points and direct same-vehicle tails.

    Retain the old two-nominal-rescue constraint but remove only the one-best
    alighting collapse and root-tail independence filter. Departures, paths,
    deadlines and pickup permissions are schedule-only.
    """
    trips,positions=blocks
    def paths(cid):
        at=graph.connections[cid].dep_stop; path=[]
        for other in trips[trip_key(graph.connections[cid])][positions[cid]:]:
            c=graph.connections[other]
            if c.dep_stop!=at or c.arr_time>root['deadline']:break
            path.append(other);at=c.arr_stop
            yield at,tuple(path)
    @cache
    def tails(stop):
        result=[]
        for c in graph.get_connections_from(stop):
            if not c.pickup_allowed or not root['scheduled_entry']<=c.dep_time<root['deadline']:continue
            for at,path in paths(c.id):
                if at==root['goal']:
                    result.append(path);break
        return tuple(result)
    result=defaultdict(list)
    # Same first-candidate window as V24. Include catalog first trips only;
    # this preserves the original observed first-offer domain for diagnostics.
    for item in root['catalog_options']:
        cid=item['first_cid']; trip=trip_key(graph.connections[cid])
        for at,path in paths(cid):
            if at==root['goal']:
                result[cid].append(RouteOption(path,{}));break
            ready=graph.connections[path[-1]].arr_time+graph.stops[at].min_transfer_time
            follow={p[0]:p for p in tails(at) if trip_key(graph.connections[p[0]])!=trip}
            legal=[t for t in follow if graph.connections[t].dep_time>=ready]
            if len({trip_key(graph.connections[t]) for t in legal})<2:continue
            result[cid].append(RouteOption(path,follow))
    return result

def with_tail_control(graph,metadata,model,params,root,option,clock,mode):
    arrival,status=physical_path(metadata,option.first_path,clock,root['deadline'])
    result=dict(arrival=arrival,status=status,tail=None,tail_trace=[])
    if status!='arrived' or not option.tails:return result
    stop=graph.connections[option.first_path[-1]].arr_stop
    ready=arrival+graph.stops[stop].min_transfer_time
    opts={cid:RouteOption(path,{}) for cid,path in option.tails.items()}
    ranks={cid:(graph.connections[p[-1]].arr_time-graph.connections[cid].dep_time,cid) for cid,p in option.tails.items()}
    offers=public_offers(metadata,tuple(opts),arrival,ready,root['deadline'])
    if offers is None:
        result.update(arrival=None,status='unresolved_tail_departure');return result
    compiler=TransferOpportunityScores(graph,model,params,root['deadline'],opts)
    one=FullMenuOpportunity(compiler,ranks,arrival,ready,root['deadline'])
    single=SingleTargetOpportunity(one)
    control=single if mode=='single' else CombinedWaitingOpportunity(single)
    result.update(arrival=root['deadline'],status='no_tail_offer')
    for event in offers:
        selected,receipt=control.choose(event['visible'],event['seen'],event['clock'])
        result['tail_trace'].append(dict(**event,choice=selected,receipt=receipt))
        if selected is not None:
            arrival,status=physical_path(metadata,opts[selected].first_path,event['clock'],root['deadline'])
            result.update(arrival=arrival,status=status,tail=selected);break
    return result

def summarize(rows,method,kind=None,source=None):
    rs=[r for r in rows if (kind is None or r['root_kind']==kind) and (source is None or r['source']==source)]
    pairs=[r for r in rs if r['base']['loss'] is not None and r['enhanced'][method]['loss'] is not None]
    gains=[r['base']['loss']-r['enhanced'][method]['loss'] for r in pairs]
    return dict(n=len(rs),paired=len(pairs),known=sum(r['enhanced'][method]['loss'] is not None for r in rs),
        mean_base=mean(r['base']['loss'] for r in pairs) if pairs else None,
        mean_new=mean(r['enhanced'][method]['loss'] for r in pairs) if pairs else None,
        mean_gain=mean(gains) if gains else None,wins=sum(x>0 for x in gains),harms=sum(x<0 for x in gains),ties=sum(x==0 for x in gains),
        statuses=dict(Counter(r['enhanced'][method]['status'] for r in rs)),
        changed=[dict(root_id=r['root_id'],gain=r['base']['loss']-r['enhanced'][method]['loss'],
                      base=r['base']['loss'],new=r['enhanced'][method]['loss']) for r in pairs if r['base']['loss']!=r['enhanced'][method]['loss']])

def main():
    began=perf_counter()
    selection=json.loads((SOURCE/'selection.json').read_text())
    fit=json.loads((SOURCE/'model_fit.json').read_text())
    sf=json.loads(Path('experiments/reconstruction_v31/segment_fit.json').read_text())
    with gzip.open('experiments/reconstruction_v31/results.json.gz','rt') as f:
        policies={r['root_id']:r for r in json.load(f)['policies']}
    data,_=load_city(SOURCE/'city_transfer_endpoints.json.gz')
    rows=[]; timings={}; counts=Counter()
    for day,group in selection['by_day'].items():
        start=perf_counter()
        graph,_,ids,_,metadata,_=dated_graph(data,day,'gtfs_station_max',True)
        blocks=build_trip_blocks(graph)
        model=SegmentDurationBelief(graph,duration_fit=fit['duration_fit'],segment_fit=sf,
            stop_identities={v:k for k,v in ids.items()},public_occurrences=False,grid_step_seconds=20,**fit['increments'])
        params={c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
        for root in group['roots']:
            fixed=options_from(root['full_options']);catalog=options_from(root['catalog_options'])
            variants=alternative_options(graph,blocks,root)
            per_variant={}
            for cid,items in variants.items():
                per_variant[cid]=[(op,TransferOpportunityScores(graph,model,params,root['deadline'],{cid:op})) for op in items]
            for source in ('single_target_nominal','combined_wait_nominal','full_first_offer'):
                base=policies[root['root_id']]['outcomes'][source]
                cid=base['action']; enhanced={}
                if cid is None:
                    for name in ('catalog_first_tail','fixed_tail_single','fixed_tail_combined','alternative_alight_mean_first_tail','alternative_alight_mean_tail_single'):
                        enhanced[name]=dict(loss=base['loss'],status=base['status'],tail=base.get('tail'),first_path=None)
                    rows.append(dict(root_id=root['root_id'],day=day,root_kind=root['root_kind'],od=root['od'],source=source,base=base,enhanced=enhanced));continue
                clock=next(e['clock'] for e in base['trace'] if e['choice']==cid)
                arrival,status,tail=route_outcome(graph,metadata,fixed[cid],clock,root['deadline'])
                assert (None if arrival is None else arrival-root['start'],status,tail)==(base['loss'],base['status'],base['tail'])
                counts['frozen_physical_outcomes_reproduced']+=1
                arrival,status,tail=route_outcome(graph,metadata,catalog[cid],clock,root['deadline'])
                enhanced['catalog_first_tail']=dict(loss=None if arrival is None else arrival-root['start'],status=status,tail=tail,first_path=catalog[cid].first_path)
                for mode in ('single','combined'):
                    outcome=with_tail_control(graph,metadata,model,params,root,fixed[cid],clock,mode)
                    enhanced['fixed_tail_'+mode]=dict(loss=None if outcome['arrival'] is None else outcome['arrival']-root['start'],
                        status=outcome['status'],tail=outcome['tail'],first_path=fixed[cid].first_path,tail_trace=outcome['tail_trace'])
                scores=[(comp.clock_nominal_score(cid,clock),op) for op,comp in per_variant[cid]]
                cost,op=min(scores,key=lambda p:(p[0],p[1].first_path))
                for name,mode in [('alternative_alight_mean_first_tail',None),('alternative_alight_mean_tail_single','single')]:
                    if mode is None:
                        arrival,status,tail=route_outcome(graph,metadata,op,clock,root['deadline'])
                        outcome=dict(arrival=arrival,status=status,tail=tail,tail_trace=[])
                    else:outcome=with_tail_control(graph,metadata,model,params,root,op,clock,mode)
                    enhanced[name]=dict(loss=None if outcome['arrival'] is None else outcome['arrival']-root['start'],
                        status=outcome['status'],tail=outcome['tail'],first_path=op.first_path,
                        predicted_score=cost,variant_scores=[dict(score=x,first_path=y.first_path) for x,y in scores],tail_trace=outcome['tail_trace'])
                rows.append(dict(root_id=root['root_id'],day=day,root_kind=root['root_kind'],od=root['od'],source=source,base=base,enhanced=enhanced))
            print(root['root_id'],len(variants),sum(map(len,variants.values())),flush=True)
        timings[day]=perf_counter()-start
    methods=tuple(rows[0]['enhanced'])
    summary={source:{kind:{m:summarize(rows,m,kind,source) for m in methods} for kind in ('origin','first_transfer')}
             for source in ('single_target_nominal','combined_wait_nominal','full_first_offer')}
    report=dict(commit='86ba89c327e5230f1290e8678bebcb589b2465a9',protocol=__doc__,counts=dict(counts),timings=timings,total_wall=perf_counter()-began,summary=summary,rows=rows)
    (OUT/'probe_board.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(counts=dict(counts),timings=timings,total_wall=report['total_wall'],summary=summary),indent=2),flush=True)

if __name__=='__main__':main()
