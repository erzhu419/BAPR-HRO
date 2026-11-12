"""Retrospective coverage diagnostic, NEVER a policy or achieved gain.

Enumerate already-public, legal root departures at which the strong frozen
baseline had not yet left. Compare each SAME first ride under old vs restored
catalog tail rules. Both tail rules are causal first-legal-tail policies;
selection among their observed outcomes in this scan is retrospective only.
"""
import gzip,json
from pathlib import Path
from collections import Counter
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from
from experiments.reconstruction_v25.run import departure,physical_path,route_outcome
from src.recourse_itinerary_router import build_trip_blocks,trip_key
OUT=Path(__file__).parent;SOURCE=Path('experiments/reconstruction_v25')
def main():
    selection=json.loads((SOURCE/'selection.json').read_text())
    with gzip.open('experiments/reconstruction_v31/results.json.gz','rt') as f:pol={r['root_id']:r for r in json.load(f)['policies']}
    data,_=load_city(SOURCE/'city_transfer_endpoints.json.gz');rows=[];counts=Counter();deleted=[];deleted_counts=Counter()
    for day,group in selection['by_day'].items():
        graph,_,_,_,metadata,_=dated_graph(data,day,'gtfs_station_max',True);blocks,positions=build_trip_blocks(graph)
        def describe(cid):
            c=graph.connections[cid];row=metadata[cid]
            return dict(cid=cid,route=c.route,trip=trip_key(c),from_stop=graph.stops[c.dep_stop].name,to_stop=graph.stops[c.arr_stop].name,
                scheduled_departure=c.dep_time,scheduled_arrival=c.arr_time,departure=row['departure'],arrival=row['arrival'])
        for root in group['roots']:
            base=pol[root['root_id']]['outcomes']['single_target_nominal'];baseclock=next((h['clock'] for h in base['trace'] if h['choice'] is not None),root['deadline'])
            full=options_from(root['full_options']);catalog=options_from(root['catalog_options']);roots_by_trip={trip_key(graph.connections[c]):c for c in catalog}
            entry=root['start'];status='arrived'
            if root['prefix']:entry,status=physical_path(metadata,root['prefix'],root['start']+30,root['deadline'])
            if status!='arrived':continue
            ready=entry+(max(30,graph.stops[root['stop']].min_transfer_time) if root['prefix'] else 30)
            for cid,option in catalog.items():
                event=departure(metadata[cid])
                removed=set(option.tails)-(set(full[cid].tails) if cid in full else set())
                for tailcid in removed:
                    target=roots_by_trip.get(trip_key(graph.connections[tailcid]))
                    if target is None:continue
                    relative='downstream_of_root' if positions[tailcid]>positions[target] else 'upstream_of_root' if positions[tailcid]<positions[target] else 'same_point'
                    deleted_counts['removed_root_tail_overlap_occurrences']+=1;deleted_counts[relative]+=1
                    detail=dict(root_id=root['root_id'],local=cid,target_root=target,tail=tailcid,relative=relative,root_legal=False,before_strong_leaves=False,tail_legally_catchable=False,target_already_left_root=False)
                    if event is not None and not event[1] and ready<=event[0]<root['deadline']:
                        detail['root_legal']=True;detail['before_strong_leaves']=event[0]<=baseclock
                        arr,ast=physical_path(metadata,option.first_path,event[0],root['deadline'])
                        tail_event=departure(metadata[tailcid]);target_event=departure(metadata[target])
                        if target_event is not None and not target_event[1]:detail['target_already_left_root']=target_event[0]<event[0]
                        if ast=='arrived' and tail_event is not None and not tail_event[1]:
                            tailready=arr+graph.stops[graph.connections[option.first_path[-1]].arr_stop].min_transfer_time
                            final,finalstatus=physical_path(metadata,option.tails[tailcid],tailready,root['deadline'])
                            detail.update(local_departure=event[0],local_arrival=arr,tailready=tailready,tail_departure=tail_event[0],final=final,finalstatus=finalstatus)
                            detail['tail_legally_catchable']=tailready<=tail_event[0]<root['deadline'] and finalstatus=='arrived'
                    if detail['root_legal'] and detail['tail_legally_catchable']:
                        deleted_counts['legally_catchable_'+relative]+=1
                        if detail['target_already_left_root']:
                            deleted_counts['target_already_left_and_catchable_'+relative]+=1
                            if detail['before_strong_leaves']:deleted_counts['target_already_left_and_catchable_before_strong_leaves_'+relative]+=1
                    deleted.append(detail)
                if event is None or event[1] or not ready<=event[0]<root['deadline']:continue
                counts['legal_catalog_boardings']+=1
                if event[0]>baseclock:continue
                counts['legal_catalog_boardings_before_strong_leaves']+=1
                new,status,tail=route_outcome(graph,metadata,option,event[0],root['deadline'])
                old,oldstatus,oldtail=(route_outcome(graph,metadata,full[cid],event[0],root['deadline']) if cid in full else (None,'filtered_root_option',None))
                if tail is None or trip_key(graph.connections[tail]) not in roots_by_trip:continue
                counts['selected_tail_overlaps_root_trip']+=1
                target=roots_by_trip[trip_key(graph.connections[tail])]
                relative='downstream_of_root' if positions[tail]>positions[target] else 'upstream_of_root' if positions[tail]<positions[target] else 'same_point'
                counts[relative]+=1
                first_arrival,firststatus=physical_path(metadata,option.first_path,event[0],root['deadline'])
                stop=graph.connections[option.first_path[-1]].arr_stop;transferready=first_arrival+graph.stops[stop].min_transfer_time if first_arrival is not None else None
                gain_vs_base=None if new is None or base['loss'] is None else base['loss']-(new-root['start'])
                if gain_vs_base is not None and gain_vs_base>0:counts['retrospectively_better_than_strong']+=1
                rows.append(dict(root_id=root['root_id'],root_kind=root['root_kind'],day=day,od=root['od'],local=describe(cid),target_at_root=describe(target),restored_tail=describe(tail),relative=relative,
                    first_path=option.first_path,tail_path=option.tails[tail],local_arrival=first_arrival,transfer_stop=graph.stops[stop].name,mct=graph.stops[stop].min_transfer_time,transferready=transferready,
                    strong_action=base['action'],strong_clock=baseclock,strong_loss=base['loss'],new_loss=None if new is None else new-root['start'],old_loss=None if old is None else old-root['start'],
                    oldstatus=oldstatus,newstatus=status,retrospective_gain_vs_strong=gain_vs_base,oldtail=oldtail))
    report=dict(protocol=__doc__,counts=dict(counts),rows=rows,deleted_counts=dict(deleted_counts),deleted_occurrences=deleted)
    (OUT/'overlap_witnesses.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
