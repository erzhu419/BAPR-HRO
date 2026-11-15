"""Post-hoc root4 mechanism probe, NOT a new general candidate algorithm.

Three cheap W arms: original actions/observations; original actions plus local
visibility of cid145 at Stadelhofen; add its public timetable through path
145->182->193 plus the same expanded visibility. No original files changed.
The candidate is retrospectively selected and therefore cannot support a
population performance claim or unchanged-domain algorithm comparison.
"""
import gzip,json,os,sys
from pathlib import Path
from time import perf_counter,process_time
REPO=Path(os.environ['BAPR_HRO_REPO']).resolve();sys.path.insert(0,str(REPO))
from experiments.reconstruction_v33.nodes import root_ride,Ride
from experiments.reconstruction_v34.discover import context,read
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.run import inputs,relevant_frames,setup
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v50.run import actions
from experiments.reconstruction_v51.run import PolicyExecutor
from frozen_live_probe import event_signature
HERE=Path(__file__).resolve().parent


def main():
    tick,cpu=perf_counter(),process_time()
    _,data,messages,fit,_=inputs()
    emission=read(REPO/'experiments/reconstruction_v35/fitted_emission.json')
    menu=read(REPO/'experiments/reconstruction_v50/menus/004.json.gz');root=menu['wide_root']
    graph,_,_,mapping,metadata,_=context(data,root['day'])
    preparation=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
    old=read(REPO/'experiments/reconstruction_v51/roots/004_cheap.json.gz')
    path=(145,182,193)
    assert all(graph.connections[cid].trip_id==graph.connections[path[0]].trip_id for cid in path)
    assert all(graph.connections[a].arr_stop==graph.connections[b].dep_stop for a,b in zip(path,path[1:]))
    assert graph.connections[path[-1]].arr_stop==root['goal']
    assert graph.connections[path[0]].pickup_allowed
    assert all(metadata[cid][kind]['status']=='REAL' for cid in path for kind in ('departure','arrival'))
    assert len({metadata[cid][kind]['official_journey'] for cid in path for kind in ('departure','arrival')})==1
    base_node=root_ride(graph,root);base_contract=ObservationContract.from_menu(base_node)
    assert 145 not in base_node.exits[111].rides and 145 not in base_contract.cids_at(2)
    rows=[];target=HERE/'root4_menu_receipts';target.mkdir(exist_ok=True)
    for arm in ('oldM_oldO','oldM_plusO','plusM_plusO'):
        tick,cpu=perf_counter(),process_time()
        model,params=setup(graph,fit);node=root_ride(graph,root)
        old_trips=node.trips(model)
        station_map=dict(base_contract.station_to_cids)
        if arm!='oldM_oldO':station_map[2]=tuple(sorted((*station_map[2],145)))
        if arm=='plusM_plusO':node.exits[111].rides[145]=Ride(145,path,{193:None})
        assert node.trips(model)==old_trips  # S5 was already present at Hardbruecke.
        observations=ObservationContract(station_map)
        frames=relevant_frames(messages['dates'][root['day']]['frames'],old_trips,mapping)
        history=PhysicalHistory(model,params,fit=emission,frames=frames,source_to_local=mapping)
        planner=MatchedContinuationPlanner(graph,model,params,root['deadline'])
        executor=PolicyExecutor(graph,planner,root['deadline'],label='cheap',rollout_limits={},
            root_index=4,cpu_start=cpu,observation_contract=observations)
        outcome=executor.run(node,root['start'],history,ArchivedWorld(metadata))
        cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick,
            shared_preparation=preparation,original_menu_generation=menu['cost'])
        assert executor.states[0]['public_history']==old['states'][0]['public_history']
        validation=dict(first_public_history_equal=True,remote_trip_set_equal=True)
        if arm=='oldM_oldO':
            validation.update(loss_equal=outcome['loss']==old['outcome']['loss'],
                status_equal=outcome['status']==old['outcome']['status'],
                actions_equal=actions(outcome)==old['actions'],
                all_physical_public_trace_fields_equal=event_signature(outcome)==event_signature(old['outcome']))
            assert all(validation.values())
        receipt=dict(arm=arm,outcome=outcome,states=executor.states,publications=executor.publications,
            actions=actions(outcome),cost=cost,validation=validation,observation=observations.describe(),
            first_station_action_menu=list(node.exits[111].rides),added_path=list(path) if arm=='plusM_plusO' else None)
        outfile=target/f'{arm}.json.gz'
        with gzip.open(outfile,'wt') as f:json.dump(receipt,f,ensure_ascii=False)
        rows.append(dict(arm=arm,loss=outcome['loss'],status=outcome['status'],actions=actions(outcome),
            cost=cost,validation=validation,first_station_action_menu=receipt['first_station_action_menu'],
            source=str(outfile.relative_to(HERE))))
        print(arm,outcome['loss'],actions(outcome),flush=True)
    result=dict(commit='a2b82be9c3fc6b61e2a5826d7cd8691d39ff4491',root_index=4,
        scope='Post-hoc mechanism probe on one selected development root. Changes the declared action/observation domain; not an unchanged-V55 comparison and not a general route-discovery rule.',
        intervention=dict(added_local_observation=[dict(stop=2,station=graph.stops[2].name,cid=145)],
            added_action_path=[145,182,193],declared_destination_exit=193,
            no_added_platform_departure_at_HB_or_Hardbruecke=True,
            remote_message_trip_set_unchanged=True,root_public_history_unchanged=True,
            timetable_chain_valid=True,complete_actual_endpoints=True,same_actual_vehicle=True),
        contract='Original P0/W/parameters/MCT/archived physical world and ideal logical publisher; CPU never shifts physical clock.',
        rows=rows)
    (HERE/'root4_menu_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
