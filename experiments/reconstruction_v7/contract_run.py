"""Causal contract counterexample and local decisions from archived city histories."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from src.background_policy import IssuedPolicy, execute_background
from src.committed_transit import build_committed_model, CommittedState
from src.information_planner import ExactPlanner
from src.platform_transit import build_platform_model, PlatformState
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import TransitGraph, Stop


def actual_rule(primary, acceptable, trip_by_cid):
    def action(state):
        if state.current_trip is not None and trip_by_cid[primary] != state.current_trip:
            return ('alight',)
        eligible = [cid for cid in state.visible_connections if cid in acceptable]
        return ('board', min(eligible)) if eligible else ('wait',)
    return IssuedPolicy(action, name='frozen_accepted_set')


def counterexample():
    from src.transit_graph import Connection
    g = TransitGraph()
    for sid in range(2):
        g.add_stop(Stop(sid, str(sid), min_transfer_time=0))
    for trip in ('A', 'B'):
        g.add_connection(Connection(-1, 'R', trip, 0, 1, 20, 30))
    bank = [(0.5, JourneyScenario({cid: VehicleEvent(t-20, False, t, t+10)
            for cid, t in enumerate(times)}, minute=60, publications={})) for times in ((20,40),(40,20))]
    return dict(committed=ExactPlanner(build_committed_model(g, bank, 0, 1, 0, 120)).solve().value,
        local_event=ExactPlanner(build_platform_model(g, bank, 0, 1, 0, 120)).solve().value)


def main():
    data, _ = load_city(ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
    fit = fit_training(data, retain_no_pickup=True)
    archive = json.loads((ROOT / 'experiments/reconstruction_v4/city_replay_results.json').read_text())
    journeys = [r for r in archive['rows'] if r['method'] == 'evolving_rollout']
    prepared = {day: dated_graph(data, day, 'gtfs_station_max', True) for day in data['holdout_dates']}
    protocol = dict(scope='Posthoc V1–V6 development: frozen first-action local next-stop decisions, not whole journey rankings.',
        cohort='All archived evolving_rollout decision states with a selected connection. The target is that connection next stop.',
        acceptable='Same next stop, legal timetable menu; root nominal expected arrival <= selected primary mean+30s. No future actual used.',
        posture='If primary is a transfer, alight at the decision clock. If primary is same car, retain only same-car alternatives.',
        old_history_posture='A positive gap since last arrival with current_trip set is not identified as physically onboard; keep an unresolved cohort row.',
        unknown='Any unresolved acceptable connection keeps the local outcome unidentified; no dropping before selection.',
        transfers=['legacy station buffer','zero transfer floor sensitivity; not asserted realistic'],
        event_visibility='Local departure at occurrence, instantaneous already-issued boarding rule; actual door/reaction times unidentified.',
        background='Both rules use shared issued-plan publication;30s is a fixed latency sensitivity, not an algorithm runtime estimate.',
        source_text={name:(ROOT/name).read_text() for name in ('src/platform_transit.py','src/background_policy.py',
                     'experiments/reconstruction_v7/contract_run.py')})
    (HERE/'contract_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]; exclusions=Counter(); start=perf_counter(); cpu=process_time()
    for archived in journeys:
        graph, world, _, _, metadata, _ = prepared[archived['date']]
        parameters={c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
        for number, decision in enumerate(archived['decisions']):
            cid=decision.get('action')
            if not isinstance(cid,int):
                exclusions['no_selected_connection']+=1;continue
            belief,state=context(graph,metadata,archived,number,fit)
            primary=graph.connections[cid]; nominal=belief.mean_events(parameters)
            limit=nominal[cid].arrival
            if limit is None:
                exclusions['zero_operating_primary']+=1;continue
            candidates=[c for c in graph.get_connections_from(state.stop)
                if c.arr_stop==primary.arr_stop and c.id not in state.unavailable_ids
                and c.dep_time<=state.time+1500 and (c.pickup_allowed or c.trip_id==state.current_trip)
                and nominal[c.id].arrival is not None and nominal[c.id].arrival<=limit+30]
            if state.current_trip is not None and primary.trip_id==state.current_trip:
                candidates=[c for c in candidates if c.trip_id==state.current_trip]
            row=dict(date=archived['date'],query=archived['query'],decision=number,time=state.time,
                stop=state.stop,primary=cid,local_goal=primary.arr_stop,accepted_ids=[c.id for c in candidates],
                unknown_ids=[c.id for c in candidates if c.id not in world.events])
            if state.current_trip is not None and state.time > state.last_arrival:
                row.update(status='unidentified_posture',last_arrival=state.last_arrival,
                    loss_bounds=[0,archived['od']['deadline_seconds']-state.time])
                rows.append(row);continue
            if row['unknown_ids']:
                row['status']='unidentified';rows.append(row);continue
            small=TransitGraph()
            for sid in {state.stop,primary.arr_stop}:
                small.add_stop(graph.stops[sid])
            mapping={c.id:i for i,c in enumerate(candidates)}
            for c in candidates:small.add_connection(replace(c,dep_distribution=None,arr_distribution=None))
            scene=JourneyScenario({mapping[c.id]:world.events[c.id] for c in candidates},minute=60,publications={})
            target=mapping[cid];deadline=archived['od']['deadline_seconds']
            base=CommittedState(state.stop,state.time,state.current_trip,state.last_arrival,initialized=True)
            committed=build_committed_model(small,[(1.,scene)],state.stop,primary.arr_stop,state.time,deadline,initial_state=base)
            out=committed.transition(committed.scenarios[0],base,('select',target))
            row['committed_first_attempt_stop_loss']=out.cost+committed.terminal_cost(committed.scenarios[0],out.state)
            row['outcomes']={}
            for name,transfer in [('station_buffer',None),('zero_floor',lambda s,c:0)]:
                initial=PlatformState(state.stop,state.time,state.current_trip,state.last_arrival,
                    last_connection=-1 if state.current_trip is not None else None,local_since=state.time)
                model=build_platform_model(small,[(1.,scene)],state.stop,primary.arr_stop,state.time,deadline,
                    transfer_time=transfer,poll_interval=1,initial_state=initial)
                rule=actual_rule(target,set(mapping.values()),{c.id:c.trip_id for c in small.connections})
                single=actual_rule(target,{target},{c.id:c.trip_id for c in small.connections})
                output={}
                for mode,publish,blocking in [('immediate',state.time,False),('background30',state.time+30,False),('blocking30',state.time+30,True)]:
                    result=execute_background(model,model.scenarios[0],single,rule,publish,blocking=blocking)
                    output[mode]=dict(cost=result['cost'],published=result['published'],rejected=result['rejected'],
                        boarded_while_computing=result['incumbent_actions_during_compute'])
                row['outcomes'][name]=output
            row['status']='identified';rows.append(row)
    result=dict(protocol=protocol,counterexample=counterexample(),rows=rows,exclusions=dict(exclusions),
        wall_seconds=perf_counter()-start,cpu_seconds=process_time()-cpu,
        summary=dict(n=len(rows),statuses=dict(Counter(r['status'] for r in rows))))
    (HERE/'contract_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('protocol','rows')},indent=2),flush=True)


if __name__=='__main__':main()
