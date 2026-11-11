"""Whole journeys with one background plan update and common platform execution."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from math import ceil
from pathlib import Path
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from src.background_policy import IssuedPolicy, execute_background
from src.committed_transit import CommittedState
from src.evolving_trip_belief import EvolvingTripBelief
from src.first_branch_belief import FirstBranchSampler
from src.indexed_nominal_tail import IndexedNominalTailRouter
from src.optimized_native_router import OptimizedNativeEventRouter, build_native_kernel, load_native_kernel
from src.platform_transit import build_platform_model
from src.recourse_itinerary_router import (RecourseItineraryRouter, _ConditionalNominal,
    build_trip_blocks, expand_trip_blocks, nominal_itinerary, trip_key)

METHODS = ('compiled_nominal', 'analytic_recourse', 'uct64', 'uct512', 'optimized_native_priorities')
CLOCKS = ('logical', 'background', 'blocking')


def path_targets(graph, primary, rescue, start):
    def targets(state):
        path = primary
        if primary and primary[0] not in state.used_connections and rescue:
            failed = any(cid == primary[0] and kind == 'published_cancel'
                or (cid == primary[0] and kind == 'departure' and time < state.time)
                for time, cid, kind in state.public_log)
            if failed or state.time > start + 720:
                path = rescue
        for cid in path:
            if cid not in state.used_connections:
                return (cid,) if graph.connections[cid].dep_stop == state.stop else ()
        return ()
    return targets


def issue(graph, name, targets, origin, first_board_not_before=None):
    def action(state):
        options = targets(state)
        if first_board_not_before is not None and not state.used_connections and state.time < first_board_not_before:
            return ('wait',)
        if (options and state.current_trip is not None and
                all(graph.connections[cid].trip_id != state.current_trip for cid in options)):
            return ('alight',)
        return next((('board', cid) for cid in options if cid in state.visible_connections), ('wait',))

    def applicable(state):
        if state.stop != origin or state.used_connections or state.current_trip is not None:
            return False
        past = {cid for time, cid, kind in state.public_log
                if kind == 'published_cancel' or (kind == 'departure' and time < state.time)}
        return any(cid not in past for cid in targets(state))

    return IssuedPolicy(action, applicable, name)


def compute(graph, fit, origin, destination, start, deadline, seed, method, kernel):
    params = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    kwargs = dict(event_belief=belief, max_time=deadline-start, seed=seed, samples=4)
    if method == 'optimized_native_priorities':
        router = OptimizedNativeEventRouter(graph, params, native_kernel=kernel, **kwargs)
    elif method.startswith('uct'):
        router = IndexedNominalTailRouter(graph, params, algorithm='uct', work_budget=int(method[3:]), **kwargs)
    else:
        router = RecourseItineraryRouter(graph, params, optimize=method == 'analytic_recourse', **kwargs)
    router.online_mean = router.online_cancel = False
    router._initial_cancels = {route: p['cancel_rate'] for route, p in params.items()}
    router.route(origin, destination, start)
    router.set_information_context(origin, start, None, start, set(), set(), set())
    selected = router.select_connection(origin, start, np.random.default_rng(seed))
    choice = selected[0].connection_id if selected else None
    primary, rescue = (), ()
    if method == 'optimized_native_priorities':
        priorities = router.export_issued_priorities()
        def targets(state):
            candidates = tuple(cid for cid in priorities.get(state.stop, ())
                if cid not in state.used_connections and graph.connections[cid].dep_time <= state.time + 1500
                and (graph.connections[cid].pickup_allowed or graph.connections[cid].trip_id == state.current_trip))
            same = tuple(cid for cid in candidates if state.current_trip is not None
                         and graph.connections[cid].trip_id == state.current_trip)
            return same or candidates
    elif method.startswith('uct'):
        state = router.information_state
        nominal = belief.mean_events(params)
        trip_blocks = build_trip_blocks(graph)
        if choice is None:
            primary = nominal_itinerary(graph, replace(state, time=start+60), destination,
                nominal, router.menu, deadline, 1500, trip_blocks=trip_blocks) or ()
        else:
            c = graph.connections[choice]
            if c.arr_stop == destination:
                primary = (choice,)
            else:
                sampler = FirstBranchSampler(belief, params, choice, start, deadline)
                if sampler.probability(True):
                    tail_law = _ConditionalNominal(nominal, belief, trip_key(c), sampler.conditional_law(True))
                    arrival = min(deadline, round(tail_law[choice].arrival))
                    after = CommittedState(c.arr_stop, arrival, trip_key(c), arrival, frozenset({choice}), initialized=True)
                    tail = nominal_itinerary(graph, after, destination, tail_law, router.menu, deadline, 1500,
                                             trip_blocks=trip_blocks)
                    if tail is not None:
                        primary = expand_trip_blocks(graph, (choice,) + tail, destination, trip_blocks=trip_blocks) or ()
        targets = path_targets(graph, primary, (), start)
    else:
        primary, rescue = router._path, router._rescue
        targets = path_targets(graph, primary, rescue, start)
    return targets, dict(primary=primary, rescue=rescue, scored_root_action=choice,
        first_board_not_before=start+60 if method.startswith('uct') and choice is None else None,
        planner=router.predictions[-1] if router.predictions else {},
        number_of_priority_stops=len(priorities) if method == 'optimized_native_priorities' else None)


def unresolved_check(graph, metadata, scene, targets_by_name):
    trips = defaultdict(list)
    for cid, row in metadata.items():
        trips[row['trip_id']].append(cid)
    following = {}
    for ids in trips.values():
        ids.sort(key=lambda cid: metadata[cid]['dep_sequence'])
        for a, b in zip(ids, ids[1:]):
            if metadata[a]['arr_sequence'] == metadata[b]['dep_sequence']:
                following[a] = b

    def unresolved(state, policy, action):
        if action[0] == 'alight':
            return ()
        wanted = targets_by_name[policy.name](state) if policy is not None else ()
        missing = {cid for cid in wanted if cid not in scene.events}
        if state.current_trip is not None and state.last_connection in following:
            before, next_cid = state.last_connection, following[state.last_connection]
            previous = metadata[before]['arrival']
            upcoming = metadata[next_cid]['departure']
            if (next_cid not in scene.events or previous['official_journey'] != upcoming['official_journey']
                    or (upcoming['status'] == 'REAL' and upcoming['time_seconds'] < state.last_arrival)):
                missing.add(next_cid)
        return tuple(sorted(missing))
    return unresolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    protocol = dict(scope='Four already-exposed July26/27 date×OD cases, frozen before new replay.',
        methods=args.methods, clocks=CLOCKS, source=str(source.relative_to(ROOT)),
        incumbent='Same prior-based compiled nominal itinerary available at departure. Compute and report its preparation separately.',
        publication='One root worker launched at departure, release after ceil(measured wall seconds). Recheck origin and first options using only public facts.',
        assumptions='Ideal local departure events, same legacy station buffers, no explicit footpaths or global actual feed.',
        comparability='Planners retain their legacy scoring contracts. UCT compiles its selected root with a common conditional nominal tail. Native exports strict priorities. This evaluates common execution of issued plans, not online optimality in the new contract.',
        censor='Unresolved issued targets or next same-service identity stop offline scoring with elapsed/cap bounds, never supplied to the planner.',
        sources={name:(ROOT/name).read_text() for name in ('src/platform_transit.py','src/background_policy.py',
                 'src/causal_planning_router.py','src/indexed_nominal_tail.py','src/recourse_itinerary_router.py','src/optimized_native_router.py',
                 'experiments/reconstruction_v8/issued_replay.py')})
    output.with_name(output.stem + '_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started = perf_counter()
    fit = fit_training(data, retain_no_pickup=True)
    fit_seconds = perf_counter()-started
    kernel, compile_seconds = None, 0.
    if 'optimized_native_priorities' in args.methods:
        library = HERE/'build'/'native_order.so'
        started = perf_counter()
        build_native_kernel(library)
        compile_seconds = perf_counter()-started
        kernel = load_native_kernel(library)
    rows, preparation = [], []
    for day in data['holdout_dates']:
        started = perf_counter()
        graph, scene, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        graph_seconds = perf_counter()-started
        for query_index, query in enumerate(data['od_queries']):
            start, deadline = query['ready_seconds'], query['deadline_seconds']
            origin, destination = stops[query['origin']], stops[query['destination']]
            seed = int(day.replace('-','')) + 1000*query_index
            started = perf_counter()
            model = build_platform_model(graph, [(1.,scene)], origin, destination, start, deadline, poll_interval=1)
            timeline_seconds = perf_counter()-started
            started = perf_counter()
            nominal_targets, nominal_work = compute(graph,fit,origin,destination,start,deadline,seed,'compiled_nominal',kernel)
            nominal_seconds = perf_counter()-started
            incumbent = issue(graph,'incumbent',nominal_targets,origin)
            preparation.append(dict(date=day, query=query_index, graph_seconds=graph_seconds,
                timeline_seconds=timeline_seconds, incumbent_seconds=nominal_seconds))
            for method in args.methods:
                wall, cpu = perf_counter(), process_time()
                targets, work = compute(graph,fit,origin,destination,start,deadline,seed,method,kernel)
                elapsed, processor = perf_counter()-wall, process_time()-cpu
                candidate = issue(graph,method,targets,origin,work['first_board_not_before'])
                unresolved = unresolved_check(graph,metadata,scene,{'incumbent':nominal_targets,method:targets})
                for mode in CLOCKS:
                    publish = start if mode == 'logical' else start + ceil(elapsed)
                    started = perf_counter()
                    result = execute_background(model,model.scenarios[0],incumbent,candidate,publish,
                        blocking=mode=='blocking',unresolved=unresolved)
                    execution_seconds = perf_counter()-started
                    state = result['state']; cap = deadline-start
                    lo, hi = ((state.time-start,cap) if result['cost'] is None else (result['cost'],result['cost']))
                    row = dict(date=day,query=query_index,method=method,clock=mode,loss_lower=lo,loss_upper=hi,
                        status='censored' if result['cost'] is None else 'arrived' if state.stop==destination else 'timeout',
                        wall_plan_seconds=elapsed,cpu_plan_seconds=processor,execution_seconds=execution_seconds,
                        publish_clock=publish,published=result['published'],rejected=result['rejected'],
                        missing=list(result['missing_connections']),plan=work,
                        incumbent_moves=result['incumbent_actions_during_compute'],
                        rides=[r for r in result['trace'] if r['ridden_connection'] is not None])
                    rows.append(row)
                print(day,query_index,method,round(elapsed,3),[(r['clock'],r['status'],r['loss_lower'],r['loss_upper']) for r in rows[-3:]],flush=True)
                output.write_text(json.dumps(dict(fit_seconds=fit_seconds,compile_seconds=compile_seconds,
                    preparation=preparation,rows=rows),indent=2)+'\n')


if __name__ == '__main__':
    main()
