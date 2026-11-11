"""Profile the compiled strict-native baseline; Python pilot remains archived."""
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from src.durner.native import native_topocsa
from src.evolving_trip_belief import EvolvingTripBelief
from src.native_event_router import minute_snapshot
from src.optimized_native_router import (NativeWorkspace, PosteriorMinuteProjection,
    build_native_kernel, optimized_native_topocsa)


def measured(function):
    wall, cpu = perf_counter(), process_time()
    value = function()
    return value, dict(wall_seconds=perf_counter() - wall, cpu_seconds=process_time() - cpu)


def compare(left, right):
    if left.cuts != right.cuts or left.hyperpath_connections != right.hyperpath_connections:
        raise AssertionError('Cut set or retained native labels changed')
    labels = 0
    for sid, source in left.stop_labels.items():
        target = right.stop_labels[sid]
        if len(source) != len(target):
            raise AssertionError('Strict label count changed')
        for a, b in zip(source, target):
            if (a.connection_id, a.mean_dest_arrival, a.feasibility, a.dest_arrival.offset) != (
                    b.connection_id, b.mean_dest_arrival, b.feasibility, b.dest_arrival.offset):
                raise AssertionError('Native label order or value changed')
            np.testing.assert_array_equal(a.dest_arrival.probs, b.dest_arrival.probs)
            labels += 1
    if left.mean_arrival != right.mean_arrival:
        raise AssertionError('Native query value changed')
    if left.dest_arrival is not None:
        np.testing.assert_array_equal(left.dest_arrival.probs, right.dest_arrival.probs)
    return dict(cuts_equal=True, label_order_values_equal=True, pmfs_bitwise_equal=True,
                native_labels=labels, native_cuts=len(left.cuts), connections=left.n_connections_processed)


def main():
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    fit, fit_cost = measured(lambda: fit_training(data, retain_no_pickup=True))
    day, qi = '2023-07-26', 0
    prepared, graph_cost = measured(lambda: dated_graph(data, day, 'gtfs_station_max', True))
    graph, world, stops, _, metadata, _ = prepared
    query = data['od_queries'][qi]
    origin, destination = stops[query['origin']], stops[query['destination']]
    ready, deadline = query['ready_seconds'], query['deadline_seconds']
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                  for c in graph.connections}
    protocol = dict(scope='Existing development date, no new confirmation labels or source download.',
        date=day, query=qi, od=query, model='Original strict independent minute-marginal recursion.',
        changes='Same DFS in C++, numerical-support positivity, trip revision projection and exact transfer/mean cache; no scenario/candidate/label change.',
        clocks='Local wall and process CPU; one process; cold independent beliefs then warmed unchanged/observed queries.',
        sources={name: (ROOT / name).read_text() for name in (
            'src/durner/native.py', 'src/native_event_router.py', 'src/optimized_native_router.py', 'src/native_order.cpp',
            'experiments/reconstruction_v7/baseline_profile.py')})
    (HERE / 'baseline_compiled_profile_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    kernel, kernel_cost = measured(lambda: build_native_kernel(HERE / 'build/native_order.so'))
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    kernel_cost['compiler_child_cpu_seconds'] = child_after.ru_utime + child_after.ru_stime - child_before.ru_utime - child_before.ru_stime
    original = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    changed = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    reference_projection, projection_cost = measured(lambda: minute_snapshot(original, parameters))
    temporary, reference_work = reference_projection
    reference, native_cost = measured(lambda: native_topocsa(temporary, origin, destination, ready // 60, deadline // 60))
    print('ORIGINAL', projection_cost, native_cost, flush=True)
    projector, projector_cost = measured(lambda: PosteriorMinuteProjection(changed))
    projected, optimized_projection_cost = measured(lambda: projector.refresh(parameters))
    optimized_graph, projection_work = projected
    workspace, workspace_cost = measured(lambda: NativeWorkspace(optimized_graph, kernel))
    optimized, optimized_query_cost = measured(lambda: optimized_native_topocsa(optimized_graph,
        origin, destination, ready // 60, deadline // 60, workspace))
    cold_work = dict(workspace.stats)
    identical = compare(reference, optimized)
    print('OPTIMIZED COLD', optimized_projection_cost, optimized_query_cost, identical, flush=True)
    warm_projection, warm_projection_cost = measured(lambda: projector.refresh(parameters))
    warm, warm_query_cost = measured(lambda: optimized_native_topocsa(warm_projection[0],
        origin, destination, ready // 60, deadline // 60, workspace))
    warm_work = dict(workspace.stats)
    compare(reference, warm)
    archive = json.loads((ROOT / 'experiments/reconstruction_v6/native_city_interned_cases/logical_2023-07-26_0.json').read_text())
    cid = archive['decisions'][0]['action']
    c, actual = graph.connections[cid], metadata[cid]
    def observe():
        for kind, planned in [('departure', c.dep_time), ('arrival', c.arr_time)]:
            changed.observe_connection_event(cid, kind, actual[kind]['time_seconds'] - planned)
    _, update_cost = measured(observe)
    observed_projection, observed_projection_cost = measured(lambda: projector.refresh(parameters, {cid}))
    observed_time = actual['arrival']['time_seconds']
    observed, observed_query_cost = measured(lambda: optimized_native_topocsa(observed_projection[0],
        c.arr_stop, destination, observed_time // 60, deadline // 60, workspace))
    observed_work = dict(workspace.stats)
    fresh, fresh_cost = measured(lambda: optimized_native_topocsa(observed_projection[0],
        c.arr_stop, destination, observed_time // 60, deadline // 60, NativeWorkspace(observed_projection[0], kernel)))
    observe_equality = compare(observed, fresh)
    result = dict(protocol='baseline_compiled_profile_protocol.json', training=fit_cost, graph_build=graph_cost,
        kernel_compilation=kernel_cost,
        original=dict(projection=projection_cost, query=native_cost, projection_work=reference_work),
        optimized_cold=dict(projector_setup=projector_cost, workspace_setup=workspace_cost,
            projection=optimized_projection_cost, query=optimized_query_cost,
            projection_work=projection_work, query_work=cold_work),
        optimized_hot=dict(projection=warm_projection_cost, query=warm_query_cost,
            projection_work=warm_projection[1], query_work=warm_work),
        after_observation=dict(connection=cid, clock=observed_time, observation=update_cost,
            projection=observed_projection_cost, query=observed_query_cost,
            projection_work=observed_projection[1], query_work=observed_work,
            fresh_query_cost=fresh_cost, fresh_query_equality=observe_equality),
        cold_result_equality=identical, process_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory_scope='Lifetime process peak includes both retained reference and optimized graphs; not a standalone algorithm memory comparison.')
    (HERE / 'baseline_compiled_profile_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'memory_scope'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
