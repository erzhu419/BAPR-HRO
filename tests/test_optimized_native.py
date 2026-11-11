import numpy as np
import pytest

from src.durner.native import native_preprocess, native_topocsa
from src.evolving_trip_belief import EvolvingTripBelief
from src.native_event_router import NativeEventRouter, minute_snapshot
from src.optimized_native_router import (NativeWorkspace, OptimizedNativeEventRouter,
    PosteriorMinuteProjection, build_native_kernel, optimized_native_preprocess, optimized_native_topocsa)
from src.pmf import PMF
from src.transit_graph import Connection, Stop, TransitGraph


def cyclic_graph(seed):
    rng = np.random.default_rng(seed)
    graph = TransitGraph()
    for sid in range(5):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=0))
    for i in range(24):
        dep, arr = rng.integers(0, 5, 2)
        time = int(rng.integers(2, 7))
        graph.add_connection(Connection(-1, 'R', f'T{i // 2}', int(dep), int(arr), time, time + 1,
            PMF(np.array([.2, .6, .2]), time - 1), PMF(np.array([.2, .6, .2]), time),
            cancel_prob=.15 if i % 7 == 0 else 0.))
    return graph


@pytest.fixture(scope='module')
def native_kernel(tmp_path_factory):
    return build_native_kernel(tmp_path_factory.mktemp('native') / 'order.so')


def equal_result(a, b):
    assert a.cuts == b.cuts
    assert a.hyperpath_connections == b.hyperpath_connections
    assert a.n_connections_processed == b.n_connections_processed
    assert a.mean_arrival == b.mean_arrival
    assert (a.dest_arrival is None) == (b.dest_arrival is None)
    if a.dest_arrival is not None:
        assert a.dest_arrival.offset == b.dest_arrival.offset
        np.testing.assert_array_equal(a.dest_arrival.probs, b.dest_arrival.probs)
    for sid in a.stop_labels:
        aa, bb = a.stop_labels[sid], b.stop_labels[sid]
        assert len(aa) == len(bb)
        for left, right in zip(aa, bb):
            assert (left.connection_id, left.mean_dest_arrival, left.feasibility) == (
                    right.connection_id, right.mean_dest_arrival, right.feasibility)
            assert left.dest_arrival.offset == right.dest_arrival.offset
            np.testing.assert_array_equal(left.dest_arrival.probs, right.dest_arrival.probs)


@pytest.mark.parametrize('seed', range(8))
def test_dfs_cycle_cut_ties_and_all_strict_labels_are_unchanged(seed, native_kernel):
    graph = cyclic_graph(seed)
    workspace = NativeWorkspace(graph, native_kernel)
    assert optimized_native_preprocess(graph, 0, 10, workspace) == native_preprocess(graph, 0, 10)
    reference = native_topocsa(graph, 0, 4, 0, 10)
    cold = optimized_native_topocsa(graph, 0, 4, 0, 10, workspace)
    warm = optimized_native_topocsa(graph, 0, 4, 1, 10, workspace)
    equal_result(reference, cold)
    equal_result(native_topocsa(graph, 0, 4, 1, 10), warm)


def event_graph():
    graph = TransitGraph()
    for sid in range(4):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=120))
    for row in [('R', 'A', 0, 1, 61, 121), ('R', 'A', 1, 2, 121, 181),
                ('R', 'B', 0, 2, 121, 241), ('S', 'C', 2, 3, 301, 361)]:
        graph.add_connection(Connection(-1, *row))
    return graph


def compare_projection(a, b):
    for left, right in zip(a.connections, b.connections):
        assert (left.dep_time, left.arr_time, left.cancel_prob, left.pickup_allowed) == (
                right.dep_time, right.arr_time, right.cancel_prob, right.pickup_allowed)
        for first, second in [(left.dep_distribution, right.dep_distribution),
                              (left.arr_distribution, right.arr_distribution)]:
            assert first.offset == second.offset
            np.testing.assert_array_equal(first.probs, second.probs)


def test_projection_reuses_only_unchanged_trip_and_parameter_factors():
    graph = event_graph()
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False)
    pars = {route: dict(mean=0., std=.7, cancel_rate=.1) for route in ('R', 'S')}
    projector = PosteriorMinuteProjection(belief)
    cold, work = projector.refresh(pars)
    compare_projection(cold, minute_snapshot(belief, pars)[0])
    assert work['refreshed_trips'] == 3
    _, work = projector.refresh(pars)
    assert work['refreshed_trips'] == 0
    retained = projector.graph.connections[2].dep_distribution
    belief.observe_connection_event(0, 'arrival', 7)
    changed, work = projector.refresh(pars, {0})
    compare_projection(changed, minute_snapshot(belief, pars, {0})[0])
    assert work['refreshed_trips'] == 1 and work['reused_trips'] == 2
    assert changed.connections[2].dep_distribution is retained
    pars['R']['mean'] = .5
    changed, work = projector.refresh(pars)
    compare_projection(changed, minute_snapshot(belief, pars)[0])
    assert work['refreshed_trips'] == 2
    assert changed.connections[0].cancel_prob < 1.


def test_optimized_router_preserves_native_choice_after_observation(native_kernel):
    graph = event_graph()
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False)
    pars = {route: dict(mean=0., std=.7, cancel_rate=.1) for route in ('R', 'S')}
    policies = [NativeEventRouter(graph, pars, event_belief=belief.copy(), max_time=600),
                OptimizedNativeEventRouter(graph, pars, event_belief=belief.copy(), max_time=600,
                                           native_kernel=native_kernel)]
    for router in policies:
        router.online_mean = router.online_cancel = False
        router._initial_cancels = {'R': .1, 'S': .1}
        router.route(0, 3, 0)
    for stop, time, trip, old, unavailable in [(0, 0, None, 0, set()), (1, 128, 'A', 128, {0})]:
        for router in policies:
            if trip:
                router.observe_connection_event(0, 'arrival', 7)
            router.set_information_context(stop, time, trip, old, unavailable, set(), set())
            router.select_connection(stop, time, np.random.default_rng(0))
        a, b = [router.predictions[-1] for router in policies]
        assert a['connection'] == b['connection']
        assert a['native_selected'] == b['native_selected']
        exported = policies[1].export_issued_priorities()
        expected = {sid: tuple(label.connection_id for label in reversed(labels))
                    for sid, labels in policies[1].last_native_result.stop_labels.items()}
        assert exported == expected
        exported.clear()
        assert policies[1].export_issued_priorities() == expected


def test_numeric_cdf_visibility_and_tiny_products_preserve_original_edges(native_kernel):
    graph = cyclic_graph(0)
    first, second = graph.connections[:2]
    workspace = NativeWorkspace(graph, native_kernel)
    for a, d in [(np.array([1., 1e-30]), np.array([1., 1e-30])),
                 (np.array([1e-310, 1.]), np.array([.5, .5])),
                 (np.array([1e-310, 1.]), np.array([1., 1e-30]))]:
        first.arr_distribution = PMF(a, 0)
        second.dep_distribution = PMF(d, 0)
        workspace.stats.update(zero_support_rejections=0, positive_support_proofs=0, positive_fallback_integrals=0)
        for shift in (-1, 0, 1, 2):
            second.dep_distribution.offset = shift
            from src.pmf import prob_reachable
            expected = prob_reachable(first.arr_distribution, second.dep_distribution, 0) * (1. - second.cancel_prob) > 0
            latest = np.zeros(len(graph.connections), dtype=np.int32)
            tails, operating = np.zeros(len(graph.connections)), np.zeros(len(graph.connections))
            last, tail, _, _ = workspace.support(second.dep_distribution)
            latest[second.id] = second.dep_distribution.offset + last
            tails[second.id], operating[second.id] = tail, 1. - second.cancel_prob
            choices = workspace.candidates(first, np.array([second.id], dtype=np.int32),
                graph, {}, latest, tails, operating)
            assert (second.id in choices) == expected
