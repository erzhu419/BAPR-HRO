from dataclasses import replace

import numpy as np
import pytest

from src.committed_transit import CommittedState
from src.evolving_trip_belief import EvolvingTripBelief
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_oracle import earliest_arrival
from src.temporal_scan import (TemporalScan, bounded_mean_lower, build_temporal_kernel,
                               sample_clocks, template_trip_map, window_trips)
from src.transit_graph import Connection, Stop, TransitGraph


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    return build_temporal_kernel(tmp_path_factory.mktemp('temporal') / 'kernel.so')


def make_graph(size, rows):
    graph = TransitGraph()
    for stop in range(size):
        graph.add_stop(Stop(stop, str(stop)))
    for trip, origin, target, dep, arr, pickup in rows:
        graph.add_connection(Connection(-1, trip, trip, origin, target, dep, arr,
                                        pickup_allowed=pickup))
    return graph


def test_equal_tick_closure_respects_onboard_and_transfer_readiness(kernel):
    graph = make_graph(4, [('D', 2, 3, 3, 4, True), ('A', 1, 2, 3, 3, False),
                           ('C', 2, 0, 3, 3, True), ('A', 0, 1, 3, 3, True),
                           ('F', 0, 3, 6, 8, True), ('G', 1, 3, 7, 9, True)])
    dep = np.array([[c.dep_time for c in graph.connections]] * 2, dtype=np.int32)
    arr = np.array([[c.arr_time for c in graph.connections]] * 2, dtype=np.int32)
    canceled = np.zeros(dep.shape, dtype=np.uint8)
    canceled[1, 0] = 1
    scan = TemporalScan(graph, kernel)
    initial = CommittedState(0, 3, None, 3)
    for state, transfer, expected in [
        (initial, 0, [4, 8]), (initial, 1, [8, 8]),
        (replace(initial, stop=1, current_trip='A'), 0, [4, 8]),
        (replace(initial, stop=1, current_trip='A', unavailable_ids=frozenset({1})), 0, [9, 9]),
        (replace(initial, current_trip='missing', last_arrival=0), 3, [8, 8]),
    ]:
        reference = []
        for d, a, z in zip(dep, arr, canceled):
            scene = JourneyScenario({c.id: VehicleEvent(0, bool(z[c.id]), int(d[c.id]), int(a[c.id]))
                                     for c in graph.connections}, minute=1, publications={})
            reference.append(earliest_arrival(graph, scene, state, 3, 10, transfer_seconds=transfer)[0])
        assert reference == expected
        assert scan.arrivals(dep, arr, canceled, state, 3, 10, transfer).tolist() == expected
    assert scan.arrivals(dep, arr, canceled, initial, 3, 10, [0, 5, 0, 0]).tolist() == [4, 8]
    assert scan.arrivals(dep, arr, canceled, initial, 3, 10, [0, 5, 2, 0]).tolist() == [8, 8]
    stale = replace(initial, current_trip='missing', last_arrival=0)
    assert scan.arrivals(dep, arr, canceled, stale, 3, 10, [2, 0, 0, 0]).tolist() == [4, 8]


def test_batch_backward_draws_match_scalar_model_on_conditioned_trips():
    graph = make_graph(3, [('A', 0, 1, 10, 12, True), ('A', 1, 2, 12, 16, False),
                           ('B', 1, 2, 12, 15, True)])
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    belief.observe_connection_event(0, 'arrival', 1)
    parameters = {'A': dict(mean=1., std=2., cancel_rate=.2),
                  'B': dict(mean=2., std=3., cancel_rate=.5)}
    dep, arr, canceled = sample_clocks(belief, parameters, np.random.default_rng(76), 32)
    for i, world in enumerate(belief.sample(parameters, np.random.default_rng(76), 32)):
        for cid, event in world.events.items():
            assert (dep[i, cid], arr[i, cid], bool(canceled[i, cid])) == (
                event.departure, event.arrival, event.canceled)


def test_platform_oracle_keeps_elapsed_transfer_and_charges_delayed_alighting(kernel):
    from src.platform_transit import PlatformState
    graph = make_graph(2, [('B', 0, 1, 12, 14, True), ('C', 0, 1, 17, 20, True),
                           ('A', 0, 1, 11, 25, False)])
    dep = np.array([[c.dep_time for c in graph.connections]], dtype=np.int32)
    arr = np.array([[c.arr_time for c in graph.connections]], dtype=np.int32)
    canceled = np.zeros(dep.shape, dtype=np.uint8)
    scan = TemporalScan(graph, kernel)
    boarded = PlatformState(0, 10, 'A', last_arrival=0, last_connection=7)
    alighted = replace(boarded, current_trip=None, last_arrival=9)
    # Still aboard: alight at10, ready15, so B is too early. Same-trip A remains available.
    assert scan.platform_arrivals(dep, arr, canceled, boarded, 1, 30, 5).tolist() == [20]
    # Already alighted at9: ready14, also misses B. At7: ready12, catches B.
    assert scan.platform_arrivals(dep, arr, canceled, alighted, 1, 30, 5).tolist() == [20]
    assert scan.platform_arrivals(dep, arr, canceled, replace(alighted, last_arrival=7),
                                  1, 30, 5).tolist() == [14]
    canceled[0, 1] = 1
    assert scan.platform_arrivals(dep, arr, canceled, boarded, 1, 30, 5).tolist() == [25]


@pytest.mark.parametrize('cancel_rate', [0., .4, 1.])
def test_native_ffbs_preserves_original_scalar_tapes_and_observed_laws(kernel, cancel_rate):
    graph = make_graph(4, [('A', 0, 1, 10, 12, True), ('A', 1, 2, 13, 16, False),
                           ('A', 3, 0, 20, 24, True)])
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    parameters = {'A': dict(mean=1., std=2., cancel_rate=cancel_rate)}
    beliefs = [belief]
    if cancel_rate < 1.:
        observed = belief.copy()
        observed.observe_connection_event(0, 'arrival', 80)
        observed.observe_forecast(2, 'arrival', 94, 1.)
        beliefs.append(observed)
    for current in beliefs:
        law = current._law('A', parameters)
        # Reversed columns are the sampler's actual, noncontiguous tape layout.
        uniforms = np.random.default_rng(821).random((512, len(law.times)))[:, ::-1]
        expected = np.stack([law._draw_uniforms_python(row) for row in uniforms])
        np.testing.assert_array_equal(law.draw_uniforms_batch(uniforms), expected)
        np.testing.assert_array_equal(np.stack([law.draw_uniforms(row) for row in uniforms[:16]]),
                                      expected[:16])


@pytest.mark.parametrize('exact_receipts', [False, True])
def test_native_ffbs_supports_ordered_duration_column_major_transitions(kernel, exact_receipts):
    from src.ordered_duration_belief import OrderedDurationBelief
    graph = make_graph(3, [('A', 0, 1, 100, 100, True), ('A', 1, 2, 100, 101, True)])
    belief = OrderedDurationBelief(graph, public_occurrences=False)
    if exact_receipts:
        for cid, kind in [(0, 'departure'), (0, 'arrival'), (1, 'departure'), (1, 'arrival')]:
            belief.observe_connection_event(cid, kind, 7)
    law = belief._law('A', {'A': dict(mean=0., std=.5, cancel_rate=.2)})
    assert all(a.flags.f_contiguous and not a.flags.c_contiguous for a in law.transitions)
    uniforms = np.random.default_rng(13).random((128, len(law.times)))[:, ::-1]
    expected = np.stack([law._draw_uniforms_python(row) for row in uniforms])
    np.testing.assert_array_equal(law.draw_uniforms_batch(uniforms), expected)
    # Reuse the cached native pointers after the packing call has returned.
    np.testing.assert_array_equal(law.draw_uniforms_batch(uniforms[:8]), expected[:8])


def test_fixed_sample_mean_bound_respects_zero_and_combines_components():
    assert bounded_mean_lower([0] * 32, 4000, .05)['lower'] == 0
    value = bounded_mean_lower([100] * 32, 4000, .05)
    assert value['product'] == pytest.approx(100 * .025 ** (1 / 32))
    assert value['lower'] == value['product']


def test_window_uses_adaptive_support_and_keeps_trip_random_tape(kernel):
    graph = make_graph(3, [('gone', 0, 1, 0, 2, True), ('onboard', 0, 1, 50, 55, True),
                           ('onboard', 1, 2, 55, 60, False), ('later', 1, 2, 200, 201, True)])
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    belief.observe_connection_event(1, 'arrival', 80)
    parameters = {c.route: dict(mean=1., std=2., cancel_rate=.2) for c in graph.connections}
    trips = window_trips(belief, parameters, 135, 144)
    assert trips == {'onboard'}
    rng = lambda: np.random.default_rng(114)
    full = sample_clocks(belief, parameters, rng(), 32)
    clipped = sample_clocks(belief, parameters, rng(), 32, trips=trips)
    for complete, trimmed in zip(full, clipped):
        np.testing.assert_array_equal(complete[:, 1:3], trimmed[:, 1:3])
    scan = TemporalScan(graph, kernel)
    state = CommittedState(1, 135, 'onboard', 135, frozenset({1}))
    np.testing.assert_array_equal(scan.arrivals(*full, state, 2, 144, transfer=2),
                                  scan.arrivals(*clipped, state, 2, 144, transfer=2))


def test_shifted_template_preserves_worlds_and_separates_gap_and_evidence():
    graph = make_graph(4, [('A', 0, 1, 10, 12, True), ('A', 1, 2, 13, 15, True),
                           ('B', 0, 1, 30, 32, True), ('B', 1, 2, 33, 35, True),
                           ('C', 0, 1, 40, 42, True), ('C', 3, 2, 43, 45, True),
                           ('D', 0, 1, 50, 52, True), ('D', 1, 2, 53, 55, True)])
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False, gap_sd=4.)
    belief.observe_connection_event(6, 'arrival', 3)
    parameters = {c.route: dict(mean=1., std=2., cancel_rate=.4) for c in graph.connections}
    templates = template_trip_map(belief, parameters)
    assert templates == {'A': ('A', 0), 'B': ('A', 20), 'C': ('C', 0)}
    full = sample_clocks(belief, parameters, np.random.default_rng(19), 32)
    reused = sample_clocks(belief, parameters, np.random.default_rng(19), 32, templates=templates)
    for expected, actual in zip(full, reused):
        np.testing.assert_array_equal(actual, expected)
    belief.observe_connection_event(2, 'arrival', 3)
    new_templates = template_trip_map(belief, parameters)
    assert 'B' not in new_templates
    expected = sample_clocks(belief, parameters, np.random.default_rng(37), 32)
    cached = sample_clocks(belief, parameters, np.random.default_rng(37), 32, templates=new_templates)
    for reference, actual in zip(expected, cached):
        np.testing.assert_array_equal(actual, reference)
