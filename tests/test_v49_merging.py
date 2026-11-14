"""Generic relevance must retain future evidence and remove expired evidence."""
import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v46.shared_oracle import shared_structure
from experiments.reconstruction_v46.small_oracle import evaluate
from experiments.reconstruction_v48.lazy_frontier import LazyFrontierOracle
from experiments.reconstruction_v49.merging import (
    OrdinaryMergedOracle, OrdinaryRelevanceMixin, RemainingDependencies,
    recursive_trip_dependencies,
)
from src import evolving_trip_belief as evolving
from src.transit_graph import Connection
from test_v46_shared import shared_case
from test_v48_frontier import forced_case


class OriginalReducerOrdinaryOracle(OrdinaryRelevanceMixin, LazyFrontierOracle):
    """Separate state merging from the bulk implementation under test."""


@pytest.mark.parametrize('reverse,forecasts,cancel', [
    (False, False, 0.), (False, True, .05), (True, True, 0.),
])
def test_relevance_all_budget_intervals_preserve_private_enumeration(monkeypatch, reverse, forecasts, cancel):
    graph, model, params, history, platform, entry = shared_case(
        monkeypatch, reverse=reverse, forecasts=forecasts, cancel=cancel)
    exact = evaluate(graph, model, params, 60, platform,
                     dict(kind='platform_start', entry=entry, clock=entry), history,
                     action='policy', max_evaluations=100000)
    assert exact['unresolved_probability'] == 0
    truth = exact['mean'] + entry
    oracle = OrdinaryMergedOracle(graph, model, params, 60, history,
                                  max_compiles=10000, verify_frontier=True)
    result = oracle.platform(platform, entry, checkpoints=tuple(range(1, 101)))
    assert result['mean'] == pytest.approx(truth, abs=1e-8)
    assert result['counters']['ordinary_projection_calls'] > 0
    assert result['counters']['unsupported_scores'] == 0
    previous = (-np.inf, np.inf)
    for snapshot in [*result['checkpoints'], result]:
        value = snapshot['actions']['policy']
        assert value['lower'] <= truth + 1e-8
        assert value['upper'] >= truth - 1e-8
        assert value['lower'] >= previous[0] - 1e-8
        assert value['upper'] <= previous[1] + 1e-8
        previous = value['lower'], value['upper']


def test_generic_relevance_retains_exact_times_then_expires_them(monkeypatch):
    _, model, _, _, platform, _ = shared_case(monkeypatch, forecasts=False)
    # Z is an independent surviving candidate. Reusing X's path would defeat
    # independence, so the liveness-only test adds a minimal third trip map.
    model._positions[9, 'departure'] = ('Z', 0)
    model._positions[9, 'arrival'] = ('Z', 1)
    platform.rides[9] = Ride(9, (9,), {9: None})
    dep = RemainingDependencies(model, platform)
    assert dep.cids == (0, 2, 9)
    # After X departs, Y can still query X's downstream trip. The histories
    # therefore stay distinct, even though their time/remaining mask agree.
    assert dep.project(0b110, ((0, 10.),)) == ((0, 10.),)
    assert dep.project(0b110, ((0, 18.),)) == ((0, 18.),)
    # Once Y also departs, only Z remains and both histories can merge.
    assert dep.project(0b100, ((0, 10.), (2, 11.))) == ()
    assert dep.project(0b100, ((0, 18.), (2, 19.))) == ()


def test_post_destination_scoring_dependency_is_not_deleted(monkeypatch):
    _, model, _, _, platform, _ = shared_case(monkeypatch)
    y = platform.rides[2]
    tail = y.exits[2]
    y.exits = {2: None, 3: tail}
    assert shared_structure(model, platform)['interface_cids'] == [0]
    assert recursive_trip_dependencies(model, y) == frozenset(('X', 'Y'))
    index = RemainingDependencies(model, platform)
    assert index.project(0b10, ((0, 10.),)) == ((0, 10.),)
    assert index.project(0b10, ((0, 18.),)) == ((0, 18.),)


def three_candidate_case(monkeypatch):
    graph, _, params, old_history, platform, entry = shared_case(monkeypatch, forecasts=False)
    # Both X and Y can be declined while Z remains, exposing the same future
    # state after two different already-observed X departure clocks.
    graph.connections[1].arr_time = 40
    graph.connections[4].dep_time = 42
    graph.connections[4].arr_time = 50
    graph.connections[3].arr_time = 55
    graph.add_connection(Connection(5, 'R', 'Z', 0, 2, 25, 26))
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False,
                                      travel_sd=.1, dwell_sd=.1, cancel_persistence=.8)
    history = PhysicalHistory(model, params, fit=old_history.fit)
    platform.rides[5] = Ride(5, (5,), {5: None})
    return graph, model, params, history, platform, entry


def test_dynamic_merging_is_real_and_matches_full_shared_value(monkeypatch):
    graph, model, params, history, platform, entry = three_candidate_case(monkeypatch)
    shared = LazyFrontierOracle(graph, model, params, 60, history, max_compiles=10000)
    ordinary = OriginalReducerOrdinaryOracle(graph, model, params, 60, history, max_compiles=10000)
    reference = shared.platform(platform, entry)
    result = ordinary.platform(platform, entry)
    assert result['mean'] == pytest.approx(reference['mean'], abs=1e-8)
    assert result['mean'] is not None
    assert ordinary.stats['ordinary_static_records_removed'] > 0
    # An expired X clock is removed before a nonempty Z continuation.
    ordinary_states = [key for key in ordinary.nodes if key[0] == 'event' and key[3] == 0b100]
    shared_states = [key for key in shared.nodes if key[0] == 'event' and key[3] == 0b100]
    assert ordinary_states and shared_states
    assert all(key[-1] == () for key in ordinary_states)
    assert any(key[-1] for key in shared_states)
    assert len(ordinary_states) < len(shared_states)


def test_both_arms_actual_stop_and_pause_preserve_new_state_graph(monkeypatch):
    graph, model, params, history, node, row = forced_case(monkeypatch)
    full = OrdinaryMergedOracle(graph, model, params, 100, history,
                                max_compiles=10000, trace=True, verify_frontier=True)
    full_result = full.forced(node, row, certificate_every=1)
    stopped = OrdinaryMergedOracle(graph, model, params, 100, history,
                                   max_compiles=10000, trace=True, verify_frontier=True)
    stop_result = stopped.forced(node, row, stop_on_certificate=True, certificate_every=1)
    assert stop_result['certified_action'] == full_result['certified_action']
    assert stop_result['stop_reason'] == 'certificate'
    assert stopped.selection_trace == full.selection_trace[:len(stopped.selection_trace)]
    for arm in ('alight', 'continue'):
        exact = evaluate(graph, model, params, 100, node, row, history,
                         action=arm, max_evaluations=100000)
        truth = exact['mean'] + row['clock']
        value = stop_result['actions'][arm]
        assert value['lower'] <= truth + 1e-8 <= value['upper'] + 2e-8


def test_original_entry_evidence_stays_in_distinct_platform_contexts(monkeypatch):
    graph, model, params, history, platform, entry = shared_case(monkeypatch)
    oracle = OrdinaryMergedOracle(graph, model, params, 60, history, max_compiles=10000)
    left = history.clone()
    right = history.clone()
    left.observe(0, 'departure', 10.)
    right.observe(0, 'departure', 18.)
    a = oracle._platform_node(platform, entry, left)
    b = oracle._platform_node(platform, entry, right)
    assert a is not b
    # Projection is only of this visit's new observations. All raw received
    # forecasts and historical evidence remain in the base history objects.
    assert left._records['X'] != right._records['X']
    assert left.raw_forecast_records('X') == right.raw_forecast_records('X')
