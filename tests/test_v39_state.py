from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v33.execute import Executor
from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from experiments.reconstruction_v38.planner import FastPlanner
from experiments.reconstruction_v39.state import DecisionEngine, raw_history_key
from src.evolving_trip_belief import _ChainLaw
from src.transit_graph import Connection, Stop, TransitGraph


def fixture():
    graph = TransitGraph()
    for stop in range(5):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=2))
    rows = [(0, 'A', 0, 1, 0, 10), (1, 'A', 1, 2, 20, 30),
            (2, 'A', 2, 4, 40, 50), (3, 'B', 1, 4, 12, 60),
            (4, 'C', 2, 4, 35, 48), (5, 'D', 2, 4, 30, 55)]
    for cid, trip, origin, target, departure, arrival in rows:
        graph.add_connection(Connection(cid, 'R', trip, origin, target, departure, arrival))
    positions, laws = {}, {}
    for trip, cids, times in (
            ('A', (0, 1, 2), [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60]]),
            ('B', (3,), [[12, 22], [60, 90]]), ('C', (4,), [[35, 45], [48, 70]]),
            ('D', (5,), [[30, 40], [55, 80]])):
        arrays = [np.asarray(row, float) for row in times]
        matrix = np.log(np.array([[.8, .2], [.3, .7]]))
        laws[trip] = _ChainLaw(arrays, np.log([.5, .5]), lambda slot, value=matrix: value,
            [np.zeros(2) for row in arrays], [np.zeros(2, bool) for row in arrays],
            (True,)*(len(arrays)-1))
        positions.update({(cid, kind): (trip, 2*i+j) for i, cid in enumerate(cids)
                          for j, kind in enumerate(('departure', 'arrival'))})
    model = SimpleNamespace(graph=graph, _positions=positions,
                            _law=lambda trip, parameters: laws[trip])
    parameters = {'R': {'std': 1.}}
    fit = dict(pooled={kind: dict(bias_seconds=0., sd_seconds=30.)
                       for kind in ('departure', 'arrival')}, route_kind={})
    history = SingleMessageHistory(model, parameters, fit=fit)
    history.apply_frame(dict(receipt_seconds=1, trip_statuses=[['C', 'SCHEDULED']],
        endpoint_updates=[[4, 'departure', 'FORECAST', 40.], [4, 'arrival', 'FORECAST', 65.]]), 1)
    for cid, departure, arrival in ((0, 5, 15), (1, 25, 35)):
        history.observe(cid, 'departure', departure)
        history.observe(cid, 'arrival', arrival)
    past = Platform(1, {3: Ride(3, (3,), {3: None})})
    future = Platform(2, {4: Ride(4, (4,), {4: None})})
    ride = Ride(0, (0, 1, 2), {0: past, 1: future, 2: None})
    return graph, model, parameters, history, ride


@pytest.mark.parametrize('method', ['rolling_fixed', 'nominal', 'one_step_rollout', 'optimal'])
def test_suffix_keeps_each_remaining_policy_value_and_current_off_grid_decision(method):
    graph, model, parameters, history, ride = fixture()
    original = FastPlanner(graph, model, parameters, 100).compile_ride(ride, history, method)
    engine = DecisionEngine(FastPlanner(graph, model, parameters, 100), 'frontier')
    before = history.public_state()
    actual, receipt = engine.onboard(ride, 1, 35, history, method)
    expected, old_receipt = original.decision(1, 35)
    assert actual == expected
    assert receipt == old_receipt
    suffix = next(iter(engine._suffixes.values()))
    assert suffix.path == (1, 2) and tuple(suffix.exits) == (1, 2)
    assert engine.planner.relevant_trips(suffix) == {'A', 'C'}
    compiled = engine.planner.compile_ride(suffix, history, method)
    for a, b in zip(compiled.policy.values, original.policy.values[1:]):
        assert a == pytest.approx(b, abs=1e-10)
    for a, b in zip(compiled.policy.decision_costs, original.policy.decision_costs[1:]):
        assert a == pytest.approx(b, abs=1e-10)
    assert history.public_state() == before
    assert ride.path == (0, 1, 2) and tuple(ride.exits) == (0, 1, 2)
    assert engine.stats['suffix_dropped_exits'] == 1


@pytest.mark.parametrize('variant, hits', [('fullmemo', 0), ('frontier', 1)])
def test_only_frontier_merges_histories_of_a_trip_that_exists_only_at_a_passed_exit(variant, hits):
    graph, model, parameters, history, ride = fixture()
    engine = DecisionEngine(FastPlanner(graph, model, parameters, 100), variant)
    first = engine.onboard(ride, 1, 35, history)
    changed = history.clone()
    changed.observe(3, 'departure', 17)
    second = engine.onboard(ride, 1, 35, changed)
    assert first == second
    assert engine.stats['onboard_requests'] == 2
    assert engine.stats['onboard_hits'] == hits
    assert engine.stats['onboard_unique'] == 2-hits
    assert all(isinstance(value, (int, float)) for value in engine.stats.values())


def test_unselected_raw_forecast_and_ordered_hard_evidence_remain_in_frontier_state():
    graph, model, parameters, history, ride = fixture()
    engine = DecisionEngine(FastPlanner(graph, model, parameters, 100), 'frontier')
    engine.onboard(ride, 1, 35, history)
    changed = history.clone()
    changed.apply_frame(dict(receipt_seconds=1, trip_statuses=[],
        endpoint_updates=[[4, 'arrival', 'FORECAST', 75.]]), 35)
    assert changed.signature(('A', 'C')) == history.signature(('A', 'C'))
    assert raw_history_key(changed, ('A', 'C')) != raw_history_key(history, ('A', 'C'))
    engine.onboard(ride, 1, 35, changed)
    assert engine.stats['onboard_hits'] == 0 and engine.stats['onboard_unique'] == 2
    changed.observe(4, 'departure', 45)
    assert changed.selected_forecast('C')[:3] == (4, 'arrival', 75.)
    reordered = history.clone()
    reordered._records['A'] = tuple(reversed(reordered._records['A']))
    assert raw_history_key(reordered, ('A',)) != raw_history_key(history, ('A',))


def test_platform_entry_and_public_offer_sets_have_distinct_decisions_without_changing_history():
    graph, model, parameters, history, ride = fixture()
    platform = ride.exits[1]
    history.observe_platform((4,), 35, 45, {4: 45})
    planner = FastPlanner(graph, model, parameters, 100)
    engine = DecisionEngine(planner, 'frontier')
    chooser = Executor(graph, planner, 100, method='rolling_fixed', waiting='combined').choose
    before = history.public_state()
    args = (platform, 35, 37, history, (4,), {4: 45}, 45, chooser)
    first = engine.platform(*args)
    assert engine.platform(*args) == first
    engine.platform(platform, 36, 38, history, (4,), {4: 45}, 45, chooser)
    assert engine.stats['platform_requests'] == 3
    assert engine.stats['platform_hits'] == 1 and engine.stats['platform_unique'] == 2
    assert first[0] == 4
    assert history.public_state() == before


def test_trip_and_endpoint_service_announcements_are_retained():
    _, _, _, history, _ = fixture()
    before = raw_history_key(history, ('C',))
    changed = history.clone()
    changed.apply_frame(dict(receipt_seconds=2, trip_statuses=[['C', 'CANCELED']],
        endpoint_updates=[[4, 'arrival', 'SKIPPED', None]]), 2)
    assert raw_history_key(changed, ('C',)) != before
    assert changed.announced_unavailable(4, 'arrival')
    assert before[1:] == (1, 7)


def test_departed_root_is_removed_from_planning_but_visible_root_and_public_history_remain():
    graph, model, parameters, history, ride = fixture()
    platform = ride.exits[1]
    platform.rides[5] = Ride(5, (5,), {5: None})
    observed = {5: 40, 4: 45}
    history.observe_platform((4, 5), 35, 45, observed)
    before = history.public_state()
    results, engines = {}, {}
    for variant in ('fullmemo', 'frontier'):
        planner = FastPlanner(graph, model, parameters, 100)
        engine = DecisionEngine(planner, variant)
        choose = Executor(graph, planner, 100, method='rolling_fixed', waiting='combined').choose
        results[variant] = engine.platform(platform, 35, 37, history, (4,), observed, 45, choose)
        engines[variant] = engine
    assert results['frontier'] == results['fullmemo']
    assert results['frontier'][0] == 4
    view = next(iter(engines['frontier']._platform_views.values()))
    assert tuple(view.rides) == (4,)
    assert engines['frontier'].planner.relevant_trips(view) == {'C'}
    assert engines['frontier'].stats['platform_dropped_roots'] == 1
    assert engines['frontier'].stats['platform_views'] == 1
    assert engines['fullmemo'].stats['platform_dropped_roots'] == 0
    assert tuple(platform.rides) == (4, 5) and history.public_state() == before
