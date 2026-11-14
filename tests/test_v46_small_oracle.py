from collections import defaultdict
from itertools import product

import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v39.executor import SnapshotExecutor
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine, PhysicalHistory
from experiments.reconstruction_v43.reference import RootConditionalLaw
from experiments.reconstruction_v46.small_oracle import PrivateLaws, evaluate
from src import evolving_trip_belief as evolving
from src.transit_graph import Connection, Stop, TransitGraph


def fixture(monkeypatch, *, ready_delay=0, cancel=.15, forecasts=True):
    monkeypatch.setattr(evolving, 'DELAYS', np.array([0., 1./60]))
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=ready_delay if stop == 1 else 0))
    for cid, trip, origin, target, departure, arrival in (
            (0, 'head', 0, 1, 0, 10), (1, 'A', 1, 2, 20, 30),
            (2, 'B', 1, 3, 20, 45), (3, 'A', 2, 3, 35, 40)):
        graph.add_connection(Connection(cid, 'R', trip, origin, target, departure, arrival))
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False,
        travel_sd=.5, dwell_sd=.5, cancel_persistence=.8)
    parameters = {'R': dict(mean=0., std=.5, cancel_rate=cancel)}
    fit = dict(route_kind={}, pooled={kind: dict(bias_seconds=0., sd_seconds=1.)
                                      for kind in ('departure', 'arrival')})
    history = PhysicalHistory(model, parameters, fit=fit)
    history.observe(0, 'departure', 0)
    history.observe(0, 'arrival', 10)
    if forecasts:
        history.apply_frame(dict(receipt_seconds=1., trip_statuses=[], endpoint_updates=[
            [3, 'arrival', 'FORECAST', 41.]]), 1.)
        history.apply_frame(dict(receipt_seconds=2., trip_statuses=[], endpoint_updates=[
            [1, 'departure', 'FORECAST', 20.]]), 2.)
    platform = Platform(1, {1: Ride(1, (1, 3), {3: None}), 2: Ride(2, (2,), {2: None})})
    node = Ride(0, (0,), {0: platform})
    row = dict(kind='onboard', cid=0, clock=10)
    return graph, model, parameters, 45, node, row, history


def joint_paths(law):
    """Raw product of initial, transition and emission factors, not marginals."""
    paths = []
    supports = [np.flatnonzero(law.marginal(slot) > 0) for slot in range(len(law.times))]
    for path in product(*supports):
        weight = law.initial[path[0]]+sum(e[state] for e, state in zip(law.emissions, path))
        weight += sum(law._transition(slot)[path[slot], path[slot+1]] for slot in range(len(path)-1))
        if np.isfinite(weight):
            paths.append((path, float(np.exp(weight))))
    total = sum(weight for _, weight in paths)
    return [(path, weight/total) for path, weight in paths]


def leaf_key(row):
    trace = tuple(tuple(sorted(r.items())) for r in row['trace'])
    return row['status'], row['loss'], trace


def independent_joint_reference(args):
    graph, model, parameters, deadline, node, row, history = args
    snapshot = freeze_snapshot(history, row['clock'])
    roots = RootConditionalLaw(snapshot, sorted(node.trips(model))).roots
    trips = tuple(roots)
    paths = [joint_paths(roots[trip]) for trip in trips]
    planner = MatchedContinuationPlanner(graph, model, parameters, deadline)
    engine = PhysicalDecisionEngine(planner, variant='frontier')
    answer, value = defaultdict(float), 0.
    for realization in product(*paths):
        assigned = {trip: path for trip, (path, _) in zip(trips, realization)}
        probability = float(np.prod([weight for _, weight in realization]))
        class World:
            def event(self, cid, kind):
                trip, slot = model._positions[cid, kind]
                law, index = roots[trip], assigned[trip][slot]
                return int(law.times[slot][index]), bool(law.canceled[slot][index]), trip
        executor = SnapshotExecutor(graph, planner, deadline, engine=engine,
                                    method='rolling_fixed', waiting='combined')
        result = executor.forced(node, row, 'alight', snapshot.clone(), World())
        assert result['loss'] is not None
        answer[leaf_key(result)] += probability
        value += probability*result['loss']
    return answer, value


@pytest.mark.parametrize('ready_delay', [0, 11])
def test_demand_oracle_equals_joint_enumeration_with_reselection_and_preready_observations(monkeypatch, ready_delay):
    args = fixture(monkeypatch, ready_delay=ready_delay)
    expected, value = independent_joint_reference(args)
    actual = evaluate(*args, action='alight', keep_terminal_records=True)
    by_outcome = defaultdict(float)
    for row in actual['terminals']:
        by_outcome[leaf_key(row)] += row['probability']
    assert actual['complete'] and actual['pending_mass'] == 0
    assert actual['probability_mass'] == pytest.approx(1., abs=1e-12)
    assert actual['mean'] == pytest.approx(value, abs=1e-10)
    assert dict(by_outcome) == pytest.approx(dict(expected), abs=1e-12)
    assert actual['status_mass'].get('service_or_deadline_failure', 0.) > 0
    assert actual['status_mass'].get('no_legal_offer', 0.) > 0
    assert actual['generated_monte_carlo_worlds'] == 0


def test_unexpanded_mass_keeps_a_valid_full_policy_interval(monkeypatch):
    args = fixture(monkeypatch)
    _, value = independent_joint_reference(args)
    for budget in (0, 1, 10):
        result = evaluate(*args, action='alight', max_evaluations=budget)
        assert result['evaluations'] == budget and not result['complete']
        assert result['mean'] is None and result['pending_mass'] > 0
        assert result['lower'] <= value <= result['upper']
        assert result['probability_mass'] == pytest.approx(1., abs=1e-12)


def test_fixture_separates_received_wait_message_from_postdeparture_working_law(monkeypatch):
    _, model, _, _, node, row, history = fixture(monkeypatch)
    snapshot = freeze_snapshot(history, row['clock'])
    root = RootConditionalLaw(snapshot, node.trips(model))
    observed = snapshot.clone()
    observed.observe(1, 'departure', 20)
    assert snapshot.selected_forecast('A')[:2] == (1, 'departure')
    assert observed.selected_forecast('A')[:2] == (3, 'arrival')
    assert not np.allclose(root.law('A', observed).marginal(3), observed.law('A').marginal(3))


def test_private_future_departures_never_enter_public_policy_history(monkeypatch):
    args = fixture(monkeypatch, cancel=0., forecasts=False)
    original = PhysicalDecisionEngine.platform
    calls = []
    def checked(self, platform, entry, ready, history, visible, observed, clock, choose, method='rolling_fixed'):
        calls.append((clock, tuple(visible)))
        for record in history.public_state():
            if record['evidence'] == 'observed':
                assert record['clock'] <= clock
        return original(self, platform, entry, ready, history, visible, observed, clock, choose, method)
    monkeypatch.setattr(PhysicalDecisionEngine, 'platform', checked)
    result = evaluate(*args, action='alight')
    assert result['complete']
    assert any(len(visible) == 2 for _, visible in calls)
    assert any(len(visible) == 1 for _, visible in calls)


def test_canceled_private_state_retains_possible_later_recovery(monkeypatch):
    args = fixture(monkeypatch)
    graph, model, parameters, deadline, node, row, history = args
    snapshot = freeze_snapshot(history, row['clock'])
    roots = RootConditionalLaw(snapshot, node.trips(model)).roots
    root = roots['A']
    canceled = int(np.flatnonzero(root.canceled[0])[0])
    actual = PrivateLaws(roots).probabilities('A', 2, {('A', 0): canceled})
    paths = [(path, p) for path, p in joint_paths(root) if path[0] == canceled]
    expected = np.zeros(len(root.times[2]))
    normalizer = sum(p for _, p in paths)
    for path, p in paths:
        expected[path[2]] += p/normalizer
    assert actual == pytest.approx(expected, abs=1e-12)
    assert actual[~root.canceled[2]].sum() > 0


def test_platform_start_and_forced_alight_evaluate_the_same_original_policy(monkeypatch):
    args = list(fixture(monkeypatch))
    forced = evaluate(*args, action='alight')
    args[4] = args[4].exits[0]
    args[5] = dict(kind='platform_start', entry=10, clock=10)
    direct = evaluate(*args)
    assert direct['complete']
    assert direct['mean'] == pytest.approx(forced['mean'], abs=1e-12)
    assert direct['status_mass'] == pytest.approx(forced['status_mass'], abs=1e-12)


@pytest.mark.parametrize('cancel', [0., .1])
def test_mixed_three_mark_dp_matches_private_original_policy_enumeration(monkeypatch, cancel):
    from experiments.reconstruction_v46.oracle import FullPolicyOracle
    from test_v46_oracle import case
    graph, model, parameters, history, platform = case(monkeypatch, cancel=cancel, forecasts=True, mct=3)
    original = evaluate(graph, model, parameters, 60, platform,
        dict(kind='platform_start', entry=10, clock=10), history)
    reduced = FullPolicyOracle(graph, model, parameters, 60, history).platform(platform, 10)
    assert original['complete'] and original['probability_mass'] == pytest.approx(1., abs=1e-10)
    assert reduced['unresolved_probability'] == 0
    assert reduced['mean']-10 == pytest.approx(original['mean'], abs=1e-9)
