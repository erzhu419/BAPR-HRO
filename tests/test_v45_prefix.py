from itertools import product

import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine, PhysicalHistory
from experiments.reconstruction_v43.reference import PrefixExecutor, RootConditionalLaw
from experiments.reconstruction_v44 import matched
from experiments.reconstruction_v45.prefix import (
    CanonicalDecisionEngine, _Query, evaluate, structure_gate,
)
from src import evolving_trip_belief as evolving
from src.transit_graph import Connection, Stop, TransitGraph


def fixture(monkeypatch, *, cancel=.05, forecasts=False, deadline=240):
    monkeypatch.setattr(evolving, 'DELAYS', np.array([0., 1., 2.]))
    graph = TransitGraph()
    for stop in range(5):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for cid, trip, origin, target, departure, arrival in (
            (0, 'head', 0, 1, 0, 10), (1, 'head', 1, 2, 20, 30),
            (2, 'head', 2, 3, 40, 50), (3, 'A', 1, 4, 35, 85),
            (4, 'B', 2, 4, 75, 100), (5, 'C', 3, 4, 90, 115)):
        graph.add_connection(Connection(cid, 'R', trip, origin, target, departure, arrival))
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False,
        travel_sd=.5, dwell_sd=.5, cancel_persistence=.8)
    parameters = {'R': dict(mean=0., std=.5, cancel_rate=cancel)}
    fit = dict(route_kind={}, pooled={kind: dict(bias_seconds=0., sd_seconds=30.)
                                      for kind in ('departure', 'arrival')})
    history = PhysicalHistory(model, parameters, fit=fit)
    history.observe(0, 'departure', 0)
    history.observe(0, 'arrival', 10)
    if forecasts:
        history.apply_frame(dict(receipt_seconds=1., trip_statuses=[], endpoint_updates=[
            [2, 'arrival', 'FORECAST', 170.]]), 1.)
        history.apply_frame(dict(receipt_seconds=2., trip_statuses=[], endpoint_updates=[
            [1, 'departure', 'FORECAST', 25.]]), 2.)
    exits = {cid: Platform(cid+1, {tail: Ride(tail, (tail,), {tail: None})})
             for cid, tail in ((0, 3), (1, 4), (2, 5))}
    node = Ride(0, (0, 1, 2), exits)
    row = dict(kind='onboard', cid=0, clock=10)
    return graph, model, parameters, deadline, node, row, history


@pytest.mark.parametrize('forecasts', [False, True])
def test_markov128_preserves_every_original_v44_prefix_and_value(monkeypatch, forecasts):
    args = fixture(monkeypatch, forecasts=forecasts)
    original = matched.evaluate(*args, seed=45003)
    compressed = evaluate(*args, seed=45003, mode='markov128')
    assert compressed['structural']['eligible'] and compressed['fallback'] is None
    assert compressed['generated_count'] == 128
    for arm in ('alight', 'continue'):
        for a, b in zip(original['rows'][arm], compressed['rows'][arm], strict=True):
            assert a['world_index'] == b['world_index']
            assert a['prefix'] == b['prefix']
            assert a['value'] == b['value']
    for key in ('mean_gain', 'mean_alight', 'mean_continue', 'se', 'full_bank_mean_gain_interval'):
        assert compressed['paired'][key] == original['paired'][key]
    assert compressed['canonical_stats']['canonical_queries'] > 0
    assert compressed['canonical_stats']['canonical_hits'] > 0


def test_same_latest_arrival_screens_different_past_and_preserves_reselection(monkeypatch):
    graph, model, params, deadline, node, row, history = fixture(monkeypatch, forecasts=True, cancel=0.)
    for trip in node.trips(model):
        history.law(trip)
    left, right = history.clone(), history.clone()
    for candidate, departure in ((left, 20), (right, 80)):
        candidate.observe(1, 'departure', departure)
        candidate.observe(1, 'arrival', 90)
    planner = MatchedContinuationPlanner(graph, model, params, deadline)
    original = PhysicalDecisionEngine(planner, variant='frontier')
    compressed = CanonicalDecisionEngine(planner, history)
    before = [h.public_state() for h in (left, right)]
    expected = [original.onboard(node, 1, 90, h) for h in (left, right)]
    actual = [compressed.onboard(node, 1, 90, h) for h in (left, right)]
    assert expected == actual
    assert expected[0] == expected[1]
    assert [h.public_state() for h in (left, right)] == before
    canonical = compressed.canonical_history(1, 90)
    assert canonical.selected_forecast('head') == left.selected_forecast('head')
    assert canonical.selected_forecast('head')[:2] == (2, 'arrival')
    assert len(canonical._records['head']) < len(left._records['head'])
    root = RootConditionalLaw(history, node.trips(model))
    assert not np.allclose(root.law('head', left).marginal(5), canonical.law('head').marginal(5))


def test_structure_rejects_shared_head_tail_and_noncontiguous_path(monkeypatch):
    graph, model, params, deadline, node, row, history = fixture(monkeypatch)
    shared = Ride(0, node.path, dict(node.exits))
    shared.exits[0] = Platform(1, {1: Ride(1, (1, 2), {2: None})})
    assert structure_gate(model, shared, row)['same_trip_tail_exits'] == [0]
    assert not structure_gate(model, shared, row)['eligible']
    skipped = Ride(0, (0, 2), {0: node.exits[0], 2: node.exits[2]})
    assert not structure_gate(model, skipped, row)['contiguous_slots']
    assert not structure_gate(model, skipped, row)['eligible']


def test_structural_fallback_keeps_the_original_sampler_and_no_population_claim(monkeypatch):
    args = list(fixture(monkeypatch))
    node = args[4]
    node.exits[0] = Platform(1, {1: Ride(1, (1, 2), {2: None})})
    original = matched.evaluate(*args, seed=45004)
    fallback = evaluate(*args, seed=45004, mode='absorbing')
    assert fallback['estimator'] == 'structural_fallback_mc128'
    assert fallback['generated_count'] == 128 and fallback['full_pi_bounds'] is None
    assert fallback['rows'] == original['rows']
    assert fallback['paired'] == original['paired']


def enumerate_prefixes(query):
    """Independent full joint path sum followed by the original executor."""
    trip = query.model._positions[query.row['cid'], 'arrival'][0]
    law = query.root_law.roots[trip]
    atoms, total = {}, 0.
    indexes = [np.flatnonzero(law.marginal(slot) > 0) for slot in range(len(law.times))]
    terms = []
    for path in product(*indexes):
        logp = law.initial[path[0]]+sum(row[state] for row, state in zip(law.emissions, path))
        logp += sum(law._transition(slot)[path[slot], path[slot+1]] for slot in range(len(path)-1))
        if np.isfinite(logp):
            terms.append((path, float(np.exp(logp))))
    normalization = sum(p for _, p in terms)
    planner = MatchedContinuationPlanner(query.graph, query.model, query.parameters, query.deadline)
    engine = PhysicalDecisionEngine(planner, variant='frontier')
    for path, probability in terms:
        weight = probability/normalization
        class World:
            def event(self, cid, kind):
                actual_trip, slot = query.model._positions[cid, kind]
                assert actual_trip == trip
                return int(law.times[slot][path[slot]]), bool(law.canceled[slot][path[slot]]), trip
        controller = PrefixExecutor(query.graph, planner, query.deadline,
            engine=engine, method='rolling_fixed', waiting='combined')
        outcome = controller.forced(query.node, query.row, 'continue', query.snapshot.clone(), World())
        if outcome['status'] == 'next_platform':
            reached = controller.reached
            key = ('platform', reached['platform'].stop, reached['entry'])
            value = query.local_value(reached['platform'], reached['entry'])['mean']
        else:
            key = ('terminal', outcome['status'], outcome['loss'])
            value = outcome['loss']
        assert value is not None
        atoms[key] = atoms.get(key, 0.)+weight
        total += weight*value
    return atoms, total


@pytest.mark.parametrize('deadline', [110, 240])
def test_absorbing_population_matches_independent_paths_and_preserves_failure_mass(monkeypatch, deadline):
    args = fixture(monkeypatch, forecasts=True, deadline=deadline)
    query = _Query(*args)
    expected_atoms, expected_value = enumerate_prefixes(query)
    exact = evaluate(*args, seed=45005, mode='absorbing')
    assert exact['estimator'] == 'exact_population_kappa'
    assert exact['generated_count'] == exact['count'] == 0
    assert exact['paired']['se'] is None
    assert exact['probability_conservation'] == pytest.approx(
        {'alight': 1., 'continue': 1.}, abs=1e-12)
    actual_atoms = {}
    for atom in exact['atoms']['continue']:
        key = (('platform', atom['stop'], atom['entry']) if atom['kind'] == 'platform'
               else ('terminal', atom['status'], atom['value']['mean']))
        actual_atoms[key] = actual_atoms.get(key, 0.)+atom['probability']
    assert actual_atoms == pytest.approx(expected_atoms, abs=1e-12)
    assert exact['actions']['continue']['mean'] == pytest.approx(expected_value, abs=1e-10)
    assert sum(a['probability'] for a in exact['atoms']['continue']
               if a['kind'] == 'terminal' and a['status'] != 'arrived') > 0
    assert exact['full_pi_bounds']['alight'] == {'lower': 0., 'upper': deadline-10.}
    assert exact['full_pi_difference_interval'][0] <= 0 <= exact['full_pi_difference_interval'][1]


def test_absorbing_never_draws_worlds(monkeypatch):
    args = fixture(monkeypatch, cancel=0.)
    def forbidden(*args, **kwargs):
        raise AssertionError('Population absorption must not generate Monte Carlo worlds')
    monkeypatch.setattr('experiments.reconstruction_v45.prefix.TripWorlds', forbidden)
    result = evaluate(*args, seed=45006, mode='absorbing')
    assert result['generated_count'] == 0 and result['requested_count'] == 128


def test_unknown_policy_score_keeps_probability_as_interval_not_physical_failure(monkeypatch):
    args = fixture(monkeypatch)
    from experiments.reconstruction_v32.conditional import UnsupportedHistory
    def unknown(*args, **kwargs):
        raise UnsupportedHistory('An undefined working policy score')
    monkeypatch.setattr(CanonicalDecisionEngine, 'onboard', unknown)
    result = evaluate(*args, seed=45007, mode='absorbing')
    unresolved = [a for a in result['atoms']['continue'] if a['kind'] == 'unresolved_prefix']
    assert unresolved and result['actions']['continue']['mean'] is None
    for atom in unresolved:
        assert atom['value']['lower'] == atom['entry']-args[5]['clock']
        assert atom['value']['upper'] == args[3]-args[5]['clock']
        assert atom['value']['unresolved_probability'] == 1.
    assert result['probability_conservation']['continue'] == pytest.approx(1., abs=1e-12)
    assert result['full_pi_bounds']['continue']['lower'] < result['full_pi_bounds']['continue']['upper']
