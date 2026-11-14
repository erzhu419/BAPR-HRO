from itertools import product

import numpy as np
import pytest

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v33.nodes import Ride
from experiments.reconstruction_v41.physical_posterior import (
    CheckingWorld, MODEL_VERSION, PhysicalDecisionEngine, PhysicalHistory, PhysicalPlanner,
    restore_physical_snapshot,
)
from src import evolving_trip_belief as evolving
from src.transit_graph import Connection, Stop, TransitGraph


def fixture(monkeypatch, cancel=.1):
    # A bounded analytical toy; production DELAYS and fitted parameters are untouched.
    monkeypatch.setattr(evolving, 'DELAYS', np.array([0., 1.]))
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    graph.add_connection(Connection(0, 'R', 'A', 0, 1, 0, 60))
    graph.add_connection(Connection(1, 'R', 'A', 1, 2, 60, 120))
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False,
        travel_sd=.5, dwell_sd=.5, cancel_persistence=.8)
    parameters = {'R': dict(mean=0., std=.5, cancel_rate=cancel)}
    fit = dict(route_kind={}, pooled={kind: dict(bias_seconds=0., sd_seconds=30.)
                                      for kind in ('departure', 'arrival')})
    return graph, model, parameters, PhysicalHistory(model, parameters, fit=fit)


def enumerate_chain(law):
    paths, mass = [], []
    for path in product(*(range(len(times)) for times in law.times)):
        logp = law.initial[path[0]]+sum(emission[state] for emission, state in zip(law.emissions, path))
        logp += sum(law._transition(slot)[path[slot], path[slot+1]] for slot in range(len(path)-1))
        if np.isfinite(logp):
            paths.append(path)
            mass.append(np.exp(logp))
    mass = np.asarray(mass)
    return np.asarray(paths), mass/mass.sum()


def test_exact_off_grid_anchor_has_no_backwards_next_departure_and_matches_enumeration(monkeypatch):
    graph, model, parameters, history = fixture(monkeypatch)
    history.observe(0, 'departure', 0)
    history.observe(0, 'arrival', 73)
    law = history.law('A')
    assert law.model_version == MODEL_VERSION
    assert history.assert_physical_support('A')
    paths, mass = enumerate_chain(law)
    for slot in range(4):
        expected = np.bincount(paths[:, slot], weights=mass, minlength=len(law.times[slot]))
        assert law.marginal(slot) == pytest.approx(expected, abs=1e-12)
    assert sum(law.marginal(1)[law.times[1] == 73]) == pytest.approx(1.)
    assert law.marginal(2)[law.times[2] < 73].sum() == 0.
    sampled = history.sample_vehicle('A', np.random.default_rng(411).random((10000, 4)))
    assert np.all(sampled['times'][:, 0] == 0) and np.all(sampled['times'][:, 1] == 73)
    assert np.all(np.diff(sampled['times'], axis=1) >= 0)
    expected = float(law.marginal(3) @ law.times[3])
    assert sampled['times'][:, 3].mean() == pytest.approx(expected, abs=1.)


def test_in_grid_simulated_observation_conditions_same_basis_and_exact_conditional_kernel(monkeypatch):
    _, _, _, history = fixture(monkeypatch, cancel=0.)
    history.observe(0, 'arrival', 73)
    before = history.law('A')
    basis = history._physical_bases['A']
    departure_slot = 2
    departure = 120
    index = np.flatnonzero(before.times[departure_slot] == departure)[0]
    expected = history.transition_kernel('A', departure_slot)[index]
    child = history.clone()
    child.observe(1, 'departure', departure)
    after = child.law('A')
    assert child._physical_bases['A'] is basis
    assert child.basis_context() == history.basis_context()
    assert all(a is b for a, b in zip(before.times, after.times))
    assert after.marginal(3) == pytest.approx(expected, abs=1e-12)
    assert history._records != child._records


def test_only_new_real_off_grid_clock_refines_the_basis_and_the_original_remains_unchanged(monkeypatch):
    _, _, _, history = fixture(monkeypatch)
    history.observe(0, 'arrival', 73)
    history.law('A')
    before = history.basis_context()
    child = history.clone()
    child.observe(1, 'departure', 81)
    law = child.law('A')
    assert child.basis_context() != before and history.basis_context() == before
    assert law.marginal(2)[law.times[2] == 81].sum() == pytest.approx(1.)
    assert child.assert_physical_support('A')


def test_unseen_window_preserves_cancellation_and_past_mass_then_exact_receipt_replaces_it(monkeypatch):
    _, _, _, history = fixture(monkeypatch)
    history.observe_platform((1,), 60, 80, {})
    law = history.law('A')
    slot = 2
    assert law.marginal(slot)[~law.canceled[slot] & (law.times[slot] >= 60) & (law.times[slot] <= 80)].sum() == 0.
    assert law.marginal(slot)[law.canceled[slot]].sum() > 0
    history.observe(1, 'departure', 81)
    law = history.law('A')
    assert history._records['A'][-1] == (1, 'departure', 'observed', 81.)
    assert law.marginal(slot)[(law.times[slot] == 81) & ~law.canceled[slot]].sum() == pytest.approx(1.)


def test_forecast_reselection_retains_raw_messages_and_same_physical_basis(monkeypatch):
    _, _, _, history = fixture(monkeypatch)
    history.apply_frame(dict(receipt_seconds=1., trip_statuses=[], endpoint_updates=[
        [0, 'arrival', 'FORECAST', 65.], [1, 'arrival', 'FORECAST', 145.]]), 1.)
    first = history.law('A')
    basis = history._physical_bases['A']
    assert history.selected_forecast('A')[:2] == (0, 'arrival')
    history.observe(0, 'arrival', 60)
    second = history.law('A')
    assert history.selected_forecast('A')[:2] == (1, 'arrival')
    assert (1, 'arrival') in history._forecasts and history._physical_bases['A'] is basis
    assert not np.array_equal(first.marginal(3), second.marginal(3))


def test_planner_and_world_reader_share_the_same_law_and_never_overwrite_clock(monkeypatch):
    graph, model, parameters, history = fixture(monkeypatch)
    history.observe(0, 'departure', 0)
    history.observe(0, 'arrival', 73)
    planner = PhysicalPlanner(graph, model, parameters, 240)
    compiled = planner.compile_ride(Ride(1, (1,), {1: None}), history, 'rolling_fixed')
    assert np.all(compiled.departure_times >= 73)
    assert planner.stats['prior_future_edge_queries'] == 0
    assert planner._edge('A', 2, history) == pytest.approx(history.transition_kernel('A', 2))
    bank = TripWorlds(history.as_belief(), parameters, [(0, 1)], 16, 411)
    assert all(CheckingWorld(bank, i, history.public_state()).event(0, 'arrival')[0] == 73 for i in range(16))
    with pytest.raises(AssertionError, match='disagrees'):
        CheckingWorld(bank, 0, [dict(cid=0, kind='arrival', evidence='observed', clock=74)]).event(0, 'arrival')


def test_contradictory_public_clocks_raise_without_clamping_or_dropping_worlds(monkeypatch):
    _, _, _, history = fixture(monkeypatch)
    history.observe(0, 'arrival', 73)
    history.observe(1, 'departure', 70)
    with pytest.raises(UnsupportedHistory, match='ordered'):
        history.law('A')


def test_snapshot_restore_preserves_complete_raw_public_evidence(monkeypatch):
    _, model, parameters, history = fixture(monkeypatch)
    history.observe(0, 'arrival', 73)
    history.observe_platform((1,), 73, 78, {})
    restored = restore_physical_snapshot(model, parameters, history.fit, history.public_state(), 78)
    assert restored.public_state() == history.public_state()
    assert restored.assert_physical_support('A')
    assert restored.frames == ()


def test_current_action_cache_distinguishes_independent_bases_but_reuses_a_clone(monkeypatch):
    graph, model, parameters, evolving_history = fixture(monkeypatch)
    # Independent reconstructions can have distinct object identities. The
    # cache safely distinguishes them; query timing must not alter their law.
    evolving_history.law('A')
    evolving_history.observe(0, 'arrival', 60)
    fresh_history = PhysicalHistory(model, parameters, fit=evolving_history.fit)
    fresh_history.observe(0, 'arrival', 60)
    evolving_history.law('A')
    fresh_history.law('A')
    assert evolving_history.public_state() == fresh_history.public_state()
    assert evolving_history.basis_context() != fresh_history.basis_context()
    assert evolving_history.law('A').marginal(3) == pytest.approx(fresh_history.law('A').marginal(3))
    planner = PhysicalPlanner(graph, model, parameters, 240)
    engine = PhysicalDecisionEngine(planner, variant='frontier')
    ride = Ride(0, (0, 1), {0: None, 1: None})
    engine.onboard(ride, 0, 60, evolving_history)
    engine.onboard(ride, 0, 60, fresh_history)
    assert engine.stats['onboard_unique'] == 2 and engine.stats['onboard_hits'] == 0
    engine.onboard(ride, 0, 60, fresh_history.clone())
    assert engine.stats['onboard_hits'] == 1


def test_early_late_queries_and_restoration_have_identical_support_marginals_and_kernels(monkeypatch):
    _, model, parameters, early = fixture(monkeypatch)
    late = PhysicalHistory(model, parameters, fit=early.fit)
    early.law('A')
    observations = ((0, 'departure', 0), (0, 'arrival', 73), (1, 'departure', 120))
    for cid, kind, clock in observations:
        early.observe(cid, kind, clock)
        early.law('A')
        late.observe(cid, kind, clock)
    restored = restore_physical_snapshot(model, parameters, early.fit, early.public_state(), 120)
    histories = (early, late, restored)
    laws = [history.law('A') for history in histories]
    assert all(history._physical_anchors['A'] == ((0, 'arrival', 73.),) for history in histories)
    for candidate in laws[1:]:
        for slot in range(4):
            assert np.array_equal(laws[0].times[slot], candidate.times[slot])
            assert candidate.marginal(slot) == pytest.approx(laws[0].marginal(slot), abs=1e-12)
        for slot in range(3):
            assert np.array_equal(laws[0]._transition(slot), candidate._transition(slot))
    for history in histories[1:]:
        for slot in range(3):
            assert history.transition_kernel('A', slot) == pytest.approx(early.transition_kernel('A', slot), abs=1e-12)
    tape = np.random.default_rng(412).random((128, 4))
    assert np.array_equal(early.sample_vehicle('A', tape)['times'], late.sample_vehicle('A', tape)['times'])


def test_sampled_clone_observations_do_not_rebuild_and_match_lazy_replay(monkeypatch):
    _, model, parameters, parent = fixture(monkeypatch)
    parent.observe(0, 'arrival', 73)
    sample = parent.sample_vehicle('A', np.random.default_rng(413).random((1, 4)))
    child = parent.clone()
    basis = child._physical_bases['A']
    child.observe(1, 'departure', int(sample['times'][0, 2]))
    child.law('A')
    assert child._physical_bases['A'] is basis
    restored = restore_physical_snapshot(model, parameters, child.fit, child.public_state(), 200)
    restored_law = restored.law('A')
    assert restored._physical_anchors['A'] == child._physical_anchors['A']
    for slot in range(4):
        assert np.array_equal(restored_law.times[slot], child.law('A').times[slot])
        assert restored_law.marginal(slot) == pytest.approx(child.law('A').marginal(slot), abs=1e-12)
