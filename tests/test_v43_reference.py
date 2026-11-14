from itertools import product

import numpy as np
import pytest

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v33.nodes import Ride
from experiments.reconstruction_v41.matched_continuation import FrozenCombinedReference
from experiments.reconstruction_v43.reference import (
    DualMeasureCombinedReference, PrefixExecutor, RootConditionalLaw, evaluate_case,
)
from src.full_menu_opportunity import FullMenuOpportunity
from src.plan_waiting_controls import CombinedWaitingOpportunity, SingleTargetOpportunity
from test_v41_matched_continuation import Compiler
from test_v41_physical_posterior import fixture as physical_fixture, enumerate_chain


def enumerate_dual(reference, physical):
    control = CombinedWaitingOpportunity(SingleTargetOpportunity(reference))
    atoms = []
    for cid in reference.cids:
        times, mass, cancel = physical[cid]
        atoms.append([(float(t), float(p)) for t, p in zip(times, mass) if p > 0]
                     + ([(None, cancel)] if cancel else []))
    value = 0.
    for world in product(*atoms):
        weight = float(np.prod([p for _, p in world]))
        events = {}
        for cid, (clock, _) in zip(reference.cids, world):
            if clock is not None and reference.entry <= clock < reference.deadline:
                events.setdefault(clock, []).append(cid)
        seen, loss = set(), reference.deadline
        for clock, visible in sorted(events.items()):
            seen.update(visible)
            if clock < reference.ready:
                continue
            choice, _ = control.choose(tuple(visible), seen, clock)
            if choice is not None:
                loss = reference.score(choice, clock)
                break
        value += weight*loss
    return value


@pytest.mark.parametrize('entry,ready', [(0., 0.), (5., 15.), (5., 20.5), (40., 50.), (101., 110.)])
def test_dual_dp_matches_independent_public_event_enumeration(entry, ready):
    rows = {0: ([0., 20., 40.], [.2, .5, .2], .1, [85., 70., 60.]),
            1: ([10., 20., 50.], [.25, .5, .25], 0., [80., 35., 70.]),
            2: ([20., 30.], [.4, .3], .3, [45., 55.])}
    reference = FullMenuOpportunity(Compiler(rows), {cid: (cid, cid) for cid in rows}, entry, ready, 100.)
    physical = {0: (np.array([0., 15., 40.]), np.array([.1, .3, .2]), .4),
                1: (np.array([10., 20., 50.]), np.array([.4, .2, .3]), .1),
                2: (np.array([20., 30.]), np.array([.6, .3]), .1)}
    result = DualMeasureCombinedReference(reference, physical)
    assert result.unresolved_probability == 0
    assert result.value == pytest.approx(enumerate_dual(reference, physical), abs=1e-10)
    assert result.lower == result.upper


def test_equal_measures_reproduce_original_exact_frozen_reference():
    rows = {0: ([10., 30.], [.5, .5], 0., [90., 40.]),
            1: ([20.], [1.], 0., [70.])}
    reference = FullMenuOpportunity(Compiler(rows), {cid: (cid, cid) for cid in rows}, 0., 0., 100.)
    result = DualMeasureCombinedReference(reference, reference.distributions)
    assert result.value == pytest.approx(FrozenCombinedReference(reference).value) == 55.


def test_physical_probability_does_not_replace_the_policys_waiting_law():
    rows = {0: ([10., 30.], [.5, .5], 0., [90., 40.]),
            1: ([20.], [1.], 0., [70.])}
    reference = FullMenuOpportunity(Compiler(rows), {cid: (cid, cid) for cid in rows}, 0., 0., 100.)
    physical = {0: (np.array([10., 30.]), np.array([.9, .1]), 0.),
                1: (np.array([20.]), np.array([1.]), 0.)}
    result = DualMeasureCombinedReference(reference, physical)
    assert result.value == pytest.approx(67.)  # early A refused; take B; late A beats B.
    assert result.value != pytest.approx(FrozenCombinedReference(reference).value)


def test_unsupported_score_retains_positive_mass_as_bounds():
    class UnsupportedCompiler(Compiler):
        def clock_nominal_score(self, cid, clock):
            if clock < 10:
                raise UnsupportedHistory('Outside frozen policy score support')
            return super().clock_nominal_score(cid, clock)
    reference = FullMenuOpportunity(UnsupportedCompiler({0: ([10.], [1.], 0., [40.])}),
                                   {0: (0, 0)}, 0., 0., 100.)
    physical = {0: (np.array([5., 10.]), np.array([.2, .8]), 0.)}
    result = DualMeasureCombinedReference(reference, physical)
    assert result.value is None
    assert result.unresolved_probability == pytest.approx(.2)
    assert result.lower == pytest.approx(32.)
    assert result.upper == pytest.approx(52.)


def test_true_future_offer_does_not_restore_zero_policy_survival():
    rows = {0: ([10.], [1.], 0., [40.]), 1: ([20.], [1.], 0., [70.])}
    reference = FullMenuOpportunity(Compiler(rows), {cid: (cid, cid) for cid in rows}, 0., 0., 100.)
    physical = {0: (np.array([30.]), np.array([1.]), 0.),
                1: (np.array([20.]), np.array([1.]), 0.)}
    result = DualMeasureCombinedReference(reference, physical)
    assert result.value == pytest.approx(enumerate_dual(reference, physical)) == 70.


def test_root_conditioning_preserves_original_soft_emission_when_policy_reselects(monkeypatch):
    _, _, _, history = physical_fixture(monkeypatch, cancel=0.)
    history.apply_frame(dict(receipt_seconds=1., trip_statuses=[], endpoint_updates=[
        [0, 'arrival', 'FORECAST', 65.], [1, 'arrival', 'FORECAST', 175.]]), 1.)
    root = RootConditionalLaw(history, ('A',))
    original = root.roots['A']
    paths, mass = enumerate_chain(original)
    next_history = history.clone()
    next_history.observe(0, 'arrival', 60)
    assert next_history.selected_forecast('A')[:2] == (1, 'arrival')
    conditional = root.law('A', next_history)
    keep = original.times[1][paths[:, 1]] == 60
    expected = np.bincount(paths[keep, 3], weights=mass[keep]/mass[keep].sum(),
                          minlength=len(original.times[3]))
    assert conditional.marginal(3) == pytest.approx(expected, abs=1e-12)
    assert not np.allclose(conditional.marginal(3), next_history.law('A').marginal(3))
    assert root.law('A', next_history) is conditional


def test_prefix_platform_does_not_read_a_future_departure(monkeypatch):
    graph, model, parameters, history = physical_fixture(monkeypatch)
    from experiments.reconstruction_v33.nodes import Platform
    from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
    from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine
    platform = Platform(1, {1: Ride(1, (1,), {1: None})})
    planner = MatchedContinuationPlanner(graph, model, parameters, 240)
    current = PrefixExecutor(graph, planner, 240, engine=PhysicalDecisionEngine(planner),
                             method='rolling_fixed', waiting='combined')
    class ForbiddenWorld:
        def event(self, cid, kind):
            raise AssertionError('No future event may be read before local reference construction')
    arrival, status = current.platform(platform, 60, history, ForbiddenWorld())
    assert (arrival, status) == (60, 'next_platform')
    assert current.reached['history'].public_state() == history.public_state()
    assert current.trace == []


def test_case_preserves_terminal_worlds_and_original_seed_without_local_platform(monkeypatch):
    graph, model, parameters, history = physical_fixture(monkeypatch, cancel=0.)
    history.observe(0, 'departure', 0)
    history.observe(0, 'arrival', 60)
    # Forced ALIGHT has a platform; CONTINUE reaches the destination directly.
    from experiments.reconstruction_v33.nodes import Platform
    platform = Platform(1, {1: Ride(1, (1,), {1: None})})
    node = Ride(0, (0, 1), {0: platform, 1: None})
    result = evaluate_case(graph, model, parameters, 240, node,
        dict(kind='onboard', cid=0, clock=60), history, seed=43007)
    assert result['count'] == result['generated_count'] == 128
    assert result['seed'] == 43007 and result['future_messages'] == 0
    assert all(row['prefix']['kind'] == 'platform' for row in result['rows']['alight'])
    assert all(row['prefix']['kind'] == 'terminal' for row in result['rows']['continue'])
    assert all(row['prefix']['trace'] == [] for row in result['rows']['alight'])
    for row in result['rows']['continue']:
        assert row['root'] == row['updated'] == row['working']
    assert result['cache_stats']['queries'] == 128 and result['cache_stats']['compiles'] == 1
    law = history.law('A')
    expected = law.marginal(3) @ law.times[3]-60
    assert result['summary']['updated']['alight']['mean'] == pytest.approx(expected, abs=1e-10)
    # The other arm keeps its original realized sample values. Replacing one
    # terminal tail by its expectation need not make the finite sample mean
    # difference exactly zero, even when the two population values coincide.
    assert result['summary']['updated']['continue']['mean'] == pytest.approx(
        np.mean([row['updated']['mean'] for row in result['rows']['continue']]))
    assert result['root_independence']['repeated_root_trips'] == 0
