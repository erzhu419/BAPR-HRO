"""Mathematical boundaries of the V39 remaining-menu sufficient state.

These are finite-law counterexamples, not added research worlds or fitted
success cases. They protect the raw evidence retained by the implementation.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v32.conditional import departure_evidence
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from experiments.reconstruction_v39.state import raw_history_key
from src.evolving_trip_belief import _ChainLaw
from src.full_menu_opportunity import FullMenuOpportunity
from src.onboard_stopping import OnboardStopping
from src.plan_waiting_controls import (
    CombinedWaitingOpportunity, SingleTargetOpportunity, TwoTargetOpportunity,
)
from src.transit_graph import Connection, Stop, TransitGraph


def finite_law(times, initial, kernels):
    with np.errstate(divide='ignore'):
        initial = np.log(initial)
        kernels = [np.log(kernel) for kernel in kernels]
    return _ChainLaw([np.asarray(row, dtype=float) for row in times], initial,
                     lambda slot: kernels[slot], [np.zeros(len(row)) for row in times],
                     [np.zeros(len(row), bool) for row in times],
                     tuple(True for _ in kernels))


def prefix_law():
    return finite_law([[0, 1], [10, 20, 30], [100, 200, 300]], [.5, .5],
                      [np.array([[.5, 0, .5], [0, 1, 0]]), np.eye(3)])


def observe(law, slot, clock):
    return departure_evidence(law, slot, clock, clock, observed=clock)


def test_latest_offgrid_clock_does_not_summarize_the_active_trip_prefix():
    prior = prefix_law()
    prefix, _ = observe(prior, 0, 0)
    complete, receipt = observe(prefix, 1, 15)
    forgotten, _ = observe(prior, 1, 15)
    assert receipt['mode'] == 'adjacent_active_projection'
    assert complete.marginal(2) == pytest.approx([.75, 0, .25])
    assert forgotten.marginal(2) == pytest.approx([.5, .5, 0])
    # The same latest clock would incorrectly flip a current stopping action.
    tail = np.array([40., 40., 140.])
    assert complete.marginal(2) @ tail == pytest.approx(65.)
    assert forgotten.marginal(2) @ tail == pytest.approx(40.)
    assert 50. < complete.marginal(2) @ tail
    assert 50. > forgotten.marginal(2) @ tail


def test_base_grid_membership_is_not_active_on_grid_observation():
    prior = prefix_law()
    prefix, _ = observe(prior, 0, 0)
    complete, active_receipt = observe(prefix, 1, 20)
    forgotten, base_receipt = observe(prior, 1, 20)
    assert active_receipt['mode'] == 'adjacent_active_projection'
    assert base_receipt['mode'] == 'on_grid'
    assert complete.marginal(2) == pytest.approx([.5, 0, .5])
    assert forgotten.marginal(2) == pytest.approx([0, 1, 0])


def test_actual_active_point_observation_separates_earlier_markov_evidence():
    prior = finite_law([[0, 1], [10, 20, 30], [100, 200]], [.5, .5],
        [np.array([[.5, 0, .5], [0, 1, 0]]),
         np.array([[.7, .3], [.5, .5], [.1, .9]])])
    prefix, _ = observe(prior, 0, 0)
    complete, receipt = observe(prefix, 1, 10)
    forgotten, _ = observe(prior, 1, 10)
    assert receipt['mode'] == 'on_grid'
    assert complete.marginal(2) == pytest.approx([.7, .3])
    assert complete.marginal(2) == pytest.approx(forgotten.marginal(2))


def forecast_history(later_forecast):
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    graph.add_connection(Connection(0, 'R', 'a', 0, 1, 0, 10))
    graph.add_connection(Connection(1, 'R', 'a', 1, 2, 21, 30))
    law = finite_law([[0], [10, 20], [21], [30, 60]], [1.],
                    [np.array([[.5, .5]]), np.ones((2, 1)), np.array([[.5, .5]])])
    positions = {(cid, kind): ('a', slot) for slot, (cid, kind) in enumerate(
        ((0, 'departure'), (0, 'arrival'), (1, 'departure'), (1, 'arrival')))}
    model = SimpleNamespace(graph=graph, _positions=positions, _law=lambda trip, params: law)
    fit = dict(pooled={kind: dict(bias_seconds=0., sd_seconds=30., n=10)
                       for kind in ('departure', 'arrival')}, route_kind={})
    history = SingleMessageHistory(model, {'R': {'std': 1.}}, fit=fit)
    history.apply_frame(dict(receipt_seconds=1, trip_statuses=[], endpoint_updates=[
        [0, 'arrival', 'FORECAST', 10], [1, 'arrival', 'FORECAST', later_forecast]]), 1)
    return history


def test_same_selected_forecast_is_not_a_full_continuation_state():
    left, right = forecast_history(30), forecast_history(60)
    assert left.signature() == right.signature()
    assert left.law('a').marginal(3) == pytest.approx(right.law('a').marginal(3))
    assert raw_history_key(left, {'a'})[0] != raw_history_key(right, {'a'})[0]
    for history in (left, right):
        history.observe(0, 'arrival', 10)
        assert history.selected_forecast('a')[:2] == (1, 'arrival')
    assert left.law('a').marginal(3)[0] > .6
    assert right.law('a').marginal(3)[0] < .4


def test_arrival_decisions_have_suffix_values_but_boarding_roots_do_not():
    times = [[0], [10, 20], [30, 40]]
    costs = [[1.], [50., 80.], [40., 90.]]
    kernels = [np.array([[.3, .5]]), np.array([[.8, .1], [.2, .6]])]
    complete = OnboardStopping(times, costs, kernels, 100).solve([1.])
    suffix = OnboardStopping(times[1:], costs[1:], kernels[1:], 100).solve([.3, .5])
    for method in ('optimal', 'rolling_fixed', 'one_step_rollout'):
        before, after = complete.policies[method], suffix.policies[method]
        for offset in range(2):
            assert after.decision_costs[offset] == pytest.approx(before.decision_costs[offset+1])
            assert after.values[offset] == pytest.approx(before.values[offset+1])
            assert np.array_equal(after.actions[offset], before.actions[offset+1])
        assert before.root_value == 1.
        assert after.root_value > 1.


@pytest.mark.parametrize('control', ['nominal', 'single', 'combined', 'pair'])
def test_observed_departed_root_can_be_removed_without_changing_wait_comparison(control):
    distributions = {
        0: (np.array([15.]), np.array([1.]), 0.),
        1: (np.array([20.]), np.array([1.]), 0.),
        2: (np.array([5., 25., 40.]), np.array([.1, .4, .4]), .1),
        3: (np.array([5., 25., 50.]), np.array([.2, .4, .3]), .1),
    }
    compiler = SimpleNamespace(departure_distribution=distributions.__getitem__,
        clock_nominal_score=lambda cid, clock: {0: 11., 1: 75., 2: clock+20., 3: clock+25.}[cid])

    def choose(cids, seen):
        reference = FullMenuOpportunity(compiler, {cid: cid for cid in cids}, 12, 12, 100)
        rule = reference
        if control != 'nominal':
            rule = SingleTargetOpportunity(reference)
        if control in ('combined', 'pair'):
            rule = CombinedWaitingOpportunity(rule)
        if control == 'pair':
            rule = TwoTargetOpportunity(rule)
        return rule.choose((1,), seen, 20)

    # Root 0 has departed; root 1 is already observed but currently visible.
    # Unseen roots retain their unobserved-past mass and the original entry.
    assert choose((0, 1, 2, 3), (0, 1)) == choose((1, 2, 3), (1,))
