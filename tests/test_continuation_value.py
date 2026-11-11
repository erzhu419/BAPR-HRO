"""Analytic checks for conditional boarding and causal rescue values."""

import numpy as np
import pytest

from src.pmf import PMF
from src.continuation_value import (
    JointOutcome, shared_trip_outcomes, independent_trip_outcomes,
    summarize_transition, combine_rescue_costs,
)


def summarize(outcomes, **changes):
    settings = dict(min_transfer_time=0, same_trip=False, journey_departure=0, cap=120)
    settings.update(changes)
    return summarize_transition(outcomes, **settings)


def test_shared_trip_delay_preserves_continuation_and_avoids_cancel_redraw():
    # A boarded trip has D=0 or 10 equally likely; the missing .2 mass is
    # impossible after observing that this very trip is already operating.
    delay = PMF(np.array([.4] + [0.] * 9 + [.4]), 0)
    outcomes = shared_trip_outcomes(delay, scheduled_incoming_arrival=10,
                                   scheduled_next_departure=10, scheduled_next_arrival=20)
    result = summarize(outcomes, same_trip=True, min_transfer_time=2)
    assert result.catch_probability == 1
    assert result.conditional_catch_loss == 25
    assert result.failure_probability == 0
    assert all(row.next_departure == row.incoming_arrival for row in outcomes)


def test_independent_two_point_transfer_has_different_joint_success():
    delay = PMF(np.array([.5] + [0.] * 9 + [.5]), 0)
    outcomes = independent_trip_outcomes(delay, delay,
        scheduled_incoming_arrival=10, scheduled_next_departure=10,
        scheduled_next_arrival=20, cancel_announcement=10)
    result = summarize(outcomes)
    assert result.catch_probability == .75
    assert result.conditional_catch_loss == pytest.approx(80 / 3)
    assert result.failure_probability == .25
    buffered = summarize(outcomes, min_transfer_time=2)
    assert buffered.catch_probability == .25
    assert buffered.conditional_catch_loss == 30


def test_catch_cost_conditions_on_the_same_delay_that_determines_boarding():
    result = summarize([
        JointOutcome(.5, 5, 0, 10),
        JointOutcome(.5, 5, 10, 20),
    ])
    assert result.catch_probability == .5
    assert result.conditional_catch_loss == 20
    branch = result.failure_branches[0]
    assert branch.state.kind == 'missed_departure'
    assert branch.state.time == 5
    assert branch.state.observed_departure is None
    assert combine_rescue_costs(result, {branch.state: 120}) == 70
    # Multiplying p_catch by the unconditional arrival mean gives 67.5.


def test_explicit_cancel_mass_is_not_a_departure_tail():
    outcomes = independent_trip_outcomes(PMF.deterministic(0), PMF(np.array([0.]), 0),
        scheduled_incoming_arrival=10, scheduled_next_departure=12,
        scheduled_next_arrival=20, cancel_announcement=11)
    result = summarize(outcomes, min_transfer_time=2)
    assert result.catch_probability == 0
    assert result.conditional_catch_loss is None
    branch = result.failure_branches[0]
    assert branch.probability == 1
    assert branch.state.kind == 'canceled'
    assert branch.state.time == 12  # transfer is still in progress at announcement
    assert combine_rescue_costs(result, {branch.state: 30}) == 30


def test_unobserved_cancellation_and_late_vehicle_share_a_timeout_belief():
    result = summarize([
        JointOutcome(.5, 0, 20, 30),
        JointOutcome(.5, 0, None, None, True, 20),
    ])
    assert result.catch_probability == 0
    assert len(result.failure_branches) == 1
    branch = result.failure_branches[0]
    assert branch.state.time == 12
    assert branch.state.kind == 'wait_timeout'
    assert branch.state.observed_departure is None
    assert branch.state.cancellation_time is None
    assert branch.outcome_weights == ((0, .5), (1, .5))


def test_rescue_uses_one_action_per_observable_branch():
    result = summarize([
        JointOutcome(.5, 0, 20, 30),
        JointOutcome(.5, 0, None, None, True, 20),
    ])
    branch = result.failure_branches[0]
    rescue_action_losses = np.array([[20., 60.], [60., 20.]])
    # One action is chosen under the timeout posterior. It cannot use which
    # hidden future happened to select a different rescue in each world.
    weights = np.array([p for _, p in branch.outcome_weights])
    causal_rescue = min(rescue_action_losses @ weights)
    assert causal_rescue == 40
    assert combine_rescue_costs(result, {branch.state: causal_rescue}) == 40
    assert (rescue_action_losses.min(axis=0) @ weights) == 20  # clairvoyant comparator


def test_failure_cannot_be_revealed_before_transfer_readiness():
    missed = summarize([JointOutcome(1, 0, 1, 10)], min_transfer_time=2)
    assert missed.failure_branches[0].state.time == 2
    assert missed.failure_branches[0].state.kind == 'missed_departure'
    late = summarize([JointOutcome(1, 0, 20, 30)], min_transfer_time=2)
    assert late.failure_branches[0].state.time == 14
    assert late.failure_branches[0].state.kind == 'wait_timeout'


def test_missed_departures_are_pooled_until_an_exact_report_is_received():
    result = summarize([
        JointOutcome(.25, 5, 0, 10),
        JointOutcome(.25, 5, 2, 12, departure_reported_at=6),
        JointOutcome(.50, 5, 3, 13, departure_reported_at=5),
    ])
    branches = {branch.state.observed_departure: branch for branch in result.failure_branches}
    assert set(branches) == {None, 3}
    assert branches[None].probability == .5
    assert branches[None].outcome_weights == ((0, .5), (1, .5))
    assert branches[3].outcome_weights == ((2, 1.),)
    assert all(branch.state.time == 5 for branch in branches.values())


def test_horizon_does_not_reveal_an_unobserved_future_incoming_arrival():
    result = summarize([JointOutcome(.5, 150, 160, 170),
                        JointOutcome(.5, 180, None, None, True, 160)])
    assert len(result.failure_branches) == 1
    state = result.failure_branches[0].state
    assert state.time == 120
    assert state.kind == 'horizon'
    assert state.incoming_arrival is None
    assert state.cancellation_time is None


def test_gaussian_shared_and_independent_delays_preserve_their_joint_difference():
    offsets = np.arange(-5, 65)
    p = np.exp(-.5 * ((offsets - 1) / 2) ** 2)
    p /= p.sum()
    delay = PMF(p, -5)
    shared = shared_trip_outcomes(delay, scheduled_incoming_arrival=10,
        scheduled_next_departure=12, scheduled_next_arrival=20)
    independent = independent_trip_outcomes(delay, delay.scale(.9),
        scheduled_incoming_arrival=10, scheduled_next_departure=12,
        scheduled_next_arrival=20, cancel_announcement=12)
    shared_value = summarize(shared, same_trip=True, min_transfer_time=2)
    independent_value = summarize(independent, min_transfer_time=2)
    assert shared_value.catch_probability == pytest.approx(1)
    # With equal independent integer delays, P(D_out >= D_in) includes ties.
    # The 12min waiting window additionally removes D_out-D_in > 12.
    expected = .9 * sum(p[i] * p[j] for i in range(len(p))
                        for j in range(len(p)) if 0 <= j - i <= 12)
    assert independent_value.catch_probability == pytest.approx(expected)
    assert independent_value.catch_probability < .6
