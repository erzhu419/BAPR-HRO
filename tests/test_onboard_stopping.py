from itertools import product

import numpy as np
import pytest

from src.onboard_stopping import OnboardStopping


def adaptation_chain(current_cost=40.):
    return OnboardStopping(
        ([0], [10, 20], [21, 22]), ([current_cost], [20., 90.], [80., 30.]),
        ([[.5, .5]], np.eye(2)), deadline=100.)


def test_future_observed_arrival_creates_adaptation_and_small_rollout_absorbs_it():
    result = adaptation_chain().solve()
    assert result.fixed_root_values == pytest.approx([40., 55., 55.])
    rolling, optimal, rollout = (result.policies[k] for k in
                                ('rolling_fixed', 'optimal', 'one_step_rollout'))
    assert rolling.root_value == 40.
    assert optimal.root_value == rollout.root_value == 25.
    assert rolling.action(0, 0) == 'alight'
    assert optimal.action(0, 0) == rollout.action(0, 0) == 'continue'
    assert optimal.action(1, 0) == 'alight'
    assert optimal.action(1, 1) == 'continue'
    # The rollout's benefit is standard policy improvement, not a new solver.
    assert optimal.root_value <= rollout.root_value <= rolling.root_value


def test_negative_case_keeps_the_cheap_action_and_value():
    result = adaptation_chain(current_cost=20.).solve()
    for policy in result.policies.values():
        assert policy.root_value == 20.
        assert policy.action(0, 0) == 'alight'


def test_killed_transition_and_initial_mass_are_absorbed_not_renormalized():
    chain = OnboardStopping(([0], [10, 20]), ([100.], [20., 40.]),
                            ([[.3, .2]],), 100.)
    result = chain.solve(initial_mass=[.8])
    # First-stop conditional value=.3*20+.2*40+.5*100=64;
    # the pre-first-stop failure contributes a further .2*100.
    assert result.fixed_commitment_values[0][1] == pytest.approx([64.])
    assert result.fixed_root_values == pytest.approx([100., 71.2])
    for policy in result.policies.values():
        assert policy.root_value == pytest.approx(71.2)
        assert policy.action(0, 0) == 'continue'


def test_unavailable_intermediate_stops_and_terminal_failure_do_not_alight():
    chain = OnboardStopping(([0], [10], [20]), ([100.], [100.], [100.]),
                            ([[1.]], [[0.]]), 100.)
    result = chain.solve()
    for policy in result.policies.values():
        assert policy.root_value == 100.
        assert policy.action(0, 0) == policy.action(1, 0) == 'continue'
        assert policy.action(2, 0) == 'deadline'


def test_rolling_commitment_is_evaluated_after_reselection():
    result = adaptation_chain(current_cost=70.).solve()
    rolling = result.policies['rolling_fixed']
    assert result.fixed_root_values == pytest.approx([70., 55., 55.])
    assert rolling.decision_costs[0] == pytest.approx(np.array([[70., 55.]]))
    assert rolling.root_value == 25.  # It switches stop after observing time 20.


def test_exact_value_matches_all_state_policies_with_branch_failure():
    chain = OnboardStopping(([0], [10, 20], [30]),
                            ([65.], [35., 80.], [45.]),
                            ([[.4, .5]], [[.8], [.4]]), 100.)
    result = chain.solve(initial_mass=[.9])
    values = []
    for bits in product((False, True), repeat=4):
        policy = ([bits[0]], bits[1:3], [bits[3]])
        values.append(chain.evaluate_policy(policy, [.9]).root_value)
    assert result.policies['optimal'].root_value == pytest.approx(min(values))
    for policy in result.policies.values():
        assert chain.evaluate_policy(policy.actions, [.9]).root_value == pytest.approx(policy.root_value)


def test_public_prefix_has_same_action_until_arrival_observation_differs():
    policy = adaptation_chain().solve().policies['optimal']
    # Two model worlds have the same public state at stop zero. Their future
    # arrival state is used only after CONTINUE, when it has become observed.
    worlds = [(0, 0, 0), (0, 1, 1)]
    before = [policy.action(0, world[0]) for world in worlds]
    after = [policy.action(1, world[1]) for world in worlds]
    assert before == ['continue', 'continue']
    assert after == ['alight', 'continue']
    # An oracle could use tomorrow's branch to reject the current cost only
    # in the fast world. The public-state controller cannot do that.
    other = OnboardStopping(([0], [5, 6]), ([25.], [10., 90.]), ([[.5, .5]],), 100.)
    public_policy = other.solve().policies['optimal']
    assert public_policy.root_value == 25. > .5 * min(25., 10.) + .5 * min(25., 90.)
    assert [public_policy.action(0, world[0]) for world in worlds] == ['alight', 'alight']


@pytest.mark.parametrize('kernel', ([[1.01]], [[-.01]], [[np.nan]]))
def test_invalid_kernel_mass_cannot_create_a_fake_gain(kernel):
    with pytest.raises(ValueError, match='sub-stochastic'):
        OnboardStopping(([0], [10]), ([50.], [20.]), (kernel,), 100.)


def test_multiple_initial_states_require_their_actual_probability_mass():
    chain = OnboardStopping(([10, 20],), ([30., 40.],), (), 100.)
    with pytest.raises(ValueError, match='explicit initial_mass'):
        chain.solve()
    with pytest.raises(ValueError, match='sub-probability'):
        chain.solve([.6, .6])
    assert chain.solve([.25, .5]).policies['optimal'].root_value == 52.5
