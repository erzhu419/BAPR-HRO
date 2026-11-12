"""Finite, observed-arrival stopping on one vehicle, with common tail costs.

State (stop index, arrival time) must be sufficient for the conditional vehicle
kernel and tail costs. In particular, there are no omitted external information
updates between these states. Under that contract, backward induction is exact
for ALIGHT versus CONTINUE with the supplied causal tail policy. It is not a
claim of optimality over other tail policies or a larger information state.

Costs are absolute arrival times truncated at ``deadline``. Sub-stochastic
transition mass that disappears is absorbed at that deadline, never conditioned
away. A disallowed alighting point is represented by a deadline-valued row.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OnboardPolicyResult:
    values: tuple
    actions: tuple  # Boolean arrays: True means ALIGHT, False means CONTINUE.
    decision_costs: tuple  # Each (n_states, 2) array is [ALIGHT, CONTINUE].
    root_value: float

    def action(self, stage, state_index):
        """Use only the current observed state; no future realization is input."""
        if self.actions[stage][state_index]:
            return 'alight'
        return 'continue' if stage + 1 < len(self.actions) else 'deadline'


@dataclass(frozen=True)
class OnboardStoppingResult:
    times: tuple
    policies: dict
    fixed_commitment_values: tuple  # [current stage][target stage] -> value row.
    fixed_root_values: np.ndarray


class OnboardStopping:
    """Compile standard stopping and two cheap controls on the same finite chain.

    ``times[r]`` and ``alight_costs[r]`` describe the observed operating states
    at stop r. ``transitions[r]`` maps them to stop r+1, with canceled/failed
    paths removed. The caller compiles all common tail costs, including MCT.

    Optimal stopping costs O(sum_r n_r*n_{r+1}); compiling all fixed-stop
    commitments costs O(m*sum_r n_r*n_{r+1}). These costs exclude tail-value and
    transition compilation, which must also be charged in an experiment.

    Proof: at the final stop, the only costs are G and deadline. If the next
    row is optimal conditional on its observed state, CONTINUE has value
    K*V_next + (1-K*1)*deadline. Taking its minimum with G is optimal at the
    current stop by conditioning on the next observation. Backward induction
    proves the claim; using a larger omitted information state breaks that
    conditioning argument. This is the standard optimal-stopping recursion.
    """

    def __init__(self, times, alight_costs, transitions, deadline):
        self.times = tuple(np.asarray(t).copy() for t in times)
        self.alight_costs = tuple(np.asarray(g, dtype=float).copy() for g in alight_costs)
        self.transitions = tuple(np.asarray(k, dtype=float).copy() for k in transitions)
        self.deadline = float(deadline)
        n = len(self.times)
        if not n or len(self.alight_costs) != n or len(self.transitions) != n - 1:
            raise ValueError('Need one cost row per stop and one kernel between adjacent stops')
        if not np.isfinite(self.deadline):
            raise ValueError('Deadline must be finite')
        for times_row, costs in zip(self.times, self.alight_costs):
            if (times_row.ndim != 1 or not len(times_row) or costs.shape != times_row.shape
                    or not np.isfinite(times_row).all() or not np.isfinite(costs).all()
                    or (costs > self.deadline + 1e-8).any()):
                raise ValueError('Finite state and cost rows must agree, with costs at most deadline')
        for r, kernel in enumerate(self.transitions):
            if (kernel.shape != (len(self.times[r]), len(self.times[r + 1]))
                    or not np.isfinite(kernel).all() or (kernel < 0).any()
                    or (kernel.sum(axis=1) > 1. + 1e-10).any()):
                raise ValueError('Transition kernels must be nonnegative and sub-stochastic')
        self.alight_costs = tuple(np.minimum(g, self.deadline) for g in self.alight_costs)

    def _initial(self, initial_mass):
        if initial_mass is None:
            if len(self.times[0]) != 1:
                raise ValueError('Multiple first-stop states require explicit initial_mass')
            initial_mass = [1.]
        mass = np.asarray(initial_mass, dtype=float)
        if (mass.shape != self.times[0].shape or not np.isfinite(mass).all()
                or (mass < 0).any() or mass.sum() > 1. + 1e-10):
            raise ValueError('Initial mass must be a nonnegative sub-probability row')
        return mass

    def _continue(self, stage, next_values):
        # H - K(H-V) includes the killed mass at cost H without renormalizing.
        return self.deadline - self.transitions[stage] @ (self.deadline - next_values)

    def _root(self, values, initial_mass):
        return float(self.deadline - initial_mass @ (self.deadline - values))

    def _choose(self, alight, continuation):
        # Alight on useful ties; G=H at an unavailable stop cannot trigger exit.
        return (alight < self.deadline) & (alight <= continuation)

    def evaluate_policy(self, actions, initial_mass=None, decision_costs=None):
        """Integrate a state-indexed policy, retaining all killed probability.

        ``decision_costs`` optionally records the heuristic's actual comparison;
        evaluated continuation values must not silently replace its scores.
        """
        initial_mass = self._initial(initial_mass)
        actions = tuple(np.asarray(a, dtype=bool) for a in actions)
        if len(actions) != len(self.times) or any(
                a.shape != t.shape for a, t in zip(actions, self.times)):
            raise ValueError('Policy must supply one action per observed state')
        values, comparisons = [None] * len(actions), [None] * len(actions)
        for r in range(len(actions) - 1, -1, -1):
            continuation = (self._continue(r, values[r + 1]) if r + 1 < len(actions)
                            else np.full(len(self.times[r]), self.deadline))
            values[r] = np.where(actions[r], self.alight_costs[r], continuation)
            comparisons[r] = np.column_stack((self.alight_costs[r], continuation))
        return OnboardPolicyResult(tuple(values), actions,
                                   tuple(comparisons if decision_costs is None else decision_costs),
                                   self._root(values[0], initial_mass))

    def solve(self, initial_mass=None):
        """Return optimal stopping, rolling fixed-stop choice, and one-step rollout.

        Rolling fixed-stop choice reselects a best commitment after each arrival;
        its value is evaluated as that executable policy, not as a fixed plan.
        Rollout uses the evaluated rolling policy as its one-step leaf value and
        is then itself evaluated over the full chain. No realized future enters
        any policy. ``initial_mass`` includes service survival to the first stop.
        """
        initial_mass = self._initial(initial_mass)
        m = len(self.times)
        fixed = [dict() for _ in range(m)]
        optimal_values, optimal_actions, optimal_costs = [None] * m, [None] * m, [None] * m
        rolling_actions, rolling_costs = [None] * m, [None] * m
        for r in range(m - 1, -1, -1):
            fixed[r][r] = self.alight_costs[r].copy()
            for target in range(r + 1, m):
                fixed[r][target] = self._continue(r, fixed[r + 1][target])
            continuation = (self._continue(r, optimal_values[r + 1]) if r + 1 < m
                            else np.full(len(self.times[r]), self.deadline))
            optimal_actions[r] = self._choose(self.alight_costs[r], continuation)
            optimal_values[r] = np.minimum(self.alight_costs[r], continuation)
            optimal_costs[r] = np.column_stack((self.alight_costs[r], continuation))
            fixed_continue = (np.minimum.reduce([fixed[r][j] for j in range(r + 1, m)])
                              if r + 1 < m else np.full(len(self.times[r]), self.deadline))
            rolling_actions[r] = self._choose(self.alight_costs[r], fixed_continue)
            rolling_costs[r] = np.column_stack((self.alight_costs[r], fixed_continue))
        optimal = OnboardPolicyResult(tuple(optimal_values), tuple(optimal_actions),
                                      tuple(optimal_costs), self._root(optimal_values[0], initial_mass))
        rolling = self.evaluate_policy(rolling_actions, initial_mass, rolling_costs)
        rollout_actions, rollout_costs = [], []
        for r in range(m):
            continuation = (self._continue(r, rolling.values[r + 1]) if r + 1 < m
                            else np.full(len(self.times[r]), self.deadline))
            rollout_actions.append(self._choose(self.alight_costs[r], continuation))
            rollout_costs.append(np.column_stack((self.alight_costs[r], continuation)))
        rollout = self.evaluate_policy(rollout_actions, initial_mass, rollout_costs)
        return OnboardStoppingResult(
            self.times, dict(optimal=optimal, rolling_fixed=rolling, one_step_rollout=rollout),
            tuple(fixed), np.array([self._root(fixed[0][j], initial_mass) for j in range(m)]))
