"""Anytime root search with finite-bank lower bounds and causal witnesses."""
from dataclasses import dataclass
from fractions import Fraction

from .causal_planning_router import BudgetedPolicySearch, PlanningResult
from .committed_transit import WAIT


@dataclass(frozen=True)
class CertifiedResult:
    planning: PlanningResult
    lower: Fraction
    upper: Fraction
    action_lower: dict
    action_upper: dict
    removed: tuple
    evaluated_worlds: dict

    @property
    def gap(self):
        return self.upper - self.lower


class CertifiedPolicySearch(BudgetedPolicySearch):
    """Spend transitions on the incumbent before evaluating alternatives.

    A fixed root action followed by ``tail_action`` is a causal witness.
    Unfinished worlds receive the deadline loss, so its upper bound exists
    even when the transition budget interrupts evaluation. The supplied
    action lower bounds relax the *optimal* causal continuation, rather than
    just this tail. All bounds refer to the supplied finite scenario bank.
    """

    def plan_certified(self, lower_bounds, remaining_cap):
        root = self.model.initial_state
        actions = tuple(self.model.actions(root))
        if not actions:
            result = PlanningResult(WAIT, None, 0, 0, 0, 0, 0, 0)
            total = sum(Fraction(w.probability) for w in self.model.scenarios)
            value = sum(Fraction(w.probability) * self.model.terminal_cost(w, root)
                        for w in self.model.scenarios) / total
            return CertifiedResult(result, value, value, {}, {}, (), {})
        fallback = self._tail(root, self.belief, actions)
        weights = [Fraction(world.probability) for world in self.model.scenarios]
        total = sum(weights)
        weights = [weight / total for weight in weights]
        cap = Fraction(remaining_cap)
        upper = dict.fromkeys(actions, cap)
        counts = dict.fromkeys(actions, 0)
        complete = set()
        incumbent = fallback

        def evaluate(action):
            nonlocal incumbent
            for world, weight in zip(self.model.scenarios, weights):
                if not weight:
                    continue
                value = self._simulate(world, first=action)
                if value is None:
                    break
                # A finite-bank witness's remaining unsimulated worlds retain
                # their worst-case loss. No private world-dependent action is
                # used by the tail policy.
                upper[action] -= weight * (cap - Fraction(value))
                counts[action] += 1
                if upper[action] < upper[incumbent]:
                    incumbent = action
            if counts[action] == sum(weight > 0 for weight in weights):
                complete.add(action)

        if callable(lower_bounds):
            # A causal witness supplies the cutoff before the relaxation does
            # work. In particular, a12-minute failed attempt already dominates
            # a75-second witness, without any downstream shortest-path query.
            evaluate(fallback)
            raw = lower_bounds(upper[fallback])
            order = [a for a in actions if a != fallback]
        else:
            raw = lower_bounds
            order = [fallback] + [a for a in actions if a != fallback]
        lower = {action: Fraction(raw[action]) for action in actions}
        order.sort(key=lambda a: (a != fallback, lower[a]))
        for action in order:
            if self.work >= self.budget:
                break
            if lower[action] > upper[incumbent]:
                continue
            if action != incumbent and lower[action] >= upper[incumbent]:
                continue
            evaluate(action)
        best_upper = upper[incumbent]
        removed = tuple(action for action in actions if lower[action] > best_upper)
        result = PlanningResult(incumbent, float(best_upper), self.work, self.rollouts,
                                len(actions), len(complete), 0, 0)
        return CertifiedResult(result, min(lower.values()), best_upper, lower,
                               upper, removed, counts)
