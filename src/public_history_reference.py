"""A bounded finite-law reference that groups hidden states by public history.

This is ordinary belief-tree dynamic programming. It is a small reference, not
a new transit algorithm. Transition callbacks enumerate exact finite-law atoms;
a sampled scenario bank is deliberately not accepted as a law-exact source.
"""

from dataclasses import dataclass, field
from math import fsum, isfinite


@dataclass(frozen=True)
class PublicTransition:
    probability: float
    next_state: object
    observation: object
    terminal_cost: float | None = None


@dataclass(frozen=True)
class FinitePublicModel:
    initial_mass: dict
    initial_observation: object
    actions: object  # callback(public history) -> legal actions
    transition: object  # callback(private state, action) -> exact atoms
    bounds: object  # callback(public history) -> valid cost bounds
    provenance: str = 'finite_transition_law'
    description: str = ''


@dataclass(frozen=True)
class PublicHistoryResult:
    lower: float
    upper: float
    action: object
    exact: bool
    unresolved_mass: float  # Under the selected upper-bound completion.
    histories: int
    transition_terms: int
    budget_exhausted: bool
    decisions: tuple = field(default_factory=tuple)
    scope: str = 'original supplied finite law'


class PublicHistoryReference:
    """Preserve sum/min order and return an interval when the budget runs out.

    At a public node, all private states share one action. Its transition atoms
    are grouped by the next public observation before a subsequent action is
    selected. Thus private state IDs never identify a future for the policy.
    ``bounds`` must hold for every feasible continuation under that history;
    deadline and current clock are simple valid transit bounds. Bounds at a
    truncated leaf describe an interval, not an executed optimal policy.
    """

    def __init__(self, model, *, max_histories=2000, max_transition_terms=2000000,
                 max_depth=32):
        if model.provenance not in ('finite_transition_law', 'complete_law_enumeration'):
            raise ValueError('A sampled bank is not an exact original finite-law reference')
        self.model = model
        self.max_histories = max_histories
        self.max_transition_terms = max_transition_terms
        self.max_depth = max_depth
        self.histories = self.terms = 0
        self.exhausted = False
        self.decisions = []

    def _bounds(self, history):
        lower, upper = map(float, self.model.bounds(history))
        if not isfinite(lower) or not isfinite(upper) or lower > upper:
            raise ValueError('Public-history bounds must be finite and ordered')
        return lower, upper

    def _leaf(self, history):
        self.exhausted = True
        lower, upper = self._bounds(history)
        return lower, upper, None, 1.

    def _action(self, history, belief, action, depth):
        lower_bound, upper_bound = self._bounds(history)
        groups = {}
        terminal, consumed = [], []
        complete = True
        for state, mass in belief.items():
            row_mass = []
            for atom in self.model.transition(state, action):
                if self.terms >= self.max_transition_terms:
                    complete = False
                    break
                self.terms += 1
                p = float(atom.probability)
                if not isfinite(p) or p < 0:
                    raise ValueError('Transition atoms must have nonnegative finite mass')
                row_mass.append(p)
                weight = mass*p
                if not weight:
                    continue
                consumed.append(weight)
                if atom.terminal_cost is not None:
                    cost = float(atom.terminal_cost)
                    if not isfinite(cost) or not lower_bound-1e-8 <= cost <= upper_bound+1e-8:
                        raise ValueError('Terminal cost is outside the declared bounds')
                    terminal.append(weight*cost)
                else:
                    posterior = groups.setdefault(atom.observation, {})
                    posterior[atom.next_state] = posterior.get(atom.next_state, 0.)+weight
            if not complete:
                break
            if abs(fsum(row_mass)-1.) > 1e-8:
                raise ValueError('Each exact transition row must retain all probability mass')
        base = fsum(terminal)
        lower = upper = base
        unresolved = 0.
        if not complete:
            # Partially collected observation groups cannot be treated as full
            # posteriors. Bound all collected and unenumerated continuations.
            self.exhausted = True
            remaining = max(0., 1.-fsum(consumed))
            lower += remaining*lower_bound
            upper += remaining*upper_bound
            unresolved += remaining
        for observation, posterior in groups.items():
            weight = fsum(posterior.values())
            child_history = history+((action, observation),)
            if complete:
                child = self._node(child_history,
                    {state: p/weight for state, p in posterior.items()}, depth+1)
                lo, hi, _, pending = child
            else:
                lo, hi = self._bounds(child_history)
                pending = 1.
            lower += weight*lo
            upper += weight*hi
            unresolved += weight*pending
        return lower, upper, unresolved

    def _node(self, history, belief, depth):
        if (depth >= self.max_depth or self.histories >= self.max_histories
                or self.terms >= self.max_transition_terms):
            return self._leaf(history)
        self.histories += 1
        actions = tuple(self.model.actions(history))
        if not actions:
            raise ValueError('Nonterminal public histories must offer a legal action')
        values = []
        for action in actions:
            if self.terms >= self.max_transition_terms:
                self.exhausted = True
                lower, upper = self._bounds(history)
                values.append((lower, upper, action, 1.))
            else:
                lower, upper, unresolved = self._action(history, belief, action, depth)
                values.append((lower, upper, action, unresolved))
        best = min(values, key=lambda value: value[1])
        lower = min(value[0] for value in values)
        self.decisions.append(dict(history=history, action=best[2], lower=lower,
                                   upper=best[1], private_support=len(belief),
                                   action_bounds=tuple((a, lo, hi) for lo, hi, a, _ in values)))
        return lower, best[1], best[2], best[3]

    def solve(self):
        initial = {state: float(p) for state, p in self.model.initial_mass.items() if p}
        if (any(not isfinite(p) or p < 0 for p in initial.values())
                or abs(fsum(initial.values())-1.) > 1e-8):
            raise ValueError('Initial law must retain total probability one')
        history = ((None, self.model.initial_observation),)
        lower, upper, action, unresolved = self._node(history, initial, 0)
        return PublicHistoryResult(lower, upper, action, upper-lower <= 1e-8,
            unresolved, self.histories, self.terms, self.exhausted, tuple(self.decisions))
