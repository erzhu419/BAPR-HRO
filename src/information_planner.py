"""Exact causal planning on a finite bank of immutable exogenous worlds.

This is a reference solver, not a scalable transit algorithm. All randomness is
drawn before a journey. A transition exposes only its returned observation,
physical state, incurred cost and terminal status; world factors stay hidden.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import isclose, isfinite
from typing import Callable, Hashable, Mapping, Sequence


@dataclass(frozen=True)
class Scenario:
    name: str
    probability: float
    factors: Mapping[str, Hashable] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    state: Hashable
    observation: Hashable
    cost: float
    terminal: bool = False


@dataclass(frozen=True)
class InformationState:
    stage: int
    state: Hashable
    # World indices are model hypotheses, not the realized hidden world.
    support: tuple[int, ...]
    belief: tuple[float, ...]
    history: tuple = ()


@dataclass
class FiniteScenarioModel:
    scenarios: Sequence[Scenario]
    initial_state: Hashable
    actions: Callable[[Hashable], Sequence[Hashable]]
    transition: Callable[[Scenario, Hashable, Hashable], Outcome]
    horizon: int
    terminal_cost: Callable[[Scenario, Hashable], float] = lambda world, state: 0.0
    # Every factor that can alter state/observation/cost/termination is included.
    dependencies: Callable[[Hashable, Hashable], set[str]] | None = None
    terminal_dependencies: Callable[[Hashable], set[str]] | None = None

    def __post_init__(self):
        self.scenarios = tuple(self.scenarios)
        if not self.scenarios or self.horizon < 0:
            raise ValueError("A nonempty scenario bank and nonnegative horizon are required")
        probabilities = [world.probability for world in self.scenarios]
        if any(not isfinite(p) or p < 0 for p in probabilities):
            raise ValueError("Scenario probabilities must be finite and nonnegative")
        if not isclose(sum(probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("Scenario probabilities must sum to one")

    @property
    def factors(self):
        return set().union(*(world.factors for world in self.scenarios))


@dataclass(frozen=True)
class Solution:
    value: float
    action: Hashable | None
    action_values: Mapping[Hashable, float]


@dataclass(frozen=True)
class Certificate:
    policy_cost: float
    full_optimum: float
    restricted_optimum: float
    support_loss: float
    within_candidate_loss: float
    gap: float
    # The full optimum has already been solved: this is a diagnostic certificate.
    kind: str = "exact_finite_bank_same_information"


class ExactPlanner:
    """Bellman recursion on exact public histories and finite posteriors.

    ``candidate_rule(info)`` returns a nonempty subset of physical actions and
    governs every future history, including counterfactual histories. Policies
    receive InformationState only. ``solve_compressed`` merges identical one-step
    outcomes only for the unrestricted model, where continuation does not depend
    on the names of past actions. All reported values are exact expectations for
    this finite bank, up to floating-point arithmetic.
    """

    def __init__(self, model: FiniteScenarioModel, candidate_rule=None):
        self.model = model
        self.candidate_rule = candidate_rule
        support = tuple(i for i, world in enumerate(model.scenarios) if world.probability > 0)
        self.root = InformationState(0, model.initial_state, support,
                                     tuple(model.scenarios[i].probability for i in support))
        self._solutions = {}
        self._q = {}
        self._branches = {}
        self._parents = defaultdict(set)
        self._direct_dependencies = {}
        self._leaf_dependencies = {}
        self.stats = defaultdict(int)

    def _actions(self, node):
        physical = tuple(self.model.actions(node.state))
        if not physical or self.candidate_rule is None:
            return physical
        allowed = tuple(self.candidate_rule(node))
        if not allowed or not set(allowed) <= set(physical):
            raise ValueError("Candidate rule must return a nonempty physical-action subset")
        return allowed

    def _terminal_value(self, node):
        return sum(p * self.model.terminal_cost(self.model.scenarios[i], node.state)
                   for i, p in zip(node.support, node.belief))

    def _outcomes(self, node, action):
        outcomes = []
        for i in node.support:
            outcome = self.model.transition(self.model.scenarios[i], node.state, action)
            if not isfinite(outcome.cost):
                raise ValueError("Use a finite common timeout/failure cost")
            outcomes.append(outcome)
            self.stats["transition_evaluations"] += 1
        return tuple(outcomes)

    def _action_value(self, node, action, compressed=False, outcomes=None):
        key = node, action
        if key in self._q:
            self.stats["q_cache_hits"] += 1
            return self._q[key]
        outcomes = outcomes if outcomes is not None else self._outcomes(node, action)
        grouped = defaultdict(list)
        for i, probability, outcome in zip(node.support, node.belief, outcomes):
            grouped[outcome].append((i, probability))
        value, branches = 0.0, []
        for outcome, members in grouped.items():
            probability = sum(p for _, p in members)
            public_event = (action, outcome.state, outcome.observation,
                            outcome.cost, outcome.terminal)
            child = InformationState(node.stage + 1, outcome.state,
                                     tuple(i for i, _ in members),
                                     tuple(p / probability for _, p in members),
                                     node.history + (public_event,))
            if outcome.terminal:
                continuation = self._terminal_value(child)
                dependencies = (self.model.terminal_dependencies(child.state)
                                if self.model.terminal_dependencies else self.model.factors)
                self._leaf_dependencies[child] = set(dependencies)
            else:
                continuation = self._solve(child, compressed).value
            self._parents[child].add(key)
            value += probability * (outcome.cost + continuation)
            branches.append((probability, outcome.cost, child, outcome.terminal))
        dependencies = (self.model.dependencies(node.state, action)
                        if self.model.dependencies else self.model.factors)
        self._direct_dependencies[key] = set(dependencies)
        self._branches[key] = branches
        self._q[key] = value
        self.stats["action_evaluations"] += 1
        return value

    def _solve(self, node, compressed=False):
        if node in self._solutions:
            self.stats["value_cache_hits"] += 1
            return self._solutions[node]
        actions = self._actions(node) if node.stage < self.model.horizon else ()
        if not actions:
            solution = Solution(self._terminal_value(node), None, {})
            dependencies = (self.model.terminal_dependencies(node.state)
                            if self.model.terminal_dependencies else self.model.factors)
            self._leaf_dependencies[node] = set(dependencies)
        else:
            values, signatures = {}, {}
            for action in actions:
                outcomes = self._outcomes(node, action) if compressed else None
                if compressed and outcomes in signatures:
                    values[action] = values[signatures[outcomes]]
                    self.stats["equivalent_actions_merged"] += 1
                else:
                    values[action] = self._action_value(node, action, compressed, outcomes)
                    if compressed:
                        signatures[outcomes] = action
            best = min(values, key=values.__getitem__)
            solution = Solution(values[best], best, values)
        self._solutions[node] = solution
        self.stats["information_nodes_solved"] += 1
        return solution

    def solve(self, node=None):
        return self._solve(self.root if node is None else node)

    def solve_compressed(self, node=None):
        if self.candidate_rule is not None:
            raise ValueError("Outcome equivalence alone is unsafe for history-dependent candidate rules")
        return self._solve(self.root if node is None else node, compressed=True)

    def q_value(self, node, action):
        if action not in self._actions(node):
            raise ValueError("Action is outside the candidate rule")
        return self._action_value(node, action)

    def optimal_action(self, node):
        return self.solve(node).action

    def evaluate(self, policy, node=None):
        """Evaluate a causal policy exactly, using the same finite world bank."""
        node = self.root if node is None else node
        if node.stage >= self.model.horizon or not self._actions(node):
            return self._terminal_value(node)
        action = policy(node)
        if action not in self._actions(node):
            raise ValueError("Policy selected an action outside its candidate rule")
        self._action_value(node, action)
        return sum(p * (cost + (self._terminal_value(child) if terminal
                                else self.evaluate(policy, child)))
                   for p, cost, child, terminal in self._branches[node, action])

    def decision_loss(self, policy, node=None):
        """Expected sum of restricted-optimum advantages on policy histories."""
        node = self.root if node is None else node
        if node.stage >= self.model.horizon or not self._actions(node):
            return 0.0
        action = policy(node)
        gap = self.q_value(node, action) - self.solve(node).value
        return gap + sum(p * self.decision_loss(policy, child)
                         for p, _, child, terminal in self._branches[node, action]
                         if not terminal)

    def certificate(self, policy):
        restricted = self.solve().value
        full = restricted if self.candidate_rule is None else ExactPlanner(self.model).solve().value
        actual = self.evaluate(policy)
        return Certificate(actual, full, restricted, restricted - full,
                           actual - restricted, actual - full)

    def decompose(self, policy):
        return self.certificate(policy)

    def clairvoyant_lower_bound(self):
        """Worldwise optimum: information relaxation, never a causal baseline.

        This may retain a positive information gap even with every action kept.
        It also enumerates planning states, so cheapness is not asserted.
        """
        result = 0.0
        for scenario in self.model.scenarios:
            single = FiniteScenarioModel((Scenario(scenario.name, 1.0, scenario.factors),),
                                         self.model.initial_state, self.model.actions,
                                         self.model.transition, self.model.horizon,
                                         self.model.terminal_cost)
            result += scenario.probability * ExactPlanner(single).solve().value
        return result

    def invalidate_all(self):
        """Required after changing prior probabilities, actions or information rules."""
        self.__init__(self.model, self.candidate_rule)

    def incremental_update(self, changed_factors):
        """Invalidate changed local primitives and every cached ancestor.

        Contract: scenario identities/probabilities, action sets, candidate rule
        and observation rule are unchanged. The declared dependencies cover all
        changed transition/cost inputs. Factor values may alter emitted messages
        under the fixed rule. Change the model's factors first, then call this.
        Changing any contract element requires ``invalidate_all`` instead.
        """
        changed_factors = set(changed_factors)
        affected_q = {key for key, deps in self._direct_dependencies.items()
                      if deps & changed_factors}
        # Compressed-away actions still have to trigger invalidation if their
        # factor changes break their former equality with the representative.
        for node, solution in self._solutions.items():
            for action in solution.action_values:
                dependencies = (self.model.dependencies(node.state, action)
                                if self.model.dependencies else self.model.factors)
                if set(dependencies) & changed_factors:
                    affected_q.add((node, action))
        nodes = {node for node, deps in self._leaf_dependencies.items() if deps & changed_factors}
        nodes.update(node for node, _ in affected_q)
        queue = deque(nodes)
        while queue:
            node = queue.popleft()
            for key in self._parents.get(node, ()):
                affected_q.add(key)
                if key[0] not in nodes:
                    nodes.add(key[0])
                    queue.append(key[0])
        for key in affected_q:
            self._q.pop(key, None)
            self._branches.pop(key, None)
            self._direct_dependencies.pop(key, None)
        for node in nodes:
            self._solutions.pop(node, None)
            self._leaf_dependencies.pop(node, None)
        report = {"invalidated_actions": len(affected_q), "invalidated_nodes": len(nodes),
                  "retained_nodes": len(self._solutions)}
        self.stats["incremental_invalidated_actions"] += len(affected_q)
        return report
