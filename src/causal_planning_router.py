"""Work-bounded causal rollout and belief-tree baselines for committed transit."""
from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from time import perf_counter

import numpy as np

from .committed_transit import WAIT, build_committed_model
from .continuation_router import ContinuationRouter, nominal_action
from .durner.topocsa import HyperpathResult
from .transit_graph import StopLabel


@dataclass(frozen=True)
class PlanningResult:
    action: tuple
    value: float | None
    transitions: int
    rollouts: int
    root_actions: int
    scored_root_actions: int
    common_rounds: int
    tree_nodes: int


class BudgetedPolicySearch:
    """Root rollout or UCT using only observed states to choose actions.

    Both methods use the same causal terminal rollout policy and count actual
    model transitions. A partially evaluated common-world round contributes
    work but no root estimate. UCT updates only from completed simulations.
    """

    def __init__(self, model, belief, tail_action, observe_step, rng, budget, cap):
        self.model = model
        self.belief = belief
        self.tail_action = tail_action
        self.observe_step = observe_step
        self.rng = rng
        self.budget, self.cap = budget, cap
        self.work = self.rollouts = 0
        self.nominal = {}
        self.tree = {}

    def _tail(self, state, belief, actions):
        if state not in self.nominal:
            action = self.tail_action(state, belief, actions)
            self.nominal[state] = action if action in actions else WAIT
        return self.nominal[state]

    def _tree_action(self, actions, nominal, stats):
        if nominal not in stats:
            return nominal
        unexplored = [action for action in actions if action not in stats]
        if unexplored:
            return unexplored[int(self.rng.integers(len(unexplored)))]
        visits = sum(stats[action][0] for action in actions)
        return min(actions, key=lambda action: stats[action][1] / (stats[action][0] * self.cap)
                   - sqrt(2 * log(visits) / stats[action][0]))

    def _simulate(self, world, first=None, use_tree=False):
        state = self.model.initial_state
        belief = self.belief.copy()
        cost = 0.0
        visits = []
        expanded = False
        for step in range(self.model.horizon):
            actions = self.model.actions(state)
            if not actions:
                break
            if self.work >= self.budget:
                return None
            if step == 0 and first is not None:
                action = first
            else:
                nominal = self._tail(state, belief, actions)
                if use_tree and state not in self.tree and not expanded:
                    self.tree[state] = {}
                    expanded = True
                stats = self.tree.get(state) if use_tree else None
                action = nominal if stats is None else self._tree_action(actions, nominal, stats)
                if stats is not None:
                    visits.append((stats, action, cost))
            outcome = self.model.transition(world, state, action)
            self.work += 1
            self.observe_step(belief, state, outcome.state, outcome.terminal)
            state = outcome.state
            cost += outcome.cost
            if outcome.terminal:
                break
        cost += self.model.terminal_cost(world, state)
        for stats, action, prefix in visits:
            n, total = stats.get(action, (0, 0.0))
            stats[action] = (n + 1, total + cost - prefix)
        self.rollouts += 1
        return cost

    def plan(self, algorithm):
        root = self.model.initial_state
        actions = tuple(self.model.actions(root))
        if not actions:
            return PlanningResult(WAIT, None, 0, 0, 0, 0, 0, 0)
        fallback = self._tail(root, self.belief, actions)
        rounds = 0
        scores = {}
        if algorithm == "rollout":
            totals = dict.fromkeys(actions, 0.0)
            # Complete every root action on a world before using that round.
            for world in self.model.scenarios:
                round_losses = {}
                for action in actions:
                    loss = self._simulate(world, first=action)
                    if loss is None:
                        break
                    round_losses[action] = loss
                if len(round_losses) != len(actions):
                    break
                rounds += 1
                for action, loss in round_losses.items():
                    totals[action] += loss
            if rounds:
                scores = {action: total / rounds for action, total in totals.items()}
        elif algorithm == "uct":
            self.tree[root] = {}
            probabilities = [world.probability for world in self.model.scenarios]
            while self.work < self.budget:
                world = self.model.scenarios[int(self.rng.choice(len(probabilities), p=probabilities))]
                if self._simulate(world, use_tree=True) is None:
                    break
            scores = {action: total / n for action, (n, total) in self.tree[root].items()}
        else:
            raise ValueError("algorithm must be rollout or uct")
        best = min(scores, key=lambda action: (scores[action], action != fallback)) if scores else fallback
        return PlanningResult(best, scores.get(best), self.work, self.rollouts, len(actions), len(scores),
                              rounds, len(self.tree))


class CausalPlanningRouter(ContinuationRouter):
    """Replan over the full menu or a fixed, trip-closed TopoCSA menu."""

    def __init__(self, graph, priors, *, menu="full", algorithm="rollout", work_budget=256,
                 samples=8, seed=0, candidate_limit=5, max_time=120, event_belief=None):
        if (menu not in ("full", "fixed", "fixed_top5") or algorithm not in ("rollout", "uct", "nominal")
                or work_budget < 1):
            raise ValueError("Choose a supported menu, search and positive work budget")
        super().__init__(graph, priors, samples=samples, seed=seed, max_time=max_time,
                         candidate_limit=candidate_limit)
        self.menu_mode, self.algorithm, self.work_budget = menu, algorithm, work_budget
        if event_belief is not None and menu != 'full':
            raise ValueError("Measured event clocks use the physical full menu")
        self.minute = 1 if event_belief is None else event_belief.minute
        if event_belief is not None:
            self.trip_belief = event_belief
            self.lookahead = 25 * self.minute

    def route(self, source, destination, departure):
        started = perf_counter()
        if self.menu_mode != "full":
            result = super().route(source, destination, departure)
            if self.menu_mode == "fixed_top5":
                self.menu = {label.connection_id for labels in result.stop_labels.values()
                             for label in labels[-5:]}
            else:
                self.menu = set(result.hyperpath_connections)
            trips = {self.graph.connections[cid].trip_id for cid in self.menu}
            self.menu.update(cid for trip in trips for cid in self._trip_connections.get(trip, ()))
        else:
            self.origin, self.destination, self.departure = source, destination, departure
            self.journey_deadline = departure + self.max_time
            if self.minute == 1:
                self._assign()  # Legacy minute-grid marginal support.
            self.estimate_ms += (perf_counter() - started) * 1000
            self.menu = {c.id for c in self.graph.connections}
        labels = {sid: [] for sid in self.graph.stops}
        for c in self.graph.connections:
            if c.id in self.menu:
                labels[c.dep_stop].append(StopLabel(c.id, c.arr_distribution, c.arr_time, 0.0))
        self.cached_result = HyperpathResult(labels, None, float("inf"), self.menu, 0, set())
        self._valued_version = self._version
        self.preprocess_ms = (perf_counter() - started) * 1000
        return self.cached_result

    def observe_delay(self, route, delay):
        super().observe_delay(route, delay)
        # The frozen menu and bounded delay support do not depend on new estimates.
        self._valued_version = self._version

    def observe_cancel(self, route, kind="true"):
        super().observe_cancel(route, kind)
        self._valued_version = self._version

    def select_connection(self, current_stop, current_time, rng, top_k=5):
        started = perf_counter()
        state = self.information_state
        parameters = self._parameters()
        private_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.query_number]))
        self.query_number += 1
        if self.algorithm == 'nominal':
            action = nominal_action(self.graph, state, self.destination, self.trip_belief.mean_events(parameters),
                                    self.menu, self.journey_deadline, self.lookahead, minute=self.minute)
            if action == WAIT:
                return None
            c = self.graph.connections[action[1]]
            return StopLabel(c.id, c.arr_distribution, c.arr_time, 0.0), 0.0
        scenes = self.trip_belief.sample(parameters, private_rng, self.samples)
        self.trip_belief.mean_events(parameters)
        model = build_committed_model(
            self.graph, [(1 / self.samples, scene) for scene in scenes], self.origin, self.destination,
            self.departure, self.journey_deadline, candidate_ids=self.menu, lookahead=self.lookahead,
            initial_state=state)

        def tail(state, belief, actions):
            boardings = {action[1] for action in actions if action[0] == "select"}
            if not boardings:
                return WAIT
            return nominal_action(self.graph, state, self.destination, belief.mean_events(parameters),
                                  self.menu, self.journey_deadline, self.lookahead, root_boardings=boardings,
                                  minute=self.minute)

        planner = BudgetedPolicySearch(model, self.trip_belief, tail, self._advance_belief,
                                       private_rng, self.work_budget, self.max_time)
        result = planner.plan(self.algorithm)
        elapsed = (perf_counter() - started) * 1000
        self.rollout_ms += elapsed
        self.rollout_steps += result.transitions
        cid = None if result.action == WAIT else result.action[1]
        self.predictions.append({"connection": cid, "stop": current_stop, "time": current_time,
                                 "planning_ms": elapsed, "work": result.transitions,
                                 "value": result.value, "completed_rollouts": result.rollouts,
                                 "root_actions": result.root_actions,
                                 "scored_root_actions": result.scored_root_actions,
                                 "complete_common_rounds": result.common_rounds,
                                 "tree_nodes": result.tree_nodes, "menu_size": len(self.menu)})
        if cid is None:
            return None
        c = self.graph.connections[cid]
        return StopLabel(cid, c.arr_distribution, c.arr_time, 0.0), result.value
