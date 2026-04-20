"""Oracle Router: perfect-information upper bound baseline.

The oracle knows the true current regime and uses the exact delay means
to select connections. This provides an upper bound on achievable performance
and quantifies the optimality gap of online methods.

The oracle is NOT implementable in practice (it requires knowing which
disruptions are occurring before observing them). It serves as a
theoretical benchmark:
  - If LCB ≈ Oracle: our method is near-optimal
  - Gap(LCB, Oracle) measures cost of Bayesian uncertainty

Oracle selection rule:
  Given true regime R, the oracle computes the true expected delay per route:
    E[delay | route, regime=R] = sum(delay * p(delay | R))
  Then picks the route minimizing:
    score_oracle(c) = label.mean_dest_arrival + E[delay | route, R]
                     + cancel_cost * p_cancel(route, R)

This is the best-in-hindsight connection selector given perfect knowledge.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Callable

from .transit_graph import TransitGraph, StopLabel
from .durner.topocsa import topocsa, HyperpathResult


class OracleRouter:
    """Oracle router with perfect regime knowledge.

    Parameters:
        graph: Transit network.
        regime_dist_fn: Function mapping regime name → dict[route → dist_params].
            Same signature as create_regime_distributions.
        regime_schedule: Callable(time) → str. Maps current time to regime name.
        cancel_cost: Cost per unit cancellation probability. Default 60.0.
    """

    def __init__(
        self,
        graph: TransitGraph,
        regime_dist_fn: Callable[[str], dict],
        regime_schedule_fn: Callable[[int], str],
        cancel_cost: float = 60.0,
    ):
        self.graph = graph
        self.regime_dist_fn = regime_dist_fn
        self.regime_schedule_fn = regime_schedule_fn
        self.cancel_cost = cancel_cost
        self.cached_result: Optional[HyperpathResult] = None
        self.total_observations: int = 0

    def route(self, s_source: int, s_dest: int, t_source: int) -> HyperpathResult:
        self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)
        return self.cached_result

    def observe_delay(self, route: str, delay: float):
        self.total_observations += 1  # Oracle ignores observations (knows truth)

    def observe_cancel(self, route: str):
        self.total_observations += 1

    def _true_expected_delay(self, route: str, current_time: int) -> float:
        """Compute true expected delay for a route under current regime."""
        regime = self.regime_schedule_fn(current_time)
        dists = self.regime_dist_fn(regime)
        if route not in dists:
            return 1.0  # fallback: prior mean
        info = dists[route]
        probs = info["delay_probs"].copy()
        probs = probs / probs.sum()
        offset = info.get("delay_offset", 0)
        delays = offset + np.arange(len(probs))
        return float(np.dot(probs, delays))

    def _true_cancel_prob(self, route: str, current_time: int) -> float:
        """Compute true cancellation probability for a route under current regime."""
        regime = self.regime_schedule_fn(current_time)
        dists = self.regime_dist_fn(regime)
        if route not in dists:
            return 0.0
        return dists[route].get("cancel_prob", 0.0)

    def oracle_score(self, label: StopLabel, route: str, current_time: int) -> float:
        """Oracle score: exact expected arrival under true current regime."""
        true_delay = self._true_expected_delay(route, current_time)
        cancel_prob = self._true_cancel_prob(route, current_time)
        delay_adj = true_delay - 1.0  # center on prior mean
        return label.mean_dest_arrival + delay_adj + self.cancel_cost * cancel_prob

    def select_connection(
        self,
        stop_id: int,
        current_time: int,
        rng: np.random.Generator,
        top_k: int = 5,
    ) -> Optional[tuple[StopLabel, float]]:
        if self.cached_result is None:
            return None
        labels = self.cached_result.stop_labels.get(stop_id, [])
        if not labels:
            return None

        candidates = []
        seen_routes: set[str] = set()
        for label in reversed(labels):
            c = self.graph.connections[label.connection_id]
            if c.dep_time < current_time - 1:
                continue
            if c.dep_time > current_time + 25:
                continue
            if c.route in seen_routes:
                continue
            seen_routes.add(c.route)
            score = self.oracle_score(label, c.route, current_time)
            candidates.append((label, c, score))
            if len(candidates) >= top_k:
                break

        if not candidates:
            return None
        best = min(candidates, key=lambda x: x[2])
        return best[0], best[2]
