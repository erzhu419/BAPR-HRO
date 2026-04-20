"""EXP3-SSP Router: Adversarial bandit for single-episode transit SSP.

Theoretically motivated by:
  Neu, Gyorgy, Szepesvari (2010) "The Online Loop-free Stochastic
  Shortest-Path Problem." COLT 2010.

  Rosenberg & Mansour (2021) "Stochastic Shortest Path with Adversarially
  Changing Costs." IJCAI 2021.

  Chen et al. (2024) "Non-Stationary Bandits with Adversarial Corruptions
  in Transit Networks." ECML 2024.

Key contrast to LCB:
  LCB: assumes stochastic delays, maintains distributional posterior,
       uses pessimism for risk control.
  EXP3: makes NO distributional assumption, treats delays as adversarial,
        uses multiplicative weight updates.

For single-shot SSP (our setting), EXP3 suffers from:
  1. Slow adaptation (needs O(K log K) samples to concentrate weights)
  2. Forced exploration (η-dependent uniform mixing) wastes time
  3. Cannot leverage the hyperpath structure for transfer across stops

We include it to demonstrate that adversarial methods are ill-suited
for single-shot stochastic transit routing, supporting our theoretical
claim that single-shot SSP calls for pessimism, not no-regret exploration.

Implementation follows EXP3 with implicit exploration (EXP3-IX):
  Kocák, Neu, Valko (2014) "Efficient Learning by Implicit Exploration in
  Bandit Problems with Side Observations."

EXP3-IX replaces the additive η/(K) exploration bonus with implicit
exploration via adjusted importance weights, reducing variance.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .transit_graph import TransitGraph, StopLabel
from .durner.topocsa import topocsa, HyperpathResult


class EXP3Router:
    """EXP3-IX adversarial bandit router.

    Maintains per-route weights updated multiplicatively after each
    cost observation (delay + cancel). Selection mixes over weights.

    Parameters:
        gamma (float): EXP3 exploration rate ∈ (0,1]. Higher = more uniform.
            Default 0.1 balances exploration and exploitation.
        eta (float): EXP3 learning rate. Default 0.05.
            Smaller = more conservative weight updates.
        cancel_cost (float): Cost assigned to cancellations. Default 60.
    """

    def __init__(
        self,
        graph: TransitGraph,
        gamma: float = 0.1,
        eta: float = 0.05,
        cancel_cost: float = 60.0,
    ):
        self.graph = graph
        self.gamma = gamma
        self.eta = eta
        self.cancel_cost = cancel_cost
        self.cached_result: Optional[HyperpathResult] = None

        # Log-weights per route (EXP3 uses multiplicative updates on weights)
        self._log_weights: dict[str, float] = {}
        # Track routes seen at least once for normalization
        self._known_routes: set[str] = set()
        self.total_observations: int = 0

    def _log_w(self, route: str) -> float:
        return self._log_weights.get(route, 0.0)

    def _all_weights(self) -> dict[str, float]:
        if not self._known_routes:
            return {}
        log_ws = {r: self._log_weights.get(r, 0.0) for r in self._known_routes}
        max_lw = max(log_ws.values())
        raw = {r: np.exp(lw - max_lw) for r, lw in log_ws.items()}
        total = sum(raw.values())
        return {r: v / total for r, v in raw.items()}

    def route(self, s_source: int, s_dest: int, t_source: int) -> HyperpathResult:
        self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)
        return self.cached_result

    def _update_weight(self, route: str, cost: float, prob: float):
        """EXP3-IX multiplicative weight update.

        Standard EXP3: w_r ← w_r * exp(-eta * cost / prob(r))
        EXP3-IX:       w_r ← w_r * exp(-eta * cost / (prob(r) + eta))
        The +eta denominator (implicit exploration) reduces variance.
        """
        self._known_routes.add(route)
        # Normalize cost to [0,1] range (max cost = cancel_cost = 60 min)
        normalized_cost = min(cost / self.cancel_cost, 1.0)
        # EXP3-IX update: divide by prob + eta to reduce variance
        safe_prob = max(prob, 1e-6)
        adj_cost = normalized_cost / (safe_prob + self.eta)
        self._log_weights[route] = self._log_w(route) - self.eta * adj_cost

    def observe_delay(self, route: str, delay: float):
        """Update weights with observed delay cost."""
        self._known_routes.add(route)
        weights = self._all_weights()
        prob = weights.get(route, 1.0 / max(len(self._known_routes), 1))
        self._update_weight(route, delay, prob)
        self.total_observations += 1

    def observe_cancel(self, route: str):
        """Update weights with cancellation (maximum cost)."""
        self._known_routes.add(route)
        weights = self._all_weights()
        prob = weights.get(route, 1.0 / max(len(self._known_routes), 1))
        self._update_weight(route, self.cancel_cost, prob)
        self.total_observations += 1

    def exp3_score(self, route: str) -> float:
        """EXP3 score: lower weight → higher score (we minimize arrival time).

        EXP3 minimizes cost, so we use the inverse weight as a score.
        Routes with high observed costs have lower weights.
        """
        weights = self._all_weights()
        if route not in weights or not weights:
            return 0.0  # unknown route: neutral score
        # Mixed distribution: (1-gamma) * softmax + gamma/K uniform
        K = max(len(self._known_routes), 1)
        mixed_prob = (1 - self.gamma) * weights.get(route, 1.0 / K) + self.gamma / K
        return -np.log(max(mixed_prob, 1e-10))  # higher prob = lower score = prefer

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

        # Collect candidates
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
            # Register routes encountered at this stop (needed for weight normalization)
            self._known_routes.add(c.route)
            candidates.append((label, c))
            if len(candidates) >= top_k:
                break

        if not candidates:
            return None

        # EXP3 mixed-strategy selection
        routes = [c.route for _, c in candidates]
        weights = self._all_weights()
        K = len(candidates)

        if weights and all(r in weights for r in routes):
            probs = np.array([(1 - self.gamma) * weights.get(r, 1.0 / K)
                              + self.gamma / K for r in routes])
            probs /= probs.sum()
        else:
            probs = np.ones(K) / K  # uniform when no history

        # Stochastic selection (EXP3 is a randomized algorithm)
        idx = rng.choice(K, p=probs)
        label, c = candidates[idx]
        return label, self.exp3_score(c.route)
