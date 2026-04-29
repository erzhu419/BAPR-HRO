"""Bandit Router V3: Topology-Aware Meta-Adaptive LCB.

V3 extends V2 with network topology awareness, following BAPR's
meta-learning principle:

  V1: fixed β (one size fits all)
  V2: β(s) = β_base + β_ood * OOD(s)  (adapts to observation uncertainty)
  V3: β(s) = (β_base + β_ood * OOD(s)) * topo_gate(s)  (adapts to network structure)

The topo_gate captures:
  - How many alternative routes exist at this stop
  - Whether alternatives are meaningfully different (not same-route variants)
  - How much "choice space" the LCB can exploit

Key insight: LCB is useless when there's only 1 route to choose from.
The topo_gate smoothly scales β from 0 (no alternatives, trust nominal)
to 1 (many alternatives, full LCB).

This is BAPR's meta-learning applied to network structure:
  BAPR outer loop: detect regime → adjust β
  V3 outer loop: detect network topology → adjust β

Formally verified: V3 bound ≤ V2 bound (BAPRHRO_V2.lean,
dynamic_dominates_fixed), because topo_gate ∈ [0,1] means
V3's β(s) ≤ V2's β(s) at every stop.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .transit_graph import TransitGraph, StopLabel
from .bandit_router_v2 import BanditRouterV2, RouteEnsembleBelief
from .durner.topocsa import topocsa, HyperpathResult


class BanditRouterV3(BanditRouterV2):
    """Topology-aware meta-adaptive LCB router.

    Inherits V2's ensemble + dynamic beta. Adds topology gating.
    """

    def __init__(
        self,
        graph: TransitGraph,
        n_estimators: int = 5,
        beta_base: float = 1.0,
        beta_ood: float = 1.0,
        cancel_penalty_weight: float = 60,
        topo_threshold: int = 2,  # min routes for full β
        seed: int = 42,
    ):
        super().__init__(graph, n_estimators, beta_base, beta_ood,
                         cancel_penalty_weight, seed)
        self.topo_threshold = topo_threshold

        # Pre-compute per-stop topology features
        self._stop_n_routes: dict[int, int] = {}
        self._precompute_topology()

    def _precompute_topology(self):
        """Count distinct routes per stop (both departing and arriving)."""
        for sid in self.graph.stops:
            routes = self.graph.get_routes_at_stop(sid)
            self._stop_n_routes[sid] = len(routes)

    def _topo_gate(self, stop_id: int, n_candidate_routes: int) -> float:
        """Topology gate: scales β based on ACTUAL alternatives in hyperpath.

        Uses the number of distinct candidate routes at decision time,
        NOT the number of routes serving the stop (which may not all
        be in the hyperpath).

        Returns a value in [0, 1]:
          0: only 1 candidate route → no choice → β = 0 (trust nominal)
          1: ≥ topo_threshold candidate routes → full LCB power
        """
        # Use actual candidates — this is what matters for decision making
        n = n_candidate_routes

        if n <= 1:
            return 0.0  # no choice → trust nominal
        elif n >= self.topo_threshold:
            return 1.0  # enough alternatives → full LCB
        else:
            # Linear interpolation
            return (n - 1) / (self.topo_threshold - 1)

    def _compute_dynamic_beta(self, routes: list[str], stop_id: int = None) -> float:
        """V3 dynamic beta = V2 beta × topo_gate.

        When stop has many routes: β = β_base + β_ood * OOD (same as V2)
        When stop has 1 route:    β ≈ 0 (trust nominal, don't penalize)
        """
        if not routes:
            return 0.0  # no candidates → no penalty

        # V2 base: OOD-driven beta
        ood_scores = [self._get_belief(r).ood_score for r in routes]
        max_ood = max(ood_scores)
        v2_beta = self.beta_base + self.beta_ood * max_ood

        # V3: scale by topology gate
        gate = self._topo_gate(stop_id, len(set(routes)))
        return v2_beta * gate

    def select_connection(
        self,
        stop_id: int,
        current_time: int,
        rng: np.random.Generator,
        top_k: int = 5,
        beta: float = None,
    ) -> Optional[tuple[StopLabel, float]]:
        """Select connection with topology-aware beta.

        When only 1 route available: β ≈ 0, score ≈ nominal (like static)
        When multiple routes: β > 0, score = nominal + LCB penalty (like V2)
        """
        if self.cached_result is None:
            return None

        labels = self.cached_result.stop_labels.get(stop_id, [])
        if not labels:
            return None

        candidates = []
        seen_routes = set()
        candidate_routes = []

        for label in reversed(labels):
            c = self.graph.connections[label.connection_id]
            if c.dep_time < current_time - 1:
                continue
            if c.dep_time > current_time + 25:
                continue
            if c.route in seen_routes:
                continue
            seen_routes.add(c.route)
            candidates.append((label, c))
            candidate_routes.append(c.route)
            if len(candidates) >= top_k:
                break

        if not candidates:
            return None

        # V3: topology-aware dynamic beta
        if beta is None:
            beta = self._compute_dynamic_beta(candidate_routes, stop_id)

        # Compute topo_gate for this decision point
        gate = self._topo_gate(stop_id, len(set(candidate_routes)))

        scored = []
        for label, c in candidates:
            belief = self._get_belief(c.route)
            delay_adj = belief.ensemble_mean - 1.0
            # V3: topo_gate scales ALL penalties, not just β
            # When only 1 route: gate=0 → score=nominal+delay_adj (like static)
            # When multiple routes: gate=1 → full LCB scoring
            # Cold-start fix (mirrors V2): use posterior_std, not ensemble_std.
            # See bandit_router_v2.py for rationale.
            std_penalty = beta * belief.posterior_std
            cancel_penalty = (self.cancel_penalty_weight * belief.cancel_rate * gate
                              if belief.n_attempts > 0 else 0.0)
            # A7 (GPT review): layered risk penalties.
            infeasibility_penalty = 60.0 * (1.0 - label.feasibility)
            if label.dest_arrival is not None:
                p_on_time = label.dest_arrival.prob_le(120)
                timeout_penalty = 60.0 * (1.0 - p_on_time)
            else:
                timeout_penalty = 0.0

            score = (label.mean_dest_arrival + delay_adj + std_penalty
                     + cancel_penalty + infeasibility_penalty + timeout_penalty)
            scored.append((label, c, score, beta))

        best = min(scored, key=lambda x: x[2])
        return best[0], best[2]
