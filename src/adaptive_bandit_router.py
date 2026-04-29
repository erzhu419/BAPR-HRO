"""Adaptive-β Bandit Router for Transit: learns optimal β online.

Wraps BanditRouter and dynamically adjusts β using EXP3 meta-bandit.
At each connection selection, β is drawn from the learned distribution.
After each journey, the total travel time updates the β weights.

This enables the same router to work in:
  - Disrupted transit (should learn β > 0, pessimistic LCB)
  - Normal transit (should learn β ≈ 0, trust the mean)
  - Any intermediate regime
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .transit_graph import TransitGraph, StopLabel
from .bandit_router import BanditRouter, RouteBeliefState
from .durner.topocsa import HyperpathResult


class AdaptiveBetaBanditRouter:
    """Bandit router with EXP3-learned β.

    Inherits all belief tracking from BanditRouter.
    Adds: meta-bandit over β grid, updated after each journey.

    A8 (GPT-5.5 review): cross-journey persistent meta-policy. With a
    single short journey (~6 stops) the EXP3 inner loop has too few
    samples to learn a useful β. We share the meta-bandit state across
    all journeys via a class-level table, so weights accumulate across
    OD pairs and days. Each journey starts from the current global
    posterior over β and contributes one observation back.
    """

    # A8: class-level shared meta-state (persistent across journeys
    # and instances). EXP3 weights, journey costs, and counter are
    # accumulated globally so a single passenger trip's small sample is
    # pooled with all other trips.
    _shared_log_weights: np.ndarray | None = None
    _shared_journey_costs: dict | None = None
    _shared_journey_count: int = 0
    _shared_n_betas: int = 0

    @classmethod
    def reset_shared_state(cls, n_betas: int):
        cls._shared_log_weights = np.zeros(n_betas)
        cls._shared_journey_costs = {i: [] for i in range(n_betas)}
        cls._shared_journey_count = 0
        cls._shared_n_betas = n_betas

    def __init__(
        self,
        graph: TransitGraph,
        beta_grid: list[float] | None = None,
        eta: float = 0.1,
        share_meta_state: bool = True,
        route_priors_override: Optional[dict] = None,
    ):
        self.graph = graph
        self._inner = BanditRouter(graph,
                                   route_priors_override=route_priors_override)
        self.beta_grid = beta_grid or [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        self.n_betas = len(self.beta_grid)
        self.eta = eta
        self.share_meta_state = share_meta_state
        self._rng = np.random.default_rng(42)
        self._current_beta_idx = self.n_betas // 2

        if share_meta_state:
            cls = type(self)
            if (cls._shared_log_weights is None
                    or cls._shared_n_betas != self.n_betas):
                cls.reset_shared_state(self.n_betas)
        else:
            self._log_weights_local = np.zeros(self.n_betas)
            self._journey_costs_local = {i: [] for i in range(self.n_betas)}
            self._journey_count_local = 0

    @property
    def _log_weights(self) -> np.ndarray:
        if self.share_meta_state:
            return type(self)._shared_log_weights
        return self._log_weights_local

    @_log_weights.setter
    def _log_weights(self, val: np.ndarray):
        if self.share_meta_state:
            type(self)._shared_log_weights = val
        else:
            self._log_weights_local = val

    @property
    def _journey_costs(self) -> dict:
        if self.share_meta_state:
            return type(self)._shared_journey_costs
        return self._journey_costs_local

    @_journey_costs.setter
    def _journey_costs(self, val: dict):
        if self.share_meta_state:
            type(self)._shared_journey_costs = val
        else:
            self._journey_costs_local = val

    @property
    def _journey_count(self) -> int:
        if self.share_meta_state:
            return type(self)._shared_journey_count
        return self._journey_count_local

    @_journey_count.setter
    def _journey_count(self, val: int):
        if self.share_meta_state:
            type(self)._shared_journey_count = val
        else:
            self._journey_count_local = val

    @property
    def cached_result(self) -> Optional[HyperpathResult]:
        return self._inner.cached_result

    @cached_result.setter
    def cached_result(self, value):
        self._inner.cached_result = value

    @property
    def route_beliefs(self):
        return self._inner.route_beliefs

    @property
    def total_observations(self):
        return self._inner.total_observations

    @property
    def current_beta(self) -> float:
        return self.beta_grid[self._current_beta_idx]

    @property
    def beta_probs(self) -> np.ndarray:
        w = self._log_weights - self._log_weights.max()
        p = np.exp(w)
        return p / p.sum()

    @property
    def expected_beta(self) -> float:
        return float(np.dot(self.beta_probs, self.beta_grid))

    def route(self, s_source: int, s_dest: int, t_source: int) -> HyperpathResult:
        return self._inner.route(s_source, s_dest, t_source)

    def observe_delay(self, route: str, delay: float):
        self._inner.observe_delay(route, delay)

    def observe_cancel(self, route: str, kind: str = 'true'):
        self._inner.observe_cancel(route, kind=kind)

    def begin_journey(self):
        """Call at start of each journey to sample β."""
        probs = self.beta_probs
        self._current_beta_idx = self._rng.choice(self.n_betas, p=probs)

    def end_journey(self, travel_time: float):
        """Call at end of journey to update β weights."""
        self._journey_costs[self._current_beta_idx].append(travel_time)
        self._journey_count += 1

        # Update EXP3 weights using journey cost
        if self._journey_count >= self.n_betas:
            # Compare: for each β, what is its average journey cost?
            probs = self.beta_probs
            for j in range(self.n_betas):
                if self._journey_costs[j]:
                    avg_cost = np.mean(self._journey_costs[j])
                else:
                    avg_cost = travel_time  # unknown → assume current

                # Normalize
                all_costs = [np.mean(c) for c in self._journey_costs.values() if c]
                if all_costs:
                    cost_range = max(max(all_costs) - min(all_costs), 1e-6)
                    normalized = (avg_cost - min(all_costs)) / cost_range
                    self._log_weights[j] -= self.eta * normalized / max(probs[j], 0.01)

    def select_connection(
        self,
        stop_id: int,
        current_time: int,
        rng: np.random.Generator,
        top_k: int = 5,
        beta: float | None = None,
    ) -> Optional[tuple[StopLabel, float]]:
        """Select connection using adaptive β.

        If beta is None (default), use the EXP3-learned β.
        """
        if beta is None:
            beta = self.beta_grid[self._current_beta_idx]

        return self._inner.select_connection(
            stop_id, current_time, rng, top_k=top_k, beta=beta,
        )
