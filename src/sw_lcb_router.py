"""Sliding-Window LCB Router for non-stationary transit bandits.

Baseline from:
  Garivier & Moulines (2011) "On Upper-Confidence Bound Policies for
  Switching Bandit Problems." ALT 2011.

  Cheung, Simchi-Levi & Zhu (2019) "Learning to Optimize under
  Non-Stationarity." AISTATS 2019.

Recent applications to transit networks:
  Drift-aware routing (Luo et al., 2024) uses SW-UCB on route arms.

Key difference vs V1-LCB:
  V1 (BanditRouter): full Bayesian posterior over all history —
    good when drift is slow, but stale when regime shifts sharply.
  SW-LCB: only last W observations per route —
    adapts faster after disruptions, but higher variance under stability.

This implements the pessimistic (LCB) variant:
  score(c) = mu_hat_W(c) + beta * sigma_hat_W(c) + gamma * p_cancel_W(c)
where statistics are computed over the sliding window only.
"""

from __future__ import annotations

from collections import deque
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Deque

from .transit_graph import TransitGraph, StopLabel
from .durner.topocsa import topocsa, HyperpathResult


@dataclass
class SWRouteBeliefState:
    """Sliding-window belief for a single route.

    Maintains a fixed-size window of recent observations.
    Statistics computed over window only — older data discarded.
    """
    window_size: int = 20

    _delay_window: Deque[float] = field(default_factory=deque)
    _cancel_window: Deque[bool] = field(default_factory=deque)

    # Prior (used when window is empty or has few observations)
    prior_mean: float = 1.0
    prior_std: float = 5.0
    prior_n: float = 2.0

    def _push(self, val, window: deque, maxlen: int):
        window.append(val)
        while len(window) > maxlen:
            window.popleft()

    def observe_delay(self, delay: float):
        self._push(delay, self._delay_window, self.window_size)

    def observe_cancel(self):
        # P1 R3 review: only push to cancel window. Earlier code
        # pushed True (=1.0 in numpy) into the delay window, which
        # compressed cancellation events into a 1-min delay sample
        # and pulled window_mean / window_std down, weakening the
        # cancel penalty.
        self._push(True, self._cancel_window, self.window_size)

    def observe_no_cancel(self):
        self._push(False, self._cancel_window, self.window_size)

    @property
    def n_obs(self) -> int:
        return len(self._delay_window)

    @property
    def window_mean(self) -> float:
        if not self._delay_window:
            return self.prior_mean
        n = len(self._delay_window)
        return (self.prior_n * self.prior_mean + sum(self._delay_window)) / (self.prior_n + n)

    @property
    def window_std(self) -> float:
        if len(self._delay_window) < 2:
            return self.prior_std
        arr = np.array(list(self._delay_window))
        obs_std = arr.std()
        n = len(arr)
        # Blend with prior std (shrinkage towards prior as n decreases)
        w = n / (n + self.prior_n)
        return w * max(obs_std, 0.5) + (1 - w) * self.prior_std

    @property
    def cancel_rate(self) -> float:
        if not self._cancel_window:
            return 0.05  # prior: ~5% cancel
        cancels = sum(self._cancel_window)
        total = len(self._cancel_window)
        # Beta(1+cancels, 9+total-cancels) posterior mean
        alpha = 1 + cancels
        beta = 9 + (total - cancels)
        return alpha / (alpha + beta)


class SWLCBRouter:
    """Sliding-Window LCB router.

    Identical interface to BanditRouter so it plugs into
    simulate_bandit_journey without modification.

    Parameters:
        window_size (int): Window W for recent observations. Default 20.
            Smaller W = faster adaptation, higher variance.
            Larger W = slower adaptation, lower variance.
        beta (float): Pessimism level. Default 1.5 (matches V1-LCB).
        gamma (float): Cancellation penalty (minutes). Default 60.
    """

    def __init__(
        self,
        graph: TransitGraph,
        window_size: int = 20,
        beta: float = 1.5,
        gamma: float = 60.0,
    ):
        self.graph = graph
        self.window_size = window_size
        self.beta = beta
        self.gamma = gamma
        self.cached_result: Optional[HyperpathResult] = None
        self.beliefs: dict[str, SWRouteBeliefState] = {}
        self.total_observations: int = 0

    def _get_belief(self, route: str) -> SWRouteBeliefState:
        if route not in self.beliefs:
            self.beliefs[route] = SWRouteBeliefState(window_size=self.window_size)
        return self.beliefs[route]

    def route(self, s_source: int, s_dest: int, t_source: int) -> HyperpathResult:
        self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)
        return self.cached_result

    def observe_delay(self, route: str, delay: float):
        b = self._get_belief(route)
        b.observe_delay(delay)
        b.observe_no_cancel()
        self.total_observations += 1

    def observe_cancel(self, route: str, kind: str = 'true'):
        b = self._get_belief(route)
        b.observe_cancel()
        self.total_observations += 1

    def lcb_score(self, label: StopLabel, route: str) -> float:
        """SW-LCB score = windowed mean + beta * windowed std + cancel penalty."""
        b = self._get_belief(route)
        delay_adj = b.window_mean - b.prior_mean
        uncertainty = self.beta * b.window_std
        cancel_pen = self.gamma * b.cancel_rate
        return label.mean_dest_arrival + delay_adj + uncertainty + cancel_pen

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
            score = self.lcb_score(label, c.route)
            candidates.append((label, c, score))
            if len(candidates) >= top_k:
                break

        if not candidates:
            return None
        best = min(candidates, key=lambda x: x[2])
        return best[0], best[2]
