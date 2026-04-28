"""Bandit Router: Thompson Sampling on Durner's hyperpath labels.

Core idea: Durner's hyperpath already contains the right SET of alternatives.
The problem is RANKING them under real-time conditions. We model this as a
contextual bandit:

- At each stop, the "arms" are the available routes in the hyperpath
- Each arm has an uncertain "reward" = -(actual arrival time at destination)
- We maintain a posterior over each arm's expected arrival time
- Thompson Sampling selects which route to try first
- After observing delays/cancels, we update the posterior

This avoids the two failure modes of the previous adaptive approach:
1. No hyperpath recomputation → no over-conservative rerouting
2. No regime detection needed → learns from direct observations

The posterior is a simple Normal-InverseGamma for each route, tracking
mean delay and variance from actual observations.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .transit_graph import TransitGraph, StopLabel
from .durner.topocsa import topocsa, HyperpathResult
from .bocd.regime_detector import DelayObservation


@dataclass
class RouteBeliefState:
    """Bayesian belief about a route's delay distribution.

    Normal-Gamma conjugate prior:
    - delay ~ N(mu, sigma^2)
    - We track posterior over (mu, sigma^2)

    Simplified: just track running mean and variance of observed delays,
    plus a cancel rate estimate.
    """
    n_obs: int = 0
    delay_sum: float = 0.0
    delay_sq_sum: float = 0.0
    n_cancels: int = 0
    n_attempts: int = 0

    # Prior parameters. The defaults reflect Swiss real-data normal-day
    # statistics: mean delay ≈ 0.7 min, std ≈ 1.3 min, cancel rate ≈
    # 0.5%. Earlier defaults (prior_var=25, Beta(1,9) for cancels) were
    # ~15× too pessimistic and effectively added a uniform 13.5-min
    # uncertainty tax to every candidate connection at cold start, which
    # biased V1 toward already-ridden routes regardless of the
    # hyperpath's nominal ordering. The reviewer flagged this as
    # over-pessimism in the GTFS-RT setting.
    prior_mean: float = 1.0    # expect ~1 min delay (matches Swiss)
    prior_var: float = 2.0     # std=1.4 (matches Swiss real-data)
    prior_n: float = 2.0       # weak prior (2 pseudo-observations)
    cancel_alpha: float = 1.0  # Beta cancel prior numerator
    cancel_beta: float = 99.0  # Beta(1,99) → expected cancel ≈ 1%

    @property
    def posterior_mean(self) -> float:
        """Posterior mean delay estimate (Bayesian update)."""
        total_n = self.prior_n + self.n_obs
        return (self.prior_n * self.prior_mean + self.delay_sum) / total_n

    @property
    def posterior_var(self) -> float:
        """Posterior variance of delay estimate."""
        total_n = self.prior_n + self.n_obs
        if total_n < 2:
            return self.prior_var
        # Combine prior variance with observed variance
        if self.n_obs > 1:
            obs_mean = self.delay_sum / self.n_obs
            obs_var = (self.delay_sq_sum / self.n_obs - obs_mean ** 2)
        else:
            obs_var = self.prior_var
        return (self.prior_n * self.prior_var + self.n_obs * obs_var) / total_n

    @property
    def cancel_rate(self) -> float:
        """Estimated cancellation probability (Beta-Binomial posterior).

        Default prior Beta(1, 99) → expected 1% (was Beta(1,9) → 10%
        which was 20× higher than Swiss-normal-day reality).
        """
        alpha = self.cancel_alpha + self.n_cancels
        beta = self.cancel_beta + (self.n_attempts - self.n_cancels)
        return alpha / (alpha + beta)

    def update_delay(self, delay: float):
        """Update with an observed delay."""
        self.n_obs += 1
        self.delay_sum += delay
        self.delay_sq_sum += delay * delay
        self.n_attempts += 1

    def update_cancel(self):
        """Update with a cancellation observation."""
        self.n_cancels += 1
        self.n_attempts += 1

    def sample_expected_arrival(self, scheduled_arrival: float, rng: np.random.Generator) -> float:
        """Thompson Sampling: sample expected arrival from posterior.

        Returns sampled arrival time accounting for delay uncertainty and cancel risk.
        """
        # Sample delay from posterior
        std = max(self.posterior_var, 0.01) ** 0.5
        sampled_delay = rng.normal(self.posterior_mean, std)

        # Factor in cancellation risk: if canceled, arrival = infinity
        if rng.random() < self.cancel_rate:
            return float('inf')

        return scheduled_arrival + sampled_delay


class BanditRouter:
    """Contextual bandit router using Thompson Sampling on hyperpath labels.

    Architecture:
    1. Compute hyperpath ONCE at origin (same as static)
    2. At each stop, maintain beliefs about each route's delay/cancel rate
    3. Use Thompson Sampling to pick which route to take
    4. Update beliefs based on actual observations (delays, cancels)

    No regime detection. No hyperpath recomputation. Just learning.
    """

    def __init__(self, graph: TransitGraph,
                 disruption_gate: bool = True,
                 cancel_threshold: float = 0.02,
                 delay_threshold: float = 3.0):
        self.graph = graph
        self.cached_result: Optional[HyperpathResult] = None
        # Per-route belief states
        self.route_beliefs: dict[str, RouteBeliefState] = {}
        self.total_observations: int = 0
        # Disruption gating: scales β by an observed regime signal.
        # If observed cancel rate ≪ cancel_threshold AND observed mean
        # delay ≪ delay_threshold, the day looks normal and we shrink
        # β toward 0 (V1 reduces to Static). If either signal exceeds
        # threshold, β ramps to its full value. This addresses the
        # reviewer's concern that fixed-β LCB hurts on normal days.
        self.disruption_gate = disruption_gate
        self.cancel_threshold = cancel_threshold
        self.delay_threshold = delay_threshold

    def _get_belief(self, route: str) -> RouteBeliefState:
        if route not in self.route_beliefs:
            self.route_beliefs[route] = RouteBeliefState()
        return self.route_beliefs[route]

    def route(self, s_source: int, s_dest: int, t_source: int) -> HyperpathResult:
        """Initial route computation (same as static)."""
        self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)
        return self.cached_result

    def observe_delay(self, route: str, delay: float):
        """Feed an observed delay to update route belief."""
        belief = self._get_belief(route)
        belief.update_delay(delay)
        self.total_observations += 1

    def observe_cancel(self, route: str):
        """Feed a cancellation observation."""
        belief = self._get_belief(route)
        belief.update_cancel()
        self.total_observations += 1

    def _disruption_factor(self) -> float:
        """Compute disruption signal in [0, 1] from accumulated observations.

        Aggregates observed cancel rate and mean delay across all routes.
        Returns 0 if conditions look normal (≪ thresholds), 1 if either
        signal saturates. This is the regime-detection mechanism that
        gates β: on a normal day we want β_eff ≈ 0; on a disrupted day,
        β_eff ≈ β. The transit V1 previously used fixed β, which biased
        the policy toward already-ridden routes irrespective of regime.
        """
        if not self.disruption_gate or self.total_observations < 3:
            # Cold start: no evidence yet; use neutral β=0 to avoid
            # over-pessimism on what might be a normal day.
            return 0.0
        total_delay, total_n, total_cancels, total_attempts = 0.0, 0, 0, 0
        for b in self.route_beliefs.values():
            total_delay += b.delay_sum
            total_n += b.n_obs
            total_cancels += b.n_cancels
            total_attempts += b.n_attempts
        avg_delay = total_delay / max(total_n, 1)
        cancel_rate = total_cancels / max(total_attempts, 1)
        cancel_score = min(cancel_rate / self.cancel_threshold, 1.0)
        delay_score = min(max(avg_delay, 0) / self.delay_threshold, 1.0)
        return float(max(cancel_score, delay_score))

    def select_connection(
        self,
        stop_id: int,
        current_time: int,
        rng: np.random.Generator,
        top_k: int = 5,
        beta: float = 1.5,
    ) -> Optional[tuple[StopLabel, float]]:
        """Select best connection using LCB (Lower Confidence Bound).

        Like BAPR's adaptive conservatism: pick the route with best
        PESSIMISTIC expected arrival. Uncertain routes are penalized.
        Routes with observed cancellations are heavily penalized.

        score(route) = mean_dest_arrival + beta * posterior_std + cancel_penalty

        Lower score = better. Pick the lowest.
        """
        if self.cached_result is None:
            return None

        labels = self.cached_result.stop_labels.get(stop_id, [])
        if not labels:
            return None

        # Apply disruption gate: shrink β when no disruption signal
        # observed yet, restore to full β when disruption detected.
        # (We tested an additional std-amplification under disruption
        # (1+4·gate) but it over-pessimized — V1 skipped buses on Oct
        # 29 and timed out, dropping reach from 49% to 38%. The
        # un-amplified version gives the best aggregate result.)
        gate = self._disruption_factor()
        beta_eff = beta * gate

        candidates = []
        seen_routes = set()
        for label in reversed(labels):
            c = self.graph.connections[label.connection_id]
            if c.dep_time < current_time - 1:
                continue
            if c.dep_time > current_time + 25:
                continue
            if c.route in seen_routes:
                continue
            seen_routes.add(c.route)

            belief = self._get_belief(c.route)

            # LCB score: nominal arrival + delay adjustment + uncertainty penalty
            delay_adj = belief.posterior_mean - 1.0  # subtract prior mean
            std_penalty = beta_eff * (belief.posterior_var ** 0.5)
            # Cancel penalty: only fire if we have evidence of cancels
            # (gate by total_attempts to avoid penalizing the prior).
            cancel_penalty = belief.cancel_rate * 60 if belief.n_attempts > 0 else 0.0

            score = label.mean_dest_arrival + delay_adj + std_penalty + cancel_penalty

            candidates.append((label, c, score))

            if len(candidates) >= top_k:
                break

        if not candidates:
            return None

        best = min(candidates, key=lambda x: x[2])
        return best[0], best[2]

    def get_route_summary(self) -> dict[str, dict]:
        """Get current belief state for all observed routes."""
        summary = {}
        for route, belief in self.route_beliefs.items():
            summary[route] = {
                "posterior_mean_delay": belief.posterior_mean,
                "posterior_std": belief.posterior_var ** 0.5,
                "cancel_rate": belief.cancel_rate,
                "n_obs": belief.n_obs,
                "n_cancels": belief.n_cancels,
            }
        return summary
