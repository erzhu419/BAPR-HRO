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
    # A3 (GPT review): typed cancel counters. true_cancels are signal
    # cancellations confirmed by GTFS-RT (delay > 30 min sentinel);
    # late_no_shows are passenger-side timeouts (waited past patience).
    # In current data we cannot distinguish "feed_missing"; we collapse
    # both observed kinds into n_cancels for the cancel_rate property
    # but keep the sub-counts for later reweighting.
    n_true_cancels: int = 0
    n_late_no_shows: int = 0

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

    def update_cancel(self, kind: str = 'true'):
        """Update with a cancellation observation.

        A3 (GPT review): kind ∈ {'true', 'late_no_show', 'feed_missing'}.
        For backward compat, default 'true'. Caller can pass kind to
        track sub-types.
        """
        self.n_cancels += 1
        self.n_attempts += 1
        if kind == 'late_no_show':
            self.n_late_no_shows += 1
        elif kind == 'true':
            self.n_true_cancels += 1
        # 'feed_missing' is structurally tracked but we treat it as
        # not strongly indicative of route reliability.

    @property
    def cancel_rate_by_type(self) -> tuple[float, float]:
        """A3: weighted cancel rate that down-weights late_no_show vs
        true_cancel. Returns (true_rate, late_rate)."""
        denom = self.cancel_alpha + self.cancel_beta + self.n_attempts
        true_rate = (1.0 + self.n_true_cancels) / denom
        late_rate = (1.0 + self.n_late_no_shows) / denom
        return float(true_rate), float(late_rate)

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

    # A4 (GPT-5.5 review): hierarchical prior. Lazy-loaded once per
    # process from data/route_priors.pkl. Each route gets a prior
    # initialized to its historical mean/std/cancel rate (averaged
    # across the 34 normal days), instead of the uniform global prior.
    _route_priors_cache: Optional[dict] = None

    @classmethod
    def _load_route_priors(cls) -> dict:
        if cls._route_priors_cache is not None:
            return cls._route_priors_cache
        try:
            import pickle
            with open('data/route_priors.pkl', 'rb') as f:
                cls._route_priors_cache = pickle.load(f)
        except Exception:
            cls._route_priors_cache = {}
        return cls._route_priors_cache

    def __init__(self, graph: TransitGraph,
                 disruption_gate: bool = True,
                 cancel_threshold: float = 0.05,
                 delay_threshold: float = 10.0,
                 max_time: int = 120,
                 infeasibility_weight: float = 60.0,
                 timeout_weight: float = 60.0,
                 use_hierarchical_prior: bool = True):
        self.graph = graph
        self.cached_result: Optional[HyperpathResult] = None
        # Per-route belief states
        self.route_beliefs: dict[str, RouteBeliefState] = {}
        self.total_observations: int = 0
        self.use_hierarchical_prior = use_hierarchical_prior
        self._route_priors = (self._load_route_priors()
                              if use_hierarchical_prior else {})
        # Disruption gating: scales β by an observed regime signal.
        # If observed cancel rate ≪ cancel_threshold AND observed mean
        # delay ≪ delay_threshold, the day looks normal and we shrink
        # β toward 0 (V1 reduces to Static). If either signal exceeds
        # threshold, β ramps to its full value. This addresses the
        # reviewer's concern that fixed-β LCB hurts on normal days.
        self.disruption_gate = disruption_gate
        self.cancel_threshold = cancel_threshold
        self.delay_threshold = delay_threshold
        # A7 (GPT-5.5 review): layered risk score. Use the hyperpath's
        # built-in `feasibility` and `dest_arrival` PMF to penalize
        # candidates that have a high probability of being infeasible
        # (user already gone past) or arriving past the timeout window.
        # Without these two terms, V1 was over-fitting the scalar
        # `mean_dest_arrival` and getting bitten when the *tail* of the
        # arrival distribution crossed `max_time`. These penalties
        # directly target reach rate (the metric V1 was losing on).
        self.max_time = max_time
        self.infeasibility_weight = infeasibility_weight
        self.timeout_weight = timeout_weight

    def _get_belief(self, route: str) -> RouteBeliefState:
        if route not in self.route_beliefs:
            # A4: hierarchical prior — use this route's historical
            # mean/std/cancel rate when available, else fall back to
            # the global default prior.
            p = self._route_priors.get(route)
            if p is None:
                self.route_beliefs[route] = RouteBeliefState()
            else:
                # Convert std -> var; clip cancel rate to a reasonable
                # Beta-prior denominator. With a per-route historical
                # cancel rate p, set Beta(α, β) so α/(α+β) = p with
                # a moderate effective sample size of 100 pseudo-obs.
                hist_var = max(p['std'] ** 2, 0.5)  # floor to avoid zero
                p_cancel = max(min(p['cancel_rate'], 0.5), 1e-4)
                pseudo_n = 100.0
                cancel_alpha = max(p_cancel * pseudo_n, 1.0)
                cancel_beta = max(pseudo_n - cancel_alpha, 1.0)
                self.route_beliefs[route] = RouteBeliefState(
                    prior_mean=float(p['mean']),
                    prior_var=float(hist_var),
                    prior_n=2.0,
                    cancel_alpha=cancel_alpha,
                    cancel_beta=cancel_beta,
                )
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

        A2 (GPT review): use bilinear combine instead of max — both
        cancel_rate and mean delay contribute multiplicatively, so a
        moderate cancel rate AND a moderate delay together can trigger
        full β even when neither alone saturates the threshold. Pure max
        was binary-ish and missed compound disruption.

        On normal day with cancel≈0.005 and delay≈0.7 min:
            cancel_score = 0.1, delay_score = 0.07 → bilinear ≈ 0.16
        On Oct 29 with cancel≈0.05 and delay≈1.5 min:
            cancel_score = 1.0, delay_score = 0.15 → bilinear ≈ 1.0

        We additionally floor by max(cancel_score, delay_score)/2 so
        a strong single signal still triggers half-β, preserving the
        old behaviour as a lower bound.
        """
        if not self.disruption_gate or self.total_observations < 3:
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
        # Bilinear: each signal contributes; both saturating ⇒ 1.0
        bilinear = 1.0 - (1.0 - cancel_score) * (1.0 - delay_score)
        # Floor by half of the strongest single signal so a clear
        # single-channel disruption still partly engages.
        floor = 0.5 * max(cancel_score, delay_score)
        return float(max(bilinear, floor))

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

        gate = self._disruption_factor()
        beta_eff = beta * gate

        # A5 (GPT review): adaptive top-k and lookahead. On normal day
        # gate≈0, top-k stays at the caller's value (default 5) and
        # lookahead is the canonical 25 minutes. On disrupted day,
        # widen both: top-k expands to 5+3=8 to give the LCB more
        # safe-route candidates to choose from, and lookahead grows to
        # 25+25=50 minutes so the agent can wait through congested
        # buses for a still-feasible later one.
        top_k_eff = int(round(top_k + 3 * gate))
        lookahead_eff = int(round(25 + 25 * gate))

        candidates = []
        seen_routes = set()
        for label in reversed(labels):
            c = self.graph.connections[label.connection_id]
            if c.dep_time < current_time - 1:
                continue
            if c.dep_time > current_time + lookahead_eff:
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
            # A7: layered risk penalties from the hyperpath label itself.
            # `feasibility` ∈ [0,1] is the probability the user is still at
            # this stop in time to board this connection — a low value means
            # the alternative is "looks great on mean but the user
            # probably already missed it". `dest_arrival.prob_le(max_time)`
            # is the probability the destination is reached before the
            # journey timeout. Both directly target reach rate without
            # changing conditional mean.
            infeasibility_penalty = self.infeasibility_weight * (1.0 - label.feasibility)
            if label.dest_arrival is not None:
                p_on_time = label.dest_arrival.prob_le(self.max_time)
                timeout_penalty = self.timeout_weight * (1.0 - p_on_time)
            else:
                timeout_penalty = 0.0

            score = (label.mean_dest_arrival + delay_adj + std_penalty
                     + cancel_penalty + infeasibility_penalty + timeout_penalty)

            candidates.append((label, c, score))

            if len(candidates) >= top_k_eff:
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
