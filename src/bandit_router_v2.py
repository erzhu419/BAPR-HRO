"""Bandit Router V2: Ensemble LCB with dynamic beta.

Key improvements over V1 (bandit_router.py):
  V1: Normal-Gamma parametric posterior, fixed beta=1.5
  V2: Ensemble disagreement for uncertainty, dynamic beta(s)

Architecture (aligned with RE-SAC):
  1. Maintain K belief models per route (ensemble of delay estimators)
  2. Uncertainty = std across ensemble predictions (model-free, no distributional assumption)
  3. Beta adapts based on OOD score: beta(s) = beta_base + beta_ood * OOD(s)
     - High OOD (unfamiliar state) → more pessimistic
     - Low OOD (well-observed state) → can be less conservative
  4. LCB score = mean_arrival + beta(s) * ensemble_std + cancel_penalty

This preserves the same structure-preserving paradigm:
  - Compute hyperpath ONCE
  - Select connections via ensemble LCB re-ranking
  - Update ensemble from delay observations
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .transit_graph import TransitGraph, StopLabel
from .durner.topocsa import topocsa, HyperpathResult
from .bocd.regime_detector import DelayObservation


@dataclass
class RouteEnsembleBelief:
    """Ensemble-based belief about a route's delay distribution.

    Instead of parametric Normal-Gamma, maintains K independent
    delay estimators using bootstrap aggregation.

    Each estimator maintains a running mean/variance from a
    different bootstrap sample of observations.
    """
    n_estimators: int = 5
    # Per-estimator sufficient statistics
    _means: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _vars: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _counts: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Cancellation tracking (shared across ensemble)
    n_cancels: int = 0
    n_attempts: int = 0

    # Prior
    prior_mean: float = 1.0
    # Tightened to Swiss real-data scale: std ≈ 1.4 min vs old 5 min.
    # Old prior_var=25 inflated cold-start std_penalty by 5×, making
    # V2 over-pessimistic on normal days (calibration fix).
    prior_var: float = 2.0
    prior_n: float = 2.0
    # design note: V2 also takes a historical cancel prior (Beta).
    # Defaults match V1 (Beta(1, 99) ≈ 1% cold-start cancel rate).
    cancel_alpha: float = 1.0
    cancel_beta: float = 99.0

    def __post_init__(self):
        if len(self._means) == 0:
            self._means = np.full(self.n_estimators, self.prior_mean)
            self._vars = np.full(self.n_estimators, self.prior_var)
            self._counts = np.full(self.n_estimators, self.prior_n)

    @property
    def ensemble_mean(self) -> float:
        """Mean across ensemble estimators."""
        return float(self._means.mean())

    @property
    def ensemble_std(self) -> float:
        """Uncertainty = std of predictions across ensemble.

        This is the model-free uncertainty estimate (RE-SAC style).
        High disagreement = high uncertainty = unfamiliar state.
        """
        return float(self._means.std())

    @property
    def posterior_std(self) -> float:
        """Combined uncertainty: epistemic (ensemble) + aleatoric (avg variance)."""
        epistemic = self.ensemble_std
        aleatoric = float(np.sqrt(self._vars.mean())) if self._vars.mean() > 0 else 1.0
        return np.sqrt(epistemic ** 2 + (aleatoric / np.sqrt(max(self.total_obs, 1))) ** 2)

    @property
    def total_obs(self) -> int:
        return int(self._counts.sum() - self.n_estimators * self.prior_n)

    @property
    def ood_score(self) -> float:
        """OOD score: high when ensemble disagrees strongly.

        Normalized by average within-estimator std so it's scale-invariant.
        """
        if self.total_obs < 2:
            return 1.0  # max uncertainty when no data
        avg_internal_std = float(np.sqrt(self._vars.mean()))
        if avg_internal_std < 1e-6:
            return 0.0
        return min(self.ensemble_std / avg_internal_std, 3.0)

    @property
    def cancel_rate(self) -> float:
        # design note: now uses the route-level Beta prior parameters
        # (cancel_alpha, cancel_beta) so V2/V3 honor the A4 historical
        # cancel rate the same way V1 does. The earlier hardcoded
        # Beta(1, 99) ignored historical cancel info even when
        # route_priors_override supplied it.
        alpha = self.cancel_alpha + self.n_cancels
        beta = self.cancel_beta + (self.n_attempts - self.n_cancels)
        return alpha / (alpha + beta)

    def update_delay(self, delay: float, rng: np.random.Generator):
        """Update ensemble with observed delay.

        Each estimator includes this observation with probability 1-1/e
        (Poisson bootstrap), giving diversity across estimators.
        """
        # Poisson bootstrap: each estimator samples weight ~ Poisson(1)
        weights = rng.poisson(1, self.n_estimators)

        for k in range(self.n_estimators):
            w = weights[k]
            if w == 0:
                continue  # this estimator skips this observation
            for _ in range(w):
                self._counts[k] += 1
                n = self._counts[k]
                old_mean = self._means[k]
                self._means[k] += (delay - old_mean) / n
                self._vars[k] += ((delay - old_mean) * (delay - self._means[k]) - self._vars[k]) / n

        self.n_attempts += 1

    def update_cancel(self, kind: str = 'true'):
        # correctness fix: typed cancel counters mirroring V1.
        # 'true' = simulator/GTFS-RT cancel signal; 'late_no_show'
        # = patience-window timeout; 'feed_missing' = inferred from
        # absent update.
        self.n_cancels += 1
        self.n_attempts += 1
        if not hasattr(self, 'n_true_cancels'):
            self.n_true_cancels = 0
            self.n_late_no_shows = 0
            self.n_feed_missing = 0
        if kind == 'late_no_show':
            self.n_late_no_shows += 1
        elif kind == 'feed_missing':
            self.n_feed_missing += 1
        else:
            self.n_true_cancels += 1

    def sample_ts(self, scheduled_arrival: float, rng: np.random.Generator) -> float:
        """Thompson Sampling: sample from a random estimator."""
        k = rng.integers(0, self.n_estimators)
        std = max(self._vars[k], 0.01) ** 0.5
        sampled_delay = rng.normal(self._means[k], std)
        if rng.random() < self.cancel_rate:
            return float('inf')
        return scheduled_arrival + sampled_delay


class BanditRouterV2:
    """Ensemble LCB router with dynamic beta.

    V2 improvements:
    1. Ensemble disagreement for uncertainty (no Normal assumption)
    2. Dynamic beta: beta(s) = beta_base + beta_ood * OOD(s)
    3. OOD detection from ensemble disagreement
    """

    def __init__(
        self,
        graph: TransitGraph,
        n_estimators: int = 5,
        beta_base: float = 1.0,
        beta_ood: float = 1.0,
        cancel_penalty_weight: float = 60,
        seed: int = 42,
        route_priors_override: Optional[dict] = None,
        infeasibility_weight: float = 60.0,
        timeout_weight: float = 60.0,
        use_hierarchical_prior: bool = True,
        use_legacy_cold_start: bool = False,
    ):
        self.graph = graph
        self.n_estimators = n_estimators
        self.beta_base = beta_base
        self.beta_ood = beta_ood
        self.cancel_penalty_weight = cancel_penalty_weight
        self.rng = np.random.default_rng(seed)
        # Component-ablation toggles (for the R16 development-history
        # audit). Defaults reproduce the deployed R16 configuration; set
        # use_legacy_cold_start=True to recover the pre-fix V2 behavior
        # in which std_penalty=beta*ensemble_std (zero at cold start),
        # set infeasibility_weight=timeout_weight=0 to disable A7,
        # and set use_hierarchical_prior=False to disable A4.
        self.infeasibility_weight = float(infeasibility_weight)
        self.timeout_weight = float(timeout_weight)
        self.use_hierarchical_prior = bool(use_hierarchical_prior)
        self.use_legacy_cold_start = bool(use_legacy_cold_start)

        self.cached_result: Optional[HyperpathResult] = None
        self.route_beliefs: dict[str, RouteEnsembleBelief] = {}
        self.total_observations: int = 0
        # A4: hierarchical route priors. route_priors_override (e.g. for
        # leave-one-day-out evaluation) takes precedence over the
        # class-level cache.
        if route_priors_override is not None:
            self._route_priors = route_priors_override
        else:
            if not hasattr(type(self), '_route_priors_cache'):
                type(self)._route_priors_cache = None
            if type(self)._route_priors_cache is None:
                try:
                    import pickle
                    with open('data/route_priors.pkl', 'rb') as f:
                        type(self)._route_priors_cache = pickle.load(f)
                except Exception:
                    type(self)._route_priors_cache = {}
            self._route_priors = type(self)._route_priors_cache

    def _get_belief(self, route: str) -> RouteEnsembleBelief:
        if route not in self.route_beliefs:
            # A4: initialize ensemble around the route's historical
            # mean (with a small per-estimator jitter to break the
            # cold-start ensemble_std=0 symmetry that previously
            # collapsed V2 to argmin nominal_arrival).
            p = self._route_priors.get(route) if self.use_hierarchical_prior else None
            if p is None:
                self.route_beliefs[route] = RouteEnsembleBelief(
                    n_estimators=self.n_estimators)
            else:
                hist_mean = float(p['mean'])
                hist_var = max(float(p['std']) ** 2, 0.5)
                # design note: also seed the cancel-rate Beta prior
                # from historical cancel rate (was hardcoded Beta(1, 99)
                # in V2/V3, ignoring A4's per-route cancel prior).
                p_cxl = max(min(float(p.get('cancel_rate', 0.0)), 0.5),
                            1e-4)
                pseudo_n = 100.0
                ca = max(p_cxl * pseudo_n, 1.0)
                cb = max(pseudo_n - ca, 1.0)
                # Tiny per-estimator jitter
                jitter = self.rng.normal(0, 0.1, self.n_estimators)
                belief = RouteEnsembleBelief(
                    n_estimators=self.n_estimators,
                    prior_mean=hist_mean,
                    prior_var=hist_var,
                    cancel_alpha=ca,
                    cancel_beta=cb,
                )
                belief.__post_init__()
                belief._means = belief._means + jitter
                self.route_beliefs[route] = belief
        return self.route_beliefs[route]

    def route(self, s_source: int, s_dest: int, t_source: int,
              max_time: int = 120) -> HyperpathResult:
        self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)
        # correctness fix: store the journey deadline so A7's
        # prob_le() compares against an absolute clock time.
        self.journey_deadline = t_source + max_time
        return self.cached_result

    def observe_delay(self, route: str, delay: float):
        belief = self._get_belief(route)
        belief.update_delay(delay, self.rng)
        self.total_observations += 1

    def observe_cancel(self, route: str, kind: str = 'true'):
        belief = self._get_belief(route)
        belief.update_cancel(kind=kind)
        self.total_observations += 1

    def _compute_dynamic_beta(self, routes: list[str]) -> float:
        """Compute state-dependent beta from ensemble OOD scores.

        beta(s) = beta_base + beta_ood * max_OOD(routes at this stop)

        When all routes are well-observed (low OOD), beta is low → less conservative.
        When any route is poorly observed (high OOD), beta rises → more pessimistic.
        """
        if not routes:
            return self.beta_base + self.beta_ood  # max conservatism

        ood_scores = [self._get_belief(r).ood_score for r in routes]
        max_ood = max(ood_scores)
        return self.beta_base + self.beta_ood * max_ood

    def select_connection(
        self,
        stop_id: int,
        current_time: int,
        rng: np.random.Generator,
        top_k: int = 5,
        beta: float = None,  # if None, use dynamic beta
    ) -> Optional[tuple[StopLabel, float]]:
        """Select best connection using Ensemble LCB with dynamic beta.

        score(route) = mean_dest_arrival
                     + beta(s) * ensemble_std
                     + cancel_penalty_weight * cancel_rate

        beta(s) = beta_base + beta_ood * OOD_score(routes at stop)
        """
        if self.cached_result is None:
            return None

        labels = self.cached_result.stop_labels.get(stop_id, [])
        if not labels:
            return None

        # A5: adaptive top-k / lookahead based on V2's max OOD score.
        # When all routes look in-distribution, narrow window; when any
        # route is OOD, widen.
        any_ood = max((self._get_belief(c.route).ood_score
                        for c_label in labels[-min(8, len(labels)):]
                        for c in [self.graph.connections[c_label.connection_id]]),
                      default=0.0)
        any_ood = float(min(any_ood, 1.0))
        top_k_eff = int(round(top_k + 3 * any_ood))
        lookahead_eff = int(round(25 + 25 * any_ood))

        candidates = []
        seen_routes = set()
        candidate_routes = []

        for label in reversed(labels):
            c = self.graph.connections[label.connection_id]
            if c.dep_time < current_time - 1:
                continue
            if c.dep_time > current_time + lookahead_eff:
                continue
            if c.route in seen_routes:
                continue
            seen_routes.add(c.route)
            candidates.append((label, c))
            candidate_routes.append(c.route)
            if len(candidates) >= top_k_eff:
                break

        if not candidates:
            return None

        # Dynamic beta based on OOD at this stop
        if beta is None:
            beta = self._compute_dynamic_beta(candidate_routes)

        scored = []
        for label, c in candidates:
            belief = self._get_belief(c.route)

            delay_adj = belief.ensemble_mean - 1.0
            # Cold-start fix: use posterior_std (epistemic ensemble disagreement
            # ⊕ aleatoric within-estimator uncertainty / sqrt(n_obs)) rather
            # than ensemble_std alone. ensemble_std is exactly 0 at cold start
            # (all K bootstrap members initialized to the same prior mean), so
            # std_penalty = beta * ensemble_std = 0 there, collapsing V2 to
            # "argmin nominal_dest_arrival" = Static. posterior_std correctly
            # carries the prior aleatoric uncertainty until the first
            # observations diversify the ensemble.
            if self.use_legacy_cold_start:
                # Pre-fix behavior: ensemble_std is identically zero at
                # cold start, so std_penalty=0 and V2 collapses to
                # nominal-mean ranking before any observations.
                std_penalty = beta * belief.ensemble_std
            else:
                std_penalty = beta * belief.posterior_std
            # Only apply cancel penalty after observing this route at
            # least once: pre-observation, the Beta prior gives a uniform
            # cross-route penalty that biases nothing but inflates scores.
            cancel_penalty = (self.cancel_penalty_weight * belief.cancel_rate
                              if belief.n_attempts > 0 else 0.0)
            # A7 layered risk penalties from the hyperpath label.
            infeasibility_penalty = self.infeasibility_weight * (1.0 - label.feasibility)
            if label.dest_arrival is not None and self.timeout_weight > 0.0:
                deadline = getattr(self, 'journey_deadline', None)
                if deadline is None:
                    deadline = current_time + 120
                p_on_time = label.dest_arrival.prob_le(deadline)
                timeout_penalty = self.timeout_weight * (1.0 - p_on_time)
            else:
                timeout_penalty = 0.0

            score = (label.mean_dest_arrival + delay_adj + std_penalty
                     + cancel_penalty + infeasibility_penalty + timeout_penalty)
            scored.append((label, c, score, beta))

        best = min(scored, key=lambda x: x[2])
        return best[0], best[2]

    def get_route_summary(self) -> dict[str, dict]:
        summary = {}
        for route, belief in self.route_beliefs.items():
            summary[route] = {
                "ensemble_mean_delay": belief.ensemble_mean,
                "ensemble_std": belief.ensemble_std,
                "ood_score": belief.ood_score,
                "cancel_rate": belief.cancel_rate,
                "n_obs": belief.total_obs,
                "dynamic_beta": self.beta_base + self.beta_ood * belief.ood_score,
            }
        return summary
