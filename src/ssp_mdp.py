"""Stochastic Shortest Path MDP for transit routing.

Formalizes transit routing as an SSP-MDP (Bertsekas & Tsitsiklis, 1991):

    State:      s = (stop_id, time)
    Action:     a = route to board from hyperpath labels at current stop
    Cost:       c(s, a) = actual travel time on route a (stochastic)
    Transition: s' = (next_stop, arrival_time) determined by delay realization
    Goal:       reach destination stop (absorbing, zero-cost state)

The delay distributions are UNKNOWN and learned online via Bayesian
posterior. This makes it a Bayesian Adaptive SSP (BA-SSP).

We solve the BA-SSP via Posterior Sampling (PSRL):
    1. Sample one delay model from posterior
    2. Evaluate hyperpath labels under sampled model
    3. Pick the best label (greedy under sampled model)
    4. Execute, observe actual delay, update posterior

This is mathematically equivalent to BAPR's mechanism:
- BAPR: posterior over MDP dynamics → sample → solve → execute
- PS-SSP: posterior over delay distributions → sample → evaluate labels → board
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats


@dataclass
class DelayPosterior:
    """Conjugate Normal-Gamma posterior for route delay distribution.

    Model: delay ~ N(mu, 1/tau)
    Prior: mu | tau ~ N(mu_0, 1/(kappa_0 * tau))
           tau ~ Gamma(alpha_0, beta_0)

    This is the standard conjugate prior for unknown mean AND variance,
    enabling proper Bayesian inference from delay observations.
    """
    # Hyperparameters (Normal-Gamma)
    mu_0: float = 1.0       # prior mean delay (minutes)
    kappa_0: float = 2.0    # prior observation count for mean
    alpha_0: float = 2.0    # prior shape for precision
    beta_0: float = 4.0     # prior rate for precision (var ~ beta/alpha = 2.0)

    # Sufficient statistics from observations
    n: int = 0
    sum_x: float = 0.0
    sum_x2: float = 0.0

    # Cancellation tracking (Beta-Binomial)
    cancel_alpha: float = 1.0   # prior: expect low cancel rate
    cancel_beta: float = 19.0   # prior: ~5% cancel

    def observe_delay(self, delay: float):
        """Update posterior with an observed delay."""
        self.n += 1
        self.sum_x += delay
        self.sum_x2 += delay * delay

    def observe_cancel(self):
        """Update cancellation posterior."""
        self.cancel_alpha += 1

    def observe_no_cancel(self):
        """Update cancellation posterior (bus showed up)."""
        self.cancel_beta += 1

    @property
    def _posterior_params(self) -> tuple[float, float, float, float]:
        """Compute Normal-Gamma posterior parameters."""
        kappa_n = self.kappa_0 + self.n
        mu_n = (self.kappa_0 * self.mu_0 + self.sum_x) / kappa_n
        alpha_n = self.alpha_0 + self.n / 2
        if self.n > 0:
            x_bar = self.sum_x / self.n
            s2 = self.sum_x2 / self.n - x_bar ** 2
            beta_n = (self.beta_0
                      + 0.5 * self.n * s2
                      + 0.5 * self.kappa_0 * self.n * (x_bar - self.mu_0) ** 2 / kappa_n)
        else:
            beta_n = self.beta_0
        return mu_n, kappa_n, alpha_n, beta_n

    @property
    def posterior_mean(self) -> float:
        mu_n, _, _, _ = self._posterior_params
        return mu_n

    @property
    def posterior_std(self) -> float:
        mu_n, kappa_n, alpha_n, beta_n = self._posterior_params
        # Marginal variance of mu: beta_n / (alpha_n * kappa_n) [Student-t variance factor]
        if alpha_n > 1:
            return float(np.sqrt(beta_n / (alpha_n * kappa_n) * alpha_n / (alpha_n - 1)))
        return float(np.sqrt(self.beta_0 / self.alpha_0))

    @property
    def cancel_rate(self) -> float:
        return self.cancel_alpha / (self.cancel_alpha + self.cancel_beta)

    def sample_delay(self, rng: np.random.Generator) -> float:
        """Sample a delay value from the posterior predictive distribution.

        This is the core of Posterior Sampling:
        1. Sample precision tau from Gamma posterior
        2. Sample mean mu from Normal posterior conditioned on tau
        3. Sample delay from N(mu, 1/tau)
        """
        mu_n, kappa_n, alpha_n, beta_n = self._posterior_params

        # Sample precision from Gamma(alpha_n, beta_n)
        tau = rng.gamma(alpha_n, 1.0 / max(beta_n, 1e-6))
        tau = max(tau, 1e-6)

        # Sample mean from N(mu_n, 1/(kappa_n * tau))
        mu = rng.normal(mu_n, 1.0 / np.sqrt(kappa_n * tau))

        # Sample delay from N(mu, 1/tau)
        delay = rng.normal(mu, 1.0 / np.sqrt(tau))
        return float(delay)

    def sample_cancel(self, rng: np.random.Generator) -> bool:
        """Sample cancellation from Beta posterior."""
        p = rng.beta(self.cancel_alpha, self.cancel_beta)
        return rng.random() < p


class PosteriorSamplingRouter:
    """Posterior Sampling for Stochastic Shortest Path (PS-SSP).

    This is the theoretically grounded replacement for Bandit-LCB.

    Architecture:
    1. Compute hyperpath ONCE at origin (Durner's TopoCSA)
    2. Maintain conjugate Bayesian posterior per route
    3. At each stop: sample delay model → evaluate labels → pick best
    4. After boarding: observe actual delay → update posterior

    Connection to BAPR:
    - BAPR: posterior over MDP dynamics → Thompson Sampling
    - PS-SSP: posterior over delay distributions → Thompson Sampling
    - Both achieve Bayesian regret O(B*S√(AK)) [Osband & Van Roy, 2017]

    Connection to Durner:
    - Durner provides the hyperpath (the set of "actions" at each stop)
    - PS-SSP provides the online selection policy over these actions
    - Together: Durner's structure + PSRL's online learning
    """

    def __init__(self, graph, dest: int = None):
        from .transit_graph import TransitGraph
        from .durner.topocsa import topocsa

        self.graph: TransitGraph = graph
        self.dest = dest
        self.cached_result = None
        self.posteriors: dict[str, DelayPosterior] = {}
        self.total_observations: int = 0

    def _get_posterior(self, route: str) -> DelayPosterior:
        if route not in self.posteriors:
            self.posteriors[route] = DelayPosterior()
        return self.posteriors[route]

    def route(self, s_source: int, s_dest: int, t_source: int):
        """Initial hyperpath computation (same as static)."""
        from .durner.topocsa import topocsa
        self.dest = s_dest
        self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)
        return self.cached_result

    def observe_delay(self, route: str, delay: float):
        """Update posterior with observed delay."""
        post = self._get_posterior(route)
        post.observe_delay(delay)
        post.observe_no_cancel()
        self.total_observations += 1

    def observe_cancel(self, route: str, kind: str = "true"):
        """Update posterior with cancellation."""
        post = self._get_posterior(route)
        post.observe_cancel()
        self.total_observations += 1

    def select_connection(
        self,
        stop_id: int,
        current_time: int,
        rng: np.random.Generator,
        top_k: int = 5,
        pessimism: float = 0.7,
    ) -> Optional[tuple]:
        """Pessimistic Posterior Sampling for SSP.

        Standard PSRL samples from the full posterior, which causes
        excessive exploration in single-shot settings (transit = one chance,
        no "next episode" to recover from bad exploration).

        We use PESSIMISTIC posterior sampling (Curi et al., 2021):
        - Sample N delay models from posterior
        - For each candidate, use the PESSIMISTIC quantile (e.g., 70th percentile)
        - Pick the candidate with best pessimistic arrival

        This combines PSRL's theoretical framework with LCB's conservatism,
        but derived from proper Bayesian posteriors rather than ad-hoc formulas.

        Args:
            pessimism: quantile to use (0.5 = median, 1.0 = worst-case)
        """
        if self.cached_result is None:
            return None

        labels = self.cached_result.stop_labels.get(stop_id, [])
        if not labels:
            return None

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

            post = self._get_posterior(c.route)

            # Pessimistic Posterior Sampling:
            # Draw multiple samples, use pessimistic quantile
            n_samples = 5
            delay_samples = []
            cancel_count = 0
            for _ in range(n_samples):
                if post.sample_cancel(rng):
                    cancel_count += 1
                else:
                    delay_samples.append(post.sample_delay(rng))

            if cancel_count > n_samples * 0.5:
                # Majority of samples were cancels
                sampled_arrival = float('inf')
            elif delay_samples:
                # Use pessimistic quantile of sampled delays
                pessimistic_delay = np.quantile(delay_samples, pessimism)
                sampled_arrival = label.mean_dest_arrival + (pessimistic_delay - post.mu_0)
                # Add cancel risk as additive penalty
                cancel_risk = cancel_count / n_samples
                sampled_arrival += cancel_risk * 30  # 30 min penalty per cancel probability
            else:
                sampled_arrival = float('inf')

            candidates.append((label, c, sampled_arrival))

            if len(candidates) >= top_k:
                break

        if not candidates:
            return None

        best = min(candidates, key=lambda x: x[2])
        return best[0], best[2]

    def get_route_summary(self) -> dict[str, dict]:
        """Current posterior state for each route."""
        return {
            route: {
                "mean": post.posterior_mean,
                "std": post.posterior_std,
                "cancel_rate": post.cancel_rate,
                "n_obs": post.n,
            }
            for route, post in self.posteriors.items()
        }
