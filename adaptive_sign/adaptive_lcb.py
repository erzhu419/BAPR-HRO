"""Adaptive-β BAPR: learns BOTH sign AND magnitude of β from data.

Evolution:
  V1-LCB:  β = +1.5 (fixed positive = always pessimistic)
  V2-LCB:  β = β_base + β_ood·OOD (dynamic magnitude, always positive)
  Adaptive-Sign: β = β₀ · sign_factor (learned sign, fixed magnitude)
  → Adaptive-β: β learned end-to-end via online gradient descent

Method: discretize β into a grid [-3, -2, -1, 0, +1, +2, +3],
treat each as a "meta-arm", track which β value produces lowest
cost via EXP3-style multiplicative weights.

This is a meta-bandit over β values, where the environment's
irrecoverability structure determines the optimal β.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class BeliefNG:
    """Normal-Gamma belief."""
    mu: float = 0.0
    kappa: float = 0.1
    alpha: float = 2.0
    beta: float = 1.0
    n_obs: int = 0

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def std(self) -> float:
        if self.n_obs == 0:
            return 10.0
        if self.alpha <= 1:
            return 5.0
        return float(np.sqrt(self.beta / (self.kappa * (self.alpha - 1))))

    def update(self, value: float):
        self.n_obs += 1
        if self.n_obs == 1:
            self.mu = value
            self.kappa = 1.0
            self.beta = max(abs(value) * 0.1, 0.1)
            return
        kn = self.kappa + 1
        mn = (self.kappa * self.mu + value) / kn
        an = self.alpha + 0.5
        bn = self.beta + 0.5 * self.kappa * (value - self.mu) ** 2 / kn
        self.mu, self.kappa, self.alpha, self.beta = mn, kn, an, bn


class AdaptiveBetaRouter:
    """Meta-bandit over β: learns optimal β (sign + magnitude) online.

    Maintains a grid of β values and EXP3 weights.
    Each episode: pick β via softmax over weights, select arm using that β,
    observe cost, update both arm beliefs AND β weights.

    β grid: [-2, -1, -0.5, 0, +0.5, +1, +2]
      negative = UCB (optimistic)
      zero = pure mean (no uncertainty)
      positive = LCB (pessimistic)
    """

    def __init__(
        self,
        n_arms: int,
        beta_grid: list[float] | None = None,
        eta: float = 0.1,   # EXP3 learning rate
        warm_costs: list[float] | None = None,
    ):
        self.n_arms = n_arms
        self.beta_grid = beta_grid or [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        self.n_betas = len(self.beta_grid)
        self.eta = eta

        # EXP3 weights over β values (log-space for stability)
        self._log_weights = np.zeros(self.n_betas)

        # Arm beliefs
        self.beliefs = [BeliefNG() for _ in range(n_arms)]
        self._episode = 0
        self._current_beta_idx = self.n_betas // 2  # start at β=0

        # Track for reporting
        self._beta_history = []

        if warm_costs:
            for i, c in enumerate(warm_costs):
                self.beliefs[i].update(c)

    @property
    def current_beta(self) -> float:
        return self.beta_grid[self._current_beta_idx]

    @property
    def beta_probs(self) -> np.ndarray:
        """Current softmax probability over β grid."""
        w = self._log_weights - self._log_weights.max()
        p = np.exp(w)
        return p / p.sum()

    @property
    def expected_beta(self) -> float:
        """Expected β under current distribution."""
        return float(np.dot(self.beta_probs, self.beta_grid))

    @property
    def mode(self) -> str:
        eb = self.expected_beta
        if eb > 0.2:
            return "LCB"
        elif eb < -0.2:
            return "UCB"
        return "~0"

    def _pick_with_beta(self, beta: float) -> int:
        """Select arm using given β value."""
        # Explore unvisited
        for i in range(self.n_arms):
            if self.beliefs[i].n_obs == 0:
                return i

        decay = 1.0 / max(self._episode, 1) ** 0.3
        beta_eff = beta * decay

        best_i, best_s = 0, float("inf")
        for i in range(self.n_arms):
            b = self.beliefs[i]
            score = b.mean + beta_eff * b.std
            if score < best_s:
                best_s = score
                best_i = i
        return best_i

    def select(self) -> int:
        """Select arm using β sampled from current meta-distribution."""
        # Sample β from softmax
        probs = self.beta_probs
        self._current_beta_idx = np.random.choice(self.n_betas, p=probs)
        beta = self.beta_grid[self._current_beta_idx]
        self._beta_history.append(beta)
        return self._pick_with_beta(beta)

    def observe(self, arm_idx: int, cost: float):
        """Update arm belief AND β weights."""
        # Update arm belief
        self.beliefs[arm_idx].update(cost)

        # Update β weights: what would each β have picked, and what would the cost be?
        # For the β that was actually used, we know the true cost.
        # For other βs, estimate using beliefs.
        probs = self.beta_probs

        for j in range(self.n_betas):
            would_pick = self._pick_with_beta(self.beta_grid[j])
            if would_pick == arm_idx:
                # Same arm → same cost
                estimated_cost = cost
            else:
                # Different arm → use belief estimate
                estimated_cost = self.beliefs[would_pick].mean

            # EXP3 update: lower weight for higher cost
            # Normalize cost to [0, 1] range for stable updates
            if self._episode > 0:
                cost_range = max(
                    max(b.mean for b in self.beliefs if b.n_obs > 0) -
                    min(b.mean for b in self.beliefs if b.n_obs > 0),
                    1e-6,
                )
                normalized = (estimated_cost - min(b.mean for b in self.beliefs if b.n_obs > 0)) / cost_range
                self._log_weights[j] -= self.eta * normalized / max(probs[j], 0.01)

        self._episode += 1


class AdaptiveBetaLinkRouter:
    """Link-level adaptive-β router for SDN/VRP/Transit.

    Same meta-bandit over β, but with link-level beliefs.
    """

    def __init__(
        self,
        beta_grid: list[float] | None = None,
        eta: float = 0.1,
    ):
        self.beta_grid = beta_grid or [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        self.n_betas = len(self.beta_grid)
        self.eta = eta

        self._log_weights = np.zeros(self.n_betas)
        self.link_beliefs: dict[tuple, BeliefNG] = {}
        self._episode = 0
        self._current_beta_idx = self.n_betas // 2
        self._beta_history = []
        self._last_paths = None
        self._last_path_mean = 0.0

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

    @property
    def mode(self) -> str:
        eb = self.expected_beta
        if eb > 0.2:
            return "LCB"
        elif eb < -0.2:
            return "UCB"
        return "~0"

    def _link_key(self, i, j):
        return (min(i, j), max(i, j))

    def _get_belief(self, i, j):
        key = self._link_key(i, j)
        if key not in self.link_beliefs:
            self.link_beliefs[key] = BeliefNG()
        return self.link_beliefs[key]

    def _score_path(self, path, beta):
        total = 0.0
        for k in range(len(path) - 1):
            b = self._get_belief(path[k], path[k + 1])
            total += b.mean + beta * b.std
        return total

    def _path_mean(self, path):
        return sum(self._get_belief(path[k], path[k+1]).mean
                   for k in range(len(path) - 1))

    def _pick_path_with_beta(self, paths, beta):
        decay = 1.0 / max(self._episode + 1, 1) ** 0.3
        beta_eff = beta * decay
        best, best_s = 0, float("inf")
        for i, path in enumerate(paths):
            score = self._score_path(path, beta_eff)
            if score < best_s:
                best_s = score
                best = i
        return best

    def select_path(self, paths, **kw) -> int:
        self._last_paths = paths

        probs = self.beta_probs
        self._current_beta_idx = np.random.choice(self.n_betas, p=probs)
        beta = self.beta_grid[self._current_beta_idx]
        self._beta_history.append(beta)

        pick = self._pick_path_with_beta(paths, beta)
        self._last_path_mean = self._path_mean(paths[pick])
        return pick

    def observe(self, path_idx, delay, paths=None, **kw):
        if paths is None:
            paths = self._last_paths

        # Update β weights (clip extreme delays from link failures)
        if paths and self._episode > 5:
            delay_clipped = min(delay, 50.0)  # cap at 50ms to avoid link-failure outliers
            probs = self.beta_probs
            for j in range(self.n_betas):
                would_pick = self._pick_path_with_beta(paths, self.beta_grid[j])
                if would_pick == path_idx:
                    est_cost = delay_clipped
                else:
                    est_cost = min(self._path_mean(paths[would_pick]), 50.0)

                costs_all = [min(self._path_mean(p), 50.0) for p in paths]
                cost_range = max(max(costs_all) - min(costs_all), 1e-6)
                normalized = (est_cost - min(costs_all)) / cost_range
                self._log_weights[j] -= self.eta * normalized / max(probs[j], 0.01)

        self._episode += 1

        # Update link beliefs
        if paths:
            path = paths[path_idx]
            n_links = len(path) - 1
            if n_links > 0:
                per_link = delay / n_links
                for k in range(n_links):
                    self._get_belief(path[k], path[k + 1]).update(per_link)
