"""Adaptive-Sign LCB/UCB: automatically detects whether the environment
rewards pessimism (LCB) or optimism (UCB).

Method: Alternating Trial.
  - Even episodes: play with LCB (β > 0)
  - Odd episodes: play with UCB (β < 0)
  - Track cumulative regret of each
  - After warmup, follow the winner exclusively

This is clean, unbiased, and directly measures which policy is better.
The sign_factor is simply: (UCB_avg_cost - LCB_avg_cost) / normalization.
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


class AdaptiveSignRouter:
    """Alternating-trial adaptive router for arm-level selection.

    Phase 1 (first warmup episodes): alternate LCB/UCB to measure which wins.
    Phase 2: follow the winner exclusively.
    """

    def __init__(self, n_arms: int, beta0: float = 2.0,
                 warmup: int = 20, warm_costs: list[float] | None = None):
        self.n_arms = n_arms
        self.beta0 = beta0
        self.warmup = warmup
        self.beliefs = [BeliefNG() for _ in range(n_arms)]
        self._episode = 0

        self._lcb_costs = []  # costs observed when using LCB
        self._ucb_costs = []  # costs observed when using UCB

        if warm_costs:
            for i, c in enumerate(warm_costs):
                self.beliefs[i].update(c)

    @property
    def sign_factor(self) -> float:
        """Positive = LCB better, Negative = UCB better."""
        if not self._lcb_costs or not self._ucb_costs:
            return 0.0
        lcb_avg = np.mean(self._lcb_costs)
        ucb_avg = np.mean(self._ucb_costs)
        total = (lcb_avg + ucb_avg) / 2
        if total < 1e-8:
            return 0.0
        # Positive when UCB costs more (LCB is better)
        return np.clip((ucb_avg - lcb_avg) / total, -1, 1)

    @property
    def mode(self) -> str:
        sf = self.sign_factor
        if sf > 0.01:
            return "LCB"
        elif sf < -0.01:
            return "UCB"
        return "~0"

    def _pick(self, sign: float) -> int:
        """Pick arm with sign-adjusted β."""
        # Explore first
        for i in range(self.n_arms):
            if self.beliefs[i].n_obs == 0:
                return i

        beta = self.beta0 * sign / max(self._episode, 1) ** 0.3
        best_i, best_score = 0, float("inf")
        for i in range(self.n_arms):
            b = self.beliefs[i]
            score = b.mean + beta * b.std
            if score < best_score:
                best_score = score
                best_i = i
        return best_i

    def select(self) -> int:
        if self._episode < self.warmup:
            # Alternate: even=LCB, odd=UCB
            if self._episode % 2 == 0:
                self._current_mode = "LCB"
                return self._pick(+1.0)
            else:
                self._current_mode = "UCB"
                return self._pick(-1.0)
        else:
            # Follow the winner
            sf = self.sign_factor
            self._current_mode = "LCB" if sf >= 0 else "UCB"
            return self._pick(+1.0 if sf >= 0 else -1.0)

    def observe(self, arm_idx: int, cost: float):
        if self._episode < self.warmup:
            if self._current_mode == "LCB":
                self._lcb_costs.append(cost)
            else:
                self._ucb_costs.append(cost)

        self.beliefs[arm_idx].update(cost)
        self._episode += 1


class AdaptiveSignLinkRouter:
    """Dual-play adaptive router: runs LCB and UCB beliefs in parallel.

    Both policies share the same observations (link delays) but make
    different path selections. The sign_factor tracks which policy
    WOULD HAVE accumulated lower total delay.
    """

    def __init__(self, beta0: float = 2.0, warmup: int = 30):
        self.beta0 = beta0
        self.warmup = warmup
        # Shared beliefs (both policies see the same data)
        self.link_beliefs: dict[tuple, BeliefNG] = {}
        self._episode = 0

        # Track virtual costs of each policy
        self._lcb_total = 0.0
        self._ucb_total = 0.0
        self._n_compared = 0

    @property
    def sign_factor(self) -> float:
        if self._n_compared < 10:
            return 0.0
        total = (self._lcb_total + self._ucb_total) / 2
        if total < 1e-8:
            return 0.0
        # Positive when UCB costs more → LCB wins
        return np.clip((self._ucb_total - self._lcb_total) / total, -1, 1)

    @property
    def mode(self) -> str:
        sf = self.sign_factor
        if sf > 0.01:
            return "LCB"
        elif sf < -0.01:
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
                   for k in range(len(path)-1))

    def select_path(self, paths, **kw) -> int:
        beta_mag = self.beta0 / max(self._episode + 1, 1) ** 0.3

        # Compute what both policies would pick
        lcb_pick = min(range(len(paths)),
                       key=lambda i: self._score_path(paths[i], +beta_mag))
        ucb_pick = min(range(len(paths)),
                       key=lambda i: self._score_path(paths[i], -beta_mag))
        self._last_lcb_pick = lcb_pick
        self._last_ucb_pick = ucb_pick
        self._last_paths = paths

        # During warmup: alternate; after: follow winner
        if self._episode < self.warmup:
            return lcb_pick if self._episode % 2 == 0 else ucb_pick
        else:
            sf = self.sign_factor
            return lcb_pick if sf >= 0 else ucb_pick

    def observe(self, path_idx, delay, paths=None, **kw):
        self._episode += 1

        if paths is None:
            paths = self._last_paths

        # Track virtual costs: what would each policy have experienced?
        if hasattr(self, '_last_lcb_pick') and paths:
            self._n_compared += 1
            if self._last_lcb_pick == path_idx:
                self._lcb_total += delay
            else:
                self._lcb_total += self._path_mean(paths[self._last_lcb_pick])

            if self._last_ucb_pick == path_idx:
                self._ucb_total += delay
            else:
                self._ucb_total += self._path_mean(paths[self._last_ucb_pick])

        # Update shared link beliefs
        if paths:
            path = paths[path_idx]
            n_links = len(path) - 1
            if n_links > 0:
                per_link = delay / n_links
                for k in range(n_links):
                    self._get_belief(path[k], path[k + 1]).update(per_link)
