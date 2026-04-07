"""Adaptive-Sign BAPR: auto-detects LCB vs UCB from environment structure.

Three-phase protocol:
  Phase 1 (explore): try each arm once to initialize beliefs
  Phase 2 (calibrate): alternate LCB/UCB on the SAME best-mean arm
     to measure whether pessimism or optimism gives lower realized cost
  Phase 3 (exploit): follow the detected winner

The key insight: during calibration, LCB and UCB may pick DIFFERENT arms.
We track the cumulative cost of each policy's picks to determine which
policy structure (pessimistic or optimistic) suits this environment.
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
    """Arm-level adaptive LCB/UCB router.

    Phase 1: explore each arm once
    Phase 2: alternate LCB(+β)/UCB(-β) picks for calibrate_eps episodes
    Phase 3: follow the winner
    """

    def __init__(self, n_arms: int, beta0: float = 2.0,
                 calibrate_eps: int = 10,
                 warm_costs: list[float] | None = None):
        self.n_arms = n_arms
        self.beta0 = beta0
        self.calibrate_eps = calibrate_eps
        self.beliefs = [BeliefNG() for _ in range(n_arms)]
        self._episode = 0
        self._current_mode = "LCB"

        # Phase 2 tracking
        self._lcb_costs = []
        self._ucb_costs = []

        if warm_costs:
            for i, c in enumerate(warm_costs):
                self.beliefs[i].update(c)
            self._episode = n_arms  # skip explore phase

    @property
    def sign_factor(self) -> float:
        if not self._lcb_costs or not self._ucb_costs:
            return 0.0
        lcb_avg = np.mean(self._lcb_costs)
        ucb_avg = np.mean(self._ucb_costs)
        total = (lcb_avg + ucb_avg) / 2
        if total < 1e-8:
            return 0.0
        return np.clip((ucb_avg - lcb_avg) / total, -1, 1)

    @property
    def mode(self) -> str:
        sf = self.sign_factor
        if sf > 0.01:
            return "LCB"
        elif sf < -0.01:
            return "UCB"
        return "~0"

    def _pick_with_sign(self, sign: float) -> int:
        beta = self.beta0 * sign / max(self._episode, 1) ** 0.3
        best_i, best_s = 0, float("inf")
        for i in range(self.n_arms):
            b = self.beliefs[i]
            score = b.mean + beta * b.std
            if score < best_s:
                best_s = score
                best_i = i
        return best_i

    def select(self) -> int:
        # Phase 1: explore
        if self._episode < self.n_arms:
            for i in range(self.n_arms):
                if self.beliefs[i].n_obs == 0:
                    self._current_mode = "explore"
                    return i

        # Phase 2: calibrate — alternate LCB/UCB
        cal_ep = self._episode - self.n_arms
        if cal_ep < self.calibrate_eps:
            if cal_ep % 2 == 0:
                self._current_mode = "LCB"
                return self._pick_with_sign(+1.0)
            else:
                self._current_mode = "UCB"
                return self._pick_with_sign(-1.0)

        # Phase 3: exploit winner
        sf = self.sign_factor
        self._current_mode = "LCB" if sf >= 0 else "UCB"
        return self._pick_with_sign(+1.0 if sf >= 0 else -1.0)

    def observe(self, arm_idx: int, cost: float):
        # Track calibration costs
        if self._current_mode == "LCB":
            self._lcb_costs.append(cost)
        elif self._current_mode == "UCB":
            self._ucb_costs.append(cost)

        self.beliefs[arm_idx].update(cost)
        self._episode += 1


class AdaptiveSignLinkRouter:
    """Link-level adaptive LCB/UCB router for SDN/VRP.

    Same three-phase protocol but at the path level with link-level beliefs.
    """

    def __init__(self, beta0: float = 2.0, calibrate_eps: int = 20):
        self.beta0 = beta0
        self.calibrate_eps = calibrate_eps
        self.link_beliefs: dict[tuple, BeliefNG] = {}
        self._episode = 0
        self._current_mode = "LCB"
        self._lcb_delays = []
        self._ucb_delays = []

    @property
    def sign_factor(self) -> float:
        if not self._lcb_delays or not self._ucb_delays:
            return 0.0
        lcb_avg = np.mean(self._lcb_delays)
        ucb_avg = np.mean(self._ucb_delays)
        total = (lcb_avg + ucb_avg) / 2
        if total < 1e-8:
            return 0.0
        return np.clip((ucb_avg - lcb_avg) / total, -1, 1)

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

    def select_path(self, paths, **kw) -> int:
        beta_mag = self.beta0 / max(self._episode + 1, 1) ** 0.3

        if self._episode < self.calibrate_eps:
            sign = +1.0 if self._episode % 2 == 0 else -1.0
            self._current_mode = "LCB" if sign > 0 else "UCB"
        else:
            sf = self.sign_factor
            sign = +1.0 if sf >= 0 else -1.0
            self._current_mode = "LCB" if sign > 0 else "UCB"

        beta = beta_mag * sign
        best, best_score = 0, float("inf")
        for i, path in enumerate(paths):
            score = self._score_path(path, beta)
            if score < best_score:
                best_score = score
                best = i
        return best

    def observe(self, path_idx, delay, paths=None, **kw):
        if self._episode < self.calibrate_eps:
            if self._current_mode == "LCB":
                self._lcb_delays.append(delay)
            elif self._current_mode == "UCB":
                self._ucb_delays.append(delay)

        self._episode += 1

        if paths:
            path = paths[path_idx]
            n_links = len(path) - 1
            if n_links > 0:
                per_link = delay / n_links
                for k in range(n_links):
                    self._get_belief(path[k], path[k + 1]).update(per_link)
