"""BAPR-HRO LCB routers for Unit Commitment.

Simplified architecture: learn beliefs over TOTAL COST of each schedule
directly from execution outcomes, exactly like transit routing learns
total travel time per bus connection.

Schedule k → execute on day d → observe total_cost → update belief(k)
Next day: pick schedule with lowest LCB of total_cost.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from uc_env import ScheduleResult


@dataclass
class CostBeliefNG:
    """Normal-Gamma posterior over total cost of a schedule."""
    mu: float = 0.0      # will be set from first observation
    kappa: float = 0.1    # low confidence initially
    alpha: float = 2.0
    beta: float = 1e8     # high variance prior for costs
    n_obs: int = 0

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def std(self) -> float:
        if self.alpha <= 1 or self.n_obs == 0:
            return 1e6  # very uncertain if no observations
        return float(np.sqrt(self.beta / (self.kappa * (self.alpha - 1))))

    def update(self, cost: float):
        self.n_obs += 1
        if self.n_obs == 1:
            # First observation: set mean to observed cost
            self.mu = cost
            self.kappa = 1.0
            self.beta = cost * 0.1  # 10% of first cost as initial variance
            return
        kn = self.kappa + 1
        mn = (self.kappa * self.mu + cost) / kn
        an = self.alpha + 0.5
        bn = self.beta + 0.5 * self.kappa * (cost - self.mu) ** 2 / kn
        self.mu, self.kappa, self.alpha, self.beta = mn, kn, an, bn

    def sample(self, rng: np.random.Generator) -> float:
        if self.n_obs == 0:
            return 1e6  # unknown schedule
        tau = rng.gamma(self.alpha, 1.0 / max(self.beta, 1e-8))
        sigma = 1.0 / np.sqrt(max(tau * self.kappa, 1e-8))
        return rng.normal(self.mu, sigma)


@dataclass
class CostBeliefEnsemble:
    """Ensemble belief over schedule cost — V2."""
    n_estimators: int = 5
    _means: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _vars: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_obs: int = 0

    def __post_init__(self):
        if len(self._means) == 0:
            self._means = np.zeros(self.n_estimators)
            self._vars = np.full(self.n_estimators, 1e10)
            self._counts = np.zeros(self.n_estimators)

    @property
    def mean(self) -> float:
        return float(self._means.mean()) if self.n_obs > 0 else 1e6

    @property
    def std(self) -> float:
        if self.n_obs < 2:
            return 1e6
        return float(self._means.std())

    @property
    def ood_score(self) -> float:
        if self.n_obs < 3:
            return 0.0
        avg_int = float(np.sqrt(self._vars).mean())
        return min(self.std / max(avg_int, 1e-6), 3.0)

    def update(self, cost: float, rng: np.random.Generator):
        self.n_obs += 1
        weights = rng.poisson(1, self.n_estimators)
        for k in range(self.n_estimators):
            for _ in range(weights[k]):
                self._counts[k] += 1
                n = self._counts[k]
                d = cost - self._means[k]
                self._means[k] += d / n
                self._vars[k] += (d * (cost - self._means[k]) - self._vars[k]) / n


# ── Routers ──────────────────────────────────────────────────────────────────

class StaticRouter:
    """Always pick the same schedule (the one that was cheapest first time)."""

    def __init__(self, n_schedules: int):
        self.n = n_schedules
        self._best = None
        self._best_cost = float("inf")

    def select_schedule(self) -> int:
        if self._best is None:
            return 0  # start with schedule 0
        return self._best

    def observe(self, schedule_idx: int, result: ScheduleResult):
        if result.total_cost < self._best_cost:
            self._best_cost = result.total_cost
            self._best = schedule_idx


class LCBRouter:
    """BAPR-HRO V1: Normal-Gamma + LCB over schedule costs."""

    def __init__(self, n_schedules: int, beta: float = 1.0, explore_top: int = 4):
        self.n = n_schedules
        self.beta = beta
        self.beliefs = [CostBeliefNG() for _ in range(n_schedules)]
        self._episode = 0
        self._explore_order = list(range(min(explore_top, n_schedules)))

    def select_schedule(self) -> int:
        if self._episode < len(self._explore_order):
            return self._explore_order[self._episode]

        best_i, best_score = 0, float("inf")
        for i in range(self.n):
            b = self.beliefs[i]
            # LCB: pessimistic = mean + beta * std (higher cost = worse)
            # We MINIMIZE cost, so we want the schedule with lowest UPPER bound
            # But LCB in BAPR-HRO is pessimistic = want to AVOID high cost
            # So: pick schedule with lowest (mean + beta * std)
            score = b.mean + self.beta * b.std
            if score < best_score:
                best_score = score
                best_i = i
        return best_i

    def observe(self, schedule_idx: int, result: ScheduleResult):
        self._episode += 1
        self.beliefs[schedule_idx].update(result.total_cost)


class LCBRouterV2:
    """BAPR-HRO V2: Ensemble + dynamic beta."""

    def __init__(self, n_schedules: int, beta_base: float = 0.8,
                 beta_ood: float = 0.8, n_estimators: int = 5, seed: int = 0,
                 explore_top: int = 4):
        self.n = n_schedules
        self.beta_base = beta_base
        self.beta_ood = beta_ood
        self.rng = np.random.default_rng(seed)
        self.beliefs = [CostBeliefEnsemble(n_estimators=n_estimators)
                        for _ in range(n_schedules)]
        self._episode = 0
        self._explore_order = list(range(min(explore_top, n_schedules)))

    def select_schedule(self) -> int:
        if self._episode < len(self._explore_order):
            return self._explore_order[self._episode]

        max_ood = max(b.ood_score for b in self.beliefs)
        beta = self.beta_base + self.beta_ood * max_ood

        best_i, best_score = 0, float("inf")
        for i in range(self.n):
            b = self.beliefs[i]
            score = b.mean + beta * b.std
            if score < best_score:
                best_score = score
                best_i = i
        return best_i

    def observe(self, schedule_idx: int, result: ScheduleResult):
        self._episode += 1
        self.beliefs[schedule_idx].update(result.total_cost, self.rng)


class TSRouter:
    """Thompson Sampling — samples cost from posterior, picks lowest."""

    def __init__(self, n_schedules: int, seed: int = 0, explore_top: int = 4):
        self.n = n_schedules
        self.rng = np.random.default_rng(seed)
        self.beliefs = [CostBeliefNG() for _ in range(n_schedules)]
        self._episode = 0
        self._explore_order = list(range(min(explore_top, n_schedules)))

    def select_schedule(self) -> int:
        if self._episode < len(self._explore_order):
            return self._explore_order[self._episode]

        best_i, best_score = 0, float("inf")
        for i in range(self.n):
            sampled = self.beliefs[i].sample(self.rng)
            if sampled < best_score:
                best_score = sampled
                best_i = i
        return best_i

    def observe(self, schedule_idx: int, result: ScheduleResult):
        self._episode += 1
        self.beliefs[schedule_idx].update(result.total_cost)
