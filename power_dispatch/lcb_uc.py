"""BAPR-HRO LCB routers for Unit Commitment.

Optimizations applied:
  1. Warm-start: initialize beliefs from a single static execution
     instead of round-robin exploration (saves K episodes of waste)
  2. Adaptive β: β = β₀ / √(n_obs) — high exploration early, low later
  3. V2 ensemble prior fix: moderate initial variance, avoids OOD inflation
  4. Hybrid UCB→LCB: use UCB in early episodes, switch to LCB after convergence
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from uc_env import ScheduleResult


@dataclass
class CostBeliefNG:
    """Normal-Gamma posterior over total cost of a schedule."""
    mu: float = 0.0
    kappa: float = 0.1
    alpha: float = 2.0
    beta: float = 1e8
    n_obs: int = 0

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def std(self) -> float:
        if self.n_obs == 0:
            return 1e6
        if self.alpha <= 1:
            return 5e4
        return float(np.sqrt(self.beta / (self.kappa * (self.alpha - 1))))

    def update(self, cost: float):
        self.n_obs += 1
        if self.n_obs == 1:
            self.mu = cost
            self.kappa = 1.0
            self.beta = cost * 0.1
            return
        kn = self.kappa + 1
        mn = (self.kappa * self.mu + cost) / kn
        an = self.alpha + 0.5
        bn = self.beta + 0.5 * self.kappa * (cost - self.mu) ** 2 / kn
        self.mu, self.kappa, self.alpha, self.beta = mn, kn, an, bn

    def sample(self, rng: np.random.Generator) -> float:
        if self.n_obs == 0:
            return 1e6
        tau = rng.gamma(self.alpha, 1.0 / max(self.beta, 1e-8))
        sigma = 1.0 / np.sqrt(max(tau * self.kappa, 1e-8))
        return rng.normal(self.mu, sigma)


@dataclass
class CostBeliefEnsemble:
    """Ensemble belief over schedule cost — V2 (fixed initialization)."""
    n_estimators: int = 5
    _means: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _vars: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_obs: int = 0

    def __post_init__(self):
        if len(self._means) == 0:
            self._means = np.zeros(self.n_estimators)
            # FIX: moderate prior variance, not 1e10
            self._vars = np.full(self.n_estimators, 1e4)
            self._counts = np.ones(self.n_estimators)  # pseudo-count=1

    def warm_start(self, cost: float):
        """Initialize all estimators from a warm-start cost."""
        self._means[:] = cost
        self._vars[:] = (cost * 0.05) ** 2  # 5% variance
        self._counts[:] = 2  # moderate confidence
        self.n_obs = 1

    @property
    def mean(self) -> float:
        return float(self._means.mean()) if self.n_obs > 0 else 1e6

    @property
    def std(self) -> float:
        if self.n_obs < 2:
            return 1e4 if self.n_obs == 0 else float(self._means.std()) + 1e3
        return float(self._means.std())

    @property
    def ood_score(self) -> float:
        if self.n_obs < 3:
            return 0.0
        avg_int = float(np.sqrt(np.maximum(self._vars, 0)).mean())
        return min(self.std / max(avg_int, 1e-3), 3.0)

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
    """Always pick the same schedule (cheapest on first observation)."""
    def __init__(self, n_schedules: int):
        self.n = n_schedules
        self._best = 0
        self._best_cost = float("inf")

    def select_schedule(self) -> int:
        return self._best

    def observe(self, schedule_idx: int, result: ScheduleResult):
        if result.total_cost < self._best_cost:
            self._best_cost = result.total_cost
            self._best = schedule_idx


class LCBRouter:
    """BAPR-HRO V1: Normal-Gamma + adaptive LCB.

    Warm-start: execute each schedule once via static sim to set priors.
    Adaptive β: β₀ / √(max(n_obs_min, 1)) decays as beliefs sharpen.
    """
    def __init__(self, n_schedules: int, beta0: float = 2.0,
                 warm_costs: list[float] | None = None):
        self.n = n_schedules
        self.beta0 = beta0
        self.beliefs = [CostBeliefNG() for _ in range(n_schedules)]

        # Warm-start: seed beliefs from static cost estimates
        if warm_costs:
            for i, c in enumerate(warm_costs):
                self.beliefs[i].update(c)

    def _adaptive_beta(self) -> float:
        min_obs = min(b.n_obs for b in self.beliefs)
        return self.beta0 / max(min_obs, 1) ** 0.5

    def select_schedule(self) -> int:
        # If any schedule has 0 observations, explore it
        for i in range(self.n):
            if self.beliefs[i].n_obs == 0:
                return i

        beta = self._adaptive_beta()
        best_i, best_score = 0, float("inf")
        for i in range(self.n):
            b = self.beliefs[i]
            score = b.mean + beta * b.std
            if score < best_score:
                best_score = score
                best_i = i
        return best_i

    def observe(self, schedule_idx: int, result: ScheduleResult):
        self.beliefs[schedule_idx].update(result.total_cost)


class LCBRouterV2:
    """BAPR-HRO V2: Ensemble + dynamic beta + warm-start + adaptive decay."""

    def __init__(self, n_schedules: int, beta_base: float = 1.5,
                 beta_ood: float = 0.5, n_estimators: int = 5, seed: int = 0,
                 warm_costs: list[float] | None = None):
        self.n = n_schedules
        self.beta_base = beta_base
        self.beta_ood = beta_ood
        self.rng = np.random.default_rng(seed)
        self.beliefs = [CostBeliefEnsemble(n_estimators=n_estimators)
                        for _ in range(n_schedules)]
        self._total_obs = 0

        if warm_costs:
            for i, c in enumerate(warm_costs):
                self.beliefs[i].warm_start(c)

    def _adaptive_beta(self) -> float:
        min_obs = min(b.n_obs for b in self.beliefs)
        max_ood = max(b.ood_score for b in self.beliefs)
        base = self.beta_base / max(min_obs, 1) ** 0.5
        return base + self.beta_ood * max_ood

    def select_schedule(self) -> int:
        for i in range(self.n):
            if self.beliefs[i].n_obs == 0:
                return i

        beta = self._adaptive_beta()
        best_i, best_score = 0, float("inf")
        for i in range(self.n):
            b = self.beliefs[i]
            score = b.mean + beta * b.std
            if score < best_score:
                best_score = score
                best_i = i
        return best_i

    def observe(self, schedule_idx: int, result: ScheduleResult):
        self._total_obs += 1
        self.beliefs[schedule_idx].update(result.total_cost, self.rng)


class TSRouter:
    """Thompson Sampling with warm-start."""
    def __init__(self, n_schedules: int, seed: int = 0,
                 warm_costs: list[float] | None = None):
        self.n = n_schedules
        self.rng = np.random.default_rng(seed)
        self.beliefs = [CostBeliefNG() for _ in range(n_schedules)]

        if warm_costs:
            for i, c in enumerate(warm_costs):
                self.beliefs[i].update(c)

    def select_schedule(self) -> int:
        for i in range(self.n):
            if self.beliefs[i].n_obs == 0:
                return i

        best_i, best_score = 0, float("inf")
        for i in range(self.n):
            sampled = self.beliefs[i].sample(self.rng)
            if sampled < best_score:
                best_score = sampled
                best_i = i
        return best_i

    def observe(self, schedule_idx: int, result: ScheduleResult):
        self.beliefs[schedule_idx].update(result.total_cost)


class HybridRouter:
    """Hybrid UCB→LCB: explore with UCB, then exploit with LCB.

    First phase (ep < switch_ep): use UCB (optimistic) for fast exploration.
    Second phase: switch to LCB (pessimistic) for safe exploitation.
    """
    def __init__(self, n_schedules: int, beta0: float = 2.0,
                 switch_ep: int = 10, warm_costs: list[float] | None = None):
        self.n = n_schedules
        self.beta0 = beta0
        self.switch_ep = switch_ep
        self.beliefs = [CostBeliefNG() for _ in range(n_schedules)]
        self._episode = 0

        if warm_costs:
            for i, c in enumerate(warm_costs):
                self.beliefs[i].update(c)

    def select_schedule(self) -> int:
        for i in range(self.n):
            if self.beliefs[i].n_obs == 0:
                return i

        beta = self.beta0 / max(min(b.n_obs for b in self.beliefs), 1) ** 0.5

        if self._episode < self.switch_ep:
            # UCB phase: pick schedule with lowest OPTIMISTIC cost
            best_i, best_score = 0, float("inf")
            for i in range(self.n):
                b = self.beliefs[i]
                score = b.mean - beta * b.std  # optimistic
                if score < best_score:
                    best_score = score
                    best_i = i
        else:
            # LCB phase: pick schedule with lowest PESSIMISTIC cost
            best_i, best_score = 0, float("inf")
            for i in range(self.n):
                b = self.beliefs[i]
                score = b.mean + beta * b.std  # pessimistic
                if score < best_score:
                    best_score = score
                    best_i = i
        return best_i

    def observe(self, schedule_idx: int, result: ScheduleResult):
        self._episode += 1
        self.beliefs[schedule_idx].update(result.total_cost)


class FlowLCBRouter:
    """LCB with schedule-commitment for `commit_duration` consecutive days.

    Adapted from sdn_routing FlowLCBRouter ("commit a path for N
    episodes"). For UC this is "commit a schedule for N days" —
    once selected, the same schedule is used until the commit window
    expires.
    """
    def __init__(self, n_schedules: int, beta0: float = 2.0,
                 commit_duration: int = 5,
                 warm_costs: list[float] | None = None):
        self.n = n_schedules
        self.beta0 = beta0
        self.commit_duration = commit_duration
        self.beliefs = [CostBeliefNG() for _ in range(n_schedules)]
        if warm_costs:
            for i, c in enumerate(warm_costs):
                self.beliefs[i].update(c)
        self._committed_idx: int | None = None
        self._commit_until = -1
        self._episode = 0

    def select_schedule(self) -> int:
        # Initial round-robin exploration
        for i in range(self.n):
            if self.beliefs[i].n_obs == 0:
                return i

        # Reuse commitment if window still active
        if self._committed_idx is not None and self._episode < self._commit_until:
            return self._committed_idx

        # Fresh LCB selection + commit
        min_obs = min(b.n_obs for b in self.beliefs)
        beta = self.beta0 / max(min_obs, 1) ** 0.5
        best_i, best_score = 0, float("inf")
        for i in range(self.n):
            b = self.beliefs[i]
            score = b.mean + beta * b.std
            if score < best_score:
                best_score = score
                best_i = i
        self._committed_idx = best_i
        self._commit_until = self._episode + self.commit_duration
        return best_i

    def observe(self, schedule_idx: int, result: ScheduleResult):
        self._episode += 1
        self.beliefs[schedule_idx].update(result.total_cost)


class AdaptiveBetaRouter:
    """BAPR-HRO Adaptive-β: V1-LCB wrapped in an EXP3 meta-bandit over a β grid.

    At the start of each day, β is sampled from the current EXP3
    distribution; after the day, the realized cost updates the β
    weights. Same pattern as the transit
    `AdaptiveBetaBanditRouter` in src/adaptive_bandit_router.py.
    """

    def __init__(self, n_schedules: int,
                 beta_grid: list[float] | None = None, eta: float = 0.1,
                 seed: int = 0,
                 warm_costs: list[float] | None = None):
        self.n = n_schedules
        self.beta_grid = beta_grid or [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        self.n_betas = len(self.beta_grid)
        self.eta = eta
        self.rng = np.random.default_rng(seed)
        self.beliefs = [CostBeliefNG() for _ in range(n_schedules)]
        if warm_costs:
            for i, c in enumerate(warm_costs):
                self.beliefs[i].update(c)
        self._log_weights = np.zeros(self.n_betas)
        self._episode = 0
        self._current_beta_idx = self.n_betas // 2
        self._episode_costs: dict[int, list[float]] = {i: [] for i in range(self.n_betas)}

    @property
    def beta_probs(self) -> np.ndarray:
        w = self._log_weights - self._log_weights.max()
        p = np.exp(w)
        return p / p.sum()

    def select_schedule(self) -> int:
        # Explore unobserved schedules first (mirrors LCBRouter)
        for i in range(self.n):
            if self.beliefs[i].n_obs == 0:
                return i

        # Sample β at the start of each day
        probs = self.beta_probs
        self._current_beta_idx = int(self.rng.choice(self.n_betas, p=probs))
        beta = self.beta_grid[self._current_beta_idx]

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
        self.beliefs[schedule_idx].update(result.total_cost)

        if self._episode > self.n:  # past initial exploration
            self._episode_costs[self._current_beta_idx].append(result.total_cost)
            if self._episode >= self.n + self.n_betas:
                probs = self.beta_probs
                all_costs = [np.mean(c) for c in self._episode_costs.values() if c]
                if all_costs:
                    cost_range = max(max(all_costs) - min(all_costs), 1e-6)
                    for j in range(self.n_betas):
                        avg = (np.mean(self._episode_costs[j])
                               if self._episode_costs[j] else result.total_cost)
                        normalized = (avg - min(all_costs)) / cost_range
                        self._log_weights[j] -= self.eta * normalized / max(probs[j], 0.01)
