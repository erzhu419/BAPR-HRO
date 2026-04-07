"""Bayesian Online Change Detection for transit delay streams.

Adapted from BAPR's BeliefTracker (bapr_components.py). The core BOCD
algorithm is identical; what changes is the surprise signal source:
- BAPR: reward z-score + Q-std spike (RL environment)
- BAPR-HRO: |actual_delay - predicted_delay| (GTFS-RT observations)

Reference: Adams & MacKay (2007), "Bayesian Online Changepoint Detection"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


class BeliefTracker:
    """BOCD run-length posterior tracker.

    Maintains posterior distribution rho(h) over run-length h, where
    h=0 means "changepoint just happened" and large h means "stable regime."

    Directly ported from BAPR with identical math:
    - Likelihood: L(h, xi) = exp(-xi^2 / (2 * sigma_h^2))
    - sigma_h^2 = base_var + var_growth * h
    - Hazard shift: rho_new(0) = sum(rho) * hazard, rho_new(h+1) = rho(h) * (1-hazard)
    """

    def __init__(
        self,
        max_run_length: int = 20,
        hazard_rate: float = 0.05,
        base_variance: float = 0.1,
        variance_growth: float = 0.05,
    ):
        self.max_H = max_run_length
        self.hazard = hazard_rate
        self.base_var = base_variance
        self.var_growth = variance_growth
        self.belief = np.ones(max_run_length) / max_run_length

    def reset(self):
        self.belief = np.ones(self.max_H) / self.max_H

    def update(self, surprise: float):
        """BOCD update step. Identical to BAPR's BeliefTracker.update()."""
        # 1. Compute likelihood
        variances = self.base_var + self.var_growth * np.arange(self.max_H)
        L = np.exp(-surprise ** 2 / (2 * variances))

        # 2. Reweight and normalize
        unnorm = self.belief * L
        Z = unnorm.sum()
        if Z > 1e-10:
            self.belief = unnorm / Z
        else:
            self.belief = np.ones(self.max_H) / self.max_H

        # 3. Hazard shift (BOCD changepoint prior)
        growth_prob = self.belief * (1 - self.hazard)
        changepoint_prob = self.belief.sum() * self.hazard
        new_belief = np.zeros(self.max_H)
        new_belief[0] = changepoint_prob
        new_belief[1:] = growth_prob[:-1]
        total = new_belief.sum()
        self.belief = new_belief / total if total > 1e-10 else np.ones(self.max_H) / self.max_H

    @property
    def effective_window(self) -> float:
        """Expected run-length = sum(h * rho(h)). Low = recent changepoint."""
        return float(np.sum(np.arange(self.max_H) * self.belief))

    @property
    def entropy(self) -> float:
        """Shannon entropy of belief. High = uncertain about regime stability."""
        p = self.belief[self.belief > 1e-10]
        return float(-np.sum(p * np.log(p)))

    @property
    def changepoint_probability(self) -> float:
        """P(changepoint in last few steps) = sum of rho(h) for small h."""
        return float(self.belief[:3].sum())

    @property
    def confidence(self) -> float:
        """Confidence that regime is stable. 0=just changed, 1=very stable."""
        max_entropy = np.log(self.max_H)
        return float(1.0 - self.entropy / max_entropy) if max_entropy > 0 else 0.0
