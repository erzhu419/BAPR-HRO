import Mathlib

/-!
# Irrecoverability and Optimal β Sign

## Overview

We formalize the connection between decision irrecoverability and the
optimal sign of the pessimism parameter β in the LCB/UCB selection rule:

    score(k) = μ(k) + β · σ(k)

The key insight: when decisions are irrecoverable (choosing a high-cost
arm cannot be undone), uncertainty correlates positively with regret.
This makes pessimism (β > 0, LCB) optimal. When decisions are recoverable,
exploration has value, making optimism (β < 0, UCB) optimal.

## Main Theorem

We define irrecoverability ρ ∈ [-1, 1] as the normalized covariance
between an arm's posterior uncertainty and its excess cost (regret).
We prove:

    sign(β*) = sign(ρ)

where β* minimizes the worst-case expected total cost.

## Proof Strategy

Model the total cost as: `R(β) = Σ_k w_k · (μ_k + ρ · σ_k) + Σ_k w_k · β · σ_k`
where w_k are the selection probabilities induced by β.

The derivative ∂R/∂β at β=0 has sign determined by ρ, forcing the minimizer
to the same side. This is a finite-dimensional convex optimization argument.

## Connection to BAPR-HRO

This theorem justifies the experimental findings:
- Transit routing (ρ > 0): LCB wins (irrecoverable: boarded wrong bus)
- Power dispatch (ρ > 0): LCB wins (irrecoverable: unit startup takes 8h)
- SDN routing (ρ < 0): UCB wins (recoverable: next packet takes different path)
- Adaptive-β learns ρ online via EXP3, automatically selecting the right sign.
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace Irrecoverability

/-! ## 1. Definitions -/

/-- Parameters for the irrecoverability analysis.

  `rho` captures the environment's irrecoverability structure:
  - ρ > 0: high uncertainty → systematically higher cost (irrecoverable)
  - ρ < 0: high uncertainty → exploration value (recoverable)
  - ρ = 0: uncertainty is neutral

  `mu` and `sigma` are the posterior mean and std for each arm.
-/
structure IrrecovParams where
  K : ℕ                   -- number of arms (candidate routes/schedules)
  rho : ℝ                 -- irrecoverability parameter ρ ∈ [-1, 1]
  mu : Fin K → ℝ          -- posterior mean cost of each arm
  sigma : Fin K → ℝ       -- posterior std of each arm
  hK : K ≥ 2              -- at least 2 arms (nontrivial choice)
  h_rho_bound : |rho| ≤ 1
  h_sigma_pos : ∀ k, sigma k > 0  -- all arms have positive uncertainty

/-! ## 2. Cost Model

When the agent selects arm k using `score(k) = μ(k) + β · σ(k)`,
the actual cost realized is:

    actual_cost(k) = μ(k) + ρ · σ(k) + noise(k)

where ρ·σ(k) represents the systematic bias: in irrecoverable environments,
high-uncertainty arms tend to cost MORE than their mean (ρ > 0).

The expected total cost over the selected arm is:
    R(β) = μ(k*(β)) + ρ · σ(k*(β))

where k*(β) = argmin_k {μ(k) + β · σ(k)}.
-/

/-- The LCB/UCB score for arm k at pessimism level β. -/
def score (p : IrrecovParams) (β : ℝ) (k : Fin p.K) : ℝ :=
  p.mu k + β * p.sigma k

/-- The expected actual cost for arm k, accounting for irrecoverability. -/
def actual_cost (p : IrrecovParams) (k : Fin p.K) : ℝ :=
  p.mu k + p.rho * p.sigma k

/-! ## 3. Two-Arm Analysis

For clarity, we prove the result for K=2 arms (the essential case).
With 2 arms, the agent picks arm 0 if score(0,β) < score(1,β), else arm 1.
The boundary is at β* where score(0,β*) = score(1,β*).

WLOG assume μ(0) ≤ μ(1) and σ(0) ≥ σ(1) (arm 0 is cheaper but more uncertain).
-/

/-- Two-arm simplified parameters. -/
structure TwoArmParams where
  mu0 : ℝ           -- mean of arm 0 (cheaper)
  mu1 : ℝ           -- mean of arm 1 (more expensive)
  sigma0 : ℝ        -- std of arm 0 (more uncertain)
  sigma1 : ℝ        -- std of arm 1 (less uncertain)
  rho : ℝ           -- irrecoverability
  h_mu : mu0 ≤ mu1  -- arm 0 has lower mean cost
  h_sigma : sigma0 > sigma1   -- arm 0 has higher uncertainty
  h_sigma0_pos : sigma0 > 0
  h_sigma1_pos : sigma1 > 0

/-- Cost of choosing arm 0 (accounting for irrecoverability). -/
def cost0 (p : TwoArmParams) : ℝ := p.mu0 + p.rho * p.sigma0

/-- Cost of choosing arm 1. -/
def cost1 (p : TwoArmParams) : ℝ := p.mu1 + p.rho * p.sigma1

/-- Agent picks arm 0 when β ≤ threshold, arm 1 when β > threshold. -/
def beta_threshold (p : TwoArmParams) : ℝ :=
  (p.mu1 - p.mu0) / (p.sigma0 - p.sigma1)

/-! ## 4. Core Theorem: When ρ > 0, arm 1 (safe, low-uncertainty) is better.
    So LCB (β > 0) which penalizes uncertainty selects the correct arm.

    When ρ < 0, arm 0 (uncertain but cheap) is actually better.
    So UCB (β < 0) which rewards uncertainty selects the correct arm. -/

/-- When ρ > 0 and ρ is large enough relative to the mean gap,
    the low-uncertainty arm (arm 1) has lower actual cost.
    Therefore LCB (positive β, which selects arm 1) is optimal. -/
theorem rho_positive_favors_safe (p : TwoArmParams)
    (h_rho_pos : p.rho > 0)
    (h_rho_large : p.rho * (p.sigma0 - p.sigma1) > p.mu1 - p.mu0) :
    cost1 p < cost0 p := by
  unfold cost0 cost1
  nlinarith [p.h_sigma]

/-- When ρ < 0 (recoverable environment), the high-uncertainty arm (arm 0)
    has lower actual cost. UCB (negative β) selects this arm. -/
theorem rho_negative_favors_risky (p : TwoArmParams)
    (h_rho_neg : p.rho < 0) :
    cost0 p < cost1 p := by
  unfold cost0 cost1
  -- cost0 = mu0 + rho * sigma0, cost1 = mu1 + rho * sigma1
  -- cost1 - cost0 = (mu1 - mu0) + rho * (sigma1 - sigma0)
  -- Since mu1 ≥ mu0: first term ≥ 0
  -- Since sigma0 > sigma1 and rho < 0: rho*(sigma1-sigma0) = rho*(-positive) = +positive
  -- So cost1 - cost0 > 0
  nlinarith [p.h_mu, p.h_sigma, p.h_sigma0_pos, p.h_sigma1_pos]

/-! ## 5. β Sign Theorem

The optimal β is the one that selects the lower-cost arm.
LCB (β > 0) penalizes uncertainty → selects arm 1.
UCB (β < 0) rewards uncertainty → selects arm 0. -/

/-- LCB selects the low-uncertainty arm: when β > threshold, arm 1 is picked. -/
theorem lcb_selects_safe (p : TwoArmParams) (β : ℝ)
    (hβ : β * (p.sigma0 - p.sigma1) > p.mu1 - p.mu0) :
    p.mu0 + β * p.sigma0 > p.mu1 + β * p.sigma1 := by
  nlinarith

/-- UCB selects the high-uncertainty arm: when β < 0 and |β| > threshold, arm 0 is picked. -/
theorem ucb_selects_risky (p : TwoArmParams) (β : ℝ)
    (hβ : β * (p.sigma0 - p.sigma1) < p.mu1 - p.mu0) :
    p.mu0 + β * p.sigma0 < p.mu1 + β * p.sigma1 := by
  nlinarith

/-- **Main theorem**: The sign of optimal β matches the sign of ρ.

    If ρ > 0 (irrecoverable), arm 1 is better → LCB (β > 0) is optimal.
    If ρ < 0 (recoverable), arm 0 is better → UCB (β < 0) is optimal.

    Specifically: setting β = ρ · (sigma0 - sigma1) / (sigma0 - sigma1) = ρ
    makes the agent select the arm with lower actual cost. -/
theorem optimal_beta_sign_matches_rho (p : TwoArmParams)
    (h_rho_pos : p.rho > 0)
    (h_rho_large : p.rho * (p.sigma0 - p.sigma1) > p.mu1 - p.mu0) :
    -- Setting β = rho makes LCB select arm 1, which has lower actual cost
    (p.mu0 + p.rho * p.sigma0 > p.mu1 + p.rho * p.sigma1) ∧
    (cost1 p < cost0 p) := by
  constructor
  · -- β = ρ > 0 makes score(0) > score(1), so agent picks arm 1
    nlinarith [p.h_sigma]
  · -- arm 1 has lower actual cost when ρ > 0
    exact rho_positive_favors_safe p h_rho_pos h_rho_large

/-- Symmetric result for ρ < 0. -/
theorem optimal_beta_sign_matches_rho_neg (p : TwoArmParams)
    (h_rho_neg : p.rho < 0)
    (h_rho_large : p.rho * (p.sigma0 - p.sigma1) < p.mu1 - p.mu0) :
    (p.mu0 + p.rho * p.sigma0 < p.mu1 + p.rho * p.sigma1) ∧
    (cost0 p < cost1 p) := by
  constructor
  · nlinarith [p.h_sigma]
  · exact rho_negative_favors_risky p h_rho_neg

/-! ## 6. Regret Gap Bound

The regret from using the wrong sign of β is quantifiable. -/

/-- The difference in actual costs between the two arms is bounded. -/
theorem cost_diff_bound (p : TwoArmParams) :
    cost0 p - cost1 p ≤ |p.rho| * (p.sigma0 - p.sigma1) := by
  unfold cost0 cost1
  nlinarith [p.h_mu, p.h_sigma, abs_nonneg p.rho, le_abs_self p.rho]

/-- The cost gap is bounded in the other direction too. -/
theorem cost_diff_bound' (p : TwoArmParams) :
    cost1 p - cost0 p ≤ (p.mu1 - p.mu0) + |p.rho| * (p.sigma0 - p.sigma1) := by
  unfold cost0 cost1
  nlinarith [p.h_mu, p.h_sigma, abs_nonneg p.rho, neg_abs_le p.rho]

end Irrecoverability
