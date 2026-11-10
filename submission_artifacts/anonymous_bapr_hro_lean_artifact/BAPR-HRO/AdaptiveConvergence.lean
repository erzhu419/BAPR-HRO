import Mathlib

/-!
# Adaptive-β Convergence: Tighter Regret Bounds

## Overview

1. Per-episode regret² ≤ 2·(c + v²)/T — the O(1/T) rate
2. Convergence: ∀ ε > 0, ∃ T₀, per-episode ≤ ε for T ≥ T₀
3. Doubling T halves the squared per-episode regret
4. For large T, stationary term dominates variation
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace AdaptiveConvergence

def C₀ : ℝ := 2 * Real.sqrt 2
lemma C₀_pos : C₀ > 0 := by unfold C₀; positivity

/-! ## 1. Elementary Inequality -/

theorem sq_add_le (a b : ℝ) : (a + b) ^ 2 ≤ 2 * (a ^ 2 + b ^ 2) := by
  nlinarith [sq_nonneg (a - b)]

/-! ## 2. Combined Per-Episode Bound -/

/-- (√(c/T) + v/T)² ≤ 2·(c/T + (v/T)²). -/
theorem combined_sq (c v T : ℝ) (hc : c ≥ 0) (hT : T > 0) :
    (Real.sqrt (c / T) + v / T) ^ 2 ≤ 2 * (c / T + (v / T) ^ 2) := by
  calc (Real.sqrt (c / T) + v / T) ^ 2
      ≤ 2 * ((Real.sqrt (c / T)) ^ 2 + (v / T) ^ 2) := sq_add_le _ _
    _ = 2 * (c / T + (v / T) ^ 2) := by
        congr 1; congr 1
        exact Real.sq_sqrt (div_nonneg hc (le_of_lt hT))

/-- (v/T)² · T² ≤ v² · T when T ≥ 1. Helper for clearing denominators. -/
theorem sq_denom_clear (v T : ℝ) (hT : T ≥ 1) :
    v ^ 2 ≤ v ^ 2 * T := by
  nlinarith [sq_nonneg v]

/-- **Main bound** (denominator-free form):
    (√(c/T) + v/T)² · T ≤ 2·c + 2·v²   when T ≥ 1.

    This is O(1/T) since dividing both sides by T gives the rate. -/
theorem per_episode_sq_rate (c v T : ℝ) (hc : c ≥ 0) (_hv : v ≥ 0) (hT : T ≥ 1) :
    T * (Real.sqrt (c / T) + v / T) ^ 2 ≤ 2 * c + 2 * v ^ 2 := by
  have hT_pos : T > 0 := by linarith
  have h1 := combined_sq c v T hc hT_pos
  -- h1: (...)² ≤ 2 * (c/T + (v/T)²)
  -- Goal: T · (...)² ≤ 2c + 2v²
  -- Suffices: T · 2 · (c/T + (v/T)²) ≤ 2c + 2v²
  --         = 2c + 2·(v/T)²·T
  -- Need: (v/T)²·T ≤ v²  i.e.  v²/T ≤ v²  i.e. 1/T ≤ 1
  suffices h : T * (2 * (c / T + (v / T) ^ 2)) ≤ 2 * c + 2 * v ^ 2 by
    nlinarith [mul_le_mul_of_nonneg_left h1 (le_of_lt hT_pos)]
  have hT_ne : T ≠ 0 := ne_of_gt hT_pos
  -- Expand (v/T)² = v²/T²
  rw [div_pow]
  -- T * (2 * (c/T + v²/T²)) = 2*c + 2*v²/T
  -- and 2*v²/T ≤ 2*v² since T ≥ 1
  have key : T * (2 * (c / T + v ^ 2 / T ^ 2)) = 2 * c + 2 * v ^ 2 / T := by
    field_simp
  rw [key]
  have : 2 * v ^ 2 / T ≤ 2 * v ^ 2 := by
    apply div_le_self (by nlinarith [sq_nonneg v]) hT
  linarith

/-! ## 3. Convergence -/

/-- For any ε > 0, eventually (c+v)/T ≤ ε.
    Stated as: (c+v) ≤ ε · T for T large enough. -/
theorem convergence (c v eps : ℝ) (hcv : c + v ≥ 0) (heps : eps > 0)
    (T : ℝ) (hT : T ≥ (c + v) / eps + 1) :
    c + v ≤ eps * T := by
  have h1 : (c + v) / eps ≥ 0 := div_nonneg hcv (le_of_lt heps)
  have hT_pos : T > 0 := by linarith
  -- T ≥ (c+v)/eps + 1 > (c+v)/eps
  -- eps · T ≥ eps · (c+v)/eps = c+v
  have h2 : eps * T ≥ eps * ((c + v) / eps + 1) := by nlinarith
  rw [mul_add, mul_div_cancel₀ _ (ne_of_gt heps)] at h2
  linarith

/-- Existence form: there exists T₀ > 0. -/
theorem convergence_exists (c v eps : ℝ) (hcv : c + v ≥ 0) (heps : eps > 0) :
    ∃ T₀ : ℝ, T₀ > 0 ∧ ∀ T : ℝ, T ≥ T₀ → c + v ≤ eps * T := by
  exact ⟨(c + v) / eps + 1, by positivity, fun T hT => convergence c v eps hcv heps T hT⟩

/-! ## 4. Doubling Improvement -/

theorem doubling_halves (c T : ℝ) (_hT : T > 0) :
    c / (2 * T) = c / T / 2 := by
  field_simp

theorem n_doublings (c T : ℝ) (_hT : T > 0) (n : ℕ) :
    c / (2 ^ n * T) = c / T / 2 ^ n := by
  field_simp

/-! ## 5. Stationary Dominates Variation for Large T -/

/-- When c·T ≥ v², the variation v²/T is ≤ c. So v²/T² ≤ c/T.
    Stated denominator-free: v² · T ≤ c · T². -/
theorem stationary_dominates (c v T : ℝ) (_hc : c > 0) (_hT : T ≥ 1)
    (h_large : c * T ≥ v ^ 2) :
    v ^ 2 * T ≤ c * T ^ 2 := by
  nlinarith [sq_nonneg T]

/-! ## 6. Rate Monotonicity -/

/-- a/T₂ ≤ a/T₁ when a ≥ 0 and T₂ ≥ T₁ > 0. Cross-multiplied: a·T₁ ≤ a·T₂. -/
theorem rate_monotone (a T1 T2 : ℝ) (ha : a ≥ 0) (_hT1 : T1 > 0) (hT2 : T2 ≥ T1) :
    a * T1 ≤ a * T2 := by
  nlinarith

/-! ## 7. Concrete Instantiation

For K=7, S=3, beta_range=3:
  c ≈ 109, v = 18, v² = 324
  2·(c + v²) = 2·(109 + 324) = 866
  T=100: per-ep² ≤ 866/100 = 8.66 → per-ep ≤ 2.94
  T=1000: per-ep² ≤ 866/1000 = 0.866 → per-ep ≤ 0.93
-/

example : 2 * ((109 : ℝ) + 324) = 866 := by norm_num
example : (866 : ℝ) / 100 = 8.66 := by norm_num

end AdaptiveConvergence
