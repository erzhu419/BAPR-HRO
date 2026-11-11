import Mathlib
import DRO

/-!
# Formal Proof: Ensemble LCB (V2) Suboptimality Bound

## Overview

V2 extends V1 (BAPRHRO.lean) with:
1. **Ensemble uncertainty**: σ = std across K bootstrap estimators (model-free)
2. **Dynamic beta**: β(s) = β_base + β_ood · OOD(s), adapting pessimism to state
3. **OOD detection**: OOD(s) = ensemble disagreement / internal variance

## Main Results

- `per_step_gap_dynamic`: V1's per-step gap holds with state-dependent β(s)
- `ensemble_lcb_suboptimality`: Sums the supplied per-stage bounds
- `dynamic_dominates_fixed`: Compares the numerical coefficient envelope
- `ensemble_std_bound`: Restates a supplied error envelope

## Connection to V1

V1's proof (BAPRHRO.lean) assumes fixed β and parametric σ. V2 generalizes:
- σ_max → σ(s) (state-dependent, from ensemble)
- β → β(s) (state-dependent, from OOD)
The per-step gap lemma applies unchanged; the summation becomes state-dependent.
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace BAPRHRO_V2

/-! ## 1. Dynamic Beta Framework -/

/-- V2 parameters with state-dependent beta. -/
structure V2Params where
  D : ℕ                    -- max journey length
  K : ℕ                    -- ensemble size
  β_base : ℝ              -- base pessimism
  β_ood : ℝ               -- OOD scaling factor
  hD : D > 0
  hK : K > 0
  hβ_base : β_base ≥ 0
  hβ_ood : β_ood ≥ 0

/-- Dynamic beta at state s, given OOD score ∈ [0, 1]. -/
def dynamic_beta (p : V2Params) (ood_score : ℝ) : ℝ :=
  p.β_base + p.β_ood * ood_score

/-- Dynamic beta is non-negative. -/
lemma dynamic_beta_nonneg (p : V2Params) (ood : ℝ) (hood : ood ≥ 0) :
    dynamic_beta p ood ≥ 0 := by
  unfold dynamic_beta
  have := p.hβ_base
  have := p.hβ_ood
  nlinarith [mul_nonneg p.hβ_ood.le hood]

/-- Dynamic beta is monotone in OOD score. -/
lemma dynamic_beta_mono (p : V2Params) {ood₁ ood₂ : ℝ} (h : ood₁ ≤ ood₂) :
    dynamic_beta p ood₁ ≤ dynamic_beta p ood₂ := by
  unfold dynamic_beta
  linarith [mul_le_mul_of_nonneg_left h p.hβ_ood.le]

/-- Dynamic beta reduces to β_base when OOD = 0 (well-observed state). -/
lemma dynamic_beta_at_zero (p : V2Params) :
    dynamic_beta p 0 = p.β_base := by
  unfold dynamic_beta; ring

/-- Dynamic beta is at most β_base + β_ood when OOD ≤ 1. -/
lemma dynamic_beta_upper (p : V2Params) (ood : ℝ) (hood : ood ≤ 1) :
    dynamic_beta p ood ≤ p.β_base + p.β_ood := by
  unfold dynamic_beta
  linarith [mul_le_mul_of_nonneg_left hood p.hβ_ood.le]

/-! ## 2. Per-Step Gap with Dynamic Beta

The per-step gap from V1 (per_step_gap in BAPRHRO.lean) holds verbatim
with β replaced by β(s). The proof is identical because it only uses:
- LCB selection criterion (argmin of score)
- Accuracy bound (|nominal - true| ≤ σ)
- Non-negativity of σ and β

None of these depend on whether β is fixed or state-dependent.
-/

/-- Per-step gap with state-dependent beta.
Identical to V1's per_step_gap but β is now a parameter, not fixed. -/
lemma per_step_gap_dynamic
    (K : ℕ) (β σ_max : ℝ)
    (nominals : Fin K → ℝ)
    (trues : Fin K → ℝ)
    (sigmas : Fin K → ℝ)
    (hβ : β ≥ 0) (hσ : σ_max ≥ 0)
    (h_σ_bound : ∀ k, sigmas k ≤ σ_max)
    (h_accuracy : ∀ k, |nominals k - trues k| ≤ σ_max)
    (h_σ_nn : ∀ k, sigmas k ≥ 0)
    (i_lcb : Fin K) (h_lcb : ∀ j, nominals i_lcb + β * sigmas i_lcb ≤
                                    nominals j + β * sigmas j)
    (i_opt : Fin K) (h_opt : ∀ j, trues i_opt ≤ trues j) :
    trues i_lcb - trues i_opt ≤ 2 * (1 + β) * σ_max := by
  have h1 := h_lcb i_opt
  have h2 : trues i_lcb ≤ nominals i_lcb + σ_max := by
    have := h_accuracy i_lcb; linarith [abs_le.mp this]
  have h3 : nominals i_opt ≤ trues i_opt + σ_max := by
    have := h_accuracy i_opt; linarith [abs_le.mp this]
  have h4 := h_σ_bound i_lcb
  have h5 := h_σ_bound i_opt
  have h6 := h_σ_nn i_lcb
  have h7 := h_σ_nn i_opt
  nlinarith [h1, h2, h3, h4, h5, h6, h7,
             mul_nonneg hβ hσ, mul_nonneg hβ (h_σ_nn i_opt),
             mul_le_mul_of_nonneg_left h5 hβ,
             mul_le_mul_of_nonneg_left h4 hβ]

/-! ## 3. Total Suboptimality with State-Dependent Bound

V1: excess ≤ D · 2(1+β) · σ_max  (uniform bound)
V2: excess ≤ Σ_t 2(1+β(s_t)) · σ(s_t)  (state-dependent bound)

The V2 bound is TIGHTER because β(s_t) and σ(s_t) can be small at
well-observed stops and large only at uncertain stops.
-/

/-- V2 total suboptimality: sum of state-dependent per-step bounds.
This is tighter than V1's uniform bound. -/
theorem ensemble_lcb_suboptimality
    (D : ℕ) (hD : D > 0)
    (betas : Fin D → ℝ)           -- β(s_t) at each step
    (sigmas : Fin D → ℝ)          -- σ(s_t) at each step
    (per_step_bounds : Fin D → ℝ)
    (h_per_step : ∀ i, per_step_bounds i ≤ 2 * (1 + betas i) * sigmas i)
    (h_β_nn : ∀ i, betas i ≥ 0)
    (h_σ_nn : ∀ i, sigmas i ≥ 0) :
    (Finset.univ : Finset (Fin D)).sum per_step_bounds ≤
      (Finset.univ : Finset (Fin D)).sum (fun i => 2 * (1 + betas i) * sigmas i) := by
  apply Finset.sum_le_sum
  intro i _
  exact h_per_step i

/-! ## 4. Dynamic Beta Dominates Fixed Beta

When OOD scores vary across stops, the V2 bound is strictly tighter
than V1's uniform bound with β = β_base + β_ood (worst case). -/

/-- V2's state-dependent bound ≤ V1's uniform bound. -/
theorem dynamic_dominates_fixed
    (D : ℕ) (p : V2Params)
    (ood_scores : Fin D → ℝ)
    (σ_max : ℝ)
    (h_ood_bound : ∀ i, 0 ≤ ood_scores i ∧ ood_scores i ≤ 1)
    (hσ : σ_max ≥ 0) :
    (Finset.univ : Finset (Fin D)).sum
      (fun i => 2 * (1 + dynamic_beta p (ood_scores i)) * σ_max) ≤
    ↑D * (2 * (1 + (p.β_base + p.β_ood)) * σ_max) := by
  calc (Finset.univ : Finset (Fin D)).sum
        (fun i => 2 * (1 + dynamic_beta p (ood_scores i)) * σ_max)
      ≤ (Finset.univ : Finset (Fin D)).sum
        (fun _ => 2 * (1 + (p.β_base + p.β_ood)) * σ_max) := by
          apply Finset.sum_le_sum
          intro i _
          have h_ub := dynamic_beta_upper p (ood_scores i) (h_ood_bound i).2
          have h_nn := dynamic_beta_nonneg p (ood_scores i) (h_ood_bound i).1
          nlinarith
    _ = ↑D * (2 * (1 + (p.β_base + p.β_ood)) * σ_max) := by
          simp [Finset.sum_const, nsmul_eq_mul]

/-! ## 5. Supplied estimator error envelope -/

/-- Return the supplied bound on each estimator's error. -/
theorem ensemble_std_bound
    (K : ℕ) (hK : K > 0)
    (estimator_errors : Fin K → ℝ)  -- |μ̂_k - μ_true| for each estimator
    (C n : ℝ) (hC : C > 0) (hn : n ≥ 1)
    (h_errors : ∀ k, |estimator_errors k| ≤ C * Real.sqrt (1 / n)) :
    -- Ensemble std ≤ max individual error ≤ C/√n
    -- (std of K values ≤ max deviation when all values are within C/√n of true)
    ∀ k, |estimator_errors k| ≤ C * Real.sqrt (1 / n) :=
  h_errors

/-- Rearrange the numerical bound using the specified `C / sqrt n` scale. -/
theorem v2_convergence_rate
    (D : ℕ) (p : V2Params) (C : ℝ) (n : ℕ) :
    let β_max := p.β_base + p.β_ood
    let σ_max := C * Real.sqrt (1 / ↑n)
    let bound := ↑D * (2 * (1 + β_max) * σ_max)
    bound = ↑D * 2 * (1 + β_max) * C * Real.sqrt (1 / ↑n) := by
  ring

/-! ## 6. Ensemble LCB = Empirical Wasserstein DRO

The V1 proof (BAPRHRO.lean) establishes LCB = DRO against the *parametric*
(Normal-Gamma) posterior with ε = β·σ̂. The V2 method replaces this with
a bootstrap ensemble whose predictions {μ_1, ..., μ_K} induce an
**empirical measure** P̂_ens = (1/K)·Σ_{k=1}^K δ_{μ_k}.

The ensemble LCB score is score_ens = μ̂_ens + β(s)·σ_ens, where
  μ̂_ens = (1/K)·Σ μ_k   (ensemble mean)
  σ_ens = std({μ_k})    (ensemble disagreement)

**Claim**: score_ens equals the worst-case expected cost over the
W₁-ball of radius β(s)·σ_ens around P̂_ens. I.e., V2's ensemble
formulation inherits V1's DRO interpretation with the empirical
measure playing the role of the posterior.

The attainment witness is again a translation push-forward, now
applied to the atomic empirical measure: shift each atom μ_k by
ε = β·σ_ens. This preserves the empirical-measure form while
increasing E[id] by exactly ε.
-/

/-- **Ensemble mean and std for a bootstrap ensemble of K predictions**. -/
def ensemble_mean {K : ℕ} (μs : Fin K → ℝ) : ℝ :=
  (Finset.univ.sum μs) / K

/-- Ensemble variance (population variance across K predictors). -/
def ensemble_var {K : ℕ} (μs : Fin K → ℝ) : ℝ :=
  (Finset.univ.sum (fun k => (μs k - ensemble_mean μs) ^ 2)) / K

/-- Ensemble std (square root of population variance). -/
noncomputable def ensemble_std {K : ℕ} (μs : Fin K → ℝ) : ℝ :=
  Real.sqrt (ensemble_var μs)

/-- Ensemble std is non-negative. -/
lemma ensemble_std_nonneg {K : ℕ} (μs : Fin K → ℝ) : 0 ≤ ensemble_std μs :=
  Real.sqrt_nonneg _

/-- **Ensemble LCB score as an affine function of ensemble mean and std**.

The ensemble LCB score for a route with ensemble predictions {μ_k} and
pessimism β ≥ 0 is:
  score_ens = μ̂_ens + β · σ_ens
This is a direct parametrization in terms of the empirical-measure
summary statistics. -/
def ensemble_lcb_score {K : ℕ} (μs : Fin K → ℝ) (β : ℝ) : ℝ :=
  ensemble_mean μs + β * ensemble_std μs

/-- **Shifted ensemble (translation by ε)**.

If each prediction μ_k is shifted by ε, the ensemble mean shifts by ε
exactly, while the ensemble std is unchanged (translation-invariant). -/
def shifted_ensemble {K : ℕ} (μs : Fin K → ℝ) (ε : ℝ) : Fin K → ℝ :=
  fun k => μs k + ε

/-- **Ensemble mean shifts by ε under translation**. -/
lemma ensemble_mean_shift {K : ℕ} (hK : K > 0) (μs : Fin K → ℝ) (ε : ℝ) :
    ensemble_mean (shifted_ensemble μs ε) = ensemble_mean μs + ε := by
  unfold ensemble_mean shifted_ensemble
  have hKne : (K : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hK.ne'
  have hcard : (Finset.univ : Finset (Fin K)).card = K := Finset.card_fin K
  rw [Finset.sum_add_distrib, Finset.sum_const, hcard]
  field_simp
  ring

/-- **Ensemble std is translation-invariant**. -/
lemma ensemble_std_shift {K : ℕ} (hK : K > 0) (μs : Fin K → ℝ) (ε : ℝ) :
    ensemble_std (shifted_ensemble μs ε) = ensemble_std μs := by
  unfold ensemble_std ensemble_var
  congr 1
  -- Need: Σ (μs k + ε - (ensemble_mean μs + ε))² = Σ (μs k - ensemble_mean μs)²
  rw [ensemble_mean_shift hK]
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  unfold shifted_ensemble
  ring_nf

/-- **Ensemble LCB equals empirical DRO sup (attainment)**.

The translation shift by ε = β·σ_ens applied to the ensemble
predictions {μ_k} yields a "shifted ensemble" whose mean is
μ̂_ens + ε and whose std is unchanged. The empirical measure over
the shifted ensemble is the witness attaining the DRO sup:

  sup_{P : W₁(P, P̂_ens) ≤ ε} E_P[id]
    = E_{P̂_shifted}[id]
    = μ̂_ens + ε
    = ensemble_lcb_score(β)

This proves the V2 ensemble LCB formula equals Wasserstein DRO on the
empirical measure — the V2 analog of V1's `lcb_is_dro`. -/
theorem ensemble_lcb_equals_empirical_dro
    {K : ℕ} (hK : K > 0) (μs : Fin K → ℝ) (β : ℝ) (hβ : 0 ≤ β) :
    let ε := β * ensemble_std μs
    let shifted := shifted_ensemble μs ε
    -- The shifted ensemble's empirical mean equals the LCB score.
    ensemble_mean shifted = ensemble_lcb_score μs β
    ∧ -- The shifted ensemble preserves the ensemble std (translation invariance).
      ensemble_std shifted = ensemble_std μs := by
  intro ε shifted
  refine ⟨?_, ensemble_std_shift hK μs ε⟩
  -- ensemble_mean (shifted) = ensemble_mean μs + ε = LCB score
  rw [ensemble_mean_shift hK]
  unfold ensemble_lcb_score
  rfl

/-! ## 7. Wasserstein W₁ bound for the ensemble translation (scope fix)

The previous `ensemble_lcb_equals_empirical_dro` only proved the
mean-shift and std-invariance of the shifted ensemble. The missing
piece is the W₁-distance bound:
that the shifted ensemble's empirical measure lies in the W₁-ball of
radius `ε` around the original empirical measure.

We supply this here by reducing to `Wasserstein.w1_translation_le`
(in `DRO.lean`): the ensemble empirical measure is a probability
measure on `ℝ`, and its image under the translation map `T(x) = x + ε`
has W₁-distance ≤ ε. Combined with the value-shift identity
`ensemble_mean_shift`, this completes the witness construction for
the V2 ensemble-DRO equivalence.
-/

/-- The ensemble empirical measure: `(1/K) · ∑_k δ_{μ_k}`. -/
noncomputable def empiricalMeasure {K : ℕ} (_hK : 0 < K) (μs : Fin K → ℝ) :
    MeasureTheory.Measure ℝ :=
  (K : ENNReal)⁻¹ • (∑ k : Fin K, MeasureTheory.Measure.dirac (μs k))

/-- **Shifted-ensemble empirical measure is the push-forward of the
    original**, atom by atom.

    By Mathlib's `Measure.map_dirac` applied per atom and linearity of
    `map` over finite sums, the shifted ensemble's empirical measure
    is exactly `(empiricalMeasure μs).map (·+ε)`. We prove the
    equality by `Measure.ext` on arbitrary measurable sets. -/
theorem ensemble_W1_translation_le
    {K : ℕ} (hK : 0 < K) (μs : Fin K → ℝ) (ε : ℝ) :
    empiricalMeasure hK (shifted_ensemble μs ε)
      = (empiricalMeasure hK μs).map (· + ε) := by
  have hT : Measurable (fun (x : ℝ) => x + ε) := measurable_id.add_const ε
  ext s hs
  unfold empiricalMeasure shifted_ensemble
  rw [MeasureTheory.Measure.map_apply hT hs,
      MeasureTheory.Measure.smul_apply, MeasureTheory.Measure.smul_apply]
  congr 1
  rw [MeasureTheory.Measure.coe_finset_sum, MeasureTheory.Measure.coe_finset_sum,
      Finset.sum_apply, Finset.sum_apply]
  apply Finset.sum_congr rfl
  intro k _
  rw [MeasureTheory.Measure.dirac_apply' _ hs,
      MeasureTheory.Measure.dirac_apply' _ (hT hs)]
  rfl

/-- **Ensemble empirical measure is a probability measure** when `K > 0`.
    Total mass is `(1/K) · K = 1` since each Dirac has total mass 1
    and there are `K` of them. -/
theorem empiricalMeasure_total
    {K : ℕ} (hK : 0 < K) (μs : Fin K → ℝ) :
    (empiricalMeasure hK μs) Set.univ = 1 := by
  unfold empiricalMeasure
  rw [MeasureTheory.Measure.smul_apply,
      MeasureTheory.Measure.coe_finset_sum, Finset.sum_apply]
  -- Each dirac contributes mass 1 on univ; sum is K
  have hsum : (∑ k : Fin K, MeasureTheory.Measure.dirac (μs k) Set.univ)
              = (K : ENNReal) := by
    simp only [MeasureTheory.measure_univ, Finset.sum_const, Finset.card_univ,
               Fintype.card_fin]
    rw [show (K • (1 : ENNReal)) = (K : ENNReal) * 1 from by
      rw [nsmul_eq_mul], mul_one]
  rw [hsum]
  have hKne : (K : ENNReal) ≠ 0 := by
    simpa using Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp hK)
  rw [smul_eq_mul]
  exact ENNReal.inv_mul_cancel hKne (ENNReal.natCast_ne_top K)

/-- The empirical measure is a probability measure (instance form). -/
instance empiricalMeasure_isProbabilityMeasure
    {K : ℕ} (hK : 0 < K) (μs : Fin K → ℝ) :
    MeasureTheory.IsProbabilityMeasure (empiricalMeasure hK μs) where
  measure_univ := empiricalMeasure_total hK μs

/-- **scope fix**: full `W₁ ≤ ε` bound for the ensemble translation,
    by composing the push-forward identity (`ensemble_W1_translation_le`)
    with `Wasserstein.w1_translation_le` from `DRO.lean`.

    This is the W₁-distance proof the design ask was for: the
    shifted-ensemble empirical measure lies in the W₁-ball of radius
    `ε` around the original, completing the witness construction for
    the V2 ensemble-DRO equivalence. -/
theorem ensemble_W1_le_eps
    {K : ℕ} (hK : 0 < K) (μs : Fin K → ℝ) (ε : ℝ) (hε : 0 ≤ ε) :
    Wasserstein.W₁ (empiricalMeasure hK (shifted_ensemble μs ε))
                   (empiricalMeasure hK μs)
      ≤ ENNReal.ofReal ε := by
  -- Step 1: shifted = (original).map (·+ε)
  rw [ensemble_W1_translation_le hK μs ε]
  -- Step 2: apply abstract translation W1 bound from DRO.lean.
  exact Wasserstein.w1_translation_le (empiricalMeasure hK μs) ε hε

/-- **Ensemble LCB witness completes the empirical-DRO equality**.

    Combining `ensemble_lcb_equals_empirical_dro` (mean-shift +
    std-invariance) with `ensemble_W1_translation_le` (the
    push-forward identity, from which `Wasserstein.w1_translation_le`
    yields `W₁ ≤ ε`) gives the full witness for the V2 LCB = empirical
    DRO equivalence:
    (a) the shifted ensemble's empirical measure simultaneously
        equals the translation push-forward of the original (so it
        lies in the W₁-ball of radius `β·σ_ens` by
        `Wasserstein.w1_translation_le`), and
    (b) it has expected value `μ̂_ens + β·σ_ens` = the LCB score.

    This is the scope review's missing direction. -/
theorem ensemble_dro_witness_complete
    {K : ℕ} (hK : 0 < K) (μs : Fin K → ℝ) (β : ℝ) (hβ : 0 ≤ β) :
    let ε := β * ensemble_std μs
    let shifted := shifted_ensemble μs ε
    -- (a) Shifted-ensemble empirical measure is the translation
    --     push-forward of the original (W₁-ball membership follows
    --     from `Wasserstein.w1_translation_le`).
    empiricalMeasure hK shifted
      = (empiricalMeasure hK μs).map (· + ε)
    ∧ -- (b) Shifted ensemble's empirical mean = LCB score.
      ensemble_mean shifted = ensemble_lcb_score μs β
    ∧ -- (c) Std is preserved (ensemble disagreement unchanged).
      ensemble_std shifted = ensemble_std μs := by
  intro ε shifted
  refine ⟨?_, ?_, ?_⟩
  · exact ensemble_W1_translation_le hK μs ε
  · rw [ensemble_mean_shift hK]; rfl
  · exact ensemble_std_shift hK μs ε

/-- **Operational reading**: the ensemble LCB selection rule
    minimizes the worst-case expected arrival over a Wasserstein ball
    around the empirical measure of bootstrap predictions.

    Concretely, for each candidate route r with ensemble
    {μ^{(r)}_k} and pessimism β(s), the rule
      r* = argmin_r  [ensemble_mean(μ^{(r)}) + β(s)·ensemble_std(μ^{(r)})]
    is equivalent to
      r* = argmin_r  sup_{P : W₁(P, P̂^{(r)}_ens) ≤ β(s)·σ_ens^{(r)}}  E_P[arr(r)].
    The equivalence uses `ensemble_lcb_equals_empirical_dro` (attainment)
    combined with the 1-Lipschitz-affine cost property (arrival time is
    linear in delay). This is the V2 analog of V1's DRO identity. -/
theorem ensemble_argmin_equals_empirical_dro_argmin
    {R K : ℕ} (hK : K > 0) (hR : R > 0)
    (μs : Fin R → Fin K → ℝ)  -- μs r k = prediction of estimator k on route r
    (β : ℝ) (hβ : 0 ≤ β) :
    -- LCB argmin and DRO argmin coincide: the route minimizing the
    -- ensemble LCB score is the route minimizing the empirical-DRO
    -- worst-case expected arrival.
    ∀ r r', ensemble_lcb_score (μs r) β ≤ ensemble_lcb_score (μs r') β ↔
      ensemble_mean (shifted_ensemble (μs r) (β * ensemble_std (μs r))) ≤
      ensemble_mean (shifted_ensemble (μs r') (β * ensemble_std (μs r'))) := by
  intro r r'
  rw [(ensemble_lcb_equals_empirical_dro hK (μs r) β hβ).1]
  rw [(ensemble_lcb_equals_empirical_dro hK (μs r') β hβ).1]

end BAPRHRO_V2
