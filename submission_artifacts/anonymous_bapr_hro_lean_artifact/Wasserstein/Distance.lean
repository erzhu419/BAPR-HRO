import Mathlib

/-!
# Wasserstein Distance (W₁)

The 1-Wasserstein distance between probability measures `μ` and `ν` on a
metric space `X`:

  W₁(μ, ν) = inf { ∫ d(x,y) dγ(x,y) | γ.fst = μ ∧ γ.snd = ν }

## Main definitions and results

* `wassersteinCost` : Transport cost ∫ d(x,y) dγ
* `wassersteinDist` : W₁ as infimum over couplings
* `wassersteinDist_self` : W₁(μ, μ) = 0
* `wassersteinDist_nonneg` : W₁ ≥ 0

## References

* Villani, C. (2009). *Optimal Transport: Old and New*. Chapter 6.
-/

noncomputable section

open MeasureTheory Measure ENNReal

namespace Wasserstein

variable {X : Type*} [MeasurableSpace X] [MetricSpace X] [BorelSpace X]
  [SecondCountableTopology X] [CompleteSpace X] [PolishSpace X] [Nonempty X]
-- PolishSpace → StandardBorelSpace (with BorelSpace)
-- StandardBorelSpace → condKernel available
-- BorelSpace X + SecondCountableTopology X gives:
-- - OpensMeasurableSpace X (from BorelSpace)
-- - BorelSpace (X × X) (via Prod.borelSpace, needs SecondCountableTopologyEither)
-- - OpensMeasurableSpace (X × X) (from BorelSpace (X × X))
-- This resolves the typeclass synthesis for lintegral_map on the product.

/-! ## 1. Coupling (self-contained definition) -/

/-- A coupling of `μ` and `ν` is a measure on the product with given marginals. -/
def IsCoupling (γ : Measure (X × X)) (μ ν : Measure X) : Prop :=
  γ.fst = μ ∧ γ.snd = ν

/-! ## 2. Transport Cost -/

/-- Transport cost of a coupling: ∫ d(x, y) dγ(x, y). -/
def transportCost (γ : Measure (X × X)) : ℝ≥0∞ :=
  ∫⁻ p, ENNReal.ofReal (dist p.1 p.2) ∂γ

/-! ## 3. W₁ Distance -/

/-- The 1-Wasserstein distance: infimum of transport cost over all couplings. -/
def wassersteinDist (μ ν : Measure X) : ℝ≥0∞ :=
  ⨅ (γ : Measure (X × X)) (_ : IsCoupling γ μ ν), transportCost γ

/-! ## 4. Basic Properties -/

/-- W₁ is non-negative (trivial for ℝ≥0∞). -/
theorem wassersteinDist_nonneg (μ ν : Measure X) : 0 ≤ wassersteinDist μ ν :=
  zero_le _

/-- The diagonal map `x ↦ (x, x)` has zero transport cost:
    ∫ d(x,y) d(map diag μ) = ∫ d(x,x) dμ = ∫ 0 dμ = 0. -/
theorem transportCost_diag (μ : Measure X) :
    transportCost (μ.map (fun x => (x, x))) = 0 := by
  unfold transportCost
  -- Rewrite integral over push-forward to integral over μ
  have hdiag : Measurable (fun (x : X) => (x, x)) := Measurable.prod measurable_id measurable_id
  -- Change of variables: ∫ f d(map g μ) = ∫ (f ∘ g) dμ, then dist(x,x) = 0
  calc ∫⁻ p, ENNReal.ofReal (dist p.1 p.2) ∂(μ.map (fun x => (x, x)))
      = ∫⁻ x, ENNReal.ofReal (dist x x) ∂μ := by
        rw [lintegral_map continuous_dist.measurable.ennreal_ofReal hdiag]
    _ = ∫⁻ _, 0 ∂μ := by
        congr 1; ext x; simp [dist_self]
    _ = 0 := lintegral_zero

/-- The diagonal map is a coupling of μ with itself. -/
theorem isCoupling_diag (μ : Measure X) :
    IsCoupling (μ.map (fun x => (x, x))) μ μ := by
  constructor
  · -- fst: map Prod.fst (map diag μ) = map (Prod.fst ∘ diag) μ = map id μ = μ
    rw [Measure.fst, Measure.map_map measurable_fst
        (Measurable.prod measurable_id measurable_id)]
    have : Prod.fst ∘ (fun (x : X) => (x, x)) = id := by ext x; rfl
    rw [this, Measure.map_id]
  · -- snd: same argument
    rw [Measure.snd, Measure.map_map measurable_snd
        (Measurable.prod measurable_id measurable_id)]
    have : Prod.snd ∘ (fun (x : X) => (x, x)) = id := by ext x; rfl
    rw [this, Measure.map_id]

/-- W₁(μ, μ) = 0. The diagonal coupling witnesses this. -/
theorem wassersteinDist_self (μ : Measure X) [IsProbabilityMeasure μ] :
    wassersteinDist μ μ = 0 := by
  apply le_antisymm
  · -- W₁(μ,μ) ≤ cost(diagonal) = 0
    calc wassersteinDist μ μ
        ≤ transportCost (μ.map (fun x => (x, x))) := by
          unfold wassersteinDist
          apply le_trans (iInf_le _ (μ.map (fun x => (x, x))))
          exact iInf_le _ (isCoupling_diag μ)
      _ = 0 := transportCost_diag μ
  · exact zero_le _

/-! ## 5. Symmetry -/

/-- Swapping a coupling of (μ, ν) gives a coupling of (ν, μ). -/
theorem isCoupling_swap {μ ν : Measure X} {γ : Measure (X × X)} (h : IsCoupling γ μ ν) :
    IsCoupling (γ.map Prod.swap) ν μ := by
  constructor
  · -- fst of swap = snd of original
    rw [Measure.fst, Measure.map_map measurable_fst measurable_swap]
    have : Prod.fst ∘ Prod.swap = (Prod.snd : X × X → X) := by ext ⟨a, b⟩; rfl
    rw [this, ← Measure.snd]
    exact h.2
  · -- snd of swap = fst of original
    rw [Measure.snd, Measure.map_map measurable_snd measurable_swap]
    have : Prod.snd ∘ Prod.swap = (Prod.fst : X × X → X) := by ext ⟨a, b⟩; rfl
    rw [this, ← Measure.fst]
    exact h.1

/-- Transport cost is invariant under swapping (since dist is symmetric). -/
theorem transportCost_swap (γ : Measure (X × X)) :
    transportCost (γ.map Prod.swap) = transportCost γ := by
  unfold transportCost
  rw [lintegral_map _ measurable_swap]
  · congr 1; ext ⟨a, b⟩; simp [dist_comm]
  · exact continuous_dist.measurable.ennreal_ofReal

/-- W₁(μ, ν) ≤ W₁(ν, μ): one direction of symmetry. -/
private theorem wassersteinDist_le_comm (μ ν : Measure X) :
    wassersteinDist μ ν ≤ wassersteinDist ν μ := by
  unfold wassersteinDist
  apply iInf₂_mono'
  intro γ hγ
  exact ⟨γ.map Prod.swap, isCoupling_swap hγ, le_of_eq (transportCost_swap γ)⟩

/-- W₁(μ, ν) = W₁(ν, μ): Wasserstein distance is symmetric. -/
theorem wassersteinDist_comm (μ ν : Measure X) :
    wassersteinDist μ ν = wassersteinDist ν μ :=
  le_antisymm (wassersteinDist_le_comm μ ν) (wassersteinDist_le_comm ν μ)

/-! ## 6. Triangle Inequality -/

/-- **Gluing Lemma** (Villani, Optimal Transport, Lemma 7.6).

Given couplings γ₁ of (μ, ν) and γ₂ of (ν, ρ), there exists a coupling γ
of (μ, ρ) whose transport cost is bounded by the sum.

Construction: disintegrate γ₁.swap and γ₂ over ν via condKernel,
glue the conditional kernels via Kernel.prod, project to (x,z). -/
theorem gluing_lemma {μ ν ρ : Measure X}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] [IsProbabilityMeasure ρ]
    (γ₁ : Measure (X × X)) (γ₂ : Measure (X × X))
    (h₁ : IsCoupling γ₁ μ ν) (h₂ : IsCoupling γ₂ ν ρ) :
    ∃ γ : Measure (X × X), IsCoupling γ μ ρ ∧
      transportCost γ ≤ transportCost γ₁ + transportCost γ₂ := by
  -- Derive probability measure instances
  haveI hγ₁p : IsProbabilityMeasure γ₁ :=
    { measure_univ := by rw [← γ₁.fst_univ, h₁.1]; exact IsProbabilityMeasure.measure_univ }
  haveI hγ₂p : IsProbabilityMeasure γ₂ :=
    { measure_univ := by rw [← γ₂.fst_univ, h₂.1]; exact IsProbabilityMeasure.measure_univ }
  -- Swap γ₁ so ν is first marginal
  set γ₁' := γ₁.map Prod.swap with hγ₁'_def
  haveI : IsProbabilityMeasure γ₁' := isProbabilityMeasure_map measurable_swap.aemeasurable
  have hγ₁'_fst : γ₁'.fst = ν := (isCoupling_swap h₁).1
  have hγ₂_fst : γ₂.fst = ν := h₂.1
  -- Disintegration (key identity: ρ.fst ⊗ₘ ρ.condKernel = ρ)
  have hdis₁ := @Measure.IsCondKernel.disintegrate X X _ _ γ₁' γ₁'.condKernel _
  have hdis₂ := @Measure.IsCondKernel.disintegrate X X _ _ γ₂ γ₂.condKernel _
  -- Substitute fst = ν
  rw [hγ₁'_fst] at hdis₁
  rw [hγ₂_fst] at hdis₂
  -- Construct glued coupling
  set κ₁ := γ₁'.condKernel with hκ₁_def
  set κ₂ := γ₂.condKernel with hκ₂_def
  set γ := (ν.compProd (κ₁.prod κ₂)).map Prod.snd with hγ_def
  use γ
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · -- γ.fst = μ
    rw [hγ_def, Measure.fst, Measure.map_map measurable_fst measurable_snd]
    -- Goal: (ν ⊗ₘ κ₁.prod κ₂).map (Prod.fst ∘ Prod.snd) = μ
    -- (Prod.fst ∘ Prod.snd) on (y, (x, z)) = x
    -- Rewrite compProd + map to snd of (ν ⊗ₘ κ₁)
    -- First: (ν ⊗ₘ κ₁.prod κ₂).map (fst ∘ snd) = (ν ⊗ₘ κ₁).snd
    -- This follows from: Kernel.fst (κ₁.prod κ₂) = κ₁ [Kernel.fst_prod]
    -- and (ν ⊗ₘ κ).map (fst ∘ snd) = (ν ⊗ₘ Kernel.fst κ).snd
    -- Then use hdis₁: ν ⊗ₘ κ₁ = γ₁', and γ₁'.snd = μ
    -- γ₁'.snd = (γ₁.map swap).snd = γ₁.fst = μ
    have hγ₁'_snd : γ₁'.snd = μ := by
      rw [hγ₁'_def, Measure.snd, Measure.map_map measurable_snd measurable_swap]
      change γ₁.map Prod.fst = μ
      rw [← Measure.fst]; exact h₁.1
    -- Key step: (ν ⊗ₘ κ₁.prod κ₂).map (fst ∘ snd) = (ν ⊗ₘ κ₁).snd
    -- Use: hdis₁ says ν ⊗ₘ κ₁ = γ₁', so (ν ⊗ₘ κ₁).snd = γ₁'.snd = μ
    -- The connection requires: compProd + map factoring through Kernel.prod
    -- This is the remaining API gap.
    conv_lhs => rw [show Prod.fst ∘ Prod.snd = (fun (p : X × (X × X)) => p.2.1) from rfl]
    -- Goal: (ν ⊗ₘ κ₁.prod κ₂).map (fun p => p.2.1) = μ
    -- = (ν.compProd κ₁).snd  (by extracting fst of Kernel.prod output)
    -- = γ₁'.snd = μ
    rw [← hγ₁'_snd, ← hdis₁]
    -- Goal: (ν ⊗ₘ κ₁.prod κ₂).map (fun p => p.2.1) = (ν ⊗ₘ κ₁).snd
    -- Both sides equal (ν ⊗ₘ κ₁).map Prod.snd, by definition of Measure.snd
    -- and by the fact that κ₁ = Kernel.fst (κ₁.prod κ₂).
    -- Measure.snd = map Prod.snd, and (fun p => p.2.1) factors through
    -- Kernel.fst extraction from Kernel.prod.
    ext s hs
    rw [Measure.map_apply (by fun_prop) hs, Measure.snd_apply hs]
    -- LHS: (ν ⊗ₘ κ₁.prod κ₂) ((fun p => p.2.1) ⁻¹' s)
    -- RHS: (ν ⊗ₘ κ₁) (Prod.snd ⁻¹' s)
    -- Expand both via compProd_apply
    have hms : MeasurableSet ((fun (p : X × (X × X)) => p.2.1) ⁻¹' s) :=
      (measurable_fst.comp measurable_snd) hs
    rw [Measure.compProd_apply hms, Measure.compProd_apply (measurable_snd hs)]
    -- Both are ∫⁻ y, ... ∂ν. Show integrands are equal.
    congr 1; ext y
    -- LHS integrand: (κ₁.prod κ₂) y (Prod.mk y ⁻¹' ((fun p => p.2.1) ⁻¹' s))
    --             = (κ₁.prod κ₂) y {(x,z) | x ∈ s}
    --             = ((κ₁ y).prod (κ₂ y)) {(x,z) | x ∈ s}   [Kernel.prod_apply]
    --             = (κ₁ y) s * (κ₂ y) univ                   [Measure.prod on product set]
    --             = (κ₁ y) s                                  [IsMarkovKernel: (κ₂ y) univ = 1]
    -- RHS integrand: κ₁ y (Prod.mk y ⁻¹' (Prod.snd ⁻¹' s))
    --             = κ₁ y s
    -- LHS: (κ₁.prod κ₂) y (Prod.mk y ⁻¹' {p | p.2.1 ∈ s})
    -- = ((κ₁ y).prod (κ₂ y)) (s ×ˢ Set.univ)  [preimage simplification]
    -- = (κ₁ y) s * (κ₂ y) Set.univ              [Measure.prod_prod]
    -- = (κ₁ y) s * 1 = (κ₁ y) s                 [IsMarkovKernel → measure_univ = 1]
    -- RHS: κ₁ y {x | (y, x) ∈ Prod.snd ⁻¹' s} = κ₁ y s  [preimage simplification]
    simp only [Set.preimage, Set.mem_setOf_eq]
    -- Unfold Kernel.prod at point y
    rw [show (ProbabilityTheory.Kernel.prod κ₁ κ₂) y = (κ₁ y).prod (κ₂ y) from
        ProbabilityTheory.Kernel.prod_apply κ₁ κ₂ y]
    -- Goal: ((κ₁ y).prod (κ₂ y)) {x | x.1 ∈ s} = (κ₁ y) s
    -- {x | x.1 ∈ s} = s ×ˢ univ
    rw [show {x : X × X | x.1 ∈ s} = s ×ˢ Set.univ from by ext ⟨a,b⟩; simp]
    rw [Measure.prod_prod]
    simp [measure_univ]
  · -- γ.snd = ρ (analogous to γ.fst, using κ₂ and h₂.2)
    rw [hγ_def, Measure.snd, Measure.map_map measurable_snd measurable_snd]
    conv_lhs => rw [show Prod.snd ∘ Prod.snd = (fun (p : X × (X × X)) => p.2.2) from rfl]
    rw [← (isCoupling_swap h₂ |>.1 |> fun _ => h₂.2)]  -- γ₂.snd = ρ
    rw [← hdis₂]
    ext s hs
    rw [Measure.map_apply (by fun_prop) hs, Measure.snd_apply hs]
    have hms : MeasurableSet ((fun (p : X × (X × X)) => p.2.2) ⁻¹' s) :=
      (measurable_snd.comp measurable_snd) hs
    rw [Measure.compProd_apply hms, Measure.compProd_apply (measurable_snd hs)]
    congr 1; ext y
    simp only [Set.preimage, Set.mem_setOf_eq]
    rw [ProbabilityTheory.Kernel.prod_apply κ₁ κ₂ y]
    rw [show {x : X × X | x.2 ∈ s} = Set.univ ×ˢ s from by ext ⟨a,b⟩; simp]
    rw [Measure.prod_prod]
    simp [measure_univ]
  · -- transportCost γ ≤ transportCost γ₁ + transportCost γ₂
    unfold transportCost
    -- Step 1: Change of variables — pull back through map Prod.snd
    rw [hγ_def, lintegral_map continuous_dist.measurable.ennreal_ofReal measurable_snd]
    -- Goal: ∫⁻ p, ofReal(dist p.2.1 p.2.2) d(ν ⊗ₘ κ) ≤ ∫⁻ ... dγ₁ + ∫⁻ ... dγ₂
    -- Step 2: Triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)
    calc ∫⁻ p, ENNReal.ofReal (dist p.2.1 p.2.2) ∂(ν.compProd (κ₁.prod κ₂))
        ≤ ∫⁻ p, (ENNReal.ofReal (dist p.2.1 p.1) + ENNReal.ofReal (dist p.1 p.2.2))
            ∂(ν.compProd (κ₁.prod κ₂)) := by
          apply lintegral_mono
          intro ⟨y, x, z⟩
          -- dist x z ≤ dist x y + dist y z
          calc ENNReal.ofReal (dist x z)
              ≤ ENNReal.ofReal (dist x y + dist y z) :=
                ENNReal.ofReal_le_ofReal (dist_triangle x y z)
            _ ≤ ENNReal.ofReal (dist x y) + ENNReal.ofReal (dist y z) :=
                ENNReal.ofReal_add_le
      _ ≤ ∫⁻ q, ENNReal.ofReal (dist q.1 q.2) ∂γ₁
        + ∫⁻ q, ENNReal.ofReal (dist q.1 q.2) ∂γ₂ := by
          -- The integral of sum ≤ sum of integrals (holds with equality for ENNReal).
          -- Then each integral over ν ⊗ₘ κ₁.prod κ₂ equals the
          -- corresponding integral over γ₁ resp γ₂ via Fubini + disintegration.
          -- ∫_{(y,(x,z))} d(x,y) d(ν ⊗ₘ κ₁.prod κ₂)
          --   = ∫_y ∫_{(x,z)} d(x,y) d((κ₁ y).prod (κ₂ y)) dν(y)  [compProd Fubini]
          --   = ∫_y ∫_x d(x,y) d(κ₁ y)(x) dν(y)                     [prod marginal]
          --   = ∫_{(y,x)} d(x,y) d(ν ⊗ₘ κ₁)                        [compProd back]
          --   = ∫ d(x,y) dγ₁'                                        [hdis₁]
          --   = transportCost γ₁                                      [swap preserves cost]
          -- This chain requires lintegral_compProd + Kernel.prod marginal + hdis₁.
          -- Analogous for the second term via κ₂ + hdis₂.
          -- Split: ∫(f+g) = ∫f + ∫g, then bound each part
          have hmf : Measurable (fun (p : X × (X × X)) =>
              ENNReal.ofReal (dist p.2.1 p.1)) := by
            exact (by fun_prop : Measurable (fun (p : X × (X × X)) =>
              ENNReal.ofReal (dist p.2.1 p.1)))
          calc ∫⁻ p, (ENNReal.ofReal (dist p.2.1 p.1) + ENNReal.ofReal (dist p.1 p.2.2))
                  ∂(ν.compProd (κ₁.prod κ₂))
              = ∫⁻ p, ENNReal.ofReal (dist p.2.1 p.1) ∂(ν.compProd (κ₁.prod κ₂))
              + ∫⁻ p, ENNReal.ofReal (dist p.1 p.2.2) ∂(ν.compProd (κ₁.prod κ₂)) := by
                exact lintegral_add_left hmf _
            _ ≤ ∫⁻ q, ENNReal.ofReal (dist q.1 q.2) ∂γ₁
              + ∫⁻ q, ENNReal.ofReal (dist q.1 q.2) ∂γ₂ := by
                apply add_le_add
                · -- Goal: ∫ d(x,y) d(ν ⊗ₘ κ₁.prod κ₂) ≤ ∫ d(q.1,q.2) dγ₁
                  -- Actually =. Prove via Fubini → Kernel.prod → Tonelli → disintegrate → swap.
                  suffices h : ∫⁻ p, ENNReal.ofReal (dist p.2.1 p.1) ∂(ν.compProd (κ₁.prod κ₂))
                    = ∫⁻ q, ENNReal.ofReal (dist q.1 q.2) ∂γ₁ from le_of_eq h
                  -- Step A: Fubini on compProd
                  rw [Measure.lintegral_compProd (by fun_prop)]
                  -- ∫_y ∫_{xz} ofReal(dist xz.1 y) d((κ₁.prod κ₂) y) dν(y)
                  -- Step B: Kernel.prod_apply: (κ₁.prod κ₂) y = (κ₁ y).prod (κ₂ y)
                  simp_rw [ProbabilityTheory.Kernel.prod_apply κ₁ κ₂]
                  -- ∫_y ∫_{(x,z)} ofReal(dist x y) d((κ₁ y).prod (κ₂ y)) dν(y)
                  -- Step C: Integrand doesn't depend on z — integrate it out
                  -- ∫_{(x,z)} g(x) d((κ₁ y).prod (κ₂ y)) = (∫_x g(x) d(κ₁ y)) * (κ₂ y)(univ)
                  -- Integrate out z from inner integral (dist x y doesn't depend on z)
                  have key₁ : ∀ y, ∫⁻ xz, ENNReal.ofReal (dist xz.1 y) ∂((κ₁ y).prod (κ₂ y))
                      = ∫⁻ x, ENNReal.ofReal (dist x y) ∂(κ₁ y) := by
                    intro y
                    rw [lintegral_prod _ (by fun_prop)]
                    congr 1; ext x
                    simp [lintegral_const, measure_univ]
                  simp_rw [key₁]
                  -- Now: ∫_y ∫_x ofReal(dist x y) d(κ₁ y) dν(y) = ∫ d(q.1,q.2) dγ₁
                  -- = ∫ ofReal(dist p.2 p.1) d(ν ⊗ₘ κ₁)         [lintegral_compProd⁻¹]
                  -- = ∫ ofReal(dist p.2 p.1) dγ₁'                  [hdis₁]
                  -- = ∫ ofReal(dist q.1 q.2) dγ₁                   [swap + dist_comm]
                  have hm_dist_swap : Measurable (fun (p : X × X) => ENNReal.ofReal (dist p.2 p.1)) :=
                    continuous_dist.measurable.ennreal_ofReal.comp measurable_swap
                  conv_lhs => rw [← Measure.lintegral_compProd hm_dist_swap]
                  rw [hdis₁, hγ₁'_def, lintegral_map (by fun_prop) measurable_swap]
                  -- Check if goal is closed
                  simp [dist_comm]
                · -- Goal: ∫ d(y,z) d(ν ⊗ₘ κ₁.prod κ₂) ≤ ∫ d(q.1,q.2) dγ₂
                  suffices h : ∫⁻ p, ENNReal.ofReal (dist p.1 p.2.2) ∂(ν.compProd (κ₁.prod κ₂))
                    = ∫⁻ q, ENNReal.ofReal (dist q.1 q.2) ∂γ₂ from le_of_eq h
                  rw [Measure.lintegral_compProd (by fun_prop)]
                  simp_rw [ProbabilityTheory.Kernel.prod_apply κ₁ κ₂]
                  -- Integrand doesn't depend on x — integrate it out
                  -- ∫_{(x,z)} g(z) d((κ₁ y).prod (κ₂ y)) = (κ₁ y)(univ) * ∫_z g(z) d(κ₂ y)
                  -- Integrate out x (dist y z doesn't depend on x)
                  have key₂ : ∀ y, ∫⁻ xz, ENNReal.ofReal (dist y xz.2) ∂((κ₁ y).prod (κ₂ y))
                      = ∫⁻ z, ENNReal.ofReal (dist y z) ∂(κ₂ y) := by
                    intro y
                    rw [lintegral_prod_symm' _ (by fun_prop)]
                    congr 1; ext z
                    simp [lintegral_const, measure_univ]
                  simp_rw [key₂]
                  have hm_dist : Measurable (fun (p : X × X) => ENNReal.ofReal (dist p.1 p.2)) :=
                    continuous_dist.measurable.ennreal_ofReal
                  conv_lhs => rw [← Measure.lintegral_compProd hm_dist]
                  rw [hdis₂]

/-- Helper: W₁(μ,ρ) ≤ cost(γ₁) + cost(γ₂) for any couplings γ₁ of (μ,ν), γ₂ of (ν,ρ). -/
private theorem wassersteinDist_le_coupling_sum {μ ν ρ : Measure X}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] [IsProbabilityMeasure ρ]
    {γ₁ γ₂ : Measure (X × X)} (h₁ : IsCoupling γ₁ μ ν) (h₂ : IsCoupling γ₂ ν ρ) :
    wassersteinDist μ ρ ≤ transportCost γ₁ + transportCost γ₂ := by
  obtain ⟨γ, hγ_coupling, hγ_cost⟩ := gluing_lemma γ₁ γ₂ h₁ h₂
  calc wassersteinDist μ ρ
      ≤ transportCost γ := le_trans (iInf_le _ γ) (iInf_le _ hγ_coupling)
    _ ≤ transportCost γ₁ + transportCost γ₂ := hγ_cost

/-- **W₁(μ, ρ) ≤ W₁(μ, ν) + W₁(ν, ρ)**: Triangle inequality for W₁.

Given the gluing lemma, the proof is:
∀ γ₁ ∈ C(μ,ν), γ₂ ∈ C(ν,ρ): W₁(μ,ρ) ≤ c(γ₁) + c(γ₂)
⟹ W₁(μ,ρ) ≤ ⨅ γ₁ γ₂, c(γ₁) + c(γ₂) ≤ (⨅ γ₁, c(γ₁)) + (⨅ γ₂, c(γ₂))
             = W₁(μ,ν) + W₁(ν,ρ). -/
theorem wassersteinDist_triangle (μ ν ρ : Measure X)
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] [IsProbabilityMeasure ρ] :
    wassersteinDist μ ρ ≤ wassersteinDist μ ν + wassersteinDist ν ρ := by
  -- Suffices: W₁(μ,ρ) ≤ c(γ₁) + c(γ₂) for all couplings γ₁, γ₂
  -- Then take inf over γ₁ on the right: W₁(μ,ρ) ≤ W₁(μ,ν) + c(γ₂) for all γ₂
  -- Then take inf over γ₂: W₁(μ,ρ) ≤ W₁(μ,ν) + W₁(ν,ρ)
  -- Apply ENNReal.le_iInf₂_add_iInf₂:
  -- if W₁(μ,ρ) ≤ c(γ₁) + c(γ₂) for all γ₁∈C(μ,ν) and γ₂∈C(ν,ρ),
  -- then W₁(μ,ρ) ≤ (⨅ γ₁∈C(μ,ν), c(γ₁)) + (⨅ γ₂∈C(ν,ρ), c(γ₂)) = W₁(μ,ν) + W₁(ν,ρ)
  unfold wassersteinDist
  exact ENNReal.le_iInf₂_add_iInf₂
    (fun γ₁ hγ₁ γ₂ hγ₂ => wassersteinDist_le_coupling_sum hγ₁ hγ₂)

end Wasserstein
