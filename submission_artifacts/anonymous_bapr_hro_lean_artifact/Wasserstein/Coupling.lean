import Mathlib

/-!
# Couplings of Probability Measures

A coupling of two probability measures `μ` on `α` and `ν` on `β` is a
probability measure `γ` on `α × β` whose marginals are `μ` and `ν`:
  γ.fst = μ  and  γ.snd = ν

This is a fundamental concept in optimal transport theory. The set of all
couplings, denoted `Coupling μ ν`, is always nonempty (the product measure
`μ.prod ν` is a coupling) and compact in the weak topology under mild
conditions (Prokhorov).

## Main definitions

* `MeasureTheory.IsCoupling γ μ ν` : Predicate that `γ` is a coupling of `μ` and `ν`
* `MeasureTheory.Coupling μ ν` : The set of all couplings (as a subtype)
* `MeasureTheory.Coupling.prod` : The product coupling (always exists)

## Main results

* `MeasureTheory.Coupling.nonempty` : Couplings always exist
* `MeasureTheory.Coupling.fst_eq` : First marginal equals `μ`
* `MeasureTheory.Coupling.snd_eq` : Second marginal equals `ν`

## References

* Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
* Santambrogio, F. (2015). *Optimal Transport for Applied Mathematicians*. Birkhäuser.
-/

noncomputable section

open MeasureTheory Measure

namespace MeasureTheory

variable {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]

/-! ## 1. Coupling Predicate -/

/-- A measure `γ` on `α × β` is a coupling of `μ` and `ν` if its marginals
match: `γ.fst = μ` and `γ.snd = ν`. -/
def IsCoupling (γ : Measure (α × β)) (μ : Measure α) (ν : Measure β) : Prop :=
  γ.fst = μ ∧ γ.snd = ν

/-- The set of all couplings of `μ` and `ν`, as a subtype of measures on the product. -/
def CouplingSet (μ : Measure α) (ν : Measure β) : Set (Measure (α × β)) :=
  { γ | IsCoupling γ μ ν }

/-! ## 2. Product Coupling -/

/-- The product measure `μ.prod ν` is always a coupling of `μ` and `ν`.
This establishes that couplings always exist. -/
theorem isCoupling_prod (μ : Measure α) [IsProbabilityMeasure μ]
    (ν : Measure β) [IsProbabilityMeasure ν] :
    IsCoupling (μ.prod ν) μ ν := by
  exact ⟨Measure.fst_prod, Measure.snd_prod⟩

/-- The set of couplings is nonempty (witnessed by the product measure). -/
theorem coupling_nonempty (μ : Measure α) [IsProbabilityMeasure μ]
    (ν : Measure β) [IsProbabilityMeasure ν] :
    (CouplingSet μ ν).Nonempty :=
  ⟨μ.prod ν, isCoupling_prod μ ν⟩

/-! ## 3. Basic Properties -/

theorem IsCoupling.fst_eq {γ : Measure (α × β)} {μ : Measure α} {ν : Measure β}
    (h : IsCoupling γ μ ν) : γ.fst = μ :=
  h.1

theorem IsCoupling.snd_eq {γ : Measure (α × β)} {μ : Measure α} {ν : Measure β}
    (h : IsCoupling γ μ ν) : γ.snd = ν :=
  h.2

/-- If `γ` is a coupling of probability measures, then `γ` is itself a probability measure
(its total mass is 1). -/
theorem IsCoupling.isProbabilityMeasure
    {γ : Measure (α × β)} {μ : Measure α} {ν : Measure β}
    (h : IsCoupling γ μ ν) [hμ : IsProbabilityMeasure μ] :
    IsProbabilityMeasure γ := by
  constructor
  rw [← γ.fst_univ, h.fst_eq]
  exact hμ.measure_univ

/-- A coupling preserves the total mass. -/
theorem IsCoupling.mass_eq
    {γ : Measure (α × β)} {μ : Measure α} {ν : Measure β}
    (h : IsCoupling γ μ ν) :
    γ Set.univ = μ Set.univ := by
  rw [← γ.fst_univ, h.fst_eq]

end MeasureTheory
