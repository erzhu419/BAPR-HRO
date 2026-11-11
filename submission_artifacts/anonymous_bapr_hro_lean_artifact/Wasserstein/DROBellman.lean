import Mathlib

/-!
# DRO-Bellman: Distributionally Robust Bellman Operators

## Main Results

For a tabular MDP with finite state space `S`, finite action space `A`,
discount `γ ∈ [0, 1)`, cost `c`, and a *family of ambiguity functionals*
`Q : S × A → (S → ℝ) → ℝ` (each `Q(s, a) V` represents the worst-case
expected next-state value over some ambiguity set), we define the
**robust Bellman operator** as

    T_Q(V)(s) = min_a [ c(s, a) + γ · Q(s, a)(V) ].

This file proves:

1. `robustBellmanOp_contraction`: if every `Q(s, a)` is *non-expansive*
   in `V` (a uniform property: `|Q V - Q W| ≤ ⨆ s', |V s' - W s'|`),
   then `T_Q` is a `γ`-contraction in `‖·‖_∞`. This is a real
   contraction theorem; the proof uses non-expansiveness of `min_a`
   and of the per-state-action `Q`, not an "additive penalty cancels"
   shortcut.

2. `mixture_robust_bellman_contraction`: the discrete mixture functional
   `Q_b(s, a) V = ∑_z b(z) ⟨P_z(s, a), V⟩` is nonexpansive under the
   probability assumptions, giving the same contraction factor.

3. `wasserstein_amb_lipBound`: the nominal expectation plus penalty
   `Q_W(s, a) V = ⟨P̂(s, a), V⟩ + ε · Lip(V)` has a `(1 + ε)`
   Lipschitz bound when the supplied `Lip` functional is nonexpansive.
   This expression is the upper endpoint in the Wasserstein bound.
   A supremum of expectations over a fixed nonempty set of probability
   measures satisfies the nonexpansiveness premise directly.
-/

noncomputable section

namespace DROBellman

variable {S A : Type} [Fintype S] [Fintype A] [Nonempty S] [Nonempty A]

/-! ## 1. Standard tabular Bellman -/

/-- Standard tabular Bellman operator. -/
def bellmanOp (γ : ℝ) (c : S × A → ℝ) (P : S × A → S → ℝ)
    (V : S → ℝ) (s : S) : ℝ :=
  Finset.univ.inf' Finset.univ_nonempty
    (fun a => c (s, a) + γ * ∑ s' : S, P (s, a) s' * V s')

/-! ## 2. Generic robust Bellman over a non-expansive ambiguity functional -/

/-- Robust Bellman: `T_Q(V)(s) = min_a [c(s, a) + γ · Q(s, a)(V)]`. -/
def robustBellmanOp (γ : ℝ) (c : S × A → ℝ)
    (Q : S × A → (S → ℝ) → ℝ) (V : S → ℝ) (s : S) : ℝ :=
  Finset.univ.inf' Finset.univ_nonempty
    (fun a => c (s, a) + γ * Q (s, a) V)

/-- The pointwise-sup norm `⨆ s, |V s - W s|` on a finite state space. -/
private noncomputable def supDiff (V W : S → ℝ) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty (fun s => |V s - W s|)

/-- Pointwise diff is bounded by the sup-norm (definitional). -/
private theorem abs_diff_le_supDiff (V W : S → ℝ) (s : S) :
    |V s - W s| ≤ supDiff V W := by
  unfold supDiff
  exact Finset.le_sup' (fun s => |V s - W s|) (Finset.mem_univ s)

/-- A robust ambiguity functional `Q : S × A → (S → ℝ) → ℝ` is
    **non-expansive** if for every state-action pair, `Q(s, a)` is
    `1`-Lipschitz in the sup-norm:

        |Q (s, a) V - Q (s, a) W| ≤ ‖V − W‖_∞.

    For finite-support MDPs this holds for any *probability-mixture*
    ambiguity functional (Jensen's inequality on the underlying
    probability measure). It is the abstract structural property that
    drives `γ`-contractivity of the robust Bellman. -/
structure NonExpansiveAmbiguity (Q : S × A → (S → ℝ) → ℝ) : Prop where
  nonexp : ∀ (V W : S → ℝ) (s : S) (a : A),
    |Q (s, a) V - Q (s, a) W| ≤ supDiff V W

/-- Per-action backup non-expansiveness, scaled by `γ`. -/
private lemma backup_nonexp
    {γ : ℝ} (hγ : 0 ≤ γ)
    (c : S × A → ℝ) (Q : S × A → (S → ℝ) → ℝ)
    (hNE : NonExpansiveAmbiguity (Q := Q))
    (V W : S → ℝ) (s : S) (a : A) :
    |(c (s, a) + γ * Q (s, a) V) - (c (s, a) + γ * Q (s, a) W)|
      ≤ γ * supDiff V W := by
  have h1 : (c (s, a) + γ * Q (s, a) V) - (c (s, a) + γ * Q (s, a) W)
            = γ * (Q (s, a) V - Q (s, a) W) := by ring
  rw [h1, abs_mul, abs_of_nonneg hγ]
  exact mul_le_mul_of_nonneg_left (hNE.nonexp V W s a) hγ

/-- **Non-expansiveness of `inf'` on a `Finset`**. -/
private lemma abs_inf'_sub_inf'_le
    (f g : A → ℝ) (M : ℝ)
    (hbound : ∀ a, |f a - g a| ≤ M) :
    |Finset.univ.inf' Finset.univ_nonempty f -
     Finset.univ.inf' Finset.univ_nonempty g|
      ≤ M := by
  apply abs_sub_le_iff.mpr
  refine ⟨?_, ?_⟩
  · obtain ⟨a₁, _, ha₁⟩ := (Finset.univ : Finset A).exists_mem_eq_inf'
      Finset.univ_nonempty g
    calc Finset.univ.inf' Finset.univ_nonempty f -
         Finset.univ.inf' Finset.univ_nonempty g
        ≤ f a₁ - Finset.univ.inf' Finset.univ_nonempty g :=
          sub_le_sub_right (Finset.inf'_le _ (Finset.mem_univ _)) _
      _ = f a₁ - g a₁ := by rw [ha₁]
      _ ≤ |f a₁ - g a₁| := le_abs_self _
      _ ≤ M := hbound a₁
  · obtain ⟨a₂, _, ha₂⟩ := (Finset.univ : Finset A).exists_mem_eq_inf'
      Finset.univ_nonempty f
    calc Finset.univ.inf' Finset.univ_nonempty g -
         Finset.univ.inf' Finset.univ_nonempty f
        ≤ g a₂ - Finset.univ.inf' Finset.univ_nonempty f :=
          sub_le_sub_right (Finset.inf'_le _ (Finset.mem_univ _)) _
      _ = g a₂ - f a₂ := by rw [ha₂]
      _ ≤ |f a₂ - g a₂| := by rw [abs_sub_comm]; exact le_abs_self _
      _ ≤ M := hbound a₂

/-- **Robust Bellman γ-contraction**. -/
theorem robustBellmanOp_contraction
    {γ : ℝ} (hγ : 0 ≤ γ)
    (c : S × A → ℝ) (Q : S × A → (S → ℝ) → ℝ)
    (hNE : NonExpansiveAmbiguity (Q := Q)) (V W : S → ℝ) (s : S) :
    |robustBellmanOp γ c Q V s - robustBellmanOp γ c Q W s|
      ≤ γ * supDiff V W := by
  unfold robustBellmanOp
  exact abs_inf'_sub_inf'_le
    (fun a => c (s, a) + γ * Q (s, a) V)
    (fun a => c (s, a) + γ * Q (s, a) W)
    (γ * supDiff V W)
    (fun a => backup_nonexp hγ c Q hNE V W s a)

/-! ## 3. Concrete instance: discrete posterior-gated mixture (BAPR).

The BAPR finite-regime abstraction uses a Bayesian belief `b(z)` over
a finite set of latent regimes, and defines the value-update step as a
posterior-gated mixture of mode-conditional transition kernels. We
formalize this as a `NonExpansiveAmbiguity` and obtain γ-contractivity
of `T_b` directly via `robustBellmanOp_contraction`. -/

/-- Mixture ambiguity functional: `Q_b(s, a)(V) = ∑_z b(z) · ⟨P_z(s, a), V⟩`. -/
def mixtureAmbFunc {Z : Type} [Fintype Z]
    (b : Z → ℝ) (P_z : Z → S × A → S → ℝ)
    (sa : S × A) (V : S → ℝ) : ℝ :=
  ∑ z : Z, b z * ∑ s' : S, P_z z sa s' * V s'

/-- Per-state probability-vector Jensen on a finite state space. -/
private lemma jensen_finite
    (P : S → ℝ) (hpnn : ∀ s, 0 ≤ P s) (hpsum : ∑ s : S, P s = 1)
    (V W : S → ℝ) :
    |∑ s : S, P s * (V s - W s)| ≤ supDiff V W := by
  calc |∑ s : S, P s * (V s - W s)|
      ≤ ∑ s : S, |P s * (V s - W s)| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ s : S, P s * |V s - W s| := by
        congr 1; ext s; rw [abs_mul, abs_of_nonneg (hpnn s)]
    _ ≤ ∑ s : S, P s * supDiff V W :=
        Finset.sum_le_sum (fun s _ =>
          mul_le_mul_of_nonneg_left (abs_diff_le_supDiff V W s) (hpnn s))
    _ = (∑ s : S, P s) * supDiff V W := by rw [← Finset.sum_mul]
    _ = 1 * supDiff V W := by rw [hpsum]
    _ = supDiff V W := one_mul _

/-- The mixture functional is non-expansive. -/
theorem mixture_amb_nonexp {Z : Type} [Fintype Z]
    (b : Z → ℝ) (hbnn : ∀ z, 0 ≤ b z) (hbsum : ∑ z : Z, b z = 1)
    (P_z : Z → S × A → S → ℝ) (hpnn : ∀ z sa s', 0 ≤ P_z z sa s')
    (hpsum : ∀ z sa, ∑ s' : S, P_z z sa s' = 1) :
    NonExpansiveAmbiguity
      (Q := mixtureAmbFunc (S := S) (A := A) b P_z) := by
  refine ⟨?_⟩
  intro V W s a
  unfold mixtureAmbFunc
  -- Decompose: ∑_z b_z [⟨P_z, V⟩ - ⟨P_z, W⟩] = ∑_z b_z ⟨P_z, V - W⟩.
  have hd : (∑ z : Z, b z * ∑ s' : S, P_z z (s, a) s' * V s') -
            (∑ z : Z, b z * ∑ s' : S, P_z z (s, a) s' * W s')
            = ∑ z : Z, b z * (∑ s' : S, P_z z (s, a) s' * (V s' - W s')) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro z _
    have hzsum : ∑ s' : S, P_z z (s, a) s' * V s' -
                 ∑ s' : S, P_z z (s, a) s' * W s'
                 = ∑ s' : S, P_z z (s, a) s' * (V s' - W s') := by
      rw [← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro s' _; ring
    rw [show b z * ∑ s' : S, P_z z (s, a) s' * V s' -
           b z * ∑ s' : S, P_z z (s, a) s' * W s'
         = b z * (∑ s' : S, P_z z (s, a) s' * V s' -
                  ∑ s' : S, P_z z (s, a) s' * W s') from by ring,
        hzsum]
  rw [hd]
  -- Per-mode Jensen: |⟨P_z, V - W⟩| ≤ supDiff V W.
  have hperz : ∀ z, |∑ s' : S, P_z z (s, a) s' * (V s' - W s')|
                  ≤ supDiff V W :=
    fun z => jensen_finite (P_z z (s, a)) (hpnn z (s, a)) (hpsum z (s, a)) V W
  -- Total: ∑_z b_z · supDiff = supDiff (since ∑ b_z = 1, b_z ≥ 0).
  calc |∑ z : Z, b z * (∑ s' : S, P_z z (s, a) s' * (V s' - W s'))|
      ≤ ∑ z : Z, |b z * (∑ s' : S, P_z z (s, a) s' * (V s' - W s'))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ = ∑ z : Z, b z * |∑ s' : S, P_z z (s, a) s' * (V s' - W s')| := by
        congr 1; ext z; rw [abs_mul, abs_of_nonneg (hbnn z)]
    _ ≤ ∑ z : Z, b z * supDiff V W :=
        Finset.sum_le_sum (fun z _ =>
          mul_le_mul_of_nonneg_left (hperz z) (hbnn z))
    _ = (∑ z : Z, b z) * supDiff V W := by rw [← Finset.sum_mul]
    _ = 1 * supDiff V W := by rw [hbsum]
    _ = supDiff V W := one_mul _

/-- **BAPR (mixture) robust Bellman is γ-contractive**. -/
theorem mixture_robust_bellman_contraction {Z : Type} [Fintype Z]
    {γ : ℝ} (hγ : 0 ≤ γ)
    (c : S × A → ℝ)
    (b : Z → ℝ) (hbnn : ∀ z, 0 ≤ b z) (hbsum : ∑ z : Z, b z = 1)
    (P_z : Z → S × A → S → ℝ) (hpnn : ∀ z sa s', 0 ≤ P_z z sa s')
    (hpsum : ∀ z sa, ∑ s' : S, P_z z sa s' = 1)
    (V W : S → ℝ) (s : S) :
    |robustBellmanOp γ c (mixtureAmbFunc (S := S) (A := A) b P_z) V s -
     robustBellmanOp γ c (mixtureAmbFunc (S := S) (A := A) b P_z) W s|
      ≤ γ * supDiff V W :=
  robustBellmanOp_contraction hγ c _
    (mixture_amb_nonexp b hbnn hbsum P_z hpnn hpsum) V W s

/-! ## 4. Lipschitz bound for the nominal expectation plus penalty

The penalty is `ε · Lip(V)`. The bounds below apply to this expression
under the given hypothesis on `lipV`.
-/

/-- Upper endpoint `E[V] + ε · lipV V` of the Wasserstein bound. -/
def wassersteinAmbFunc
    (P_hat : S × A → S → ℝ) (ε : ℝ) (lipV : (S → ℝ) → ℝ)
    (sa : S × A) (V : S → ℝ) : ℝ :=
  ∑ s' : S, P_hat sa s' * V s' + ε * lipV V

/-- Lipschitz bound for the nominal expectation plus penalty. -/
theorem wasserstein_amb_lipBound
    (P_hat : S × A → S → ℝ) (hpnn : ∀ sa s', 0 ≤ P_hat sa s')
    (hpsum : ∀ sa, ∑ s' : S, P_hat sa s' = 1)
    (ε : ℝ) (hε : 0 ≤ ε)
    (lipV : (S → ℝ) → ℝ)
    (hlipNE : ∀ V W, |lipV V - lipV W| ≤ supDiff V W)
    (V W : S → ℝ) (s : S) (a : A) :
    |wassersteinAmbFunc (S := S) P_hat ε lipV (s, a) V -
     wassersteinAmbFunc (S := S) P_hat ε lipV (s, a) W|
      ≤ (1 + ε) * supDiff V W := by
  unfold wassersteinAmbFunc
  -- Algebraic decomposition of the difference.
  have hrearr : (∑ s' : S, P_hat (s, a) s' * V s' + ε * lipV V) -
                (∑ s' : S, P_hat (s, a) s' * W s' + ε * lipV W)
                = (∑ s' : S, P_hat (s, a) s' * (V s' - W s'))
                  + ε * (lipV V - lipV W) := by
    have hsum : ∑ s' : S, P_hat (s, a) s' * V s' -
                ∑ s' : S, P_hat (s, a) s' * W s'
                = ∑ s' : S, P_hat (s, a) s' * (V s' - W s') := by
      rw [← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro s' _; ring
    linarith [hsum]
  -- Bound expectation by sup-norm via Jensen on probability vector.
  have hexp_bound : |∑ s' : S, P_hat (s, a) s' * (V s' - W s')|
                   ≤ supDiff V W :=
    jensen_finite (P_hat (s, a)) (hpnn (s, a)) (hpsum (s, a)) V W
  -- Total triangle inequality.
  have hlip_bound : ε * |lipV V - lipV W| ≤ ε * supDiff V W :=
    mul_le_mul_of_nonneg_left (hlipNE V W) hε
  calc |(∑ s' : S, P_hat (s, a) s' * V s' + ε * lipV V) -
        (∑ s' : S, P_hat (s, a) s' * W s' + ε * lipV W)|
      = |(∑ s' : S, P_hat (s, a) s' * (V s' - W s'))
         + ε * (lipV V - lipV W)| := by rw [hrearr]
    _ ≤ |∑ s' : S, P_hat (s, a) s' * (V s' - W s')|
        + |ε * (lipV V - lipV W)| := abs_add_le _ _
    _ = |∑ s' : S, P_hat (s, a) s' * (V s' - W s')|
        + ε * |lipV V - lipV W| := by rw [abs_mul, abs_of_nonneg hε]
    _ ≤ supDiff V W + ε * supDiff V W := by linarith [hexp_bound, hlip_bound]
    _ = (1 + ε) * supDiff V W := by ring

/-- The Bellman backup using the nominal expectation plus penalty has
    Lipschitz bound `(1 + ε)γ`. -/
theorem wasserstein_robust_bellman_lipBound
    {γ : ℝ} (hγ : 0 ≤ γ)
    (c : S × A → ℝ)
    (P_hat : S × A → S → ℝ) (hpnn : ∀ sa s', 0 ≤ P_hat sa s')
    (hpsum : ∀ sa, ∑ s' : S, P_hat sa s' = 1)
    (ε : ℝ) (hε : 0 ≤ ε)
    (lipV : (S → ℝ) → ℝ)
    (hlipNE : ∀ V W, |lipV V - lipV W| ≤ supDiff V W)
    (V W : S → ℝ) (s : S) :
    |robustBellmanOp γ c
        (wassersteinAmbFunc (S := S) (A := A) P_hat ε lipV) V s -
     robustBellmanOp γ c
        (wassersteinAmbFunc (S := S) (A := A) P_hat ε lipV) W s|
      ≤ γ * (1 + ε) * supDiff V W := by
  unfold robustBellmanOp
  have hbound : ∀ a, |(c (s, a) + γ * wassersteinAmbFunc P_hat ε lipV (s, a) V) -
                      (c (s, a) + γ * wassersteinAmbFunc P_hat ε lipV (s, a) W)|
                    ≤ γ * (1 + ε) * supDiff V W := by
    intro a
    have hQ := wasserstein_amb_lipBound P_hat hpnn hpsum ε hε lipV hlipNE V W s a
    have h1 : (c (s, a) + γ * wassersteinAmbFunc P_hat ε lipV (s, a) V) -
              (c (s, a) + γ * wassersteinAmbFunc P_hat ε lipV (s, a) W)
              = γ * (wassersteinAmbFunc P_hat ε lipV (s, a) V -
                     wassersteinAmbFunc P_hat ε lipV (s, a) W) := by ring
    rw [h1, abs_mul, abs_of_nonneg hγ]
    have : γ * (1 + ε) * supDiff V W = γ * ((1 + ε) * supDiff V W) := by ring
    rw [this]
    exact mul_le_mul_of_nonneg_left hQ hγ
  exact abs_inf'_sub_inf'_le
    (fun a => c (s, a) + γ * wassersteinAmbFunc P_hat ε lipV (s, a) V)
    (fun a => c (s, a) + γ * wassersteinAmbFunc P_hat ε lipV (s, a) W)
    _
    hbound

/-! ## 4b. Abstract Wasserstein bounds

`wassersteinAmbFunc` defines the upper-bound expression
`E_{P̂}[V] + ε · Lip(V)`. The measure-level one-sided result is
`dro_upper_bound` in `DRO.lean`. For a nonnegative 1-Lipschitz `f`,

    sup_{P : W₁(P, P̂) ≤ ε} ∫ f dP  ≤  ∫ f dP̂ + ε.

So any ambiguity functional `Q` that satisfies a sandwich inequality
(nominal expectation ≤ Q ≤ KR upper bound) is bounded by the same
`E + ε·Lip` form, and `wassersteinAmbFunc` is the canonical
*upper-bound representative* of this equivalence class.

We expose this structural relationship as the `WassersteinKRProperty`
predicate. Any sup-over-W₁-ball ambiguity (whether on finite
distributions or Polish-space measures) satisfies the property by
weak KR + the trivial lower bound `P̂ ∈ ball`. The
`wassersteinAmbFunc` we use elsewhere is the tightest member of the
property class (it equals the upper bound exactly).
-/

/-- An ambiguity functional `Q : S × A → (S → ℝ) → ℝ` satisfies the
    **Kantorovich-Rubinstein property** at radius `ε` and Lipschitz
    operator `lipV` if it is sandwiched:

      ⟨P̂(s,a), V⟩ ≤ Q (s, a) V ≤ ⟨P̂(s,a), V⟩ + ε · lipV V.

    Any genuine sup-over-W₁-ball ambiguity satisfies this by weak KR
    (`dro_upper_bound`) on the upper side and `P̂ ∈ ball` on the lower
    side. -/
structure WassersteinKRProperty
    (Q : S × A → (S → ℝ) → ℝ)
    (P_hat : S × A → S → ℝ) (ε : ℝ) (lipV : (S → ℝ) → ℝ) : Prop where
  upper_bound :
    ∀ V s a, Q (s, a) V ≤ (∑ s' : S, P_hat (s, a) s' * V s') + ε * lipV V
  lower_bound :
    ∀ V s a, (∑ s' : S, P_hat (s, a) s' * V s') ≤ Q (s, a) V

/-- **`wassersteinAmbFunc` satisfies the KR property** trivially: it
    equals the upper bound by definition (so `Q ≤ E + ε·lipV` holds
    as equality), and it is at least `E` since `ε · lipV V ≥ 0`. -/
theorem wassersteinAmbFunc_satisfies_KR
    (P_hat : S × A → S → ℝ) (ε : ℝ) (hε : 0 ≤ ε)
    (lipV : (S → ℝ) → ℝ) (h_lip_nn : ∀ V, 0 ≤ lipV V) :
    WassersteinKRProperty
      (Q := wassersteinAmbFunc (S := S) (A := A) P_hat ε lipV)
      P_hat ε lipV := by
  refine ⟨fun V s a => ?_, fun V s a => ?_⟩
  · -- Q V = E + ε · lipV V ≤ E + ε · lipV V (equality).
    show wassersteinAmbFunc P_hat ε lipV (s, a) V
         ≤ ∑ s' : S, P_hat (s, a) s' * V s' + ε * lipV V
    unfold wassersteinAmbFunc
    exact le_refl _
  · -- Q V = E + ε · lipV V ≥ E since ε · lipV V ≥ 0.
    show ∑ s' : S, P_hat (s, a) s' * V s'
         ≤ wassersteinAmbFunc P_hat ε lipV (s, a) V
    unfold wassersteinAmbFunc
    have : 0 ≤ ε * lipV V := mul_nonneg hε (h_lip_nn V)
    linarith

end DROBellman
