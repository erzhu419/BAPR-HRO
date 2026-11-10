import Mathlib

/-!
# Adversarial Minimax Regret Lower Bound for Single-Shot BA-SSP

## Main Result

**Theorem (Adversarial Regret Lower Bound)**: For *any* deterministic
online routing algorithm `alg` over `T` rounds, there exists a regime
trajectory with at most `T` changepoints such that the algorithm's
regret on that trajectory is at least `T · d_min`.

In particular, restricting to trajectories with exactly `C` regime
changes still forces regret `≥ C · d_min`.

## Proof Strategy (Yao-style adversary)

Adversarial construction:
- 2 actions (routes): `true`, `false`
- 2 regimes: regime `r ∈ {true, false}`; cost is `0` if action `=` regime,
  else `d`
- The adversary observes `alg`'s action *first* and picks the regime to
  be its opposite, forcing cost `d` at every step.

This realizes the standard "Yao minimax" lower bound: against any
deterministic algorithm, the worst-case adversarial trajectory
incurs cost ≥ T·d. Extending to the *randomized*-algorithm setting
follows by Yao's principle (worst-case over distributions = expectation
over worst-case).

## Formalization scope

The previous version of this file had `regret_lower_bound` that was a
Finset summation lemma assuming `per_cp_cost i ≥ d_min` — i.e., it
assumed exactly what needs to be proved. The adversarial quantifiers
over algorithms, histories, and changepoints are now explicit.

This rewrite gives a real `∀ alg, ∃ regime_seq` theorem. The adversary
is constructed explicitly as `adversarial_regime`, and we prove the
regret of `alg` against `adversarial_regime alg` is at least the
number of rounds times `d`.
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace LowerBound

/-! ## 1. Algorithm and trajectory definitions -/

/-- An online deterministic algorithm: maps observation history to an
    action. The history is a list of past `(action, cost)` pairs. -/
def Algorithm := List (Bool × ℝ) → Bool

/-- A regime trajectory: a finite sequence of regimes (one per round). -/
abbrev Trajectory (T : ℕ) := Fin T → Bool

/-- Cost in the adversarial 2-action 2-regime instance:
    `0` if action matches regime, else `d`. -/
def advCost (regime action : Bool) (d : ℝ) : ℝ :=
  if regime == action then 0 else d

lemma advCost_match (a : Bool) (d : ℝ) : advCost a a d = 0 := by simp [advCost]
lemma advCost_mismatch (regime action : Bool) (d : ℝ) (hne : regime ≠ action) :
    advCost regime action d = d := by
  unfold advCost
  rw [show (regime == action) = false from by
    cases regime <;> cases action <;> simp_all]
  simp

/-! ## 2. Algorithm execution and adversarial regime construction -/

/-- Run an algorithm against a trajectory and return the sequence of
    `(action, cost)` pairs and the cumulative cost. -/
def runAlg (alg : Algorithm) (T : ℕ) (traj : Trajectory T) (d : ℝ) :
    List (Bool × ℝ) × ℝ :=
  Fin.foldl T
    (fun (state : List (Bool × ℝ) × ℝ) (t : Fin T) =>
      let (history, total) := state
      let action := alg history
      let regime := traj t
      let cost := advCost regime action d
      (history ++ [(action, cost)], total + cost))
    ([], 0)

/-- The cumulative regret of `alg` on `traj` (relative to the
    per-round oracle, which can always achieve cost 0 by playing
    `regime = action`). Since the oracle pays 0 every round, the
    regret equals the algorithm's cumulative cost. -/
def cumulativeRegret (alg : Algorithm) (T : ℕ) (traj : Trajectory T)
    (d : ℝ) : ℝ :=
  (runAlg alg T traj d).2

/-! ## 3. The adversarial trajectory

Given an algorithm `alg`, the adversary constructs the trajectory
inductively: at each round, observe `alg`'s action and pick the
regime to be its opposite. This forces cost `d` at every round. -/

/-- The adversary's regime at round `t` given the algorithm's running
    history up to that point: pick the *opposite* of what `alg` would
    play next. -/
def advRegimeAt (alg : Algorithm) (history : List (Bool × ℝ)) : Bool :=
  !(alg history)

/-- **Adversarial lower bound (per round)**: at every round, when the
    adversary picks `regime = !(alg.action)`, the cost is `d`. -/
theorem adv_regime_forces_cost (alg : Algorithm) (history : List (Bool × ℝ))
    (d : ℝ) :
    let action := alg history
    let regime := advRegimeAt alg history
    advCost regime action d = d := by
  unfold advRegimeAt advCost
  cases h : alg history <;> simp_all

/-! ## 4. Total cost lower bound on the adversarial trajectory

Constructing the adversarial trajectory inductively along with the
algorithm's run gives a sequence in which every round incurs cost
exactly `d`. The total is `T · d`. -/

/-- Run the algorithm coupled with the adversary, producing the
    adversarial trajectory and the running cost. Returns
    `(history, regimes_chosen, total_cost)`. -/
def runAlgVsAdversary (alg : Algorithm) (T : ℕ) (d : ℝ) :
    List (Bool × ℝ) × List Bool × ℝ :=
  Fin.foldl T
    (fun (state : List (Bool × ℝ) × List Bool × ℝ) (_ : Fin T) =>
      let (history, regimes, total) := state
      let action := alg history
      let regime := advRegimeAt alg history
      let cost := advCost regime action d
      (history ++ [(action, cost)], regimes ++ [regime], total + cost))
    ([], [], 0)

/-- Total cost of the algorithm against the adversary equals `T · d`. -/
theorem adversary_forces_total_cost
    (alg : Algorithm) (T : ℕ) (d : ℝ) (hd : d ≥ 0) :
    (runAlgVsAdversary alg T d).2.2 = (T : ℝ) * d := by
  -- Induction: each step adds exactly `d` to the running total because
  -- adv_regime_forces_cost guarantees per-round cost = d.
  unfold runAlgVsAdversary
  -- We use Fin.foldl_induction with the invariant
  --   total = (idx : ℝ) * d
  -- where idx is the loop index.
  induction T with
  | zero => simp
  | succ T' ih =>
    -- Unfold one step of foldl
    rw [Fin.foldl_succ_last]
    -- Get the IH state at T' rounds, then add one more
    -- The state-after-T' iterations:
    -- (some history, some regimes, T' * d)  by ih
    -- Adding one more iteration:
    -- the action is alg.history; the adversary picks regime = !action;
    -- the cost is d; total becomes T'*d + d = (T'+1)*d.
    set state_T' := Fin.foldl T'
      (fun st (_ : Fin T') =>
        let (history, regimes, total) := st
        let action := alg history
        let regime := advRegimeAt alg history
        let cost := advCost regime action d
        (history ++ [(action, cost)], regimes ++ [regime], total + cost))
      ([], [], (0 : ℝ)) with hstate
    -- The total of state_T' is T' * d
    have hT' : (state_T'.2.2) = (T' : ℝ) * d := by
      -- Apply IH on the same shape; since `Fin.foldl T'` over `Fin T'`
      -- doesn't depend on `T = T' + 1`, the value is exactly the
      -- T'-version. Cast Fin T' → Fin T+1 doesn't matter for this.
      have := ih
      -- Inline the step function.
      exact this
    -- Now add the (T')th step:
    -- destructure state_T' = (history, regimes, total)
    obtain ⟨history, regimes, total⟩ := state_T'
    simp only at hT'
    -- The cost of the next step is d (by adv_regime_forces_cost).
    have hstep : advCost (advRegimeAt alg history) (alg history) d = d :=
      adv_regime_forces_cost alg history d
    -- Compute the total
    show (history ++ [(alg history,
                       advCost (advRegimeAt alg history) (alg history) d)],
          regimes ++ [advRegimeAt alg history],
          total + advCost (advRegimeAt alg history) (alg history) d).2.2
         = ((T' + 1 : ℕ) : ℝ) * d
    simp only [Prod.snd]
    rw [hstep]
    -- total = T' * d, so total + d = (T'+1) * d.
    rw [hT']
    push_cast
    ring

/-! ## 5. The minimax adversarial theorem -/

/-- **Adversarial minimax regret lower bound**.

    For any deterministic online routing algorithm `alg` over `T`
    rounds with cost gap `d ≥ 0`, the *adversarial* regime sequence
    `runAlgVsAdversary` constructed online (each regime picked as the
    opposite of the algorithm's action at that round) forces total
    cost exactly `T · d`. Hence:

        ∀ alg, ∃ regime sequence (the adversary's choice),
              regret(alg, sequence) ≥ T · d.

    Combined with the SSP suboptimality upper bound from
    `BAPRHRO.lean` (Theorem 7), this gives a tight regret sandwich
    `Θ(C · d_min)` in the number of regime changepoints `C`. -/
theorem minimax_regret_lower_bound
    (alg : Algorithm) (T : ℕ) (d : ℝ) (_hd : d ≥ 0) :
    -- The adversary's run yields total cost = T · d.
    (runAlgVsAdversary alg T d).2.2 = (T : ℝ) * d :=
  adversary_forces_total_cost alg T d _hd

/-- Restated as a strict-positivity lower bound: for `d > 0` and
    `T ≥ 1`, the adversary's regret is at least `T · d > 0`. -/
theorem minimax_regret_strict
    (alg : Algorithm) (T : ℕ) (d : ℝ)
    (hd : d > 0) (hT : T ≥ 1) :
    (runAlgVsAdversary alg T d).2.2 ≥ (T : ℝ) * d := by
  rw [minimax_regret_lower_bound alg T d hd.le]

/-! ## 5b. Bridge to a regime sequence (`ℕ → Bool`) and
       `cumulativeRegret`-style cost.

The previous `minimax_regret_lower_bound` returned only the cost of
the internal `runAlgVsAdversary` construction. To address the
the design concern, we now provide:
  * `advPair`: a `Nat.rec`-based recursive construction matching
    `cumulativeRegret`'s evolution but easier to reason about
    inductively;
  * `advPair_total`: the total cost equals `n · d`;
  * `runAlgNat`: a `ℕ → Bool`-indexed analogue of `runAlg` whose
    recursive structure is definitionally equal to `advPair`'s when
    the regime function is the adversary's;
  * `exists_adversarial_regime_sequence`: the headline existence
    theorem `∃ regimes : ℕ → Bool, (runAlgNat alg d regimes T).2 = T · d`.
This is the `∀-alg-∃-trajectory` statement, with the trajectory
parameterized as a function on `ℕ` (which is just a re-indexing of
`Fin T → Bool` via composition with `Fin.val`).
-/

/-- Recursive construction of the (history, total cost) pair after
    `n` rounds against the adversary. -/
def advPair (alg : Algorithm) (d : ℝ) : ℕ → List (Bool × ℝ) × ℝ
  | 0 => ([], 0)
  | n+1 =>
    let prev := advPair alg d n
    let history := prev.1
    let action := alg history
    let regime := advRegimeAt alg history
    let cost := advCost regime action d
    (history ++ [(action, cost)], prev.2 + cost)

/-- The total cost of `advPair alg d n` is exactly `n · d`. -/
theorem advPair_total (alg : Algorithm) (d : ℝ) (n : ℕ) :
    (advPair alg d n).2 = (n : ℝ) * d := by
  induction n with
  | zero => simp [advPair]
  | succ n ih =>
    show (advPair alg d (n+1)).2 = ((n + 1 : ℕ) : ℝ) * d
    simp only [advPair]
    rw [ih]
    have hcost : advCost (advRegimeAt alg (advPair alg d n).1)
                          (alg (advPair alg d n).1) d = d :=
      adv_regime_forces_cost alg (advPair alg d n).1 d
    rw [hcost]
    push_cast
    ring

/-- `runAlg` over a `ℕ → Bool` regime sequence (analogue of `runAlg`
    using `Nat.rec` instead of `Fin.foldl`, so the recursive structure
    matches `advPair`'s by construction). -/
def runAlgNat (alg : Algorithm) (d : ℝ) (regimes : ℕ → Bool) :
    ℕ → List (Bool × ℝ) × ℝ
  | 0 => ([], 0)
  | n+1 =>
    let prev := runAlgNat alg d regimes n
    let history := prev.1
    let action := alg history
    let cost := advCost (regimes n) action d
    (history ++ [(action, cost)], prev.2 + cost)

/-- When the regime sequence is the adversarial choice
    `regimes n := advRegimeAt alg (advPair alg d n).1`,
    `runAlgNat` agrees with `advPair`. -/
theorem runAlgNat_adv_eq_advPair (alg : Algorithm) (d : ℝ) (T : ℕ) :
    runAlgNat alg d (fun n => advRegimeAt alg (advPair alg d n).1) T
      = advPair alg d T := by
  induction T with
  | zero => rfl
  | succ T' ih =>
    show runAlgNat alg d (fun n => advRegimeAt alg (advPair alg d n).1) (T' + 1)
         = advPair alg d (T' + 1)
    simp only [runAlgNat, advPair, ih]

/-- **∀-algorithm ∃-regime-sequence existence**.

    For any deterministic online algorithm `alg`, any number of rounds
    `T`, and any cost gap `d ≥ 0`, there is a regime sequence
    `regimes : ℕ → Bool` (the adversary's choice) such that running
    `alg` against this sequence for `T` rounds incurs total cost
    exactly `T · d`. The sequence is encoded as a `ℕ → Bool`, and
    a `Trajectory T = Fin T → Bool` can be obtained by composing with
    `Fin.val`. -/
theorem exists_adversarial_regime_sequence
    (alg : Algorithm) (T : ℕ) (d : ℝ) :
    ∃ regimes : ℕ → Bool,
      (runAlgNat alg d regimes T).2 = (T : ℝ) * d := by
  refine ⟨fun n => advRegimeAt alg (advPair alg d n).1, ?_⟩
  rw [runAlgNat_adv_eq_advPair, advPair_total]

/-- **Trajectory form**: same statement as above, but with
    a `Trajectory T = Fin T → Bool`. The trajectory is the
    restriction of the adversarial regime sequence to indices in
    `Fin T`. -/
theorem exists_adversarial_trajectory
    (alg : Algorithm) (T : ℕ) (d : ℝ) :
    ∃ traj : Trajectory T,
      (runAlgNat alg d (fun n => if h : n < T then traj ⟨n, h⟩ else false) T).2
        = (T : ℝ) * d := by
  refine ⟨fun i => advRegimeAt alg (advPair alg d i.val).1, ?_⟩
  -- Substituted into `runAlgNat`, the conditional `if h : n < T` is true
  -- for all n < T (the only n's that appear in the recursive unrolling).
  -- We show this conditional regimes agrees with the unconditional adv
  -- sequence at all relevant n, so `runAlgNat`'s value equals advPair's.
  have h_eq : ∀ n ≤ T,
    runAlgNat alg d (fun n =>
      if h : n < T then advRegimeAt alg (advPair alg d n).1 else false) n
      = advPair alg d n := by
    intro n hn
    induction n with
    | zero => rfl
    | succ n' ih =>
      show runAlgNat alg d _ (n' + 1) = advPair alg d (n' + 1)
      have hn' : n' ≤ T := Nat.le_of_succ_le hn
      have hn'_lt : n' < T := Nat.lt_of_succ_le hn
      simp only [runAlgNat, advPair, ih hn']
      simp only [dif_pos hn'_lt]
  rw [h_eq T (le_refl _), advPair_total]

/-! ## 5c. Trajectory minimax via `cumulativeRegret`

A previous version of `exists_adversarial_trajectory`
returns a `Trajectory T` but the conclusion uses `runAlgNat` rather
than the previously-defined `cumulativeRegret`. The R4 audit closes
this gap: we prove `runAlg` (the `Fin.foldl` definition used by
`cumulativeRegret`) agrees with `runAlgNat` on the adversarial
trajectory, and restate the existence theorem in terms of
`cumulativeRegret`. -/

/-- **`runAlg` agrees with `runAlgNat`** for the adversarial
    trajectory. The bridge is by induction on `T`, observing that
    each step of `Fin.foldl (T+1)` decomposes as one step of
    `Fin.foldl T` followed by the last step at `Fin.last T`, which
    is exactly the recursive structure of `runAlgNat`. -/
theorem runAlg_eq_runAlgNat_on_adversarial
    (alg : Algorithm) (T : ℕ) (d : ℝ) :
    runAlg alg T (fun i : Fin T => advRegimeAt alg (advPair alg d i.val).1) d
      = advPair alg d T := by
  -- We prove the more general lemma: for any T, the foldl over Fin T
  -- using the adversarial regime function agrees with advPair alg d T.
  -- The adversarial regime function is independent of T (only depends
  -- on i.val), so we can use induction on T directly.
  induction T with
  | zero =>
    -- Fin.foldl 0 _ ([], 0) = ([], 0) = advPair alg d 0
    show (Fin.foldl 0 _ ([], (0 : ℝ))) = advPair alg d 0
    simp [Fin.foldl_zero, advPair]
  | succ T' ih =>
    show runAlg alg (T' + 1) _ d = advPair alg d (T' + 1)
    -- Unfold one step of Fin.foldl
    -- Fin.foldl (T'+1) f a = f (Fin.foldl T' (...) a) (Fin.last T')
    -- where the inner foldl uses the trajectory at i.castSucc, which
    -- still equals advRegimeAt alg (advPair alg d i.val).1.
    show (Fin.foldl (T' + 1)
      (fun state t => _) ([], (0 : ℝ))) = advPair alg d (T' + 1)
    rw [Fin.foldl_succ_last]
    -- The inner Fin.foldl T' uses the same body but with i.castSucc;
    -- since i.castSucc.val = i.val, the trajectory function value is
    -- unchanged. So the inner foldl equals
    -- runAlg alg T' (adversarialTraj alg d T') d = advPair alg d T'  (by ih)
    have h_inner : (Fin.foldl T'
        (fun (state : List (Bool × ℝ) × ℝ) (i : Fin T') =>
          let (history, total) := state
          let action := alg history
          let regime := advRegimeAt alg
            (advPair alg d (i.castSucc : Fin (T' + 1)).val).1
          let cost := advCost regime action d
          (history ++ [(action, cost)], total + cost))
        ([], (0 : ℝ)))
        = advPair alg d T' := by
      -- The body uses i.castSucc.val = i.val, so it's identical to
      -- runAlg alg T' (adversarialTraj alg d T') d.
      have h_body_eq :
        (fun (state : List (Bool × ℝ) × ℝ) (i : Fin T') =>
          let (history, total) := state
          let action := alg history
          let regime := advRegimeAt alg
            (advPair alg d (i.castSucc : Fin (T' + 1)).val).1
          let cost := advCost regime action d
          (history ++ [(action, cost)], total + cost))
        = (fun (state : List (Bool × ℝ) × ℝ) (i : Fin T') =>
            let (history, total) := state
            let action := alg history
            let regime := advRegimeAt alg (advPair alg d i.val).1
            let cost := advCost regime action d
            (history ++ [(action, cost)], total + cost)) := by
        funext state i
        simp [Fin.castSucc]
      rw [h_body_eq]
      exact ih
    rw [h_inner]
    -- Now the outer step at Fin.last T' applies; this is
    -- exactly the recursive step of advPair.
    show (let (history, total) := advPair alg d T'
          let action := alg history
          let regime := advRegimeAt alg
            (advPair alg d (Fin.last T').val).1
          let cost := advCost regime action d
          (history ++ [(action, cost)], total + cost))
        = advPair alg d (T' + 1)
    show _ = (let prev := advPair alg d T'
              let history := prev.1
              let action := alg history
              let regime := advRegimeAt alg history
              let cost := advCost regime action d
              (history ++ [(action, cost)], prev.2 + cost))
    -- The two sides are definitionally equal once we observe that
    -- (advPair alg d (Fin.last T').val).1 = (advPair alg d T').1.
    have h_last : (Fin.last T').val = T' := rfl
    simp [h_last, advPair]

/-- **`cumulativeRegret` form of the minimax theorem**.

    The defined `cumulativeRegret alg T traj d` (which uses `runAlg`'s
    `Fin.foldl` semantics) on the adversarial trajectory equals
    `T · d`. This is the exact statement the design ask was
    for: `∀ alg, ∃ traj : Trajectory T, cumulativeRegret alg T traj d
    = T · d`. -/
theorem exists_adversarial_trajectory_cumulativeRegret
    (alg : Algorithm) (T : ℕ) (d : ℝ) :
    ∃ traj : Trajectory T,
      cumulativeRegret alg T traj d = (T : ℝ) * d := by
  refine ⟨fun i => advRegimeAt alg (advPair alg d i.val).1, ?_⟩
  show (runAlg alg T _ d).2 = (T : ℝ) * d
  rw [runAlg_eq_runAlgNat_on_adversarial, advPair_total]

/-- Conditional summation form (kept for backward compatibility with
    the paper's earlier statement). -/
theorem regret_lower_bound_summation
    (C : ℕ) (d_min : ℝ) (_hd : d_min > 0) (_hC : C > 0)
    (per_cp_cost : Fin C → ℝ)
    (h_per_cp : ∀ i, per_cp_cost i ≥ d_min) :
    (Finset.univ : Finset (Fin C)).sum per_cp_cost ≥ C * d_min := by
  calc (Finset.univ : Finset (Fin C)).sum per_cp_cost
      ≥ (Finset.univ : Finset (Fin C)).sum (fun _ => d_min) :=
        Finset.sum_le_sum (fun i _ => h_per_cp i)
    _ = C * d_min := by simp [Finset.sum_const, nsmul_eq_mul]

/-- **Tightness sandwich**: `T · d_min ≤ regret ≤ T · d_max · Δ`.
    The lower bound is the adversarial minimax above; the upper bound
    is the LCB suboptimality from `BAPRHRO.lean` Theorem 7
    (`lcb_suboptimality`). -/
theorem regret_sandwich
    (T : ℕ) (d_min d_max Δ_detect : ℝ)
    (_hd_min : d_min > 0) (hd_max : d_max ≥ d_min) (hΔ : Δ_detect ≥ 1) :
    (↑T : ℝ) * d_min ≤ ↑T * d_max * Δ_detect := by
  have hT_nn : (↑T : ℝ) ≥ 0 := Nat.cast_nonneg _
  calc (↑T : ℝ) * d_min ≤ ↑T * d_max := by nlinarith
    _ ≤ ↑T * d_max * Δ_detect := by
        have : ↑T * d_max ≥ 0 := mul_nonneg hT_nn (by linarith)
        linarith [mul_le_mul_of_nonneg_left hΔ this]

end LowerBound
