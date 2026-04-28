## Overall re-evaluation

The revision is **materially better than round 1**, especially on the explicit Wasserstein translation coupling and the generic robust Bellman abstraction. However, many high-level claims are still **not what the Lean theorems actually state**. Several new “fixes” are theorem-shaped but prove only nonnegativity, definitions, or witness fragments rather than the advertised OR results.

**New score: 4/10**  
**Verdict: NOT READY for Operations Research**

---

## (1) Round-1 weaknesses: status and evidence

| # | Status | Evidence |
|---|---|---|
| **1. “LCB = Wasserstein DRO, tight” theorem not proved** | **PARTIALLY FIXED** | `DRO.lean` now genuinely proves the missing witness distance: `w1_translation_le` constructs the swapped translation coupling `(T,id)#Phat` and computes cost `ε` to show `W₁ (Phat.map T) Phat ≤ ofReal ε` around §4b / lines ~400–465. But `lcb_equals_dro_attained` only returns **witness feasibility + value shift**, not an actual `sup/iSup` equality; it does not formally state `sup_{W₁≤ε} E_P[f] = E_Phat[f]+ε`. |
| **2. No real robust-MDP/SSP theorem** | **PARTIALLY FIXED** | `DROBellman.lean` is a substantial improvement: `robustBellmanOp`, `NonExpansiveAmbiguity`, and `robustBellmanOp_contraction` prove a real finite-action min-backup nonexpansiveness result, and `mixture_amb_nonexp` is a genuine finite Jensen argument. But this is still a discounted finite MDP theorem over an **abstract Q**, not a formal robust SSP/hyperpath theorem with rectangular Wasserstein ambiguity sets and `Q(s,a)V = sup_{p∈U(s,a)} pᵀV`. `wassersteinAmbFunc` postulates the dual form `E + ε·lipV`; it is not derived from a Wasserstein ball. |
| **3. Explicit `axiom stationary_exp3_bound`** | **FIXED narrowly** | The explicit axiom is gone from `EXP3Regret.lean`; there is no visible `axiom` or `sorry`. This is a genuine syntactic fix. However, it is not replaced by a true EXP3 regret proof; see #7. |
| **4. Ensemble = empirical DRO only proved mean/std translation** | **PARTIALLY FIXED** | `BAPRHRO_V2.lean` now defines an actual atomic `empiricalMeasure` and proves `ensemble_W1_translation_le`, but despite the name this theorem proves only the **push-forward identity** `empiricalMeasure shifted = (empiricalMeasure μs).map (·+ε)`. It does **not** prove `W₁ ≤ ε`, does not prove the empirical measure is a probability measure, and does not define or prove a DRO supremum equality. |
| **5. Posterior contraction / Bayes-risk link weak** | **NOT FIXED** | The new `chebyshev_posterior_numeric` is just algebra from the assumed premise `prob * t^2 ≤ variance`; it does not prove Chebyshev from a posterior distribution. Worse, `bayes_risk_per_step_gap` proves only `(K : ℝ)/k^2 ≥ 0 ∧ 2(1+β)kσ ≥ 0`; it does not call `per_step_gap`, does not quantify over nominal/true costs, and does not derive the deterministic accuracy hypothesis from a Normal-Gamma posterior. This is essentially superficial. |
| **6. Minimax lower bound was only conditional summation** | **PARTIALLY FIXED** | `LowerBound.lean` is improved: `advRegimeAt`, `runAlgVsAdversary`, `adv_regime_forces_cost`, and `adversary_forces_total_cost` give a real adaptive adversary forcing cost `T·d`. But `minimax_regret_lower_bound` still does **not** state `∀ alg, ∃ traj, cumulativeRegret alg traj ≥ T d`; it returns equality for the internal adaptive run object. The declared `Trajectory T` and `cumulativeRegret` are not connected to the adversarial list. Randomized algorithms/Yao are also only in comments. |
| **7. EXP3 axiom not linked to actual algorithm** | **NOT FIXED** | `StationaryRegretCertificate` is just a record containing a scalar `bound` satisfying `bound ≤ C₀√(Kτ log K)`; it has no algorithm, no losses, no regret definition. `hedge_bound` is merely the expression `log K / η + ηT/2`; the file proves nonnegativity/monotonicity, not the potential-function regret inequality for Hedge. The claim that Hedge discharges the certificate is not formalized. |
| **8. Empirical evaluation limited** | **PARTIALLY FIXED** | The new multi-seed/bootstrap CI evaluation is a real improvement: 5 seeds × 30 journeys and 95% CIs are much better than round 1. But the disrupted-day results remain statistically weak: e.g., reach rates `Static 28% [12,46]` vs DRO `33% [15,53]` have heavily overlapping intervals, and conditional mean CIs also overlap. OD heterogeneity and timeout/censoring still limit conclusions. |
| **9. Complexity story underdeveloped** | **PARTIALLY FIXED** | `Architecture.lean` gives useful algebraic comparisons such as `precompute_dominates` and `cost_ratio_bound`. But this is still not a full complexity analysis of hyperpath generation, re-ranking, posterior updates, ensemble maintenance, EXP3/Hedge adaptation, or robust Bellman computation. |
| **10. Novelty overstated relative to standard DRO literature** | **FIXED** | The paper-side changes reportedly remove the “first formalization” and broad “tight equivalence” claims and now attribute Gao-Kleywegt, Esfahani-Kuhn, Blanchet-Murthy, and Villani. This is the right correction. Some Lean comments still overclaim, but the novelty framing is much more honest. |

---

## (2) New OR score and verdict

**Score: 4/10**

**Verdict: NOT READY**

The revision deserves credit for removing the explicit axiom and for adding some genuinely nontrivial Lean proofs. But the paper still presents several results as formally established when the Lean theorem statements prove much weaker objects. In particular, the Bayes-risk link, EXP3/Hedge regret, ensemble empirical-DRO equivalence, and minimax lower-bound quantifiers remain insufficient for an OR flagship submission.

---

## (3) Remaining critical weaknesses, ranked, with minimum fixes

### 1. Formal theorem statements still do not match the advertised claims

The biggest remaining problem is not lack of Lean lines; it is **specification mismatch**. Several comments say “DRO equality,” “Bayes-risk theorem,” “Hedge regret,” or “∀ algorithm ∃ trajectory,” but the actual theorem proves a witness fragment, nonnegativity, or an internal construction.

**Minimum fix:** For each headline claim, define the actual mathematical object and prove the actual statement. E.g., define

```lean
DROValue Phat ε f := ⨆ P with W₁ P Phat ≤ ε, ∫ f dP
```

and prove `DROValue = ∫f dPhat + ε`, rather than only witness feasibility.

---

### 2. Bayes-risk / posterior-contraction connection is still essentially absent

`bayes_risk_per_step_gap` and `bayes_risk_per_step_gap_expected` are not Bayes-risk theorems. They do not involve posterior probability, Normal-Gamma distributions, route-level union bounds as events, or expected regret.

**Minimum fix:** Formalize a posterior probability space for route means, prove or import Chebyshev as a measure-theoretic theorem, derive

```lean
P(∀ k, |nominal k - true k| ≤ kappa * sigma_post k) ≥ 1 - K/kappa^2
```

then call `per_step_gap` under that event, and separately prove an expected-gap bound.

---

### 3. EXP3/Hedge regret is still not formalized

The axiom is gone, but no actual EXP3 or Hedge algorithm is defined. `hedge_bound` is a formula, not a regret theorem. Also, the claim that transit feedback is full-information is not justified: a passenger observes the route taken, not necessarily the counterfactual losses of all β-grid arms.

**Minimum fix:** Either:

1. define Hedge with weights, full-information losses, learner distribution, and regret, then prove the potential bound; or  
2. define EXP3/EXP3-IX with bandit feedback and prove/cite the actual stationary regret theorem as an explicit external assumption, clearly not as a formalized result.

If using Hedge, the paper must justify full counterfactual loss observation.

---

### 4. Ensemble empirical-DRO theorem is still incomplete

The V2 file now has an empirical measure and a push-forward identity, which is progress. But `ensemble_W1_translation_le` does not prove `W₁ ≤ ε`; `ensemble_dro_witness_complete` does not invoke `Wasserstein.w1_translation_le`; and no empirical DRO supremum is defined.

**Minimum fix:** Prove:

1. `empiricalMeasure hK μs` is a probability measure.  
2. `W₁ (empiricalMeasure shifted) (empiricalMeasure μs) ≤ ofReal ε`.  
3. The expected identity cost under `empiricalMeasure μs` equals `ensemble_mean μs`.  
4. A formal `iSup`/DRO equality theorem.

---

### 5. Lower bound is still adaptive-run equality, not the advertised minimax theorem

`runAlgVsAdversary` is a good construction, but the final theorem does not produce a `Trajectory T` or connect to `cumulativeRegret`.

**Minimum fix:** Convert the adversarial regime list into a `Trajectory T`, prove its length is `T`, prove equivalence between `runAlg` on that trajectory and `runAlgVsAdversary`, then state:

```lean
∀ alg, ∃ traj : Trajectory T,
  cumulativeRegret alg T traj d ≥ T * d
```

If randomized algorithms are claimed, add a precise Yao-style theorem or remove the claim.

---

### 6. Robust Bellman theory is improved but still not a robust SSP/hyperpath theorem

The generic contraction theorem is useful, but Operations Research readers will expect a theorem over rectangular ambiguity sets or SSP/hyperpath structure, not only an abstract `Q`.

**Minimum fix:** Define rectangular ambiguity sets `U(s,a)`, define

```lean
Q(s,a,V) = sup_{p ∈ U(s,a)} ∑ s' p s' V s'
```

prove nonexpansiveness for probability ambiguity sets, and separately handle SSP properness / transience rather than only discounted `γ`.

---

### 7. Empirical evidence remains weak

The revised CIs are welcome, but the effect sizes are small and intervals are wide. The disrupted-day reach improvements are not statistically compelling, and timeout/censoring plus OD heterogeneity remain serious.

**Minimum fix:** Add paired analyses by OD/seed, a hierarchical or blocked bootstrap, explicit timeout/survival treatment, larger OD coverage, and report effect sizes with uncertainty rather than relying on aggregate means.

---

## (4) Specific praise

1. **The new translation W₁ witness is genuinely good.**  
   `w1_translation_le` in `DRO.lean` is exactly the kind of missing measure-theoretic detail I asked for in round 1: it constructs the swapped coupling, verifies marginals, and computes the transport cost. This is a real improvement.

2. **The robust Bellman rewrite is conceptually much cleaner.**  
   `NonExpansiveAmbiguity`, `robustBellmanOp_contraction`, and `mixture_amb_nonexp` give a reusable abstraction and a real finite-state contraction argument. Also, the authors are now honest that the Wasserstein penalty gives a `(1+ε)γ` Lipschitz bound rather than pretending it is automatically γ-contractive. This is a meaningful step forward.
