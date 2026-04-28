## (1) Per-issue R4 status

| R4 fix | Status | Assessment |
|---|---:|---|
| **#1 DRO iSup** | **GENUINELY FIXED** | `Wasserstein.dro_iSup_real_equality` now contains a literal `⨆` over measures, probability-measure witnesses, W₁-ball membership, and nonnegative-support hypotheses. This is no longer the R3 “existence + pointwise upper bound” surrogate. The proof structure is also the right one: `iSup_le` upper bound via `dro_upper_bound`, lower bound via translated witness `Phat.map (· + ε)` plus `w1_translation_le` and `translation_is_dro_witness_real`. Caveat: this is still an `ENNReal`/`∫⁻ ofReal` theorem for affine costs with support restrictions, not a fully general real-expectation DRO duality theorem. But for the audited claim, it is structurally correct. |
| **#5 posterior Chebyshev** | **GENUINELY FIXED, with integration caveat** | `BAPRHRO.posterior_chebyshev_route` now starts from an actual probability measure `π_r : Measure ℝ` with `[IsProbabilityMeasure π_r]` and derives a tail-probability bound from a second-moment/variance bound using measure-theoretic Markov/Chebyshev machinery. This directly addresses the R3 complaint that the probability was merely assumed. However, `posterior_union_bound` is still an algebraic sum bound over supplied `route_prob : Fin K → ℝ`; it does not construct a joint posterior measure or prove a finite-union event bound inside a probability space. Also, `bayes_risk_per_step_gap_high_prob` still takes `h_prob` as an assumption in the current source. So the route-level Chebyshev theorem is genuine; the full Bayes-risk pipeline is not yet fully integrated. |
| **#6 cumulativeRegret lower bound** | **GENUINELY FIXED** | `LowerBound.exists_adversarial_trajectory_cumulativeRegret` has the exact desired shape: `∃ traj : Trajectory T, cumulativeRegret alg T traj d = T * d`. This uses the defined `cumulativeRegret`, not the old `runAlgNat` surrogate. The bridge theorem `runAlg_eq_runAlgNat_on_adversarial` also addresses the semantic gap between the `Fin.foldl` execution and the recursive adversarial construction. This is a real specification repair. Caveat: still deterministic-adversary only; randomized/Yao claims remain informal. |
| **#7 Hedge per-round inequality** | **GENUINELY FIXED for the audited theorem** | `EXP3Regret.hedge_per_round_unnormalized` is now a real per-round potential inequality over actual Hedge weights, losses, and exponential updates. The supporting lemmas `exp_neg_le_inv_one_add`, `inv_one_add_le_quad`, and `exp_neg_le_one_minus_plus_sq` are substantive. This is no longer merely a formula identity. Caveat: the full cumulative Hedge regret theorem is still not proved. `hedge_bound` remains a closed-form expression, and `hedge_regret_bound_form` is essentially definitional. So the per-round audit target is fixed, but the paper should not yet claim a fully formalized Hedge regret proof unless the telescoping/log step is added. |

**Bottom line on the R4 audit:** the four flagged theorem statements are no longer just theorem-shaped fragments. They now contain the right core mathematical structures: literal `⨆`, defined `cumulativeRegret`, probability-measure Chebyshev, and real Hedge potential inequality.

---

## (2) Previously fixed issues from rounds 1–3: still fixed?

Yes, I do not see regressions in the items that were already genuinely repaired.

Representative confirmations:

1. **Robust Bellman / ambiguity operator repair**  
   Still genuinely fixed. `DROBellman.robustBellmanOp_contraction` uses the actual robust Bellman form  
   `min_a [c(s,a) + γ Q(s,a)V]`, and the proof goes through `NonExpansiveAmbiguity`, not the old “additive penalty cancels” trick.  
   `mixture_robust_bellman_contraction` remains a real finite-mixture nonexpansiveness theorem.  
   `wasserstein_robust_bellman_lipBound` honestly states the Wasserstein ambiguity is `(1+ε)γ`-Lipschitz rather than falsely contractive.

2. **Ensemble empirical W₁ witness**  
   Still fixed. `BAPRHRO_V2.ensemble_W1_translation_le`, `empiricalMeasure_isProbabilityMeasure`, `ensemble_W1_le_eps`, and `ensemble_dro_witness_complete` still give the missing empirical-measure translation/W₁ witness structure.

3. **EXP3 axiom removal / honest certificate framing**  
   Still fixed in the limited sense previously accepted. There is no longer an axiom pretending to prove stationary EXP3. The file uses `StationaryRegretCertificate` as an explicit hypothesis and then proves algebraic/piecewise consequences. That is honest scope. The remaining issue is not regression; it is that full EXP3 bandit regret is still outside the formal development.

4. **Irrecoverability sign logic**  
   Still fixed at the stylized-theory level. `Irrecoverability.optimal_beta_sign_matches_rho`, `optimal_beta_sign_matches_rho_neg`, and the bridge theorems in `IrrecoverabilityBridge.lean` still encode a coherent two-arm/parametric sign argument. It remains a model abstraction, not an empirical theorem about the four domains, but the formal statements are not fake.

5. **Architecture / computational-cost claims**  
   Still fixed as algebraic cost-comparison theorems. `Architecture.precompute_dominates`, `cost_ratio_lt_one`, `per_journey_cost_v1_linear_decomp`, and the component-wise cost lemmas remain concrete. These are simple, but they are real.

6. **BAPRHRO deterministic per-step and cumulative suboptimality skeleton**  
   Still fixed. `BAPRHRO.per_step_gap` and `BAPRHRO.lcb_suboptimality` remain meaningful deterministic lemmas. The newly added Chebyshev theorem improves the probabilistic link, although the full posterior-good-event-to-gap theorem is still not fully closed.

So: the prior genuine fixes are still intact. R4 did not break them.

---

## (3) New score and verdict

**Score: 7/10.**

**Verdict: NOT READY for OR acceptance yet, but the work has made a real jump.**

This is the first round where the main specification-audit complaint is substantially addressed. The four R4 target theorems now mostly state the actual mathematical objects the paper talks about. That is a meaningful improvement over the previous “theorem-shaped fragment” problem.

However, I would still not call the formalization OR-ready because several paper-level claims remain only locally formalized rather than end-to-end formalized.

---

## (4) Score trajectory: 3 → 4 → 5 → 7

This round is **not** just diminishing returns. The R4 changes directly target the biggest R3 criticism and mostly succeed.

My interpretation of the trajectory:

- **3/10:** early formalization had many placeholders / misaligned theorem statements.
- **4/10:** some structure appeared, but several theorems still encoded weaker or unrelated claims.
- **5/10:** many local lemmas were real, but the four most important claims were still not stated in the right form.
- **7/10 now:** the main audited claims have the right formal shape, and several are substantively proved.

That said, the next improvements will become harder. The easy gains from “state the theorem correctly” are mostly exhausted. The remaining work is integration: proving complete paper-level pipelines rather than isolated local lemmas.

---

## (5) Single biggest remaining gap

The biggest remaining gap is:

> **End-to-end integration from the probabilistic/statistical model and online algorithm to the final OR-level performance theorem.**

Concrete examples:

1. **Bayes-risk pipeline is still incomplete.**  
   `posterior_chebyshev_route` is real, but the code does not yet fully derive a joint high-probability good event over routes and then feed that event into `per_step_gap` / `lcb_suboptimality`. `bayes_risk_per_step_gap_high_prob` still assumes `h_prob`.

2. **Hedge regret is still not fully formalized.**  
   The per-round potential lemma is now real, but the cumulative log-potential/telescoping theorem is missing. The file still does not prove  
   `hedgeRegret ≤ log K / η + η T / 2`.

3. **EXP3 remains conditional.**  
   The certificate framing is honest, but the paper must be careful: full EXP3/EXP3-IX bandit regret is not formalized.

4. **DRO theorem is correct in the restricted affine/ENNReal/support setting, not full KR duality.**  
   That is acceptable if the paper states the restriction clearly. It is not a general Wasserstein DRO duality theorem.

5. **Transit-system theorem remains stylized.**  
   The formal model is a simplified finite/affine/route-level abstraction. The paper should avoid implying the Lean development verifies the full GTFS/hyperpath implementation.

### If this were an OR revision decision

I would recommend **major revision, not rejection**. The formal contribution is now credible enough to be worth salvaging, but the paper needs to tighten its claims around what is actually proved.

The next single most important formal addition should be one integrated theorem of the form:

```lean
posterior_good_event_implies_lcb_suboptimality_high_prob
```

or equivalently:

- construct route posterior measures,
- derive per-route Chebyshev bounds,
- prove finite-union good-event probability,
- obtain deterministic `h_acc`,
- apply `per_step_gap`,
- sum via `lcb_suboptimality`.

That would convert the current collection of local lemmas into a recognizable OR-style theorem.
