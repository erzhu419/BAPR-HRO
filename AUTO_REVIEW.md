# BAPR-HRO External Review Log

External reviewer: **gpt-5.5** via ruoli.dev (OpenAI-compatible proxy).
Target venue: **Operations Research** (INFORMS flagship).

The reviewer received: paper structure (abstract + theorems + tables) +
**all 12 Lean source files verbatim** (2,659 lines, 107 KB).

---

## Round 1 — 2026-04-28

### Assessment summary

- **Score: 3/10**
- **Verdict: NOT READY**
- **Single biggest fix:** *"A fully stated and proved finite-horizon
  transit-SSP theorem showing that the proposed LCB reranking policy is
  exactly the robust Bellman policy for action-wise rectangular
  Wasserstein ambiguity sets, including the affine translation witness,
  the W1 distance proof, treatment of cancellations/timeouts, and
  explicit computational complexity."*

### Top 10 weaknesses (verbatim from reviewer)

| Rank | Weakness | Specific Lean evidence |
|---:|---|---|
| 1 | **"Tight LCB=DRO" theorem is not actually proved.** | `translation_is_dro_witness_real` proves the value shift but **not** `W₁(P*,P̂) ≤ ε`. `lcb_equals_dro` *assumes* both the distance condition and the value equality. No `sSup` appears anywhere. |
| 2 | **No actual robust-MDP/SSP theorem.** | `DROBellman.lean`'s "DRO Bellman" is defined as `Bellman + ε·L`; no maximization over a Wasserstein ball, no rectangularity, no SSP machinery. |
| 3 | **EXP3Regret.lean contains an explicit `axiom`.** | Line 54: `axiom stationary_exp3_bound …`. Paper claims this is "formalized". |
| 4 | **Ensemble = empirical DRO is not formalized.** | `ensemble_lcb_equals_empirical_dro` only proves mean-shift + std-invariance under translation. No empirical measure, no W1 distance, no DRO supremum. |
| 5 | **Suboptimality + posterior contraction are weaker than stated.** | `per_step_gap` assumes deterministic calibration `|nominal − true| ≤ σ_max` — not derived from the Normal-Gamma posterior. `posterior_variance_bound` is algebra, not Bayes-risk. |
| 6 | **Minimax lower bound is conditional, not adversarial.** | `regret_lower_bound` is just: if each changepoint costs ≥ d_min then sum ≥ C·d_min. No adversarial construction over algorithms. |
| 7 | **EXP3 axiom is existential and not linked to algorithm.** | The axiom asserts existence of *some* regret satisfying the bound, not the regret of the actual EXP3 process. |
| 8 | **Empirical evaluation is not OR-grade.** | 30 runs / focused OD; 17 ODs filtered; heavy timeout censoring; no CIs; no demand weighting; conditional means among completions can mislead. |
| 9 | **Complexity story is underdeveloped.** | OR needs theorems in network size for preprocessing/reranking/posterior-update/DRO-evaluation. `Architecture.lean` only proves cost inequalities. |
| 10 | **Novelty over DRO literature is overstated.** | For affine 1-Lipschitz costs `sup_{W₁≤ε} E[f] = E[f]+ε` is standard from Gao-Kleywegt, Esfahani-Kuhn, Blanchet-Murthy. |

### File-by-file Lean assessment (verbatim summary)

- **Coupling.lean**: legitimate definitions, not a major contribution.
- **Distance.lean** *(most substantive file)*: defines primal ENNReal-valued W₁
  and proves nonneg / self / sym / triangle via gluing. **Does not** prove
  W₁ is a metric on P₁(X), no identity-of-indiscernibles, no KR duality.
  "First formalization of Wasserstein" claim should be toned down.
- **DRO.lean** *(central problem)*: redefines its own `W₁` separate from
  `Distance.lean`. `dro_upper_bound` is an ENNReal-only weak KR. `lcb_equals_dro`
  *assumes* both conditions. `translation_is_dro_witness_real` proves only the
  integral shift. `bapr_score_equals_dro` is `rfl`.
- **DROBellman.lean**: doesn't formalize a DRO Bellman operator.
  `dro_bellman_bound` is `rfl`.
- **BAPRHRO.lean**: deterministic comparison lemma under strong assumption;
  posterior contraction lemmas are algebraic on simplified expressions,
  not the actual posterior predictive variance in expectation.
- **BAPRHRO_V2.lean**: dynamic-β monotonicity is elementary;
  `ensemble_lcb_equals_empirical_dro` doesn't mention Wasserstein or
  empirical measures.
- **Irrecoverability.lean**: two-arm arithmetic, not "sign of optimal β"
  in any optimization sense.
- **IrrecoverabilityBridge.lean**: largely definitional;
  `right_excess = left_excess` symmetry assumption disconnects from the
  later "right-skewed" discussion.
- **EXP3Regret.lean**: explicit `axiom`. Materially invalidates abstract
  claim "formalized EXP3 regret bounds".
- **AdaptiveConvergence.lean**: proves `(√(c/T)+v/T)² ≤ 2(c/T+(v/T)²)` —
  an O(1/T) **squared** bound, not an O(1/T) regret bound.
- **LowerBound.lean**: conditional summation lemma; adversarial
  construction is in comments only.
- **Architecture.lean**: simple algebra about precompute-vs-resolve costs.

### Spot-check verification

I independently confirmed three of the most damaging factual claims:

1. **EXP3Regret.lean line 54**: `axiom stationary_exp3_bound (K : ℕ) (tau : ℝ)` — confirmed real axiom, not theorem.
2. **DRO.lean** redefines `W₁` locally (line 53) instead of importing from Distance.lean — confirmed fragmented infrastructure.
3. **translation_is_dro_witness_real** body (DRO.lean ~line 320–370) only proves the integral identity. It does NOT prove `W₁(Phat.map (·+ε), Phat) ≤ ε` — confirmed.

The reviewer's specific factual criticisms are correct.

### Reviewer raw response

<details>
<summary>Click to expand full reviewer response (round 1)</summary>

Saved separately at: `/tmp/review_round1.txt` (13 KB).
Full markdown table + per-file Lean assessment + verdict.

</details>

### Status

- **Stopping the autonomous loop** before implementing fixes.
- The minimum fixes the reviewer requested are major:
  1. New theorem proving `W₁(translation push-forward, P̂) ≤ ε` (small, tractable).
  2. Removing `axiom stationary_exp3_bound` and replacing with proof, OR
     re-labelling the abstract claim honestly (small).
  3. Adding empirical-measure W₁ statement to ensemble theorem (medium).
  4. Full SSP-MDP theorem with action-wise rectangular Wasserstein
     ambiguity (large).
  5. Multi-day Zurich evaluation with CIs and demand weighting (large).
  6. Re-positioning novelty claim relative to Gao-Kleywegt / Esfahani-Kuhn (small).

Items 1, 2, 3, 6 are tractable in a 1-2 round loop. Items 4 and 5 are
substantive new work. Pausing here for user direction.

---

## Round 2 — 2026-04-28

### Assessment summary

- **Score: 4/10** (round-1: 3/10, **+1** improvement)
- **Verdict: NOT READY** (no change)
- **Single biggest remaining fix:** *"Specification mismatch — several theorems are theorem-shaped but prove witness fragments, nonnegativity, or definitions rather than the advertised statements (DRO sup equality, real Bayes-risk, real Hedge regret, ∀-alg-∃-trajectory minimax)."*

### Per-issue status (verbatim from reviewer)

| # | Issue | Round 1 | Round 2 |
|---|---|---|---|
| 1 | LCB=DRO tight equivalence | NOT FIXED | **PARTIALLY FIXED** — W1 bound real, but no formal `iSup` equality |
| 2 | Real DRO Bellman | NOT FIXED | **PARTIALLY FIXED** — abstract Q nonexp + mixture instance real, but Wasserstein ball not derived |
| 3 | EXP3 axiom | NOT FIXED | **FIXED narrowly** — axiom syntactically gone, but not replaced by real EXP3 proof (see #7) |
| 4 | Ensemble = empirical DRO | NOT FIXED | **PARTIALLY FIXED** — push-forward identity real, but no W1≤ε proof, no DRO sup |
| 5 | Posterior contraction → Bayes-risk | NOT FIXED | **NOT FIXED** — `bayes_risk_per_step_gap` proves only nonnegativity, doesn't invoke `per_step_gap` |
| 6 | Adversarial minimax | NOT FIXED | **PARTIALLY FIXED** — real adaptive adversary, but `minimax_regret_lower_bound` doesn't return a `Trajectory T` |
| 7 | EXP3 linked to algorithm | NOT FIXED | **NOT FIXED** — `hedge_bound` is just a formula, no weights/losses/regret defined |
| 8 | Empirical CIs | PARTIAL | **PARTIALLY FIXED** — bootstrap CIs real, but disrupted-day intervals overlap heavily |
| 9 | Complexity story | UNDEVELOPED | **PARTIALLY FIXED** — Architecture.lean now has real cost ratios, but no full hyperpath/posterior complexity |
| 10 | Novelty over DRO lit | OVERSTATED | **FIXED** — abstract attribution to Gao-Kleywegt et al. now correct |

### Praise (verbatim)

1. *"The new translation W₁ witness is genuinely good.  
   `w1_translation_le` in `DRO.lean` is exactly the kind of missing measure-theoretic detail I asked for in round 1: it constructs the swapped coupling, verifies marginals, and computes the transport cost. This is a real improvement."*

2. *"The robust Bellman rewrite is conceptually much cleaner.  
   `NonExpansiveAmbiguity`, `robustBellmanOp_contraction`, and `mixture_amb_nonexp` give a reusable abstraction and a real finite-state contraction argument. Also, the authors are now honest that the Wasserstein penalty gives a `(1+ε)γ` Lipschitz bound rather than pretending it is automatically γ-contractive."*

### Status

- **Continuing the loop?** Round-1 fixes addressed structural concerns (no axioms, no obvious bogus rfl theorems). Round-2 reveals **specification-level gaps**: theorems prove the right *facts* but the wrong *quantifiers / objects*. Tractable round-3 fixes would target (in order):
  1. `lcb_equals_dro_attained_iSup`: actual `iSup` equality (~30min)
  2. `minimax_regret_lower_bound_traj`: ∀-alg-∃-trajectory with `Trajectory T` type (~30min)
  3. `bayes_risk_per_step_gap_real`: invoke `per_step_gap` under high-prob event (~45min)
  4. `ensemble_W1_le`: compose `ensemble_W1_translation_le` with `Wasserstein.w1_translation_le` (~15min)
  5. `hedge_regret_proof`: define weights/losses/regret, prove potential bound (~3h, heaviest)

Items 1-4 are 2 hours total. Item 5 is the heaviest.


---

## Round 3 — 2026-04-28

### Assessment summary

- **Score: 5/10** (round 1: 3/10, round 2: 4/10, **+1**)
- **Verdict: NOT READY** (no change)
- Verdict trajectory: 3 → 4 → 5 over three rounds.

### Per-issue status (round 3 verbatim from reviewer)

| # | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| 1 LCB=DRO tight | NOT FIXED | PARTIAL | **SUPERFICIAL** — still no `⨆` in `dro_iSup_equals_value_plus_eps` |
| 2 Real DRO Bellman | NOT FIXED | PARTIAL | **SUPERFICIAL** — `WassersteinKRProperty` is sandwich predicate, no actual sup-over-ball |
| 3 EXP3 axiom | NOT FIXED | FIXED narrow | **GENUINELY FIXED narrow** ✅ |
| 4 Ensemble = empirical DRO | NOT FIXED | PARTIAL | **SUPERFICIAL** — empirical measure + W₁≤ε real, but no sup-equality |
| 5 Bayes-risk gap | NOT FIXED | NOT FIXED | **SUPERFICIAL** — invokes `per_step_gap` now, but assumes `h_acc` and `h_prob` rather than deriving |
| 6 Adversarial minimax | NOT FIXED | PARTIAL | **SUPERFICIAL** — has `Trajectory T` type, but uses `runAlgNat` not defined `cumulativeRegret` |
| 7 EXP3 linked to alg | NOT FIXED | NOT FIXED | **SUPERFICIAL** — Hedge algorithm + regret + `hedge_W_ge_best` real, but no `hedgeRegret ≤ ...` theorem |
| 8 Empirical CIs | PARTIAL | PARTIAL | **GENUINELY FIXED methodologically** ✅ — paired bootstrap shows honest negative result |
| 9 Complexity | UNDEVELOPED | PARTIAL | **SUPERFICIAL** — cost definitions + identities, no algorithm-level complexity |
| 10 Novelty | OVERSTATED | FIXED | **GENUINELY FIXED** ✅ |

### Reviewer's diagnosis (verbatim)

> "The single most important fix is an exact-statement audit. The
> authors need to replace theorem-shaped fragments with the actual
> target theorems. Until the theorem statements exactly encode the
> paper claims, the formal component remains overclaimed."

### Single biggest remaining fix

A SPECIFICATION AUDIT making each theorem literally encode the
advertised claim:
- `dro_iSup_equals_value_plus_eps` → state and prove an actual
  `⨆ P with W₁ P P̂ ≤ ε, ∫f dP = ∫f dP̂ + ε` over a well-defined
  W₁ ambiguity set
- Hedge: prove `hedgeRegret hK η ℓ T ≤ ln K / η + η T / 2` under
  explicit loss assumptions (full potential-function chain)
- `exists_adversarial_trajectory` → conclude with the already-defined
  `cumulativeRegret alg T traj d`
- Bayes-risk: derive the good event from an actual posterior model

### Status

- Score 3 → 4 → 5: real upward trend, each round adds ~1 point
- Rate of progress: ~+1 per round despite substantial Lean work
- Estimated rounds to reach READY (6/10+): 1-2 more rounds of
  focused specification audits (~3-4 hours each)

