# Anonymous Lean 4 Artifact for BAPR-HRO

This archive contains the Lean 4 formal proof artifact submitted as
supplementary material for review. It is anonymized for double-anonymous
review. Public repository and archival DOI information are intentionally
withheld during review and will be released after acceptance.

## Contents

- `lean-toolchain`: Lean version pin (`leanprover/lean4:v4.28.0`).
- `lakefile.toml`: Minimal Lake configuration for this artifact.
- `lake-manifest.json`: Dependency lock file, including Mathlib commit
  `8f9d9cff6bd728b17a24e163c9402775d9e6a365`.
- `Main.lean`: Minimal entry point importing all theorem files in this
  artifact.
- `Wasserstein/`: Coupling, Wasserstein distance, DRO identity, and robust
  Bellman lemmas.
- `BAPR-HRO/`: BAPR-HRO posterior-risk, irrecoverability, regret,
  convergence, lower-bound, and architecture lemmas.

## Build Instructions

From the root of this archive, run:

```bash
lake exe cache get
lake build
```

The default target links the `artifact` executable. To check `Main.lean` and
all imported theorem files without native linking, run `lake build +Main:olean`.

If Mathlib cache retrieval is unavailable, `lake build` will build dependencies
from source, which can take substantially longer.

## Verification Scope

This artifact checks abstract statements with the hypotheses in their Lean
signatures. The following distinctions govern their use in the transit model.

| Source | Checked statement and scope |
| --- | --- |
| `DRO.lean` | `dro_upper_bound` gives a one-sided Lipschitz bound. `dro_iSup_real_equality` concerns the affine arrival cost on the real line, with nonnegative support and a feasible translation witness. |
| `DROBellman.lean` | `robustBellmanOp_contraction` assumes a nonexpansive ambiguity functional. The mixture is an instance. `wassersteinAmbFunc` is the nominal expectation plus a Lipschitz penalty, and its bound concerns that expression. |
| `BAPRHRO.lean` | `posterior_std` denotes the Normal–Gamma marginal Student-t scale. The selection and product-posterior bounds assume numerical error or moment conditions and sum externally specified per-stage gaps. A link to causal journey costs requires a separate argument. |
| `BAPRHRO_V2.lean` | `ensemble_std_bound` restates the supplied error envelope. `v2_convergence_rate` rearranges a bound with an assumed square-root rate. |
| `IrrecoverabilityBridge.lean` | The coefficient signs follow from the costs specified in the environment definitions. |
| `EXP3Regret.lean` and `AdaptiveConvergence.lean` | The epoch argument takes a stationary regret bound as a hypothesis. The separate Hedge theorem uses the full loss vector. The convergence module establishes inequalities for numerical rate expressions. |
| `LowerBound.lean` and `Architecture.lean` | The adversary can calculate the action of a deterministic algorithm. The sandwich orders prescribed endpoints, and the cost formulas count abstract operations. |

The model-to-runtime correspondence, causal continuation losses, calibrated
prediction intervals and measured implementation cost require separate
evidence. The current source comments describe this scope. The submitted ZIP
is the historical artifact.

## File Map

The theorem files imported by `Main.lean` are:

- `Wasserstein/Coupling.lean`
- `Wasserstein/Distance.lean`
- `Wasserstein/DRO.lean`
- `Wasserstein/DROBellman.lean`
- `BAPR-HRO/BAPRHRO.lean`
- `BAPR-HRO/BAPRHRO_V2.lean`
- `BAPR-HRO/EXP3Regret.lean`
- `BAPR-HRO/Irrecoverability.lean`
- `BAPR-HRO/IrrecoverabilityBridge.lean`
- `BAPR-HRO/AdaptiveConvergence.lean`
- `BAPR-HRO/LowerBound.lean`
- `BAPR-HRO/Architecture.lean`
