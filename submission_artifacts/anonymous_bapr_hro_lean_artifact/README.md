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

The expected result is a successful Lean build of the `artifact` executable
target and all imported theorem files.

If Mathlib cache retrieval is unavailable, `lake build` will build dependencies
from source, which can take substantially longer.

## Verification Scope

This artifact formalizes the mathematical core used in the manuscript:

- Wasserstein coupling and distance facts.
- The per-candidate LCB/DRO identity and related bounds.
- Robust Bellman comparison bounds.
- Posterior-risk selection and per-step gap bounds.
- Irrecoverability and pessimism lemmas.
- Hedge/EXP3-style regret bound used for comparison.
- Posterior contraction, lower-bound, and implementation-cost lemmas.

The artifact does not formalize the GTFS-RT data parser, simulator runtime,
plotting scripts, or empirical statistical pipeline. Those components are
submitted separately with the code/data supplement where required by the
journal workflow.

## File Map

The theorem names and file locations correspond to the paper-to-Lean
cross-reference table in the manuscript. The main files are:

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
