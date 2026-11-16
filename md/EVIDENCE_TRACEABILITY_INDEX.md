# Current manuscript and evidence

The current paper is the TR-C development revision, not a new optimality or action-certification claim. Historical receipts remain unchanged. Corrected April replays are development evidence; the original 72 roots receive no additional budget.

- [Current manuscript](../paper/paper_reconstruction.tex) and [PDF](../paper/paper_reconstruction.pdf).
- [Sole current generator](../experiments/trc_revision_v1/build_manuscript.py), [template](../experiments/trc_revision_v1/manuscript.tex.in), [references](../experiments/trc_revision_v1/references.tex), and [build instructions](../experiments/trc_revision_v1/MANUSCRIPT.md).
- [Frozen development protocol](../experiments/trc_revision_v1/protocol.json), [execution contract](../experiments/trc_revision_v1/execution_contract.tex), [executor](../experiments/trc_revision_v1/executor.py), [interruptible rollout](../experiments/trc_revision_v1/rollout.py), and [calibration](../experiments/trc_revision_v1/calibration.py).
- [Current result summary](../experiments/trc_revision_v1/summary.json), [calibration summary](../experiments/trc_revision_v1/calibration_summary.json), and [manuscript source receipts](../experiments/trc_revision_v1/manuscript_sources.json).
- [Execution-record checks](../experiments/trc_revision_v1/execution_audit.json) and [saved scoring and runtime cases](../experiments/trc_revision_v1/scoring_decision_cases.json) give the local attribution evidence.
- Historical evidence remains in [control identity](../experiments/control_identity_v1/summary.json), [natural events](../experiments/natural_gap_v2/summary_compact.json), and [original five-day comparison](../experiments/bounded_gate_5d_v1/summary.json). Their contracts and dates differ; cross-stage gains are not additive causal effects.
- [Previous index](../experiments/trc_revision_v1/history/EVIDENCE_TRACEABILITY_INDEX_before_revision.md) and [previous manuscript and generators](../experiments/trc_revision_v1/history/manuscript_before_revision.tar.gz) are preserved for historical reconstruction only.

Run `python3 -m experiments.trc_revision_v1.build_manuscript` from the repository root. The old five-day manuscript generator is historical and does not generate the current paper.
