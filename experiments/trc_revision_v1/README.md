# TR-C development revision

This stage preserves the original model, menu, date panel, and declared budgets while repairing the execution and reporting contract. All reruns are development evidence; historical receipts are retained, and the original 72-root panel receives no additional budget.

The current manuscript is generated only by `python3 -m experiments.trc_revision_v1.build_manuscript`; see [MANUSCRIPT.md](MANUSCRIPT.md). The old `bounded_gate_5d_v1` manuscript and figure generators remain for historical reconstruction and must not be used to regenerate the current paper. The previous manuscript, figures, and generators are preserved in `history/manuscript_before_revision.tar.gz`.

The server retains the original inputs under `experiments/bounded_gate_5d_v1`, with new journey receipts under this stage's `roots/`. `run.py --source-directory <original-stage> --directory <new-stage> --workers 4` reads those inputs without changing them. Completed cells are retained on resume. Development attempts are stored separately under `development_attempt_*`; their receipts are excluded from the final paired panel.

`summary.json` and `paired_journeys.csv` retain all 60 roots; `execution_audit.json` checks issued-action clocks; `calibration_summary.json` distinguishes transfer contracts and the two weighting conventions. The raw endpoint/message archives and full journey receipts remain on the server. Local deliverables contain the small summaries, comparison tables, code, and manuscript.
