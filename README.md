# BAPR-HRO: archival evaluation of adaptive transit routing

The current manuscript targets **Transportation Research Part C** and studies how source alignment, candidate menus, transfer assumptions, and action timing affect routing evaluations.

- Manuscript: [paper/paper_reconstruction.tex](paper/paper_reconstruction.tex) and its compiled PDF.
- Current code, protocol, and development results: [experiments/trc_revision_v1](experiments/trc_revision_v1/README.md).
- Evidence map: [md/EVIDENCE_TRACEABILITY_INDEX.md](md/EVIDENCE_TRACEABILITY_INDEX.md).

Generate the current paper with `python3 -m experiments.trc_revision_v1.build_manuscript`. Earlier experiment directories retain their historical implementations and receipts; the previous manuscript and generators are also archived in the revision stage.

The corrected replay uses the already exposed 60-root April panel. It retains every gain, harm, tie, and unresolved outcome. The original 72-root October panel receives no additional control budget. Raw archives and complete journey receipts remain on the compute server; local artifacts contain code and compact evidence.
