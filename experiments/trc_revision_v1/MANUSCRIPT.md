Current manuscript source: `manuscript.tex.in`, `execution_contract.tex`, `references.tex`, and the final development result table generated from this stage's small summaries.

Generate from the repository root, then compile twice from the paper directory:

```bash
python3 -m experiments.trc_revision_v1.build_manuscript
cd paper
pdflatex -interaction=nonstopmode -halt-on-error paper_reconstruction.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_reconstruction.tex
```

Compilation from the repository root also works with `pdflatex -output-directory=paper paper/paper_reconstruction.tex` twice. The former `bounded_gate_5d_v1/build_manuscript_and_traceability.py` and `build_paper_figures.py` remain historical reconstruction tools; they are not the source for the current paper.

`history/manuscript_before_revision.tar.gz` preserves the previous tex, PDF, figures, generators and traceability index. The new generator only reads earlier experimental receipts. No previous scientific result is overwritten. The same-date corrected replay is development evidence; it is not fresh independent confirmation.
