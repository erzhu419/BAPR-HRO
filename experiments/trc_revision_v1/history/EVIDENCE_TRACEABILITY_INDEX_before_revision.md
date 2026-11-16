# BAPR-HRO Evidence Traceability Index (Elsevier TR-C Manuscript)

This index maps every claim, proposition, table, figure, and trip-level audit statistic in `paper/paper_reconstruction.tex` (`paper/paper_reconstruction.pdf`) to its exact frozen receipt JSON/CSV file, code implementation, and automated unit test. Zero historical directories (`reconstruction_v50`, `decision_gap_v1`, `decision_calibration_v1`, `decision_calibration_v2`, `control_identity_v1`, `natural_gap_v2`) were modified, `bounded_gate_5d_v1` was executed once under pre-frozen governance, and `original_72_additional_budget = 0` holds throughout.

---

## 1. Manuscript Sections, Models & Propositions Traceability

| Manuscript Element | Mathematical / Empirical Content | Code Implementation | Primary Frozen Receipt(s) | Verification Test |
|---|---|---|---|---|
| **Section 3.2** (`sec:posterior`) | Exact-anchor correlated trip-chain Markov delay + persistent cancellation ($ho_{\mathrm{cxl}}=0.9$) posterior (`EvolvingTripBelief` + `PhysicalHistory`) | `src/evolving_trip_belief.py`; `experiments/reconstruction_v41/physical_posterior.py` | `experiments/control_identity_v1/summary.json`, `experiments/bounded_gate_5d_v1/contract.json` | `tests/test_v41_matched_continuation.py`, `tests/test_control_identity.py` |
| **Section 3.3** (`sec:contracts`) | 3-tier platform-pair MCT hierarchy (published `transfers.txt` -> assumed $	au_{\mathrm{same}}=90$s `[60,90,120]s` -> station fallback) & `ActionPublisher` physical dwell token (`fallback_expired` / `deadline_expired`) | `experiments/bounded_gate_5d_v1/pair_contract.py` (`DayPlatformIndex`, `BoundedArmExecutor`) | `experiments/bounded_gate_5d_v1/platform_contract_summary.json`, `experiments/bounded_gate_5d_v1/sensitivity_tau_same.json`, `experiments/bounded_gate_5d_v1/dwell_audit.json` | `tests/test_bounded_gate_5d_v1.py` |
| **Section 4.1** (`eq:missing_bounds`) | Operational accounting decomposition (distinguishing $J^*$ vs. policy-conditional differences and $	au_{\mathrm{same}}$ boundary non-monotonicity) & sharp $[0, 3600]$s partial identification bounds | `experiments/bounded_gate_5d_v1/summarize.py` | `experiments/reconstruction_v50/summary.json`, `experiments/decision_gap_v1/summary.json`, `experiments/control_identity_v1/summary.json`, `experiments/bounded_gate_5d_v1/summary.json` | `tests/test_bounded_gate_5d_v1.py` |
| **Proposition 1** (`prop:collapse`) | (a) $O(K\max_k|\mathcal{D}_k|)$ selective-waiting envelope `PA-DSW` (`min(first_offer, single_target)`) without `1 - CDF` cancellation; (b) algebraic collapse on $m \le 2$ exits (`238/238` ties) | `experiments/bounded_gate_5d_v1/pair_contract.py` (`PairPhysicalPlanner`, `PairMatchedPlanner`) | `experiments/control_identity_v1/summary.json` (`72/72`), `experiments/natural_gap_v2/summary_compact.json` (`166/166`) | `tests/test_v41_matched_continuation.py`, `tests/test_bounded_gate_5d_v1.py` |
| **Proposition 2** (`prop:anytime`) | Dwell-expiration nullification & paired Student-$t$ override control ($M_0=64$ worlds, $\gamma=90$s, $z_{0.975}=1.96$; `403s` October MC flip harm blocked; `74.3%`--`79.8%` gates pruned) | `experiments/g2_reset_v1/certified_gate.py` (`should_invoke_rollout`, `choose_certified`) | `experiments/bounded_gate_5d_v1/contract.json`, `experiments/g2_reset_v1/OFFLINE_AUDIT_SUMMARY.json` (`finding_4`) | `tests/test_g2_reset_v1.py` |
| **Section 6.4** (Oracle Headroom) | Hindsight oracle switching headroom between PA-DSW and Rollout-128: `+1.33s` (`0.09%`, Oct), `+3.93s` (`0.29%`, Mar--Apr), `+5.24s` (`0.43%` at 90s; `0.00s` at 60s, Apr) | `experiments/g2_reset_v1/offline_audit.py`; `experiments/bounded_gate_5d_v1/summarize.py` | `experiments/control_identity_v1/summary.json`, `experiments/natural_gap_v2/summary_compact.json`, `experiments/bounded_gate_5d_v1/summary.json` | `tests/test_g2_reset_v1.py` |

---

## 2. Tables Traceability

| Table | Scope | Key Verified Metrics | Source Receipt(s) |
|---|---|---|---|
| **Table 1** (`tab:lit`) | Closest-Neighbor Literature Comparison | Contrasts Gentile (2005), Chen & Nie (2015), Kim et al. (2005), Rambha et al. (2016), Dibbelt et al. (2018, DOI `10.1145/3274661`), Durner (2024), Silver & Veness (2010), Somani et al. (2013) vs. PA-DSW | Verified bibliography in `paper/paper_reconstruction.tex` | -- |
| **Table 2** (`tab:cohorts`) | 39-Operating-Day Study Overview | `300` paired control trips (`72` Oct + `168` Mar--Apr + `60` Apr); `3,063` total opportunities (`1,325` DC2 = `1,222` res + `103` unres; `1,328` NG2 = `1,309` res + `19` pre-cancel; `410` BG5); `2,941` resolved | `experiments/g2_reset_v1/OFFLINE_AUDIT_SUMMARY.json`, `experiments/bounded_gate_5d_v1/calibration_split_summary.json`, `experiments/bounded_gate_5d_v1/audit.json` | `tests/test_g2_reset_v1.py` |
| **Table 3** (`tab:oct_cascade`) | October Controlled Attribution Cascade (`72` trips) | `1410.93s` -> `1410.93s` -> `1390.75s` (`-20.18s`) -> `1386.10s` (`-4.65s`) -> `1385.18s` (`-0.92s`, total `-25.75s`); `Determ.-Mean = 1409.85s` (`+24.67s`); `Rollout-128 = 1396.03s` (`-10.85s`) | `experiments/reconstruction_v50/summary.json`, `experiments/decision_gap_v1/summary.json`, `experiments/control_identity_v1/summary.json` | `tests/test_control_identity.py` |
| **Table 4** (`tab:bg5_arms`) | April 5-Day Confirmation — 8 Arms (`60` trips) | `StatMax-DSW = 1241.80s`, `Mean-CSA = 1234.74s` (`0.159s` CPU), `PA-DSW = 1224.16s` (`0.418s` CPU), `Multi-Hop-DSW = 1247.76s` (`24.375s` CPU), `Cert-Rollout-64 = 1252.40s` (`29.629s` CPU, `24` gates), `Rollout-128 = 1218.91s` (`17.485s` CPU, `119` gates) | `experiments/bounded_gate_5d_v1/summary.json`, `experiments/bounded_gate_5d_v1/paired_journeys.csv` | `tests/test_bounded_gate_5d_v1.py` |
| **Table 5** (`tab:bg5_contrasts`) | Pre-Declared Contrasts & Date-Cluster CIs | `PA-DSW` vs `StatMax-DSW`: `+10.59s [-10.67, +25.83]`; `PA-DSW` vs `Mean-CSA`: `+10.59s [-10.67, +25.83]`; `Cert-Rollout-64` vs `PA-DSW`: `-28.24s [-49.52, -8.38]`; `Rollout-128` vs `PA-DSW`: `+5.24s [+0.00, +15.72]` | `experiments/bounded_gate_5d_v1/summary.json`, `experiments/bounded_gate_5d_v1/paired_comparisons.csv` | `tests/test_bounded_gate_5d_v1.py` |
| **Table 6** (`tab:bg5_micro`) | Trip-Level Micro-Audit of All 9 Non-Tied / Unresolved Trips | `51/60` trips tie across all 8 arms; `Trips 17, 29, 41` (`+293, +286, +339s`); `Trip 3` (`-304s` at 90s, `0s` at 60s); `Trip 9` (`+555s` base CPU dwell overrun); `Trips 31, 49` (`-789, -580s` multi-hop error); `Trips 52, 55` (missing sensors) | `experiments/bounded_gate_5d_v1/paired_journeys.csv`, `experiments/bounded_gate_5d_v1/selection.json`, `experiments/bounded_gate_5d_v1/sensitivity_tau_same.json` | `tests/test_bounded_gate_5d_v1.py` |
| **Table C.7** (`tab:codemap`) | Manuscript-to-Repository Identifier Map | Exact mapping from `PA-DSW`, `StatMax-DSW`, `Mean-CSA`, `Multi-Hop-DSW`, `Cert-Rollout-64`, `Rollout-128` to code classes and JSON keys | `experiments/bounded_gate_5d_v1/contract.json` | -- |

---

## 3. Figures Traceability

| Figure | File Path | Generator Function | Underlying Frozen JSON Receipts |
|---|---|---|---|
| **Figure 1** (`fig:cascade`) | `paper/figures/fig1_attribution_cascade.pdf` | `build_paper_figures.py::build_fig1_attribution_cascade` | `reconstruction_v50/summary.json`, `decision_gap_v1/summary.json`, `control_identity_v1/summary.json`, `natural_gap_v2/summary_compact.json`, `bounded_gate_5d_v1/summary.json` |
| **Figure 2** (`fig:cal_plat`) | `paper/figures/fig2_calibration_platform.pdf` | `build_paper_figures.py::build_fig2_calibration_platform` | `g2_reset_v1/OFFLINE_AUDIT_SUMMARY.json`, `bounded_gate_5d_v1/calibration_split_summary.json`, `bounded_gate_5d_v1/platform_contract_summary.json`, `bounded_gate_5d_v1/sensitivity_tau_same.json` |
| **Figure 3** (`fig:pareto`) | `paper/figures/fig3_pareto_frontier.pdf` | `build_paper_figures.py::build_fig3_pareto_frontier` | `bounded_gate_5d_v1/summary.json` |
| **Figure 4** (`fig:dwell`) | `paper/figures/fig4_dwell_anytime.pdf` | `build_paper_figures.py::build_fig4_dwell_anytime` | `bounded_gate_5d_v1/dwell_audit.json`, `g2_reset_v1/OFFLINE_AUDIT_SUMMARY.json` |

---

## 4. End-to-End Reproducibility Commands

```bash
python3 -m pytest tests/test_bounded_gate_5d_v1.py tests/test_g2_reset_v1.py tests/test_control_identity.py tests/test_v41_matched_continuation.py tests/test_v51_rollout.py
python3 -m experiments.bounded_gate_5d_v1.build_paper_figures
python3 -m experiments.bounded_gate_5d_v1.build_manuscript_and_traceability
cd paper && pdflatex -interaction=nonstopmode paper_reconstruction.tex && pdflatex -interaction=nonstopmode paper_reconstruction.tex
```
