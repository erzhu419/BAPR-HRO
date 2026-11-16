"""Synthesize the complete Elsevier Transportation Research Part C manuscript
(paper/paper_reconstruction.tex) and Evidence Traceability Index
(md/EVIDENCE_TRACEABILITY_INDEX.md) directly from frozen JSON/CSV receipts.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
PAPER_DIR = ROOT / "paper"
MD_DIR = ROOT / "md"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_s(val, digits=2, signed=False):
    if val is None:
        return "--"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(float(val))


def build_all():
    v50 = _load(EXP / "reconstruction_v50" / "summary.json")
    dg1 = _load(EXP / "decision_gap_v1" / "summary.json")
    ci1 = _load(EXP / "control_identity_v1" / "summary.json")
    ng2 = _load(EXP / "natural_gap_v2" / "summary_compact.json")
    off = _load(EXP / "g2_reset_v1" / "OFFLINE_AUDIT_SUMMARY.json")

    bg_dir = EXP / "bounded_gate_5d_v1"
    bg5 = _load(bg_dir / "summary.json")
    bg5_audit = _load(bg_dir / "audit.json")
    bg5_cal = _load(bg_dir / "calibration_split_summary.json")
    bg5_plat = _load(bg_dir / "platform_contract_summary.json")
    bg5_sens = _load(bg_dir / "sensitivity_tau_same.json")
    bg5_contract = _load(bg_dir / "contract.json")
    dwell = _load(bg_dir / "dwell_audit.json")

    # Verify exact hyperparameters from contract.json (E1)
    bc_hp = bg5_contract["controller_hyperparameters"]["bounded_cert"]
    bc_worlds = int(bc_hp["worlds_per_query"])
    bc_margin = float(bc_hp["ambiguity_margin_seconds"])
    bc_budget = float(bc_hp["max_process_cpu_seconds"])

    # Extract exact metrics from October cohort (V50 -> DG1 -> CI1)
    v50_narrow_nat = v50["aggregate"]["narrow_native"]["mean_loss_bounds"][0]
    v50_narrow_com = v50["aggregate"]["narrow_common"]["mean_loss_bounds"][0]
    v50_wide_com = v50["aggregate"]["wide_common"]["mean_loss_bounds"][0]
    v50_gain = v50["comparisons"]["common_menu"]["complete_pair_mean_gain"]

    dg1_cheap = dg1["aggregate"]["new_cheap"]["complete_mean_loss"]
    dg1_menu_gain = dg1["attribution"]["menu_change_given_new_observation"]["complete_pair_mean_gain"]

    ci1_old_cheap = ci1["models"]["old_fit"]["aggregate"]["new_cheap"]["complete_mean_loss"]
    ci1_rep_cheap = ci1["models"]["repaired_fit"]["aggregate"]["new_cheap"]["complete_mean_loss"]
    ci1_rep_mean = ci1["models"]["repaired_fit"]["aggregate"]["new_mean"]["complete_mean_loss"]
    ci1_rep_f128 = ci1["models"]["repaired_fit"]["aggregate"]["new_full128"]["complete_mean_loss"]
    ci1_rep_cheap_cpu = ci1["models"]["repaired_fit"]["aggregate"]["new_cheap"]["mean_cpu"]
    ci1_rep_f128_cpu = ci1["models"]["repaired_fit"]["aggregate"]["new_full128"]["mean_cpu"]
    ci1_stoch_gain = ci1["models"]["repaired_fit"]["comparisons"]["new_mean__new_single"]["complete_pair_mean_gain"]
    ci1_f128_gain = ci1["models"]["repaired_fit"]["comparisons"]["new_cheap__new_full128"]["complete_pair_mean_gain"]
    total_oct_unlock = v50_narrow_nat - ci1_rep_cheap

    # Extract exact metrics from March-April cohort (NG2)
    ng2_cheap = ng2["aggregate"]["new_cheap"]["complete_mean_loss"]
    ng2_f128 = ng2["aggregate"]["new_full128"]["complete_mean_loss"]
    ng2_cheap_cpu = ng2["aggregate"]["new_cheap"]["mean_cpu"]
    ng2_f128_cpu = ng2["aggregate"]["new_full128"]["mean_cpu"]
    ng2_f128_gain = ng2_cheap - ng2_f128
    ng2_cheap_bounds = ng2["aggregate"]["new_cheap"]["all_root_mean_loss_bounds"]

    # Exact Calibration counts across 33 out-of-period days (E4: no hardcoded 1328 for DC2)
    dc2_c2 = off["finding_1_codex_calibration_split_verdict"]["decision_calibration_v2_split"]["conditional_summary"]
    dc2_reached_res = int(dc2_c2["reached_resolved_opportunities"])
    dc2_reached_unres = int(dc2_c2["reached_unresolved_opportunities"])
    dc2_onboard_total = dc2_reached_res + dc2_reached_unres  # 1222 + 103 = 1325

    ng2_c2 = off["finding_1_codex_calibration_split_verdict"]["natural_gap_v2_conditional_onboard"]
    ng2_reached_res = int(ng2_c2["conditional_reached_opportunities"])
    ng2_reg_total = int(ng2_c2["total_registered_opportunities"])  # 1328

    bg5_c1 = bg5_cal["metric_c1_initial_availability"]
    bg5_c2 = bg5_cal["metric_c2_conditional_onboard_calibration"]
    bg5_reached_res = int(bg5_c2["reached_resolved_opportunities"])  # 410
    bg5_reg_total = int(bg5_c1["registered_opportunities"])  # 410

    total_reached_opp = dc2_reached_res + ng2_reached_res + bg5_reached_res  # 2941
    total_eval_opp = dc2_onboard_total + ng2_reg_total + bg5_reg_total  # 3063

    # 5-Day Confirmation Experiment metrics (BG5)
    bg_start = bg5["selected_dates"][0]
    bg_end = bg5["selected_dates"][-1]
    bg_dates_str = f"{bg_start}--{bg_end}"
    arms = bg5["arms"]
    comps = bg5["comparisons"]

    g1_comp = comps["cheap_statmax_ideal__vs__cheap_plat_async"]
    csa_raw = comps["cheap_plat_async__vs__snapshot_csa_plat_async"]
    stoch_bg5_saving = -csa_raw["complete_mean_saving_seconds"]
    stoch_bg5_ci = [
        -csa_raw["date_cluster_bootstrap"]["ci95_saving_seconds"][1],
        -csa_raw["date_cluster_bootstrap"]["ci95_saving_seconds"][0],
    ]
    stoch_bg5_bounds = [
        -csa_raw["all_roots_missingness_saving_bounds"][1],
        -csa_raw["all_roots_missingness_saving_bounds"][0],
    ]
    rec_comp = comps["cheap_plat_async__vs__recursive_plat_async"]
    bc_comp = comps["cheap_plat_async__vs__bounded_cert_plat_async"]
    f128_ideal_comp = comps["cheap_plat_async__vs__full128_plat_ideal"]
    f128_async_comp = comps["cheap_plat_async__vs__full128_plat_async"]

    dq = dwell["dwell_quantiles_seconds"]
    f128_d = dwell["arms"]["full128_plat_async"]
    bc_d = dwell["arms"]["bounded_cert_plat_async"]
    cheap_d = dwell["arms"]["cheap_plat_async"]

    plat_rt_bg5_pct = 100.0 * (bg5_plat["gtfs_rt_platform_announcements"]["static_agreement_fraction"] or 0.0)
    plat_rt_chg_count = bg5_plat["gtfs_rt_platform_announcements"]["counts"]["rt_track_change_announced"]
    plat_rt_tot_count = bg5_plat["gtfs_rt_platform_announcements"]["platform_specific_updates"]
    plat_rt_chg_pct = 100.0 * plat_rt_chg_count / max(1, plat_rt_tot_count)
    plat_mct_delta_bg5 = abs(bg5_plat["mean_mct_change_vs_station_max_seconds"])

    sens_cheap_60 = bg5_sens["cheap_plat_async"]["60"]["complete_mean_loss_seconds"]
    sens_cheap_90 = bg5_sens["cheap_plat_async"]["90"]["complete_mean_loss_seconds"]
    sens_cheap_120 = bg5_sens["cheap_plat_async"]["120"]["complete_mean_loss_seconds"]
    sens_rec_60 = bg5_sens["recursive_plat_async"]["60"]["complete_mean_loss_seconds"]
    sens_rec_90 = bg5_sens["recursive_plat_async"]["90"]["complete_mean_loss_seconds"]
    sens_rec_120 = bg5_sens["recursive_plat_async"]["120"]["complete_mean_loss_seconds"]

    pruned_bg5_pct = 100.0 * (1.0 - bc_d["total_gates_invoked"] / max(1, f128_d["total_gates_invoked"]))

    p10_apr06 = 100.0 * bg5_c2["per_day_mean_residual"]["2023-04-06"]["10"]
    p10_apr07 = 100.0 * bg5_c2["per_day_mean_residual"]["2023-04-07"]["10"]
    p10_apr08 = 100.0 * bg5_c2["per_day_mean_residual"]["2023-04-08"]["10"]
    p10_apr09 = 100.0 * bg5_c2["per_day_mean_residual"]["2023-04-09"]["10"]
    p10_apr10 = 100.0 * bg5_c2["per_day_mean_residual"]["2023-04-10"]["10"]

    # Read paired_journeys.csv for exact 8-root micro-audit table
    pj_rows = list(csv.DictReader((bg_dir / "paired_journeys.csv").open(encoding="utf-8")))
    by_root = {}
    for r in pj_rows:
        by_root.setdefault(int(r["root_index"]), {})[r["arm"]] = r

    # Format 8-arm table rows for Section 6
    arm_display = [
        ("cheap_statmax_ideal", "StatMax-DSW", "Station-Max", "Ideal ($0$\\,s)"),
        ("full128_statmax_ideal", "Rollout-128", "Station-Max", "Ideal ($0$\\,s)"),
        ("snapshot_csa_plat_async", "Mean-CSA", "Platform-Pair", "Physical Dwell"),
        ("cheap_plat_async", "\\textbf{PA-DSW (Proposed)}", "Platform-Pair", "Physical Dwell"),
        ("recursive_plat_async", "Multi-Hop-DSW", "Platform-Pair", "Physical Dwell"),
        ("bounded_cert_plat_async", f"Cert-Rollout-{bc_worlds}", "Platform-Pair", "Physical Dwell"),
        ("full128_plat_ideal", "Rollout-128", "Platform-Pair", "Ideal ($0$\\,s)"),
        ("full128_plat_async", "Rollout-128", "Platform-Pair", "Physical Dwell"),
    ]
    table4_rows = []
    for key, name, mct_str, clk_str in arm_display:
        a = arms[key]
        lo_b, hi_b = a["all_60_loss_bounds"]
        table4_rows.append(
            f"{name} & {mct_str} & {clk_str} & {a['complete_roots']}/60 & "
            f"{_fmt_s(a['complete_mean_loss_seconds'], 2)} & "
            f"$[{_fmt_s(lo_b, 1)},\\,{_fmt_s(hi_b, 1)}]$ & "
            f"{_fmt_s(a['mean_cpu_seconds'], 3)} & {_fmt_s(a['max_cpu_seconds'], 2)} & {a['total_gates']} \\\\"
        )
    table4_tex = "\n".join(table4_rows)

    # Table 5: Pre-declared contrasts
    table5_rows = [
        (
            f"Platform-Pair vs.\\ Station-Max (StatMax-DSW vs.\\ PA-DSW)$^\\dagger$ & "
            f"{g1_comp['wins_count']}/{g1_comp['losses_count']}/{g1_comp['ties_count']} & "
            f"{_fmt_s(g1_comp['complete_mean_saving_seconds'], 2, True)} & "
            f"$[{_fmt_s(g1_comp['date_cluster_bootstrap']['ci95_saving_seconds'][0], 2, True)},\\,"
            f"{_fmt_s(g1_comp['date_cluster_bootstrap']['ci95_saving_seconds'][1], 2, True)}]$ & "
            f"$[{_fmt_s(g1_comp['all_roots_missingness_saving_bounds'][0], 1, True)},\\,"
            f"{_fmt_s(g1_comp['all_roots_missingness_saving_bounds'][1], 1, True)}]$ & "
            f"{_fmt_s(g1_comp['mean_cpu_delta_seconds'], 3, True)} \\\\"
        ),
        (
            f"Distributional Tail vs.\\ Point Mean (Mean-CSA vs.\\ PA-DSW) & "
            f"{csa_raw['losses_count']}/{csa_raw['wins_count']}/{csa_raw['ties_count']} & "
            f"{_fmt_s(stoch_bg5_saving, 2, True)} & "
            f"$[{_fmt_s(stoch_bg5_ci[0], 2, True)},\\,{_fmt_s(stoch_bg5_ci[1], 2, True)}]$ & "
            f"$[{_fmt_s(stoch_bg5_bounds[0], 1, True)},\\,{_fmt_s(stoch_bg5_bounds[1], 1, True)}]$ & "
            f"{_fmt_s(-csa_raw['mean_cpu_delta_seconds'], 3, True)} \\\\"
        ),
    ]
    other_contrasts = [
        ("cheap_statmax_ideal__vs__full128_statmax_ideal", "Station-Max Rollout (StatMax-DSW vs.\\ Rollout-128)"),
        ("cheap_plat_async__vs__recursive_plat_async", "Multi-Hop Recursion (PA-DSW vs.\\ Multi-Hop-DSW)"),
        ("cheap_plat_async__vs__bounded_cert_plat_async", f"Certified {bc_worlds}w Gate (PA-DSW vs.\\ Cert-Rollout-{bc_worlds})"),
        ("cheap_plat_async__vs__full128_plat_ideal", "Oracle-Clock Rollout (PA-DSW vs.\\ Rollout-128 Ideal)"),
        ("cheap_plat_async__vs__full128_plat_async", "Physical-Dwell Rollout (PA-DSW vs.\\ Rollout-128 Dwell)"),
        ("full128_plat_ideal__vs__full128_plat_async", "Dwell-Clock Difference (Rollout-128 Ideal vs.\\ Dwell)"),
    ]
    for key, desc in other_contrasts:
        c = comps[key]
        ci = c["date_cluster_bootstrap"]["ci95_saving_seconds"]
        b_lo, b_hi = c["all_roots_missingness_saving_bounds"]
        table5_rows.append(
            f"{desc} & {c['wins_count']}/{c['losses_count']}/{c['ties_count']} & "
            f"{_fmt_s(c['complete_mean_saving_seconds'], 2, True)} & "
            f"$[{_fmt_s(ci[0], 2, True)},\\,{_fmt_s(ci[1], 2, True)}]$ & "
            f"$[{_fmt_s(b_lo, 1, True)},\\,{_fmt_s(b_hi, 1, True)}]$ & "
            f"{_fmt_s(c['mean_cpu_delta_seconds'], 3, True)} \\\\"
        )
    table5_tex = "\n".join(table5_rows)

    # Table 6: Micro-audit of all 9 non-tied or unresolved roots out of 60 in BG5
    micro_specs = [
        (3, "Boundary $\\tau_{\\mathrm{same}}=90$\\,s loss ($1382$\\,s at $\\tau_{\\mathrm{same}}=60$\\,s)"),
        (9, "Base recursive CPU ($50.7$\\,s) overran dwell ($+555$\\,s)"),
        (17, "Same-platform/pair transfer unlocked ($+293$\\,s vs.\\ StatMax/CSA)"),
        (29, "Same-platform/pair transfer unlocked ($+286$\\,s vs.\\ StatMax/CSA)"),
        (31, "Multi-hop independent-belief recursion error ($-789$\\,s)"),
        (41, "Same-platform/pair transfer unlocked ($+339$\\,s vs.\\ StatMax/CSA)"),
        (49, "Multi-hop independent-belief recursion error ($-580$\\,s)"),
        (52, "Plat-pair unlocks conn.\\ 152 with missing sensor (bounded $[0,3600]$)"),
        (55, "Initial/downstream conn.\\ 179 missing sensor across all 8 arms"),
    ]
    micro_rows = []
    for ridx, mech in micro_specs:
        d = by_root[ridx]
        day = d["cheap_plat_async"]["day"][5:]
        orig_s = d["cheap_plat_async"]["origin"].replace("Zürich ", r"Z.\ ").replace("ü", r'\"u')
        goal_s = d["cheap_plat_async"]["goal"].replace("Zürich ", r"Z.\ ").replace("ü", r'\"u')
        od = f"{orig_s} $\\to$ {goal_s}"

        def _cell(arm_k):
            val = d[arm_k]["loss"]
            return "Unres." if val == "" else f"{int(float(val))}"

        micro_rows.append(
            f"{ridx} & {day} & {od} & {_cell('cheap_statmax_ideal')} & {_cell('snapshot_csa_plat_async')} & "
            f"\\textbf{{{_cell('cheap_plat_async')}}} & {_cell('recursive_plat_async')} & "
            f"{_cell('bounded_cert_plat_async')} & {_cell('full128_plat_async')} & {mech} \\\\"
        )
    table6_tex = "\n".join(micro_rows)

    tex = rf"""\documentclass[preprint,12pt,authoryear]{{elsarticle}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{microtype}}
\setlength{{\emergencystretch}}{{2em}}
\usepackage{{amsmath,amssymb,amsthm}}
\usepackage{{graphicx,booktabs,multirow,url}}
\usepackage{{setspace}}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{{hyperref}}

\setstretch{{1.22}}
\graphicspath{{{{figures/}}}}
\providecommand{{\newblock}}{{\ }}

\newtheorem{{proposition}}{{Proposition}}

\journal{{Transportation Research Part C: Emerging Technologies}}

\begin{{document}}

\begin{{frontmatter}}

\title{{Real-Time Adaptive Transit Routing Under Archival Scrutiny: Platform-Aware Distributional Selective Waiting and Four Offline Evaluation Pitfalls}}

\author[anon]{{Anonymous Author(s)}}
\affiliation[anon]{{organization={{Affiliation withheld for double-anonymous review}}}}

\begin{{abstract}}
Real-time delay feeds in dense urban rail networks have motivated increasingly
compute-intensive online routing controllers, including multi-hop dynamic
programming and Monte Carlo rollout search. Yet offline simulation studies
frequently conflate genuine lookahead value with four non-algorithmic artifacts:
heuristic candidate-menu truncation, mid-week General Transit Feed Specification
(GTFS) trip-identifier drift, conservative station-wide minimum connection times
(MCTs), and idealized zero-latency decision clocks. We conduct a multi-stage
causal attribution and deployment-feasibility study across 39 real operating
days of the Z\"urich S-Bahn network in 2023 (300 paired control journeys on 25
control days and {total_eval_opp:,} transfer opportunities on 33 out-of-period
days), coupling archived 1-second realization records (\emph{{ist-daten}}) with
historical GTFS-Realtime (GTFS-RT) forecast streams. Within a release-matched
correlated trip-chain delay and persistent-cancellation posterior, we formulate
\emph{{Platform-Aware Distributional Selective Waiting}} (\textbf{{PA-DSW}}), a
sub-second analytical controller that integrates arrival-time uncertainty across
sequential platform boarding options and single-transfer continuation envelopes
under a three-tier platform-pair MCT hierarchy. Controlled ablations on the
72-journey October development panel show that lifting a four-train candidate
cap, prefix-closing continuation menus, and repairing mid-week \texttt{{trip\_id}}
renamings (which alter 598 of 634 physical train identifiers at weekly schedule
cutovers) reduce mean journey time by {_fmt_s(total_oct_unlock, 2)}\,s per trip,
while distributional selective waiting outperforms deterministic mean-delay
routing by {_fmt_s(ci1_stoch_gain, 2)}\,s per trip at {_fmt_s(ci1_rep_cheap_cpu, 2)}\,s
CPU. Algebraically, we show that on ride legs with at most two transfer exits,
four distributional continuation rules collapse to identical scores, explaining
their 238-for-238 empirical tie across radial S-Bahn corridors. Separating
pre-onboard cancellations ($C_1$) from conditional onboard catch probabilities
($C_2$) across {total_reached_opp:,} resolved opportunities yields +43.1\% to
+50.5\% Brier skill over climatology while revealing a +3.1 percentage-point
weekday morning-peak skew in selective upstream catches. Finally, on a
pre-frozen 5-day April confirmation panel (60 trips, where 52 calm-operating
trips tie across all 8 arms), PA-DSW ({_fmt_s(arms['cheap_plat_async']['mean_cpu_seconds'], 2)}\,s
mean CPU; $\le {_fmt_s(cheap_d['max_single_decision_cpu_seconds'], 2)}$\,s per
decision against a {_fmt_s(dq['min'], 0)}\,s minimum and {_fmt_s(dq['median'], 0)}\,s
median station dwell) achieves {_fmt_s(g1_comp['complete_mean_saving_seconds'], 2, True)}\,s
point savings over both station-max MCTs and deterministic snapshot Connection
Scan Algorithm routing (95\% date-cluster CI
$[{_fmt_s(g1_comp['date_cluster_bootstrap']['ci95_saving_seconds'][0], 2, True)},
{_fmt_s(g1_comp['date_cluster_bootstrap']['ci95_saving_seconds'][1], 2, True)}]$\,s,
driven by 286--339\,s gains on affected corridors alongside one 304\,s
$\tau_{{\mathrm{{same}}}}=90$\,s boundary loss eliminated at $\tau_{{\mathrm{{same}}}}=60$\,s).
Conversely, multi-hop independent-belief recursion and {bc_worlds}-world
certified gating degrade mean travel time by
{_fmt_s(abs(rec_comp['complete_mean_saving_seconds']), 2)}\,s and
{_fmt_s(abs(bc_comp['complete_mean_saving_seconds']), 2)}\,s (95\% CI
$[{_fmt_s(bc_comp['date_cluster_bootstrap']['ci95_saving_seconds'][0], 2, True)},
{_fmt_s(bc_comp['date_cluster_bootstrap']['ci95_saving_seconds'][1], 2, True)}]$\,s,
including a +555\,s station-dwell overrun), and 128-world closed-loop rollouts
exhibit an oracle switching headroom of only 0.09\%--0.43\% across all 300
control journeys at a $40\times$ CPU cost.
\end{{abstract}}

\begin{{highlights}}
\item 39-day Z\"urich S-Bahn replay separates representation fidelity from online search.
\item Weekly GTFS cutovers rename 94.3\% of \texttt{{trip\_id}}s and cause 38 train collisions.
\item PA-DSW runs in $\le 0.56$\,s per decision, $30\times$ below minimum station dwell.
\item Four continuation rules collapse algebraically on legs with at most two exits.
\item 128-world rollouts add $<0.5\%$ oracle headroom over PA-DSW across 300 trips.
\end{{highlights}}

\begin{{keyword}}
public transit routing \sep GTFS-Realtime \sep adaptive hyperpaths \sep selective waiting \sep minimum connection time \sep offline policy evaluation
\end{{keyword}}

\end{{frontmatter}}

\section{{Introduction}}
\label{{sec:intro}}

High-frequency metropolitan rail networks—such as the Z\"urich S-Bahn, Paris
RER, and Tokyo suburban rail—operate at 5- to 15-minute headways through
multi-level transfer hubs where minor primary delays propagate into missed
connections \citep{{corman2018}}. With the widespread adoption of General Transit
Feed Specification Realtime (GTFS-RT) feeds, Advanced Traveler Information
Systems (ATIS) increasingly aim to guide passengers dynamically en route rather
than returning a single static pre-trip itinerary \citep{{gentile2005,chenNie,rambha,durner}}.
When an onboard passenger approaches an outer transfer hub on a delayed train,
the routing engine must decide—before the doors close and the train departs—whether
to recommend \emph{{alighting}} immediately for a local transfer alternative or
\emph{{continuing}} onboard toward a downstream central hub.

A growing literature formulates en-route transit guidance as a sequential
decision problem under uncertainty and proposes deeper online lookahead,
including multi-stage dynamic programming, Monte Carlo rollout, and partially
observable tree search \citep{{silver,somani2013,rambha}}. Yet a striking discrepancy
separates stylized simulation benchmarks—which often report travel-time
reductions of 5\% to 20\% for complex online search—from deployment realities
in well-instrumented rail networks. When evaluating adaptive transit controllers
on historical archives, apparent performance gaps between lightweight analytical
rules and heavy online simulation can be artificially inflated or inverted by
four non-algorithmic evaluation pitfalls:
\begin{{enumerate}}
  \item \textbf{{Candidate-menu truncation and objective mismatch}}: pre-pruning
  feasible transfer options (e.g., capping candidate departures per hub or
  omitting prefix-shared continuations) or comparing risk-penalized controllers
  against deterministic baselines that ignore timeout penalties;
  \item \textbf{{Mid-week GTFS timetable identity drift}}: joining archived
  GTFS-RT updates to static timetables by literal \texttt{{trip\_id}} strings
  across weekly schedule releases that silently rename or collide physical train
  identifiers;
  \item \textbf{{Station-wide transfer-buffer distortion}}: enforcing a uniform
  parent-station fallback Minimum Connection Time (MCT)—such as 420\,s at
  Z\"urich Hauptbahnhof (HB)—even on cross-island transfers sharing the exact
  same physical platform; and
  \item \textbf{{Idealized zero-latency decision clocks}}: crediting a
  multi-second simulation with an instantaneous action at the exact second of
  train arrival, ignoring physical station dwells that can be as short as
  17--28\,s at suburban rail stops.
\end{{enumerate}}

This paper investigates two linked questions for emerging real-time transit
guidance technologies:
\begin{{quote}}
\emph{{(1) How much realized travel-time value is attributable to candidate
completeness, timetable-identity alignment, platform-pair transfer rules, and
distributional selective waiting versus multi-hop recursion or closed-loop Monte
Carlo rollout? (2) What controller architecture lies on the practical
quality--latency Pareto frontier under physical station-dwell constraints?}}
\end{{quote}}

To answer both questions rigorously, we conduct a pre-registered, multi-stage
archival replay study across \textbf{{39 real operating days}} of the Z\"urich
S-Bahn network in 2023 (comprising 300 paired multi-arm control journeys on 25
control days and {total_eval_opp:,} transfer opportunities on 33 out-of-period
days), pairing official 1-second Swiss realization records (\emph{{ist-daten}})
with archived GTFS-RT protobuf streams. All historical stages are preserved
immutably without post-hoc parameter tuning, date manifests and candidate menus
are frozen prior to reading realized arrival or forecast payloads, and journeys
with missing sensor records are retained in every denominator via sharp
worst-case bounds.

\subsection{{Contributions and Organization}}

Our study makes four practical and methodological contributions to real-time
public transit routing:
\begin{{enumerate}}
  \item \textbf{{Sub-Dwell Platform-Aware Distributional Selective Waiting
  (PA-DSW)}}: Coupling an exact-anchor correlated trip-chain posterior with a
  three-tier platform-pair MCT contract, PA-DSW integrates discrete arrival and
  cancellation distributions over sequential platform offers and single-transfer
  commitment envelopes in $0.25$--$0.42$\,s mean CPU ($\le 0.56$\,s maximum
  single-decision latency, $>30\times$ below the $17$\,s shortest measured
  S-Bahn station dwell). Across cohorts, PA-DSW (and its station-max counterpart
  during controlled ablations) outperforms deterministic mean-delay routing by
  $+24.67$\,s per trip on the 72-trip October panel and $+10.59$\,s point
  saving on the 5-day April confirmation panel (286--339\,s on affected trips).

  \item \textbf{{Systematic Quantification of Four Offline Evaluation Pitfalls}}:
  Controlled ablations on October (`72` complete trips) demonstrate that lifting
  a 4-train hub cap saves {_fmt_s(v50_gain, 2)}\,s per trip, prefix-closing
  continuation menus saves {_fmt_s(dg1_menu_gain, 2)}\,s, and repairing mid-week
  GTFS \texttt{{trip\_id}} drift saves {_fmt_s(ci1_old_cheap - ci1_rep_cheap, 2)}\,s
  (totaling {_fmt_s(total_oct_unlock, 2)}\,s per trip), while audit of March
  GTFS schedule cutovers uncovers 598/634 literal \texttt{{trip\_id}} renamings
  and 38 physical-train collisions.

  \item \textbf{{Structural Explanation of Near-Zero Closed-Loop Search Headroom
  and Multi-Hop Degradation}}: We show algebraically that on ride legs with
  $m \le 2$ active transfer exits, four distributional continuation scoring
  rules collapse identically, explaining 238/238 ties across October and
  March--April panels. Across all 300 paired control trips, even an oracle
  switcher between PA-DSW and 128-world closed-loop rollout has only
  $0.09\%$--$0.43\%$ headroom ($1.3$--$5.2$\,s per trip; $0.00$\,s at
  $\tau_{{\mathrm{{same}}}}=60$\,s), whereas extending independent-trip backward
  induction to $m \ge 3$ hops degrades mean travel time by $23.60$--$28.24$\,s
  due to compound tail optimism and a realized $+555$\,s station-dwell overrun.

  \item \textbf{{Conditional-Onboard Calibration Separation ($C_1/C_2$) and
  Paired Override Certification}}: Separating pre-onboard cancellations ($C_1$)
  from conditional onboard four-partition catch calibration ($C_2$) across
  {total_reached_opp:,} resolved transfer events establishes $+43.1\%$ to
  $+50.5\%$ Brier skill over climatology while identifying a $+3.1$
  percentage-point weekday morning-peak skew on selective upstream catches
  ($P_{{10}}$). We also show that paired Student-$t$ override gating blocks
  403\,s of uncertified Monte Carlo flip losses on October while pruning
  $74.3\%$--$79.8\%$ of rollout queries.
\end{{enumerate}}

The remainder of the paper is organized as follows. Section~\ref{{sec:lit}}
positions our work against closest-neighbor transit routing and online planning
methods. Section~\ref{{sec:model}} formalizes the archival connection network,
the correlated trip-chain posterior model, platform-pair transfer contracts, and
physical dwell-window execution. Section~\ref{{sec:framework}} presents the
PA-DSW controller, operational accounting decomposition, two-exit algebraic
collapse, and dwell/certification analysis. Section~\ref{{sec:forensics}} reports
archival forensics across 33 out-of-period days. Section~\ref{{sec:results}}
presents empirical results across the October, March--April, and April panels,
including a complete trip-by-trip micro-audit of all non-tied April journeys.
Section~\ref{{sec:discussion}} synthesizes practical engineering guidelines and
study limitations, and Section~\ref{{sec:conclusion}} concludes.

\section{{Related Work and Positioning}}
\label{{sec:lit}}

Table~\ref{{tab:lit}} contrasts our framework and evaluation protocol against five
closest-neighbor methodological streams in public transit routing and online
planning under uncertainty.

\begin{{table}}[t]
\centering
\caption{{Comparison with Closest-Neighbor Real-Time Transit Routing and Online Planning Studies}}
\label{{tab:lit}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llllll}}
\toprule
Study / Framework & Delay \& Service Model & Recourse Policy Class & Transfer MCT Rule & Decision Clock & Empirical Evaluation Basis \\
\midrule
\citet{{gentile2005}}; \citet{{chenNie}} & Headway / stop countdowns & Frequency hyperpath + selective wait & Uniform stop node & Zero-latency & Analytical / synthetic networks \\
\citet{{kim2005}}; \citet{{rambha}} & Discrete Markov link/schedule states & Closed-loop MDP policy & Fixed station scalar & Zero-latency & Synthetic stochastic networks \\
\citet{{dibbelt2018}} (CSA-MEAT) & Independent parametric delay CDF & Decision-graph MEAT DP & Station footpath table & Zero-latency ($<0.2$\,s) & Perturbed static schedules \\
\citet{{durner}} (TopoCSA) & GTFS-RT delay + cancel PMFs & Precomputed stochastic hyperpath & Static GTFS transfers & Zero-latency & Calibrated European GTFS simulation \\
\citet{{silver}}; \citet{{somani2013}} & Sampled scenario / belief trees & POMCP / regularized DESPOT rollout & Domain-agnostic & Synchronous step & Stylized POMDP benchmarks \\
\midrule
\textbf{{This Study (PA-DSW + Audit)}} & \textbf{{Release-matched trip-chain}} & \textbf{{Sub-dwell selective waiting vs.}} & \textbf{{3-tier dynamic}} & \textbf{{Physical \emph{{ist-daten}}}} & \textbf{{39-day 1\,s archival replay +}} \\
 & \textbf{{Markov + persistent cancel}} & \textbf{{multi-hop \& certified rollout}} & \textbf{{platform-pair}} & \textbf{{dwell token}} & \textbf{{causal pitfall attribution}} \\
\bottomrule
\end{{tabular}}}}
\end{{table}}

\paragraph{{Adaptive Hyperpaths and Selective Waiting.}}
Classical frequency-based assignment models passenger recourse via attractive
line sets or hyperpaths \citep{{nguyen1988,spiess1989}}, extended to stochastic
time-dependent networks by \citet{{hickman1997}}, \citet{{millerHooks2000}}, and
\citet{{gao2006}}. \citet{{gentile2005}} and \citet{{chenNie}} formalized how real-time
arrival countdowns at stops induce \emph{{selective waiting}}—skipping an
immediately available slower vehicle when a faster service has high probability
of arriving shortly. In timetable networks, \citet{{dibbelt2018}} introduced the
Connection Scan Algorithm (CSA) and its Minimum Expected Arrival Time (MEAT)
decision-graph extension, while \citet{{geissmann}} and \citet{{durner}} incorporated
stochastic connection failures and GTFS-RT delay predictions. \citet{{bohmova2013}}
and \citet{{keyhani2013}} further showed on Swiss timetables that journey
reliability and transfer patterns exhibit strong structural regularity under
historical delays. Our PA-DSW controller builds directly on this analytical
selective-waiting lineage rather than claiming a new dynamic-programming
paradigm; our contribution lies in pairing exact-anchor trip-chain posteriors
with dynamic platform-pair readiness times and quantifying how data-identity and
contract artifacts govern realized archival performance.

\paragraph{{Online MDP/POMDP Lookahead and Anytime Execution.}}
\citet{{kim2005}} analyzed value-of-information pruning in real-time vehicle
routing, and \citet{{rambha}} formulated schedule-based adaptive transit routing
as a finite-horizon MDP. For large belief spaces, Monte Carlo rollout
\citep{{silver}} and regularized sparse scenario trees \citep{{somani2013}}
approximate closed-loop value of future information online. However, standard
evaluations assume planning completes instantaneously between environment steps.
In urban rail onboard guidance, solver latency competes directly against physical
door closure at intermediate stations ($17$--$60$\,s dwell), and finite-sample
rollouts on near-tied actions risk overriding sound analytical choices due to
sampling variance—motivating our physical dwell-token executor and paired
Student-$t$ override gate alongside proper probabilistic forecast verification
\citep{{gneiting2007}}.

\section{{Archival System Model, Delay Posterior, and Operational Contracts}}
\label{{sec:model}}

\subsection{{Connection Network, Causal Information Filtration, and Loss Bounds}}

Consider operating day $d$ indexed in local civil service seconds
$t \in \mathbb{{R}}_+$ (accounting for Europe/Zurich daylight-saving transitions).
A published static GTFS release $g_d$ defines stops $\mathcal{{S}}$, physical
vehicle trips $\mathcal{{V}}$, and elementary connections $\mathcal{{C}}$. Each
connection $c \in \mathcal{{C}}$ represents a nonstop movement of trip
$v(c) \in \mathcal{{V}}$ from departure stop/platform $s^-(c) \in \mathcal{{S}}$
at scheduled integer second $\tilde{{t}}^-(c)$ to arrival stop/platform
$s^+(c) \in \mathcal{{S}}$ at scheduled second $\tilde{{t}}^+(c)$.

In the realized operating day $\omega$, each connection $c$ either operates with
realized departure and arrival timestamps $(T^-(c,\omega), T^+(c,\omega))$
recorded at 1-second resolution in the official Swiss \emph{{ist-daten}} archive
\citep{{swissactual}}, or is canceled ($\chi(c,\omega) = 1$). Concurrently,
archived GTFS-RT protobuf snapshots emit real-time delay and status updates. To
prevent anticipative information leakage, an entity update $e$ inside archive
snapshot $m$ with feed header timestamp $t_{{\mathrm{{hdr}}}}(m)$, capture
timestamp $t_{{\mathrm{{cap}}}}(m)$, and entity timestamp $t_{{\mathrm{{upd}}}}(e)$
enters the traveler's causal filtration $\mathcal{{F}}_t$ only at receipt clock
\begin{{equation}}
  t_{{\mathrm{{rec}}}}(m,e) \;=\; \max\bigl\{{t_{{\mathrm{{cap}}}}(m),\; t_{{\mathrm{{hdr}}}}(m),\; t_{{\mathrm{{upd}}}}(e)\bigr\}} \;\le\; t ,
  \label{{eq:filtration}}
\end{{equation}}
and silence in the feed is never treated as confirmation of punctual operation.

A routing query $q = (o, d_{{\mathrm{{goal}}}}, t_0, C)$ specifies origin $o$,
destination $d_{{\mathrm{{goal}}}}$, start time $t_0$, and loss cap $C = 3600$\,s.
At an \emph{{onboard state}} upon arriving via $c$ at $s^+(c) \neq d_{{\mathrm{{goal}}}}$
at $t = T^+(c,\omega)$, if trip $v(c)$ continues via $c_{{\mathrm{{next}}}}$, the
controller selects $a \in \{{\texttt{{continue}}, \texttt{{alight}}\}}$ (with
\texttt{{alight}} mandatory when $v(c)$ terminates). At a \emph{{platform state}}
at hub $S$, the controller commits to waiting for a candidate boarding connection
within the pre-frozen route menu $\mathcal{{H}}(q)$ (subject to a 12-minute
per-commitment patience timeout before re-evaluating remaining options, held
identical across all compared arms). Realized journey loss is
$L(\pi,\omega) = T_{{\mathrm{{arr}}}}(\pi,\omega) - t_0$ if $d_{{\mathrm{{goal}}}}$
is reached by $t_0 + C$, and $L(\pi,\omega) = C$ otherwise. Whenever an archival
sensor timestamp along the realized trajectory is missing (\texttt{{MISSING}} or
unrecorded \texttt{{PROGNOSE}}), we neither drop the trip nor impute values,
retaining exact worst-case bounds
$L(\pi,\omega) \in [\ell(\pi,\omega), u(\pi,\omega)] \subseteq [0, C]$.

\subsection{{Correlated Trip-Chain Delay and Service-State Posterior}}
\label{{sec:posterior}}

To ensure that online delay and cancellation beliefs reflect railway physics
rather than conjugate algebraic convenience, both planning and rollout world
sampling share a unified discrete-support hidden Markov model along each physical
trip $v \in \mathcal{{V}}$ (\texttt{{EvolvingTripBelief}} with exact integer-second
anchor conditioning):

\begin{{enumerate}}
  \item \textbf{{Ordered Trip Event Chain and Exact Anchor Support}}. Trip $v$
  consists of $2L_v$ ordered event slots
  $(e_{{v,0}}, e_{{v,1}}, \dots, e_{{v,2L_v-1}})$ alternating departures and
  arrivals with scheduled seconds $\tilde{{t}}_{{v,\ell}}$. Each slot carries
  latent state $(T_{{v,\ell}}, \chi_{{v,\ell}}) \in \mathcal{{T}}_{{v,\ell}} \times \{{0,1\}}$,
  where $\mathcal{{T}}_{{v,\ell}}$ augments the discrete delay-shift grid around
  $\tilde{{t}}_{{v,\ell}}$ with all exact integer-second timestamps directly
  observed by the traveler on trip $v$ up to clock $t$. Thus first-hand arrival
  and departure observations condition the chain as exact point masses without
  grid-rounding error.

  \item \textbf{{Initial Route Prior and Monotone Segment Transitions}}. For trip
  $v$ on route $r(v)$, historical calibration on pre-period release-matched
  archives supplies route prior mean delay $\mu_r$, standard deviation
  $\sigma_r$, and baseline cancellation probability $q_r$. Slot $\ell=0$ is
  initialized with $\Pr(\chi_{{v,0}}=1) = q_r$ and discretized Gaussian clock
  log-weights centered at $\tilde{{t}}_{{v,0}} + \mu_r$ with scale
  $\max(\sigma_r, 30\,\text{{s}})$. Between consecutive slots $\ell-1 \to \ell$,
  service availability evolves via a persistent two-state Markov kernel (with
  cancellation persistence $\rho_{{\mathrm{{cxl}}}} = 0.90$, capturing partial or
  full-line service suspensions), while operating clocks enforce physical
  monotonicity ($T_{{v,\ell}} \ge T_{{v,\ell-1}}$) with Gaussian increment
  dispersions $(\sigma_{{\mathrm{{run}}}}, \sigma_{{\mathrm{{dwell}}}})$
  distinguishing inter-station running segments from station dwells.

  \item \textbf{{Causal Forecast Emissions and Wait-Censorship Conditioning}}.
  Each release-matched GTFS-RT \texttt{{StopTimeUpdate}} received by clock $t$
  for slot $(v,\ell)$ with predicted timestamp $\hat{{t}}_{{v,\ell}}$ and
  historical forecast-age residual scale $\hat{{s}}_{{v,\ell}} \ge 1$\,s enters
  as log-likelihood $-\frac{{1}}{{2}}\bigl((T_{{v,\ell}} - \hat{{t}}_{{v,\ell}})/\hat{{s}}_{{v,\ell}}\bigr)^2$
  on non-canceled states ($\chi_{{v,\ell}}=0$), while explicit GTFS-RT
  \texttt{{CANCELED}} / \texttt{{SKIPPED}} flags clamp $\chi_{{v,\ell}}=1$. If a
  traveler waits on a platform over interval $[t_{{\mathrm{{entry}}}}, t]$
  without observing connection $c$, operating support inside
  $[t_{{\mathrm{{entry}}}}, t]$ is ruled out—correctly preserving both late-running
  ($T^- > t$) and canceled ($\chi = 1$) posterior hypotheses without the classic
  $1 - \mathrm{{CDF}}$ mass-loss bug. Exact marginal distributions
  $\bigl(p_{{v,\ell}}(\tau), q_{{v,\ell}}\bigr)$ and posterior forward-filtering
  backward-sampling (FFBS) trajectories are obtained in linear chain time via
  stabilized \texttt{{logsumexp}} message passing.
\end{{enumerate}}

\subsection{{Three-Tier Platform-Pair Transfer Contract and Physical Dwell Clock}}
\label{{sec:contracts}}

In Swiss GTFS schedules, hub stop identifiers distinguish parent stations (e.g.,
\texttt{{8503000}} for Z\"urich HB) from platform-specific track children (e.g.,
\texttt{{8503000:0:41/42}} for underground S-Bahn island tracks 41/42). Because
GTFS \texttt{{transfers.txt}} omits self-transfer rows from a platform child to
itself, naive station-level routing assigns the parent station's worst-case
cross-level buffer $\tau_{{\mathrm{{stat}}}}(S)$ (420\,s at Z\"urich HB) even
when incoming connection $c_{{\mathrm{{in}}}}$ and outgoing connection
$c_{{\mathrm{{out}}}}$ use the same island platform.

To separate published timetable geometry from modeling assumptions transparently,
our platform-pair transfer contract defines
$\tau_{{\mathrm{{pair}}}}(c_{{\mathrm{{in}}}}, c_{{\mathrm{{out}}}}, h_t)$ via a
three-tier hierarchy using pre-decision platform identifiers
$\sigma^\pm(\cdot, h_t)$ (set to the latest GTFS-RT platform announcement
received by time $t$ if present, and the static GTFS platform child otherwise):
\begin{{equation}}
  \tau_{{\mathrm{{pair}}}}(c_{{\mathrm{{in}}}}, c_{{\mathrm{{out}}}}, h_t) \;=\;
  \begin{{cases}}
    \tau_{{\mathrm{{explicit}}}}\bigl(\sigma^+(c_{{\mathrm{{in}}}}, h_t),\,\sigma^-(c_{{\mathrm{{out}}}}, h_t)\bigr),
      & \text{{published \texttt{{transfers.txt}} pair,}} \\
    \tau_{{\mathrm{{same}}}},
      & \sigma^+(c_{{\mathrm{{in}}}}, h_t) = \sigma^-(c_{{\mathrm{{out}}}}, h_t) \text{{ (same platform),}} \\
    \tau_{{\mathrm{{parent}}}}(S),
      & \text{{otherwise (station fallback),}}
  \end{{cases}}
  \label{{eq:pair_mct}}
\end{{equation}}
where $\tau_{{\mathrm{{same}}}} = 90$\,s is an explicit modeling parameter tested
over pre-registered sensitivity grid $\tau_{{\mathrm{{same}}}} \in \{{60, 90, 120\}}$\,s.
Connection $c_{{\mathrm{{out}}}}$ is physically boardable in world $\omega$ iff
$\chi(c_{{\mathrm{{out}}}},\omega)=0$ and
$T^-(c_{{\mathrm{{out}}}},\omega) \ge T^+(c_{{\mathrm{{in}}}},\omega) + \tau_{{\mathrm{{pair}}}}(c_{{\mathrm{{in}}}}, c_{{\mathrm{{out}}}}, h_t)$.

At each continuation onboard state arriving at $t_{{\mathrm{{arr}}}} = T^+(c,\omega)$,
the physical door-open dwell interval is
$\mathcal{{W}}(c,\omega) = [t_{{\mathrm{{arr}}}},\, t_{{\mathrm{{close}}}}(c,\omega)]$
with $t_{{\mathrm{{close}}}}(c,\omega) = \max\{{T^-(c_{{\mathrm{{next}}}},\omega),\, t_{{\mathrm{{arr}}}}+1\,\text{{s}}\}}$.
Under our non-anticipative \texttt{{ActionPublisher}} execution clock, the
controller never observes $t_{{\mathrm{{close}}}}(c,\omega)$:
\begin{{enumerate}}
  \item Base planning consumes measured process CPU $\tau_{{\mathrm{{base}}}}$; if
  $t_{{\mathrm{{arr}}}} + \tau_{{\mathrm{{base}}}} < t_{{\mathrm{{close}}}}(c,\omega)$,
  it registers fallback action token $a_{{\mathrm{{base}}}}$, whereas if base
  planning itself overruns $t_{{\mathrm{{close}}}}(c,\omega)$, the train has
  already departed and \texttt{{continue}} is physically forced (\texttt{{fallback\_expired}}).
  \item Any optional rollout gate consuming $\tau_{{\mathrm{{gate}}}}$ updates the
  executed action to $a_{{\mathrm{{prop}}}}$ if and only if
  $t_{{\mathrm{{arr}}}} + \tau_{{\mathrm{{base}}}} + \tau_{{\mathrm{{gate}}}} \le t_{{\mathrm{{close}}}}(c,\omega)$;
  otherwise \texttt{{deadline\_expired}} is logged and $a_{{\mathrm{{base}}}}$ stands.
\end{{enumerate}}

\section{{Analytical Framework: PA-DSW Controller, Structural Collapse, and Certification}}
\label{{sec:framework}}

\subsection{{Operational Accounting Decomposition and Partial Identification}}

Let $J(\pi; \mathcal{{H}}, \tau) = \mathbb{{E}}_\omega[L(\pi,\omega; \mathcal{{H}}, \tau)]$
denote expected journey loss of causal policy $\pi$ executed under candidate menu
$\mathcal{{H}}$ and transfer contract $\tau$, and let $J^*(\mathcal{{H}}, \tau)$
denote the exact causal optimum within $(\mathcal{{H}}, \tau)$. Standard
telescoping addition and subtraction decomposes suboptimality relative to the
uncapped platform-pair benchmark $J^*(\mathcal{{H}}_{{\mathrm{{wide}}}}, \tau_{{\mathrm{{pair}}}})$
into menu restriction $\Delta^*_{{\mathrm{{menu}}}} = J^*(\mathcal{{H}}_0,\tau_0) - J^*(\mathcal{{H}}_{{\mathrm{{wide}}}},\tau_0)$,
contract distortion $\Delta^*_{{\mathrm{{MCT}}}} = J^*(\mathcal{{H}}_{{\mathrm{{wide}}}},\tau_0) - J^*(\mathcal{{H}}_{{\mathrm{{wide}}}},\tau_{{\mathrm{{pair}}}})$,
and within-contract policy/belief approximation $J(\pi;\mathcal{{H}}_0,\tau_0) - J^*(\mathcal{{H}}_0,\tau_0)$.

Two methodological distinctions are essential when interpreting empirical
ablations against this accounting identity:
\begin{{enumerate}}
  \item \emph{{Policy-conditional vs.\ oracle differences}}: Because computing
  the exact closed-loop POMDP optimum $J^*$ over large real-time networks is
  intractable, sequential archival ablations holding a fixed base controller
  $\pi^{{\mathrm{{DSW}}}}$ measure \emph{{policy-conditional}} increments—such as
  $J(\pi^{{\mathrm{{DSW}}}};\mathcal{{H}}_0,\tau_0) - J(\pi^{{\mathrm{{DSW}}}};\mathcal{{H}}_{{\mathrm{{wide}}}},\tau_0)$—and
  approximate residual closed-loop lookahead headroom via high-budget paired
  Monte Carlo rollouts.
  \item \emph{{Non-monotonicity of buffer relaxation for approximate policies}}:
  Pointwise $\tau_{{\mathrm{{pair}}}} \le \tau_{{\mathrm{{stat}}}}$ guarantees
  $J^*(\mathcal{{H}},\tau_{{\mathrm{{pair}}}}) \le J^*(\mathcal{{H}},\tau_{{\mathrm{{stat}}}})$
  for exact optimal policies because every feasible strategy under
  $\tau_{{\mathrm{{stat}}}}$ remains feasible under $\tau_{{\mathrm{{pair}}}}$.
  For an open-loop approximate controller, however, unlocking a tight connection
  near an assumed $\tau_{{\mathrm{{same}}}}$ boundary can occasionally induce an
  inferior realized path on an individual realization (as documented on Trip~3 in
  Section~\ref{{sec:confirm}}).
\end{{enumerate}}
When $U$ of $N$ journeys have unresolved sensor endpoints bounded in $[0,C]$,
the sample mean paired saving $\bar{{G}}_N = \frac{{1}}{{N}}\sum_{{i=1}}^N [L(\pi_0,\omega_i) - L(\pi_1,\omega_i)]$
is sharply partially identified without dropping any trip via
\begin{{equation}}
  \frac{{1}}{{N}}\sum_{{i=1}}^N \bigl[\ell(\pi_0, \omega_i) - u(\pi_1, \omega_i)\bigr]
  \;\le\; \bar{{G}}_N \;\le\;
  \frac{{1}}{{N}}\sum_{{i=1}}^N \bigl[u(\pi_0, \omega_i) - \ell(\pi_1, \omega_i)\bigr] .
  \label{{eq:missing_bounds}}
\end{{equation}}

\subsection{{Platform-Aware Distributional Selective Waiting (PA-DSW) and Two-Exit Collapse}}

Consider a traveler arriving via $c_{{\mathrm{{in}}}}$ at hub $S$ at time
$t_{{\mathrm{{arr}}}}$, facing ordered candidate rides $\mathcal{{M}} = (m_1,\dots,m_K)$
ranked by downstream nominal continuation score. Under~\eqref{{eq:pair_mct}},
candidate $m_k$ has pair-specific walk-readiness time
$r_k = t_{{\mathrm{{arr}}}} + \tau_{{\mathrm{{pair}}}}(c_{{\mathrm{{in}}}}, c(m_k), h_t)$,
posterior operating departure mass $p_k(t)$ on $\mathcal{{D}}_k$, cancellation
mass $q_k$, and downstream continuation score $V_k(t)$ given departure at $t$.
The probability that higher-ranked candidate $m_j$ ($j < k$) fails to provide a
walk-reachable departure by clock $t$ is
\begin{{equation}}
  S_j^{{\mathrm{{unavail}}}}(t) \;=\;
  \begin{{cases}}
    1, & t < r_j , \\
    q_j \;+\; \displaystyle\sum_{{\tau \in \mathcal{{D}}_j:\, \tau < r_j}} p_j(\tau) \;+\; \displaystyle\sum_{{\tau \in \mathcal{{D}}_j:\, \tau > t}} p_j(\tau), & t \ge r_j ,
  \end{{cases}}
  \label{{eq:surv_exact}}
\end{{equation}}
which sums nonnegative probability masses directly rather
than evaluating $1 - \mathrm{{CDF}}$. Under sequential priority boarding
(\texttt{{first\_offer}}), the traveler boards the earliest walk-reachable
departure according to rank order, with expected loss
\begin{{equation}}
  V_{{\mathrm{{first}}}}(h_t; \mathcal{{M}}) \;=\;
  \sum_{{k=1}}^K \sum_{{t \in \mathcal{{D}}_k:\, r_k \le t \le t_0+C}}
    p_k(t)\, V_k(t) \prod_{{j=1}}^{{k-1}} S_j^{{\mathrm{{unavail}}}}(t)
  \;+\; C \prod_{{j=1}}^K S_j^{{\mathrm{{unavail}}}}(t_0+C) .
  \label{{eq:first_offer_val}}
\end{{equation}}
To allow intentional skipping of an immediate slower train when a faster service
is expected shortly, PA-DSW pairs~\eqref{{eq:first_offer_val}} with single-target
commitment scores
$V_{{\mathrm{{target}}}}(h_t; m_k) = \sum_{{t \in \mathcal{{D}}_k:\, r_k \le t \le t_0+C}} p_k(t) V_k(t) + C\, S_k^{{\mathrm{{unavail}}}}(t_0+C)$
and evaluates each next transfer hub via the selective-waiting envelope
\begin{{equation}}
  V_{{\mathrm{{PA\text{{-}}DSW}}}}(h_t; \mathcal{{M}}) \;=\;
  \min\Bigl\{{V_{{\mathrm{{first}}}}(h_t; \mathcal{{M}}),\; \min_{{1 \le k \le K}} V_{{\mathrm{{target}}}}(h_t; m_k)\Bigr\}} ,
  \label{{eq:padsw_env}}
\end{{equation}}
executing~\eqref{{eq:padsw_env}} at platform states via combined waiting control
(\texttt{{combined}}) and scoring onboard \texttt{{alight}} vs.\
\texttt{{continue}} decisions across downstream exits along the current ride leg.

Four analytical variants of onboard continuation scoring have been considered in
our diagnostic progression: fixed best-exit continuation (\texttt{{cheap}} /
PA-DSW), independent multi-exit minimum (\texttt{{single}}), adaptive exit-pair
recourse (\texttt{{pair}}), and one-step arrival/message lookahead to the next
exit (\texttt{{one\_step}}). Proposition~\ref{{prop:collapse}} explains why all
four variants behave identically on radial two-hub corridors (proof in
Appendix~\ref{{app:proofs}}).

\begin{{proposition}}[Heterogeneous Selective-Waiting Complexity and Two-Exit Collapse]
\label{{prop:collapse}}
\begin{{enumerate}}
  \item[\emph{{(a)}}] Given trip marginals under conditional independence given
  $h_t$, both $V_{{\mathrm{{first}}}}(h_t;\mathcal{{M}})$ in~\eqref{{eq:first_offer_val}}
  and $V_{{\mathrm{{PA\text{{-}}DSW}}}}(h_t;\mathcal{{M}})$ in~\eqref{{eq:padsw_env}}
  are computable in $O\bigl(K \max_k |\mathcal{{D}}_k|\bigr)$ operations using
  prefix and suffix mass sums without floating-point tail cancellation.
  \item[\emph{{(b)}}] Suppose an active onboard ride leg has $m \le 2$ transfer
  exits $(e_1, e_2)$—an upstream transfer hub $e_1$ and a mandatory terminal
  alighting exit $e_2$—with deterministic incremental running time between
  observed arrival $T^+(e_1)=t_1$ and $e_2$ under fixed posterior law
  $\mathbb{{P}}(\cdot \mid h_{{t_1}})$. Then at $e_1$, the action values and
  selected actions of fixed-exit (PA-DSW), \texttt{{single}}, \texttt{{pair}},
  and \texttt{{one\_step}} coincide identically for both
  $a \in \{{\texttt{{alight}}, \texttt{{continue}}\}}$.
\end{{enumerate}}
\end{{proposition}}

Because radial S-Bahn trips entering central Z\"urich typically traverse one
outer interchange ($e_1$, e.g., Oerlikon or Hardbr\"ucke) followed by one central
hub ($e_2$, e.g., Z\"urich HB or Stadelhofen), $m \le 2$ holds on nearly all
onboard states, explaining the exact 238-for-238 trajectory tie across October
and March--April panels. Differentiating continuation rules requires $m \ge 3$
active exits on the same ride leg; however, as tested via \textbf{{Multi-Hop-DSW}}
in Section~\ref{{sec:confirm}}, recursively taking minima across $m \ge 3$
downstream hubs under independent trip marginals both inflates tree-evaluation
CPU and compounds optimistic tail errors when corridor delays are positively
correlated.

\subsection{{Dwell-Window Feasibility and Paired Student-$t$ Override Certification}}

When an optional $M_0$-world paired rollout (\textbf{{Cert-Rollout-64}},
$M_0 = {bc_worlds}$) is invoked on ambiguous onboard states satisfying
$|\Delta_{{\mathrm{{base}}}}| = |Q_{{\mathrm{{base}}}}(\texttt{{continue}}) - Q_{{\mathrm{{base}}}}(\texttt{{alight}})| \le \gamma$
($\gamma = {_fmt_s(bc_margin, 0)}$\,s, CPU budget {_fmt_s(bc_budget, 1)}\,s), let
$\{{D_m\}}_{{m=1}}^{{M_0}}$ denote common-random-number world loss differences
$D_m = \hat{{L}}_m(\texttt{{continue}}) - \hat{{L}}_m(\texttt{{alight}})$ with
sample mean $\bar{{D}}_{{M_0}}$ and sample standard deviation $\hat{{s}}_{{M_0}}$.
Standard properties of asynchronous execution and paired $t$-pivots characterize
when online simulation can—and cannot—improve upon $a_{{\mathrm{{base}}}}$
(derivation in Appendix~\ref{{app:proofs}}):

\begin{{proposition}}[Dwell-Expiration Nullification and Paired Override Error Control]
\label{{prop:anytime}}
Under the physical dwell window $W = t_{{\mathrm{{close}}}} - t_{{\mathrm{{arr}}}} > 0$
with base latency $\tau_{{\mathrm{{base}}}} < W$ and rollout latency
$\tau_{{M_0}}$:
\begin{{enumerate}}
  \item[\emph{{(i)}}] On the event $\tau_{{\mathrm{{base}}}} + \tau_{{M_0}} > W$,
  any proposed override misses physical door closure, yielding zero realized
  change relative to $a_{{\mathrm{{base}}}}$ while expending $\tau_{{M_0}}$ CPU
  seconds (whereas if base planning itself exceeds $W$, \texttt{{continue}} is
  forced unconditionally).
  \item[\emph{{(ii)}}] Suppose an override of $a_{{\mathrm{{base}}}}$ is
  authorized only when $\tau_{{\mathrm{{base}}}} + \tau_{{M_0}} \le W$,
  $|\Delta_{{\mathrm{{base}}}}| \le \gamma$, $|\bar{{D}}_{{M_0}}| \ge \delta_{{\min}}$
  ($5$\,s), and the two-sided $(1-\alpha)$ paired Student-$t$ interval
  $\bar{{D}}_{{M_0}} \pm z_{{1-\alpha/2}} \hat{{s}}_{{M_0}} / \sqrt{{M_0}}$
  ($z_{{0.975}} = 1.96$) excludes zero in the direction opposing
  $a_{{\mathrm{{base}}}}$. Under approximate normality of $\bar{{D}}_{{M_0}}$
  around $\mu_D = \mathbb{{E}}[D_1 \mid h_t]$, the conditional probability of a
  noise-induced false override when $\mu_D$ agrees in sign with
  $a_{{\mathrm{{base}}}}$ is at most $\alpha/2 = 0.025$.
\end{{enumerate}}
\end{{proposition}}

Auditing all 136 saved rollout gates on the October cohort confirms both parts of
Proposition~\ref{{prop:anytime}}: the 90\,s ambiguity screen prunes 74.3\%
(101/136) of gates on October and {pruned_bg5_pct:.1f}\% (95/119) on April, while
Student-$t$ certification blocks four uncertified coin-flip overrides on October
($|t| \in \{{0.14, 0.38, 0.47, 1.01\}}$), preventing 403\,s of cumulative
realized travel-time loss (263\,s on Trip~4 and 140\,s on Trip~32) while
retaining the sole certified October rollout win (+96\,s on Trip~62, $t = -3.83$).

\section{{Archival Forensics Across 33 Out-of-Period Operating Days}}
\label{{sec:forensics}}

Table~\ref{{tab:cohorts}} summarizes the four chronological archival panels
comprising our 39-day study. All heavy raw archives remain on a Linux compute
server; compact JSON/CSV statistical receipts are audited locally with zero
additional replay budget on previously exposed panels.

\begin{{table}}[t]
\centering
\caption{{Overview of the 39-Operating-Day Z\"urich S-Bahn Archival Study (2023)}}
\label{{tab:cohorts}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrll}}
\toprule
Archival Panel & Service Dates (2023) & Days & Trips & Primary Experimental / Forensic Role & Governance Status \\
\midrule
October Development Panel & Oct 25--28, 30--31$^\dagger$ & 6 & 72 & Menu cap, prefix closure \& ID repair ablations & Locked ($0$ extra budget) \\
March Calibration Panel & Mar 09--Mar 22 & 14 & 168 & Out-of-period catch calibration \& GTFS ID audit & Frozen calibration receipt \\
March--April Station-Max Panel & Mar 23--Apr 05 & 14 & 168 & Unexposed 8-arm station-max control evaluation & Frozen historical receipt \\
April Confirmation Panel & {bg_dates_str} & 5 & 60 & Unexposed platform-pair + physical-dwell test & Single pre-frozen run \\
\midrule
\textbf{{Total Archival Program}} & \textbf{{Mar--Apr \& Oct 2023}} & \textbf{{39}} & \textbf{{468}}$^\ddagger$ & \textbf{{300 paired control trips; {total_eval_opp:,} transfer events}} & \textbf{{Zero post-hoc tuning}} \\
\bottomrule
\end{{tabular}}}}
\par\vspace{{3pt}}\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes:}} $^\dagger$Six October control days (12 morning-peak trips/day = 72 trips) follow single-day schema checks. $^\ddagger$Comprises 300 paired multi-arm control trips (72 Oct + 168 Mar--Apr + 60 Apr) and 168 calibration trips (Mar 09--22). All 72 previously exposed project dates were excluded prior to freezing {bg_dates_str}. Across the 33 out-of-period days, {total_eval_opp:,} onboard/registered transfer opportunities are audited ({dc2_onboard_total:,} in Mar 09--22 [{dc2_reached_res:,} resolved, {dc2_reached_unres} unresolved sensor], {ng2_reg_total:,} in Mar 23--Apr 05 [{ng2_reached_res:,} resolved, 19 pre-onboard canceled], and {bg5_reg_total} in Apr 06--10 [{bg5_reached_res} resolved]).\end{{minipage}}
\end{{table}}

\subsection{{Pitfall 2 Forensics: Mid-Week GTFS Schedule Cutovers and \texttt{{trip\_id}} Collisions}}

Auditing archived GTFS-RT feeds against weekly static GTFS releases on the March
Calibration Panel uncovered a major archival pitfall for public transit
researchers. Swiss national static GTFS snapshots are published every Wednesday
at \texttt{{04:15}} and cut over in the live GTFS-RT stream around Wednesday
\texttt{{14:00}} local time. Matching all 634 active corridor trips across March
16--19 by physical train number, operator, ordered stop sequence, and scheduled
timestamps confirmed exact 1-to-1 physical service continuity, yet \textbf{{598
of 634 trips (94.3\%) changed their literal \texttt{{trip\_id}} key}}, and
\textbf{{38 literal \texttt{{trip\_id}} strings were reassigned to completely
different physical trains}}. Joining historical GTFS-RT archives to a single
static release by literal \texttt{{trip\_id}} corrupts up to 68\% of training or
online forecast updates. Likewise, on October 25 morning (prior to the Wednesday
afternoon cutover), matching to the active October 18 release required remapping
900 of 6,124 connection endpoints. All subsequent panels freeze release-matched
service-signature mappings strictly before decoding real-time protobuf payloads.

\subsection{{Separating Service Availability ($C_1$) from Conditional Onboard Calibration ($C_2$)}}

At each registered first-transfer opportunity, binary catch indicators
$Y_{{\mathrm{{up}}}}, Y_{{\mathrm{{dn}}}} \in \{{0,1\}}$ at the upstream and
downstream hubs partition outcomes into four mutually exclusive joint events
$\mathcal{{K}} = \{{00, 01, 10, 11\}}$, where $10$ ($Y_{{\mathrm{{up}}}}=1, Y_{{\mathrm{{dn}}}}=0$)
marks the selective-alighting regime in which alighting upstream avoids a missed
central-hub transfer.

On the March--April Panel, two trips (Trips 151 and 160) experienced official
pre-boarding cancellations (\texttt{{FAELLT\_AUS\_TF = true}}), leaving 19
downstream opportunities physically unreached. Mixing unreached pre-boarding
cancellations as $[-1, +1]$ bounds into onboard calibration inflates
identification width to $0.0603$, wider than any $\pm 0.02$ tolerance band
irrespective of posterior accuracy. Separating calibration into two orthogonal
metrics resolves this confounding:
\begin{{itemize}}
  \item \textbf{{Metric $C_1$ (Initial Service Availability)}}: fraction of
  public queries whose initial scheduled train operates to the first onboard
  decision stop ($153/154 = 99.35\%$ in Mar 09--22; $166/168 = 98.81\%$ in Mar
  23--Apr 05; $60/60 = 100.00\%$ in Apr 06--10, with 55 transfer-bearing trips
  registering {bg5_reg_total} transfer opportunities and $0$ pre-onboard
  cancellations).
  \item \textbf{{Metric $C_2$ (Conditional-Onboard Four-Partition Calibration)}}:
  evaluated on physically reached onboard states with resolved sensor endpoints
  ($n = {dc2_reached_res:,}$ in Mar 09--22, $n = {ng2_reached_res:,}$ in Mar
  23--Apr 05, and $n = {bg5_reached_res}$ in Apr 06--10; total
  $n = {total_reached_opp:,}$, with exact algebraic recovery from saved cluster
  bounds documented in Appendix~\ref{{app:audit}}).
\end{{itemize}}

Figure~\ref{{fig:cal_plat}}(a) reports 20,000-replicate whole-date cluster
bootstrap residuals $\bar{{y}}_k - \bar{{p}}_k$ across all 33 out-of-period days.
First, the release-matched trip-chain posterior demonstrates substantial
out-of-sample probabilistic skill: 4-class Brier score is $0.1233$ in Mar 09--22
(+43.1\% skill vs.\ empirical climatology $0.2166$), $0.1167$ in Mar 23--Apr 05,
and ${_fmt_s(bg5_c2['mean_brier_4class'], 4)}$ in Apr 06--10
(${_fmt_s(100.0 * bg5_c2['brier_skill_vs_climatology'], 1, True)}\%$ skill vs.\
climatology ${_fmt_s(bg5_c2['climatology_brier_4class'], 4)}$). Second, across
both weekday-dominated March panels, $P_{{10}}$ exhibits a systematic positive
residual of $+3.25$\,pp in Mar 09--22 (95\% CI $[+1.67, +4.77]$\,pp) and
$+3.12$\,pp in Mar 23--Apr 05 (95\% CI $[+1.93, +4.32]$\,pp, positive on 13 of 14
days, but dropping to $+0.99$\,pp and $+1.49$\,pp on weekend March 25--26). In an
exploratory comparison consistent with \citet{{corman2018}}'s hub-throat secondary
congestion mechanism, the April panel spans workday Maundy Thursday
(\texttt{{2023-04-06}}, $P_{{10}}$ residual $+{_fmt_s(p10_apr06, 2)}$\,pp)
followed by four Easter holiday/weekend days (\texttt{{2023-04-07}} to
\texttt{{2023-04-10}}), on which daily $P_{{10}}$ residuals shift to
${_fmt_s(p10_apr07, 2)}$, ${_fmt_s(p10_apr08, 2)}$, ${_fmt_s(p10_apr09, 2)}$, and
${_fmt_s(p10_apr10, 2)}$\,pp (5-day mean
${_fmt_s(100.0 * bg5_c2['date_cluster_inference']['mean_residual']['10'], 2, True)}$\,pp,
95\% CI $[{_fmt_s(100.0 * bg5_c2['date_cluster_inference']['ci95_marginal']['10'][0], 2, True)},
{_fmt_s(100.0 * bg5_c2['date_cluster_inference']['ci95_marginal']['10'][1], 2, True)}]$\,pp).
Because only one workday is present in the 5-day April window, we treat this
weekday-versus-holiday contrast as an exploratory diagnostic motivating
peak-regime delay conditioning in future emission models.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\textwidth]{{fig2_calibration_platform.pdf}}
\caption{{Conditional-Onboard Joint Calibration ($C_2$, $n={total_reached_opp:,}$ Resolved Opportunities Across 33 Days) and Platform-Pair Contract Composition}}
\label{{fig:cal_plat}}
\end{{figure}}

\subsection{{Pitfall 3 Forensics: Platform-Pair Coverage and Live Track Reassignments}}

Inspection of 2023 Swiss \emph{{ist-daten}} CSV headers confirms 21 columns
(\texttt{{BETRIEBSTAG}} through \texttt{{DURCHFAHRT\_TF}}) with \textbf{{no
realized platform or track column}}, making retrospective track-circuit
verification impossible from public 2023 realization files alone. Under the
three-tier pre-decision contract~\eqref{{eq:pair_mct}}, Figure~\ref{{fig:cal_plat}}(b)
shows that 100.0\% of the 2,656 registered transfer pairs in each 14-day March
panel (1,156 published \texttt{{transfers.txt}} pairs [43.5\%] + 1,500
same-platform pairs [56.5\%]) and 100.0\% of the {bg5_plat['total_pairs']} pairs
in the April panel ({bg5_plat['counts_by_rule']['explicit_gtfs_transfers_txt']}
published \texttt{{transfers.txt}} pairs [46.6\%] +
{bg5_plat['counts_by_rule']['same_physical_platform']} same-platform pairs
[53.4\%]) resolve without triggering parent-station fallbacks. At the baseline
assumption $\tau_{{\mathrm{{same}}}} = 90$\,s, replacing station-max buffers
removes an average of 126.9\,s of transfer padding in March and
{_fmt_s(plat_mct_delta_bg5, 1)}\,s in April (330\,s on Z\"urich HB island tracks
41/42). Moreover, while GTFS-RT platform announcements matched static GTFS
platforms on 99.46\% ($4,211/4,234$) of updates in routine March operations,
Easter holiday engineering works in April generated \textbf{{{plat_rt_chg_count:,}
live GTFS-RT track reassignments out of {plat_rt_tot_count:,} platform-specific
updates ({plat_rt_chg_pct:.2f}\%; {plat_rt_bg5_pct:.2f}\% static agreement)}},
confirming the practical importance of prioritizing real-time GTFS-RT platform
fields before falling back to static schedules.

\section{{Empirical Results: Attribution Cascade, Confirmation Panel, and Headroom Audit}}
\label{{sec:results}}

\subsection{{Controlled Attribution Cascade on the October Development Panel}}
\label{{sec:october}}

Table~\ref{{tab:oct_cascade}} and Figure~\ref{{fig:cascade}}(a) trace our
controlled ablations across the 72-journey October Development Panel (all 72
trips have complete sensor endpoints; all arms share station-max MCTs and ideal
clocks to isolate menu, identity, and scoring effects one factor at a time).

\begin{{table}}[t]
\centering
\caption{{Controlled Attribution Cascade on the October Development Panel (6 Operating Days, 72 Complete Trips)}}
\label{{tab:oct_cascade}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrl}}
\toprule
Ablation Stage & Controller \& Representation Contract & Complete & Mean Loss (s) & Step $\Delta$ (s) & CPU (s) & Isolated Mechanism \\
\midrule
1. Capped / Native Obs & StatMax-DSW, 4-train hub cap, native obs & 72/72 & {_fmt_s(v50_narrow_nat, 2)} & -- & 0.082 & Truncated candidate baseline \\
2. Capped / Wide Obs & StatMax-DSW, 4-train hub cap, wide obs & 72/72 & {_fmt_s(v50_narrow_com, 2)} & $+0.00$ & 0.082 & Passive observations alone add $0$\,s \\
3. Uncapped Window & StatMax-DSW, uncapped menu, wide obs & 72/72 & {_fmt_s(v50_wide_com, 2)} & $-{_fmt_s(v50_gain, 2)}$ & 0.145 & Removing 4-train cap saves $20.18$\,s \\
4. Prefix-Closed Menu & StatMax-DSW, prefix-closed, literal ID & 72/72 & {_fmt_s(dg1_cheap, 2)} & $-{_fmt_s(dg1_menu_gain, 2)}$ & 0.264 & 4 continuation rules tie 72/72 \\
5. Release-Matched ID & \textbf{{StatMax-DSW, prefix-closed, repaired ID}} & 72/72 & \textbf{{{_fmt_s(ci1_rep_cheap, 2)}}} & $-{_fmt_s(dg1_cheap - ci1_rep_cheap, 2)}$ & {_fmt_s(ci1_rep_cheap_cpu, 3)} & Cumulative $-{_fmt_s(total_oct_unlock, 2)}$\,s structural gain \\
\midrule
6. Deterministic Mean & Determ.-Mean routing, repaired ID & 72/72 & {_fmt_s(ci1_rep_mean, 2)} & $+{_fmt_s(ci1_stoch_gain, 2)}^\dagger$ & 0.255 & Point-mean routing loses $24.67$\,s \\
7. Closed-Loop Rollout & Rollout-128 (128 worlds), repaired ID & 72/72 & {_fmt_s(ci1_rep_f128, 2)} & $+{_fmt_s(abs(ci1_f128_gain), 2)}^\dagger$ & {_fmt_s(ci1_rep_f128_cpu, 3)} & 1 win (+96\,s), 4 harms ($35\times$ CPU) \\
\bottomrule
\end{{tabular}}}}
\par\vspace{{3pt}}\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes:}} $^\dagger$Measured relative to Stage~5 (StatMax-DSW with repaired GTFS identity, {_fmt_s(ci1_rep_cheap, 2)}\,s); positive $\Delta$ denotes higher (worse) mean journey time.\end{{minipage}}
\end{{table}}

Three clear empirical lessons emerge from Table~\ref{{tab:oct_cascade}}:
\begin{{enumerate}}
  \item \textbf{{Candidate-menu completeness and identity hygiene dominate}}:
  Expanding passive stop observations on a 4-train capped menu changes nothing
  ($0.00$\,s on all 72 trips), whereas lifting the 4-train cap saves
  {_fmt_s(v50_gain, 2)}\,s per trip (absorbing four wins previously misattributed
  to rollout lookahead), closing prefix-shared continuations saves
  {_fmt_s(dg1_menu_gain, 2)}\,s, and repairing Wednesday \texttt{{trip\_id}}
  drift saves a further {_fmt_s(ci1_old_cheap - ci1_rep_cheap, 2)}\,s (totaling
  {_fmt_s(total_oct_unlock, 2)}\,s per trip).
  \item \textbf{{Distributional selective waiting substantially outperforms
  point-mean routing}}: Replacing distributional selective waiting
  ({_fmt_s(ci1_rep_cheap, 2)}\,s) with deterministic conditional-mean routing
  (\textbf{{Determ.-Mean}}, {_fmt_s(ci1_rep_mean, 2)}\,s) degrades mean journey
  time by {_fmt_s(ci1_stoch_gain, 2)}\,s per trip at virtually identical CPU
  (0.25\,s), confirming that accounting for arrival-time dispersion and
  cancellation tails is operationally important.
  \item \textbf{{Uncertified 128-world rollout degrades performance}}:
  \textbf{{Rollout-128}} ({_fmt_s(ci1_rep_f128, 2)}\,s, {_fmt_s(ci1_rep_f128_cpu, 2)}\,s
  CPU) performs {_fmt_s(abs(ci1_f128_gain), 2)}\,s worse than StatMax-DSW (1 win
  of +96\,s on Trip~62 vs.\ 4 harms: Trips~4 [$-263$\,s] and~32 [$-140$\,s] from
  uncertified Monte Carlo sampling noise, and Trips~9 [$-329$\,s] and~18
  [$-145$\,s] where rollout trusted the underpredicted $P_{{10}}$ throat risk and
  skipped safe upstream transfers at Oerlikon).
\end{{enumerate}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\textwidth]{{fig1_attribution_cascade.pdf}}
\caption{{Controlled Attribution Cascade on the October Development Panel and Paired Point-Estimate Contrasts Across All 25 Control Operating Days}}
\label{{fig:cascade}}
\end{{figure}}

\subsection{{Out-of-Period 14-Day Station-Max Control Panel (March 23--April 5)}}

On the unexposed 14-day March--April Station-Max Panel (168 trips, 1,344 paired
replays; 166 sensor-complete, 2 pre-onboard cancellations retained in bounds),
all four two-exit distributional continuation rules again tie on 166/166 complete
trips at {_fmt_s(ng2_cheap, 2)}\,s ({_fmt_s(ng2_cheap_cpu, 2)}\,s CPU; all-168
bounds $[{_fmt_s(ng2_cheap_bounds[0], 1)}, {_fmt_s(ng2_cheap_bounds[1], 1)}]$\,s),
confirming Proposition~\ref{{prop:collapse}}(b) out of period. \textbf{{Rollout-128}}
achieves {_fmt_s(ng2_f128, 2)}\,s (4 wins: +247, +83, +83, +240\,s across three
incident clusters; 1 harm: $-172$\,s; 161 ties; complete-pair mean saving
$+{_fmt_s(ng2_f128_gain, 2)}$\,s; all-168 bounds $[-40.0, +45.7]$\,s crossing
zero), but requires \textbf{{{_fmt_s(ng2_f128_cpu, 2)}\,s mean CPU per trip}}
(up to $57.88$\,s on a single gate, with an 87.5\% fallback from prefix
compression to full rollout due to static candidate-count checks). Pooled across
all 238 complete station-max journeys (October + March--April), Rollout-128
records 5 wins, 5 harms, 228 ties, and a net mean difference of $-1.26$\,s per
trip at $40\times$ higher CPU cost.

\subsection{{Pre-Frozen 5-Day April Confirmation Panel and Trip-Level Micro-Audit}}
\label{{sec:confirm}}

Our final confirmation experiment evaluates \textbf{{PA-DSW}} and seven
comparators under the platform-pair contract ($\tau_{{\mathrm{{same}}}}=90$\,s,
with sensitivity over $\{{60, 90, 120\}}$\,s) and physical \emph{{ist-daten}}
station-dwell clocks on 5 chronologically frozen unexposed days
(\texttt{{{bg_start}}}--\texttt{{{bg_end}}}, 60 trips, 480 paired replays). All
7 pre-registration and data-governance checks in \texttt{{audit.json}} passed
(\texttt{{true}}); the operational latency check
\texttt{{bounded\_cert\_sub\_2s\_mean\_cpu}} evaluated to \texttt{{false}} because
exact-combined multi-exit tree evaluation in \textbf{{Multi-Hop-DSW}} averaged
{_fmt_s(arms['recursive_plat_async']['mean_cpu_seconds'], 2)}\,s CPU per trip
(bringing \textbf{{Cert-Rollout-{bc_worlds}}} to
{_fmt_s(arms['bounded_cert_plat_async']['mean_cpu_seconds'], 2)}\,s CPU), leaving
\textbf{{PA-DSW}} ({_fmt_s(arms['cheap_plat_async']['mean_cpu_seconds'], 3)}\,s
mean CPU) and \textbf{{Mean-CSA}}
({_fmt_s(arms['snapshot_csa_plat_async']['mean_cpu_seconds'], 3)}\,s mean CPU) as
the only sub-2-second controllers.

\begin{{table}}[t]
\centering
\caption{{Eight-Arm Performance and Computation on the 5-Day April Confirmation Panel ({bg_dates_str}, 60 Public Trips)}}
\label{{tab:bg5_arms}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lllrrrrrr}}
\toprule
Controller & MCT Contract & Action Clock & Comp. & Mean Loss (s) & All-60 Bounds (s) & Mean CPU (s) & Max CPU (s) & Gates \\
\midrule
{table4_tex}
\bottomrule
\end{{tabular}}}}
\par\vspace{{3pt}}\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes:}} Process CPU includes candidate-menu construction. Trips with missing \emph{{ist-daten}} departure sensors (Trip~55 under Station-Max; Trips~52 and~55 under Platform-Pair) are retained in All-60 Bounds via $[0, 3600]$\,s worst-case intervals. Cert-Rollout-{bc_worlds} uses $M_0={bc_worlds}$ worlds, margin $\gamma=90$\,s, and target gate budget $1.8$\,s on top of Multi-Hop-DSW.\end{{minipage}}
\end{{table}}

\begin{{table}}[t]
\centering
\caption{{Pre-Declared Paired Contrasts on the 5-Day April Confirmation Panel (20,000-Replicate Date-Cluster Bootstrap)}}
\label{{tab:bg5_contrasts}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrr}}
\toprule
Paired Contrast (Baseline vs.\ Subject) & W / L / T & Mean Saving (s) & 95\% Date CI (s) & All-60 Bounds (s) & $\Delta$CPU (s) \\
\midrule
{table5_tex}
\bottomrule
\end{{tabular}}}}
\par\vspace{{3pt}}\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes:}} Positive saving indicates Subject achieves lower realized travel time than Baseline ($\text{{Loss}}_{{\text{{base}}}} - \text{{Loss}}_{{\text{{subj}}}} > 0$) across complete pairs ($n=59$ for Station-Max; $n=58$ for Platform-Pair). $^\dagger$StatMax-DSW uses ideal clock while PA-DSW enforces physical dwell; because PA-DSW's maximum single-decision CPU ($0.559$\,s) is $30\times$ below the $17.0$\,s shortest realized dwell ($0$ fallback or gate expirations), PA-DSW's ideal-clock and physical-dwell trajectories are identical.\end{{minipage}}
\end{{table}}

\begin{{table}}[t]
\centering
\caption{{Complete Trip-Level Micro-Audit of All Non-Tied and Unresolved Journeys on the April Confirmation Panel (Remaining 51 of 60 Trips Tie Identically Across All 8 Arms)}}
\label{{tab:bg5_micro}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{rllrrrrrrl}}
\toprule
Trip & Date & Corridor ($o \to d_{{\mathrm{{goal}}}}$) & StatMax & Mean-CSA & \textbf{{PA-DSW}} & Multi-Hop & Cert-{bc_worlds} & Rollout-128 & Verified Physical / Algorithmic Mechanism \\
\midrule
{table6_tex}
\bottomrule
\end{{tabular}}}}
\par\vspace{{3pt}}\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes:}} Entries report realized journey loss $L(\pi,\omega)$ in seconds (\texttt{{Unres.}} denotes missing \emph{{ist-daten}} departure timestamp retained in $[0,3600]$\,s bounds). Rollout-128 (Ideal) matches Rollout-128 (Dwell) on all 60 trips.\end{{minipage}}
\end{{table}}

Tables~\ref{{tab:bg5_arms}}--\ref{{tab:bg5_micro}} and
Figures~\ref{{fig:pareto}}--\ref{{fig:dwell}} reveal the exact operational
structure of the April confirmation cohort:

\begin{{enumerate}}
  \item \textbf{{Event Sparsity on Calm Rail Days and Point Savings of PA-DSW}}.
  Because 90.5\% of S-Bahn transfer opportunities during the April window catch
  both upstream and downstream connections normally ($P_{{11}}$,
  Section~\ref{{sec:forensics}}), \textbf{{51 of 60 trips (and 51 of 58 complete
  platform-pair trips, 87.9\%) produce identical trajectories across all 8
  arms}}. Consequently, sample-wide averages dilute large per-incident effects by
  roughly an order of magnitude, and $N=5$ operating days yield wide date-cluster
  confidence intervals. Across the 4 complete trips where PA-DSW differs from
  StatMax-DSW and Mean-CSA (Table~\ref{{tab:bg5_micro}}), PA-DSW saves $+293$,
  $+286$, and $+339$\,s ($\approx 5$\,min per trip) across three days on the
  \emph{{Stettbach $\to$ Altstetten}} corridor (Trips~17, 29, 41)—where both
  conservative station-max buffers and deterministic Mean-CSA miss a fast
  platform-pair connection that PA-DSW catches by integrating upper-tail
  departure mass—while losing $-304$\,s on Trip~3 (\emph{{Tiefenbrunnen $\to$
  Altstetten}}), yielding a complete-pair point saving of
  $+{_fmt_s(g1_comp['complete_mean_saving_seconds'], 2)}$\,s (95\% date CI
  $[{_fmt_s(g1_comp['date_cluster_bootstrap']['ci95_saving_seconds'][0], 2, True)},
  {_fmt_s(g1_comp['date_cluster_bootstrap']['ci95_saving_seconds'][1], 2, True)}]$\,s,
  spanning zero). Sensitivity over $\tau_{{\mathrm{{same}}}} \in \{{60, 90, 120\}}$\,s
  pinpoints Trip~3 directly: at $\tau_{{\mathrm{{same}}}} = 60$\,s, PA-DSW
  recovers the $1382$\,s path on Trip~3, lowering complete-root mean loss to
  {_fmt_s(sens_cheap_60, 2)}\,s and matching \textbf{{Rollout-128}}
  ({_fmt_s(arms['full128_plat_ideal']['complete_mean_loss_seconds'], 2)}\,s) to
  $0.00$\,s across all 58 complete trips.

  \item \textbf{{Statistically Significant Degradation of Multi-Hop Recursion and
  Realized Dwell Overrun}}. Unlike the inconclusive positive contrasts on 5 calm
  days, extending independent-belief recursion across $m \ge 3$ hubs degrades
  performance: \textbf{{Multi-Hop-DSW}} loses
  {_fmt_s(abs(rec_comp['complete_mean_saving_seconds']), 2)}\,s per trip (0 wins,
  2 losses of $-789$\,s on Trip~31 and $-580$\,s on Trip~49; 95\% CI
  $[{_fmt_s(rec_comp['date_cluster_bootstrap']['ci95_saving_seconds'][0], 2, True)}, +0.00]$\,s),
  and \textbf{{Cert-Rollout-{bc_worlds}}} loses
  \textbf{{{_fmt_s(abs(bc_comp['complete_mean_saving_seconds']), 2)}\,s per trip}}
  (0 wins, 3 losses, 55 ties; \textbf{{95\% CI
  $[{_fmt_s(bc_comp['date_cluster_bootstrap']['ci95_saving_seconds'][0], 2, True)},
  {_fmt_s(bc_comp['date_cluster_bootstrap']['ci95_saving_seconds'][1], 2, True)}]$\,s,
  strictly below zero}}). Crucially, Trip~9 (\emph{{Flughafen $\to$ Stadelhofen}})
  demonstrates a realized physical dwell-clock failure: base multi-exit tree
  evaluation consumed $50.73$\,s CPU, overrunning the physical station dwell
  window (\texttt{{fallback\_expired = True}}) and forcing an unintended
  \texttt{{continue}} that added $+555$\,s of travel time ($1311 \to 1866$\,s).

  \item \textbf{{Physical Dwell Safety Margin vs.\ Multi-Second Solver Latency}}.
  Across all {dwell['measured_continuation_onboard_dwells']} realized continuation
  onboard states on the April panel, \emph{{ist-daten}} station dwell has
  $\min = {_fmt_s(dq['min'], 1)}$\,s, $P_{{10}} = {_fmt_s(dq['p10'], 1)}$\,s,
  median {_fmt_s(dq['median'], 1)}\,s, mean {_fmt_s(dq['mean'], 1)}\,s, and
  $P_{{90}} = {_fmt_s(dq['p90'], 1)}$\,s (Figure~\ref{{fig:dwell}}(a)). Although
  all 119 background gates of \textbf{{Rollout-128}} completed before train
  departure at the specific hub stops visited during April ($0.0\%$ background
  gate expiration, so Ideal and Dwell Rollout-128 coincide on this panel), its
  maximum single-decision CPU reached {_fmt_s(f128_d['max_single_decision_cpu_seconds'], 2)}\,s
  (and $57.88$\,s on March--April)—exceeding minimum ($17$\,s) and lower-decile
  ($28$\,s) S-Bahn dwells. By contrast, \textbf{{PA-DSW}} finishes every single
  decision in $\le {_fmt_s(cheap_d['max_single_decision_cpu_seconds'], 3)}$\,s,
  providing a $>30\times$ sub-dwell safety margin.
\end{{enumerate}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\textwidth]{{fig3_pareto_frontier.pdf}}
\caption{{Quality--Compute Frontier and 95\% Date-Cluster Bootstrap Contrasts on the 5-Day April Confirmation Panel}}
\label{{fig:pareto}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\textwidth]{{fig4_dwell_anytime.pdf}}
\caption{{Realized S-Bahn Station Dwell Quantiles vs.\ Solver Latency and Ambiguity-Margin Pruning / Certification Diagnostics}}
\label{{fig:dwell}}
\end{{figure}}

\subsection{{Why Adaptive Solver Switching Cannot Unlock Additional Gain: Oracle Headroom Audit}}

A natural question is whether an environment-adaptive meta-controller—running
fast \textbf{{PA-DSW}} on routine states and switching selectively to heavy
\textbf{{Rollout-128}} during disruptions—could extract substantial gains. We
test this upper bound directly by computing the \emph{{hindsight oracle
switching headroom}}: a hypothetical clairvoyant selector that invokes
Rollout-128 if and only if Rollout-128 achieves strictly lower realized travel
time than PA-DSW (or StatMax-DSW), while avoiding 100\% of rollout losses and
incurring zero dwell overruns:
\begin{{itemize}}
  \item \textbf{{October Panel (72 complete trips)}}: Rollout-128 beats
  StatMax-DSW on 1 trip (+96\,s on Trip~62), giving an oracle switching ceiling
  of $96 / 72 = \mathbf{{+1.33\,\text{{s/trip}}}}$ ($\mathbf{{0.09\%}}$ of mean
  journey time);
  \item \textbf{{March--April Panel (166 complete trips)}}: Rollout-128 beats
  StatMax-DSW on 4 trips (+653\,s total), giving an oracle switching ceiling of
  $653 / 166 = \mathbf{{+3.93\,\text{{s/trip}}}}$ ($\mathbf{{0.29\%}}$);
  \item \textbf{{April Confirmation Panel (58 complete trips)}}: Rollout-128
  beats PA-DSW on 1 trip (+304\,s on Trip~3 at $\tau_{{\mathrm{{same}}}}=90$\,s),
  giving an oracle ceiling of $304 / 58 = \mathbf{{+5.24\,\text{{s/trip}}}}$
  ($\mathbf{{0.43\%}}$), and $\mathbf{{0.00\,\text{{s/trip}}}}$
  ($\mathbf{{0.00\%}}$) at $\tau_{{\mathrm{{same}}}}=60$\,s.
\end{{itemize}}
Across all 296 sensor-complete paired control journeys, open-loop distributional
selective waiting already captures $>99.6\%$ of the travel-time value achievable
by 128-world closed-loop simulation under the shared posterior model. With less
than $0.5\%$ theoretical headroom, any non-clairvoyant online switching rule
loses more from a single false override or dwell overrun than it can accumulate
across dozens of calm operating days.

\section{{Engineering Guidelines and Limitations}}
\label{{sec:discussion}}

\subsection{{Practical Recommendations for Real-Time Transit Guidance}}

\begin{{enumerate}}
  \item \textbf{{Invest in Candidate Completeness, Release-Matched GTFS-RT Joins,
  and Platform-Pair Transfer Tables Before Complex Tree Search}}. Removing
  heuristic candidate caps ($+20.18$\,s), prefix-closing continuations
  ($+4.65$\,s), matching GTFS-RT updates by physical service signatures across
  mid-week schedule cutovers, and modeling platform-pair transfer readiness
  yield substantial per-incident travel-time reductions at sub-second compute
  cost. Transit agencies can further improve downstream routing accuracy by
  publishing explicit same-platform transfer rows in GTFS \texttt{{transfers.txt}}
  (eliminating reliance on assumed $\tau_{{\mathrm{{same}}}}$ buffers) and
  populating platform-specific \texttt{{stop\_id}} fields in GTFS-RT
  \texttt{{StopTimeUpdate}} messages—particularly during track engineering works
  when live platform reassignments exceed 5\%.

  \item \textbf{{Use Single-Layer Distributional Selective Waiting (PA-DSW) as
  the Default Sub-Dwell Controller}}. Point-mean snapshot routing sacrifices
  $24.67$\,s per trip on October and misses tight platform-pair connections on
  April ($286$--$339$\,s on affected trips), whereas extending independent-trip
  backward induction to $m \ge 3$ transfers degrades travel time by $23.60$\,s
  and risks overrunning station dwells. PA-DSW captures arrival-tail recourse in
  $\le 0.56$\,s per decision.

  \item \textbf{{Enforce Physical Dwell Deadlines and Paired Statistical
  Certification in Offline Benchmarks}}. Offline transit evaluations should
  replace zero-latency clocks with physical station-dwell tokens and guard any
  sampling-based override with ambiguity-margin pre-filtering and paired
  Student-$t$ intervals so finite-sample Monte Carlo variance cannot masquerade
  as policy behavior.
\end{{enumerate}}

\subsection{{Limitations and Scope Conditions}}

Four limitations bound our empirical conclusions and suggest directions for
future work:
\begin{{enumerate}}
  \item \textbf{{Network Topology and Calm-Day Event Sparsity}}: Our experiments
  focus on morning-peak Z\"urich S-Bahn radial corridors, where punctuality is
  high ($>90\%$ $P_{{11}}$ catches) and most ride legs traverse $m \le 2$
  transfer hubs. On the 5-day April confirmation cohort (1 workday + 4 Easter
  holiday/weekend days), 51 of 60 trips tie across all controllers, leaving
  positive PA-DSW contrasts with confidence intervals spanning zero. Evaluating
  PA-DSW on dense surface tram/bus grids or targeted multi-day severe disruption
  panels with higher incident density is an important extension.
  \item \textbf{{Assumed Same-Platform Buffer $\tau_{{\mathrm{{same}}}}$}}: While
  explicit \texttt{{transfers.txt}} pairs are published schedule data,
  $\tau_{{\mathrm{{same}}}}=90$\,s is a modeling assumption because 2023 Swiss
  GTFS omits self-platform transfer rows and \emph{{ist-daten}} omits realized
  track occupation; Trip~3 illustrates sensitivity between $60$\,s and $90$\,s
  at tight connection margins.
  \item \textbf{{Python Implementation Latency}}: Process CPU times reflect a
  single-threaded Python/NumPy research implementation (with C-accelerated FFBS
  chain draws). Compiled C++/Rust implementations of CSA-MEAT or rollout engines
  would reduce absolute wall-clock times, although our oracle headroom audit
  ($<0.5\%$ even under an idealized zero-latency clock) confirms that faster
  rollout execution alone would not unlock large travel-time gains on these
  corridors.
  \item \textbf{{Pre-Frozen Candidate Menus and Unmodeled Peak Throat Skew}}:
  All controllers operate within pre-frozen candidate menus (on the March--April
  panel, full-graph deterministic scanning found off-menu alternatives on 6 of
  168 trips) and use time-homogeneous route delay priors that exhibit a $+3.1$\,pp
  weekday morning-peak $P_{{10}}$ residual. Conditioning delay emissions on
  real-time corridor congestion states rather than deepening tree search is the
  more promising avenue for further accuracy gains.
\end{{enumerate}}

\section{{Conclusion}}
\label{{sec:conclusion}}

Through a 39-operating-day archival replay and attribution program on official
Swiss railway sensor and GTFS-RT streams, this paper demonstrates that realized
performance in real-time urban rail guidance is governed primarily by
\emph{{representation and contract fidelity}}—uncapped prefix-closed candidate
menus, release-matched GTFS trip identities, dynamic platform-pair transfer
rules, and sub-second distributional selective waiting (PA-DSW)—rather than deep
multi-hop recursion or closed-loop Monte Carlo lookahead. Accounting for the four
offline evaluation pitfalls documented here provides a transparent, reproducible
foundation for benchmarking next-generation real-time transit routing systems.

\appendix

\section{{Proofs of Analytical Propositions}}
\label{{app:proofs}}

\subsection{{Proof of Proposition~\ref{{prop:collapse}}}}
\emph{{Part (a):}} Under sequential priority list $\mathcal{{M}} = (m_1,\dots,m_K)$,
candidate $m_k$ is boarded at integer epoch $t \in \mathcal{{D}}_k$ iff
$r_k \le t \le t_0 + C$, $\chi(c(m_k)) = 0$ with $T^-(c(m_k)) = t$, and no
higher-priority $m_j$ ($j < k$) offered a walk-reachable boarding on $[r_j, t]$.
If $t < r_j$, $[r_j, t]$ is empty so unavailability has probability
$S_j^{{\mathrm{{unavail}}}}(t) = 1$; if $t \ge r_j$, partitioning the mutually
exclusive sample space of trip $v(c(m_j))$ into cancellation ($q_j$), pre-readiness
departure ($\sum_{{\tau < r_j}} p_j(\tau)$), legal boarding $[r_j, t]$, and late
departure ($\sum_{{\tau > t}} p_j(\tau)$) yields~\eqref{{eq:surv_exact}} via
nonnegative additions. Conditional trip independence given $h_t$ yields
boarding probability $p_k(t)\prod_{{j<k}} S_j^{{\mathrm{{unavail}}}}(t)$ and
timeout probability $\prod_{{j=1}}^K S_j^{{\mathrm{{unavail}}}}(t_0+C)$, giving~\eqref{{eq:first_offer_val}}
and~\eqref{{eq:padsw_env}} in $O(K\max_k |\mathcal{{D}}_k|)$ operations via
precomputed prefix/suffix sums.

\emph{{Part (b):}} If $m = 1$, $e_1$ is terminal, \texttt{{alight}} is mandatory,
and all four variants trivially coincide. If $m = 2$ with current exit $e_1$ at
observed arrival $T^+(e_1)=t_1$ and sole remaining mandatory exit $e_2$: for
$a = \texttt{{alight}}$, all four variants evaluate the identical platform score
at $e_1$; for $a = \texttt{{continue}}$, $e_2$ is the singleton remaining exit
($|\{{e_j : j \ge 2\}}|=1$), so fixed-exit (PA-DSW) evaluates $e_2$,
\texttt{{single}} minimizes over singleton $\{{e_2\}}$, \texttt{{pair}} has no
further decision beyond mandatory exit $e_2$, and \texttt{{one\_step}} under
deterministic running increment $T^+(e_2) - t_1 = \tilde{{t}}^+(e_2) - \tilde{{t}}^+(e_1)$
and unchanged posterior $\mathbb{{P}}(\cdot \mid h_{{t_1}})$ evaluates the exact
same point continuation value at $e_2$. \qed

\subsection{{Proof of Proposition~\ref{{prop:anytime}}}}
When $\tau_{{\mathrm{{base}}}} < W < \tau_{{\mathrm{{base}}}} + \tau_{{M_0}}$,
the \texttt{{ActionPublisher}} commits $a_{{\mathrm{{base}}}}$ before $t_{{\mathrm{{close}}}}$
and rejects the late rollout token at $t_{{\mathrm{{arr}}}} + \tau_{{\mathrm{{base}}}} + \tau_{{M_0}} > t_{{\mathrm{{close}}}}$,
executing $a_{{\mathrm{{base}}}}$ with zero loss difference; if
$\tau_{{\mathrm{{base}}}} \ge W$, no fallback token is registered prior to door
closure and \texttt{{continue}} executes physically, establishing~(i). For~(ii),
suppose $a_{{\mathrm{{base}}}} = \texttt{{continue}}$ and $\mu_D \le 0$. An
override to \texttt{{alight}} requires $\bar{{D}}_{{M_0}} + z_{{1-\alpha/2}} \hat{{s}}_{{M_0}}/\sqrt{{M_0}} < 0$
(when oriented as $L(\texttt{{alight}}) - L(\texttt{{continue}})$) or
equivalently $(\bar{{D}}_{{M_0}} - \mu_D)/(\hat{{s}}_{{M_0}}/\sqrt{{M_0}}) \ge \bar{{D}}_{{M_0}}/(\hat{{s}}_{{M_0}}/\sqrt{{M_0}}) > z_{{1-\alpha/2}}$
when overriding toward the positive tail, an event of probability at most
$\alpha/2$ under asymptotic normality (and strictly smaller under the additional
gates $|\Delta_{{\mathrm{{base}}}}| \le \gamma$, $|\bar{{D}}_{{M_0}}| \ge \delta_{{\min}}$,
and $\tau_{{\mathrm{{base}}}} + \tau_{{M_0}} \le W$). Symmetry for
$a_{{\mathrm{{base}}}} = \texttt{{alight}}$ ($\mu_D \ge 0$) completes the bound. \qed

\section{{Algebraic Recovery of Conditional-Onboard Residuals}}
\label{{app:audit}}

In the March--April Panel, compact cluster summaries store worst-case bounds
$[\ell_{{c,k}}, u_{{c,k}}]$ on cluster mean residuals $y_k - p_k$ ($k \in \{{00,01,10,11\}}$)
across clusters of size $N_c$ containing $U_c$ unreached pre-onboard events
(bounded symmetrically in $[-1, +1]$). Because resolved events contribute point
residuals while each unreached event contributes $[-1/N_c, +1/N_c]$, the exact
sum of reached onboard residuals in cluster $c$ is recovered without
approximation via
\begin{{align}}
  \sum_{{i \in \mathcal{{R}}_c}} (y_{{i,k}} - p_{{i,k}}) &\;=\; \frac{{N_c (\ell_{{c,k}} + u_{{c,k}})}}{{2}}, \nonumber\\
  u_{{c,k}} - \ell_{{c,k}} \;=\; \frac{{2 U_c}}{{N_c}}, \qquad
  &\sum_{{k \in \mathcal{{K}}}} \sum_{{i \in \mathcal{{R}}_c}} (y_{{i,k}} - p_{{i,k}}) \;=\; 0 ,
\end{{align}}
verified to $<10^{{-11}}$ across all 14 days ($1,309 = 1,328 - 19$ reached
opportunities).

\section{{Reproducibility Governance and Code-to-Manuscript Mapping}}
\label{{app:repro}}

To maintain complete traceability to frozen repository artifacts without
cluttering the main text, Table~\ref{{tab:codemap}} maps manuscript controller
and panel names to their exact immutable directory and arm identifiers. All
chronological freeze timestamps (\texttt{{date\_manifest.json}} $\prec$
\texttt{{selection.json}} $\prec$ \texttt{{rt\_decode\_summary.json}},
\texttt{{actual\_filter\_summary.json}}) are verified automatically in
\texttt{{audit.json}} and unit-tested in \texttt{{tests/test\_bounded\_gate\_5d\_v1.py}}.

\begin{{table}}[h]
\centering
\caption{{Mapping Between Manuscript Terminology and Frozen Repository Identifiers}}
\label{{tab:codemap}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lll}}
\toprule
Manuscript Name & Repository Stage / Arm Identifier & Primary Module / Receipt \\
\midrule
StatMax-DSW & \texttt{{new\_cheap}} / \texttt{{cheap\_statmax\_ideal}} & \texttt{{v41/matched\_continuation.py}} \\
PA-DSW (Proposed) & \texttt{{cheap\_plat\_async}} & \texttt{{bounded\_gate\_5d\_v1/pair\_contract.py}} \\
Mean-CSA / Determ.-Mean & \texttt{{snapshot\_csa\_plat\_async}} / \texttt{{new\_mean}} & \texttt{{PairMeanSnapshotCSA}} / \texttt{{v51/baselines.py}} \\
Multi-Hop-DSW & \texttt{{recursive\_plat\_async}} & \texttt{{PairRecursivePlanner}} \\
Cert-Rollout-{bc_worlds} & \texttt{{bounded\_cert\_plat\_async}} ($M_0={bc_worlds}$) & \texttt{{g2\_reset\_v1/certified\_gate.py}} \\
Rollout-128 & \texttt{{new\_full128}} / \texttt{{full128\_plat\_async}} & \texttt{{v51/rollout.py}}, \texttt{{pair\_contract.py}} \\
Oct / Mar / Mar--Apr / Apr Panels & \texttt{{V50+DG1+CI1}} / \texttt{{DC2}} / \texttt{{NG2}} / \texttt{{BG5}} & \texttt{{experiments/*/summary.json}} \\
\bottomrule
\end{{tabular}}}}
\end{{table}}

\begin{{thebibliography}}{{20}}

\bibitem[B\"ohmov\'a et al.(2013)]{{bohmova2013}}
B\"ohmov\'a, K., Mihal\'ak, M., Pr\"oger, T., \v{{S}}r\'amek, R., Widmayer, P., 2013.
Robust routing in urban public transportation: How to find reliable journeys
based on past observations. In: \emph{{13th Workshop on Algorithmic Approaches
for Transportation Modelling, Optimization, and Systems (ATMOS 2013)}}, OASIcs,
Vol.~33, Schloss Dagstuhl, pp.~27--41.
\url{{https://doi.org/10.4230/OASIcs.ATMOS.2013.27}}.

\bibitem[Chen and Nie(2015)]{{chenNie}}
Chen, P.W., Nie, Y.M., 2015. Optimal transit routing with partial online
information. \emph{{Transportation Research Part B: Methodological}} 72, 40--58.
\url{{https://doi.org/10.1016/j.trb.2014.11.007}}.

\bibitem[Corman and Kecman(2018)]{{corman2018}}
Corman, F., Kecman, P., 2018. Stochastic prediction of train delays in real-time
using Bayesian networks. \emph{{Transportation Research Part C: Emerging
Technologies}} 95, 599--615.
\url{{https://doi.org/10.1016/j.trc.2018.08.003}}.

\bibitem[Delling et al.(2015)]{{delling2015}}
Delling, D., Pajor, T., Werneck, R.F., 2015. Round-based public transit routing.
\emph{{Transportation Science}} 49~(3), 591--604.
\url{{https://doi.org/10.1287/trsc.2014.0534}}.

\bibitem[Dibbelt et al.(2018)]{{dibbelt2018}}
Dibbelt, J., Pajor, T., Strasser, B., Wagner, D., 2018. Connection scan
algorithm. \emph{{ACM Journal of Experimental Algorithmics}} 23, 1.2:1--1.2:56.
\url{{https://doi.org/10.1145/3274661}}.

\bibitem[Durner(2024)]{{durner}}
Durner, R., 2024. \emph{{Stochastic Strategies for Public Transport Journeys
Based on Realtime Delay Predictions}}. Master's thesis, University of Stuttgart.
\url{{https://doi.org/10.18419/opus-14802}}.

\bibitem[Gao and Chabini(2006)]{{gao2006}}
Gao, S., Chabini, I., 2006. Optimal routing policy problems in stochastic
time-dependent networks. \emph{{Transportation Research Part B: Methodological}}
40~(2), 93--122. \url{{https://doi.org/10.1016/j.trb.2005.02.001}}.

\bibitem[Geissmann and Gianinazzi(2019)]{{geissmann}}
Geissmann, B., Gianinazzi, L., 2019. Routing in stochastic public transit
networks. In: \emph{{19th Symposium on Algorithmic Approaches for Transportation
Modelling, Optimization, and Systems (ATMOS 2019)}}, OASIcs, Vol.~75, Schloss
Dagstuhl, pp.~4:1--4:15. \url{{https://doi.org/10.4230/OASIcs.ATMOS.2019.4}}.

\bibitem[Gentile et al.(2005)]{{gentile2005}}
Gentile, G., Nguyen, S., Pallottino, S., 2005. Route choice on transit networks
with online information at stops. \emph{{Transportation Science}} 39~(3),
289--297. \url{{https://doi.org/10.1287/trsc.1040.0109}}.

\bibitem[Gneiting and Raftery(2007)]{{gneiting2007}}
Gneiting, T., Raftery, A.E., 2007. Strictly proper scoring rules, prediction,
and estimation. \emph{{Journal of the American Statistical Association}}
102~(477), 359--378. \url{{https://doi.org/10.1198/016214506000001437}}.

\bibitem[Hickman and Bernstein(1997)]{{hickman1997}}
Hickman, M.D., Bernstein, D.H., 1997. Transit service and path choice models in
stochastic and time-dependent networks. \emph{{Transportation Science}} 31~(2),
129--146. \url{{https://doi.org/10.1287/trsc.31.2.129}}.

\bibitem[Keyhani et al.(2013)]{{keyhani2013}}
Keyhani, M.H., Schnee, M., Weihe, K., Zorn, H.P., 2013. Delay-robustness of
transfer patterns in public transportation route planning. In: \emph{{13th
Workshop on Algorithmic Approaches for Transportation Modelling, Optimization,
and Systems (ATMOS 2013)}}, OASIcs, Vol.~33, Schloss Dagstuhl, pp.~42--54.
\url{{https://doi.org/10.4230/OASIcs.ATMOS.2013.42}}.

\bibitem[Kim et al.(2005)]{{kim2005}}
Kim, S., Lewis, M.E., White, C.C., 2005. Optimal vehicle routing with real-time
traffic information. \emph{{IEEE Transactions on Intelligent Transportation
Systems}} 6~(2), 178--188. \url{{https://doi.org/10.1109/TITS.2005.848362}}.

\bibitem[Miller-Hooks and Mahmassani(2000)]{{millerHooks2000}}
Miller-Hooks, E.D., Mahmassani, H.S., 2000. Least expected time paths in
stochastic, time-varying transportation networks. \emph{{Transportation Science}}
34~(2), 198--215. \url{{https://doi.org/10.1287/trsc.34.2.198.12304}}.

\bibitem[Nguyen and Pallottino(1988)]{{nguyen1988}}
Nguyen, S., Pallottino, S., 1988. Equilibrium traffic assignment for large scale
transit networks. \emph{{European Journal of Operational Research}} 37~(2),
176--186. \url{{https://doi.org/10.1016/0377-2217(88)90327-X}}.

\bibitem[Open Data Platform Mobility Switzerland(2023)]{{swissactual}}
Open Data Platform Mobility Switzerland, 2023. Actual data (\emph{{ist-daten}})
and timetable GTFS 2020 archives, 2023 operating days. Official Swiss federal
mobility archive. \url{{https://archive.opentransportdata.swiss/}}.

\bibitem[Rambha et al.(2016)]{{rambha}}
Rambha, T., Boyles, S.D., Waller, S.T., 2016. Adaptive transit routing in
stochastic time-dependent networks. \emph{{Transportation Science}} 50~(3),
1043--1059. \url{{https://doi.org/10.1287/trsc.2015.0613}}.

\bibitem[Silver and Veness(2010)]{{silver}}
Silver, D., Veness, J., 2010. Monte-Carlo planning in large POMDPs. In:
\emph{{Advances in Neural Information Processing Systems}} 23, pp.~2164--2172.

\bibitem[Somani et al.(2013)]{{somani2013}}
Somani, A., Ye, N., Hsu, D., Lee, W.S., 2013. DESPOT: Online POMDP planning with
regularization. In: \emph{{Advances in Neural Information Processing Systems}}
26, pp.~1772--1780.

\bibitem[Spiess and Florian(1989)]{{spiess1989}}
Spiess, H., Florian, M., 1989. Optimal strategies: A new assignment model for
transit networks. \emph{{Transportation Research Part B: Methodological}}
23~(2), 83--102. \url{{https://doi.org/10.1016/0191-2615(89)90034-9}}.

\end{{thebibliography}}

\end{{document}}
"""
    tex = tex.replace("—", "---")
    (PAPER_DIR / "paper_reconstruction.tex").write_text(tex, encoding="utf-8")

    # Build updated md/EVIDENCE_TRACEABILITY_INDEX.md
    trace_md = f"""# BAPR-HRO Evidence Traceability Index (Elsevier TR-C Manuscript)

This index maps every claim, proposition, table, figure, and trip-level audit statistic in `paper/paper_reconstruction.tex` (`paper/paper_reconstruction.pdf`) to its exact frozen receipt JSON/CSV file, code implementation, and automated unit test. Zero historical directories (`reconstruction_v50`, `decision_gap_v1`, `decision_calibration_v1`, `decision_calibration_v2`, `control_identity_v1`, `natural_gap_v2`) were modified, `bounded_gate_5d_v1` was executed once under pre-frozen governance, and `original_72_additional_budget = 0` holds throughout.

---

## 1. Manuscript Sections, Models & Propositions Traceability

| Manuscript Element | Mathematical / Empirical Content | Code Implementation | Primary Frozen Receipt(s) | Verification Test |
|---|---|---|---|---|
| **Section 3.2** (`sec:posterior`) | Exact-anchor correlated trip-chain Markov delay + persistent cancellation ($\rho_{{\mathrm{{cxl}}}}=0.9$) posterior (`EvolvingTripBelief` + `PhysicalHistory`) | `src/evolving_trip_belief.py`; `experiments/reconstruction_v41/physical_posterior.py` | `experiments/control_identity_v1/summary.json`, `experiments/bounded_gate_5d_v1/contract.json` | `tests/test_v41_matched_continuation.py`, `tests/test_control_identity.py` |
| **Section 3.3** (`sec:contracts`) | 3-tier platform-pair MCT hierarchy (published `transfers.txt` -> assumed $\tau_{{\mathrm{{same}}}}=90$s `[60,90,120]s` -> station fallback) & `ActionPublisher` physical dwell token (`fallback_expired` / `deadline_expired`) | `experiments/bounded_gate_5d_v1/pair_contract.py` (`DayPlatformIndex`, `BoundedArmExecutor`) | `experiments/bounded_gate_5d_v1/platform_contract_summary.json`, `experiments/bounded_gate_5d_v1/sensitivity_tau_same.json`, `experiments/bounded_gate_5d_v1/dwell_audit.json` | `tests/test_bounded_gate_5d_v1.py` |
| **Section 4.1** (`eq:missing_bounds`) | Operational accounting decomposition (distinguishing $J^*$ vs. policy-conditional differences and $\tau_{{\mathrm{{same}}}}$ boundary non-monotonicity) & sharp $[0, 3600]$s partial identification bounds | `experiments/bounded_gate_5d_v1/summarize.py` | `experiments/reconstruction_v50/summary.json`, `experiments/decision_gap_v1/summary.json`, `experiments/control_identity_v1/summary.json`, `experiments/bounded_gate_5d_v1/summary.json` | `tests/test_bounded_gate_5d_v1.py` |
| **Proposition 1** (`prop:collapse`) | (a) $O(K\max_k|\mathcal{{D}}_k|)$ selective-waiting envelope `PA-DSW` (`min(first_offer, single_target)`) without `1 - CDF` cancellation; (b) algebraic collapse on $m \le 2$ exits (`238/238` ties) | `experiments/bounded_gate_5d_v1/pair_contract.py` (`PairPhysicalPlanner`, `PairMatchedPlanner`) | `experiments/control_identity_v1/summary.json` (`72/72`), `experiments/natural_gap_v2/summary_compact.json` (`166/166`) | `tests/test_v41_matched_continuation.py`, `tests/test_bounded_gate_5d_v1.py` |
| **Proposition 2** (`prop:anytime`) | Dwell-expiration nullification & paired Student-$t$ override control ($M_0={bc_worlds}$ worlds, $\gamma=90$s, $z_{{0.975}}=1.96$; `403s` October MC flip harm blocked; `74.3%`--`79.8%` gates pruned) | `experiments/g2_reset_v1/certified_gate.py` (`should_invoke_rollout`, `choose_certified`) | `experiments/bounded_gate_5d_v1/contract.json`, `experiments/g2_reset_v1/OFFLINE_AUDIT_SUMMARY.json` (`finding_4`) | `tests/test_g2_reset_v1.py` |
| **Section 6.4** (Oracle Headroom) | Hindsight oracle switching headroom between PA-DSW and Rollout-128: `+1.33s` (`0.09%`, Oct), `+3.93s` (`0.29%`, Mar--Apr), `+5.24s` (`0.43%` at 90s; `0.00s` at 60s, Apr) | `experiments/g2_reset_v1/offline_audit.py`; `experiments/bounded_gate_5d_v1/summarize.py` | `experiments/control_identity_v1/summary.json`, `experiments/natural_gap_v2/summary_compact.json`, `experiments/bounded_gate_5d_v1/summary.json` | `tests/test_g2_reset_v1.py` |

---

## 2. Tables Traceability

| Table | Scope | Key Verified Metrics | Source Receipt(s) |
|---|---|---|---|
| **Table 1** (`tab:lit`) | Closest-Neighbor Literature Comparison | Contrasts Gentile (2005), Chen & Nie (2015), Kim et al. (2005), Rambha et al. (2016), Dibbelt et al. (2018, DOI `10.1145/3274661`), Durner (2024), Silver & Veness (2010), Somani et al. (2013) vs. PA-DSW | Verified bibliography in `paper/paper_reconstruction.tex` | -- |
| **Table 2** (`tab:cohorts`) | 39-Operating-Day Study Overview | `300` paired control trips (`72` Oct + `168` Mar--Apr + `60` Apr); `{total_eval_opp:,}` total opportunities (`{dc2_onboard_total:,}` DC2 = `{dc2_reached_res:,}` res + `{dc2_reached_unres}` unres; `{ng2_reg_total:,}` NG2 = `{ng2_reached_res:,}` res + `19` pre-cancel; `{bg5_reg_total}` BG5); `{total_reached_opp:,}` resolved | `experiments/g2_reset_v1/OFFLINE_AUDIT_SUMMARY.json`, `experiments/bounded_gate_5d_v1/calibration_split_summary.json`, `experiments/bounded_gate_5d_v1/audit.json` | `tests/test_g2_reset_v1.py` |
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
"""
    (MD_DIR / "EVIDENCE_TRACEABILITY_INDEX.md").write_text(trace_md, encoding="utf-8")
    print("Generated TR-C paper/paper_reconstruction.tex and md/EVIDENCE_TRACEABILITY_INDEX.md")


if __name__ == "__main__":
    build_all()
