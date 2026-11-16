"""Summarize the 5-day bounded experiment: paired gains, dwell expiration, G1/G2 & calibration split."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, median

import numpy as np

from experiments.decision_calibration_v1.platform_pairs import static_sources, transfer_pair
from experiments.decision_calibration_v2.calibration_summary import summarize as legacy_calibration
from experiments.decision_gap_v1.panel import read, write
from experiments.g2_reset_v1.calibration_split import (
    split_event_calibration,
    summarize_platform_contract,
)
from experiments.natural_gap_v2.events import root_events
from experiments.reconstruction_v34.discover import context
from experiments.reconstruction_v50.summarize import bounds

HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def date_cluster_bootstrap(by_day_diffs, *, replicates=20000, seed=20261116):
    """Whole-day cluster bootstrap on complete-pair journey savings (baseline_loss - subject_loss)."""
    days = sorted(d for d, vals in by_day_diffs.items() if vals)
    if not days:
        return dict(days=0, complete_pairs=0, mean_saving=None, ci95=[None, None])
    day_arrays = [np.asarray(by_day_diffs[d], dtype=float) for d in days]
    day_sums = np.array([arr.sum() for arr in day_arrays], dtype=float)
    day_counts = np.array([len(arr) for arr in day_arrays], dtype=float)
    point = float(day_sums.sum() / day_counts.sum())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), size=(replicates, len(days)))
    boot_sums = day_sums[idx].sum(axis=1)
    boot_counts = day_counts[idx].sum(axis=1)
    boot_means = boot_sums / boot_counts
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return dict(
        days=len(days),
        complete_pairs=int(day_counts.sum()),
        mean_saving_seconds=point,
        ci95_saving_seconds=[float(lo), float(hi)],
        positive_days=int(sum(float(arr.mean()) > 0.0 for arr in day_arrays)),
        negative_days=int(sum(float(arr.mean()) < 0.0 for arr in day_arrays)),
    )


def compare_pair(banks, baseline_label, subject_label, *, replicates=20000, seed=20261116):
    """Compare subject against baseline; positive saving = baseline_loss - subject_loss > 0."""
    complete_savings = []
    by_day = defaultdict(list)
    low_bounds, high_bounds = [], []
    wins, losses, ties, unresolved = [], [], [], []
    cpu_deltas = []

    for bank in banks:
        base_row = bank[baseline_label]
        subj_row = bank[subject_label]
        idx = base_row["root_index"]
        day = base_row["day"]
        b_loss = base_row["outcome"]["loss"]
        s_loss = subj_row["outcome"]["loss"]
        b_lo, b_hi = bounds(base_row)
        s_lo, s_hi = bounds(subj_row)
        # saving = base - subj -> lower bound is b_lo - s_hi, upper bound is b_hi - s_lo
        low_bounds.append(float(b_lo - s_hi))
        high_bounds.append(float(b_hi - s_lo))
        cpu_deltas.append(
            float(
                subj_row["cost"]["process_cpu_including_menu"]
                - base_row["cost"]["process_cpu_including_menu"]
            )
        )
        if b_loss is None or s_loss is None:
            unresolved.append(idx)
            continue
        diff = float(b_loss - s_loss)
        complete_savings.append(diff)
        by_day[day].append(diff)
        if diff > 1e-9:
            wins.append(dict(root_index=idx, day=day, saving_seconds=diff))
        elif diff < -1e-9:
            losses.append(dict(root_index=idx, day=day, saving_seconds=diff))
        else:
            ties.append(idx)

    boot = date_cluster_bootstrap(by_day, replicates=replicates, seed=seed)
    return dict(
        baseline=baseline_label,
        subject=subject_label,
        total_roots=len(banks),
        complete_pairs=len(complete_savings),
        unresolved_roots=unresolved,
        wins_count=len(wins),
        losses_count=len(losses),
        ties_count=len(ties),
        wins=wins,
        losses=losses,
        complete_mean_saving_seconds=float(mean(complete_savings)) if complete_savings else None,
        all_roots_missingness_saving_bounds=[float(mean(low_bounds)), float(mean(high_bounds))],
        date_cluster_bootstrap=boot,
        mean_cpu_delta_seconds=float(mean(cpu_deltas)),
    )


def evaluate_calibration_and_platforms(directory):
    directory = Path(directory)
    manifest = read(directory / "date_manifest.json")
    selection = read(directory / "selection.json")
    data = read(directory / "data.json.gz")
    physical_fit = read(directory / "physical_fit.json")
    emission = read(directory / "repaired_fit.json")
    endpoint_cov = read(directory / "endpoint_coverage.json")["days"]

    all_events, all_roots, all_pairs = [], [], []
    eval_days = []
    rt_platform_checks = Counter()

    for day_info in manifest["days"]:
        day = day_info["day"]
        graph, _, _, _, metadata, _ = context(data, day)
        msg_obj = read(directory / "messages" / f"{day}.json.gz")
        msg_summary = msg_obj["summary"]
        static_dir = directory / "inputs" / f"static_{day_info['static_version']}" / "data"
        plat_idx, stops, transfers = static_sources(
            data["dates"][day]["connections"], static_dir / "zurich_gtfs", static_dir / "swiss_gtfs"
        )

        # Audit RT platform announcements vs static platform stop_ids
        static_stop_by_trip_seq = {
            (trip, seq): rows[0]["stop_id"]
            for (trip, seq), rows in plat_idx.items()
            if len(rows) == 1
        }
        for frame in msg_obj.get("frames", []):
            t_id = frame.get("trip_id")
            for u in frame.get("platform_stop_updates", []):
                pub = u.get("published_stop_id") or ""
                assigned = u.get("assigned_stop_id") or ""
                effective = assigned if ":" in assigned else pub
                seq = u.get("stop_sequence")
                stat_id = static_stop_by_trip_seq.get((t_id, seq))
                if not stat_id or ":" not in stat_id:
                    continue
                if ":" not in effective:
                    rt_platform_checks["rt_station_only_or_missing"] += 1
                elif effective == stat_id:
                    rt_platform_checks["rt_matches_static_platform"] += 1
                else:
                    rt_platform_checks["rt_track_change_announced"] += 1

        day_events, day_roots, day_pairs = [], [], []
        for index, sel in enumerate(selection["selected"]):
            if sel["day"] != day:
                continue
            menu = read(directory / "menus" / f"{index:03}.json.gz")
            receipt = read(directory / "roots" / f"{index:03}_cheap_statmax_ideal.json.gz")
            vals, root_rec = root_events(
                graph, metadata, menu, receipt, index, physical_fit, emission, msg_summary
            )
            day_events.extend(vals)
            day_roots.append(root_rec)
            for ev_row in vals:
                ev = ev_row["event"]
                for role, inc, out, mct in (
                    ("upstream", ev["current_cid"], ev["upstream_cid"], ev["upstream_mct"]),
                    ("downstream", ev["source_arrival_cid"], ev["hub_cid"], ev["downstream_mct"]),
                ):
                    day_pairs.append(
                        dict(
                            day=day,
                            root_index=index,
                            event_index=ev["event_index"],
                            role=role,
                            **transfer_pair(metadata[inc], metadata[out], mct, plat_idx, stops, transfers),
                        )
                    )
        write(
            directory / "events" / f"{day}.json.gz",
            dict(day=day, events=day_events, roots=day_roots, platform_pairs=day_pairs),
        )
        all_events.extend(day_events)
        all_roots.extend(day_roots)
        all_pairs.extend(day_pairs)
        eval_days.append(
            dict(
                day=day,
                available=True,
                denominator_known=True,
                expected_opportunities=len(day_events),
                actual_available=endpoint_cov[day]["actual_source_available"],
                messages_available=True,
            )
        )

    evaluation = dict(manifest, days=eval_days)
    legacy_cal = legacy_calibration(all_events, evaluation)
    write(directory / "calibration_legacy_raw.json", legacy_cal)

    split_res = split_event_calibration(all_events, seed=20261116, replicates=20000)

    cal_split_out = dict(
        stage="bounded_gate_5d_v1",
        dates=manifest["selected_dates"],
        metric_c1_initial_availability=dict(
            **split_res["service_availability"],
            initial_state_statuses=dict(Counter(r["initial_state"]["status"] for r in all_roots)),
            registered_opportunities=len(all_events),
        ),
        metric_c2_conditional_onboard_calibration=split_res["conditional_onboard"],
    )
    write(directory / "calibration_split_summary.json", cal_split_out)

    plat_summary = summarize_platform_contract(all_pairs, tau_same_platform=90)
    plat_specific_rt = (
        rt_platform_checks["rt_matches_static_platform"]
        + rt_platform_checks["rt_track_change_announced"]
    )
    plat_summary["gtfs_rt_platform_announcements"] = dict(
        counts=dict(rt_platform_checks),
        platform_specific_updates=plat_specific_rt,
        static_agreement_fraction=(
            rt_platform_checks["rt_matches_static_platform"] / plat_specific_rt
            if plat_specific_rt
            else None
        ),
    )
    write(directory / "platform_contract_summary.json", plat_summary)
    return cal_split_out, plat_summary


def audit_dwell_and_gates(banks):
    """Measure physical ist-daten station dwell vs solver CPU across onboard gates."""
    dwells = []
    seen_physical_states = set()
    by_arm = {}

    for arm in ("cheap_plat_async", "recursive_plat_async", "bounded_cert_plat_async", "full128_plat_async"):
        gate_cpus = []
        decision_cpus = []
        submitted = 0
        accepted = 0
        expired = 0
        overrides_proposed = 0
        overrides_executed = 0

        for bank in banks:
            row = bank[arm]
            for st in row["states"]:
                if st["kind"] == "onboard" and st.get("has_continuation"):
                    key = (row["root_index"], st["cid"], st["clock"])
                    if key not in seen_physical_states and st.get("raw_dwell_seconds") is not None:
                        seen_physical_states.add(key)
                        dwells.append(float(st["raw_dwell_seconds"]))
            for dc in row["decision_costs"]:
                decision_cpus.append(float(dc["process_cpu"]))
            for pub in row["publications"]:
                if pub.get("background_submitted"):
                    submitted += 1
                    if pub.get("background_accepted"):
                        accepted += 1
                    elif pub.get("background_rejection_reason") == "deadline_expired":
                        expired += 1
            for g in row["gates"]:
                gate_cpus.append(float(g["cost"]["process_cpu"]))
                if g.get("proposed_action") != g.get("base_action"):
                    overrides_proposed += 1
                if g.get("selected_action") != g.get("base_action"):
                    overrides_executed += 1

        by_arm[arm] = dict(
            total_gates_invoked=len(gate_cpus),
            mean_gate_cpu_seconds=float(mean(gate_cpus)) if gate_cpus else 0.0,
            max_gate_cpu_seconds=float(max(gate_cpus)) if gate_cpus else 0.0,
            max_single_decision_cpu_seconds=float(max(decision_cpus)) if decision_cpus else 0.0,
            background_rollouts_submitted=submitted,
            background_rollouts_accepted=accepted,
            background_rollouts_expired_after_departure=expired,
            expiration_rate=(expired / submitted) if submitted else 0.0,
            overrides_proposed=overrides_proposed,
            overrides_executed=overrides_executed,
        )

    dwell_arr = np.asarray(dwells, dtype=float) if dwells else np.array([])
    return dict(
        measured_continuation_onboard_dwells=len(dwells),
        dwell_quantiles_seconds=(
            {
                "min": float(dwell_arr.min()),
                "p10": float(np.quantile(dwell_arr, 0.10)),
                "median": float(np.median(dwell_arr)),
                "mean": float(dwell_arr.mean()),
                "p90": float(np.quantile(dwell_arr, 0.90)),
                "max": float(dwell_arr.max()),
            }
            if len(dwell_arr)
            else None
        ),
        arms=by_arm,
    )


def summarize_sensitivity(directory):
    directory = Path(directory)
    rows = [read(directory / "sensitivity_roots" / f"{i:03}.json") for i in range(60)]
    out = {}
    for arm in ("cheap_plat_async", "recursive_plat_async"):
        out[arm] = {}
        for tau in ("60", "90", "120"):
            losses = [r["sensitivity"][arm][tau]["loss"] for r in rows]
            known = [float(x) for x in losses if x is not None]
            lo = [0.0 if x is None else float(x) for x in losses]
            hi = [float(r["loss_cap"]) if x is None else float(x) for r, x in zip(rows, losses)]
            # Compare against tau=90
            diffs_vs_90 = [
                float(r["sensitivity"][arm]["90"]["loss"] - r["sensitivity"][arm][tau]["loss"])
                for r in rows
                if r["sensitivity"][arm]["90"]["loss"] is not None
                and r["sensitivity"][arm][tau]["loss"] is not None
            ]
            out[arm][tau] = dict(
                complete_roots=len(known),
                complete_mean_loss_seconds=float(mean(known)) if known else None,
                all_60_loss_bounds=[float(mean(lo)), float(mean(hi))],
                mean_saving_vs_90s=float(mean(diffs_vs_90)) if diffs_vs_90 else 0.0,
                changed_roots_vs_90s=sum(abs(d) > 1e-9 for d in diffs_vs_90),
            )
    write(directory / "sensitivity_tau_same.json", out)
    return out


def make_quality_time_plot(arm_summaries, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    palette = {
        "cheap_statmax_ideal": ("#7f8c8d", "s", "Legacy Cheap (Station-Max, Ideal)"),
        "full128_statmax_ideal": ("#95a5a6", "^", "Legacy Full128 (Station-Max, Ideal)"),
        "cheap_plat_async": ("#1f77b4", "o", "Cheap Selective (Platform-Pair, Async)"),
        "snapshot_csa_plat_async": ("#ff7f0e", "D", "MeanSnapshotCSA (Platform-Pair, Async)"),
        "recursive_plat_async": ("#2ca02c", "P", "Recursive Selective (Platform-Pair, Async)"),
        "bounded_cert_plat_async": ("#d62728", "*", "Bounded Certified <2s (Platform-Pair, Async)"),
        "full128_plat_ideal": ("#9467bd", "v", "Full128 (Platform-Pair, Ideal Clock)"),
        "full128_plat_async": ("#8c564b", "X", "Full128 (Platform-Pair, Real Async Dwell)"),
    }
    for label, info in arm_summaries.items():
        color, marker, title = palette.get(label, ("#333333", "o", label))
        x = info["mean_cpu_seconds"]
        y = info["complete_mean_loss_seconds"]
        ax.scatter(x, y, s=110 if marker == "*" else 75, color=color, marker=marker, label=title, zorder=4)
    ax.axvline(2.0, color="#c0392b", linestyle="--", linewidth=1.1, alpha=0.75, label="2.0s Sub-Dwell Online Budget")
    ax.set_xscale("log")
    ax.set_xlabel("Mean Query CPU Time Including Menu Construction (seconds, log scale)")
    ax.set_ylabel("Mean Realized Journey Travel Time on Complete Roots (seconds)")
    ax.set_title("5-Day Unexposed Bounded Experiment (60 Roots, 2023-04-06..10): Quality vs. Compute")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(directory=HERE):
    directory = Path(directory)
    contract = read(directory / "contract.json")
    protocol = read(directory / "protocol.json")
    manifest = read(directory / "date_manifest.json")
    selection = read(directory / "selection.json")
    labels = [a["label"] for a in contract["arms"]]

    banks = []
    flat_rows = []
    for idx in range(contract["roots"]):
        bank = {lbl: read(directory / "roots" / f"{idx:03}_{lbl}.json.gz") for lbl in labels}
        banks.append(bank)
        for lbl in labels:
            r = bank[lbl]
            lo, hi = bounds(r)
            flat_rows.append(
                dict(
                    root_index=idx,
                    day=r["day"],
                    origin=r["origin"],
                    goal=r["goal"],
                    arm=lbl,
                    controller=r["arm"]["controller"],
                    mct_contract=r["arm"]["mct_contract"],
                    clock_contract=r["arm"]["clock_contract"],
                    status=r["outcome"]["status"],
                    loss=r["outcome"]["loss"],
                    loss_lower=lo,
                    loss_upper=hi,
                    cpu_seconds=round(r["cost"]["process_cpu_including_menu"], 6),
                    wall_seconds=round(r["cost"]["wall_including_menu"], 6),
                    states=len(r["states"]),
                    gates=len(r["gates"]),
                )
            )

    write_csv(directory / "paired_journeys.csv", flat_rows)

    arm_summaries = {}
    for lbl in labels:
        losses = [b[lbl]["outcome"]["loss"] for b in banks]
        known = [float(x) for x in losses if x is not None]
        lo_all = [bounds(b[lbl])[0] for b in banks]
        hi_all = [bounds(b[lbl])[1] for b in banks]
        cpus = [float(b[lbl]["cost"]["process_cpu_including_menu"]) for b in banks]
        walls = [float(b[lbl]["cost"]["wall_including_menu"]) for b in banks]
        arm_summaries[lbl] = dict(
            complete_roots=len(known),
            unresolved_roots=len(losses) - len(known),
            status_counts=dict(Counter(b[lbl]["outcome"]["status"] for b in banks)),
            complete_mean_loss_seconds=float(mean(known)) if known else None,
            all_60_loss_bounds=[float(mean(lo_all)), float(mean(hi_all))],
            mean_cpu_seconds=float(mean(cpus)),
            median_cpu_seconds=float(median(cpus)),
            max_cpu_seconds=float(max(cpus)),
            mean_wall_seconds=float(mean(walls)),
            total_gates=sum(len(b[lbl]["gates"]) for b in banks),
        )

    key_pairs = [
        ("cheap_statmax_ideal", "cheap_plat_async"),
        ("cheap_statmax_ideal", "full128_statmax_ideal"),
        ("cheap_plat_async", "snapshot_csa_plat_async"),
        ("cheap_plat_async", "recursive_plat_async"),
        ("cheap_plat_async", "bounded_cert_plat_async"),
        ("recursive_plat_async", "bounded_cert_plat_async"),
        ("snapshot_csa_plat_async", "bounded_cert_plat_async"),
        ("full128_plat_ideal", "full128_plat_async"),
        ("cheap_plat_async", "full128_plat_ideal"),
        ("cheap_plat_async", "full128_plat_async"),
        ("full128_plat_async", "bounded_cert_plat_async"),
    ]
    comparisons = {}
    comp_csv_rows = []
    for base_lbl, subj_lbl in key_pairs:
        res = compare_pair(banks, base_lbl, subj_lbl)
        comparisons[f"{base_lbl}__vs__{subj_lbl}"] = res
        comp_csv_rows.append(
            dict(
                baseline=base_lbl,
                subject=subj_lbl,
                complete_pairs=res["complete_pairs"],
                wins=res["wins_count"],
                losses=res["losses_count"],
                ties=res["ties_count"],
                mean_saving_seconds=round(res["complete_mean_saving_seconds"], 4)
                if res["complete_mean_saving_seconds"] is not None
                else None,
                ci95_low=round(res["date_cluster_bootstrap"]["ci95_saving_seconds"][0], 4),
                ci95_high=round(res["date_cluster_bootstrap"]["ci95_saving_seconds"][1], 4),
                all_60_bound_low=round(res["all_roots_missingness_saving_bounds"][0], 4),
                all_60_bound_high=round(res["all_roots_missingness_saving_bounds"][1], 4),
                mean_cpu_delta_seconds=round(res["mean_cpu_delta_seconds"], 4),
            )
        )
    write_csv(directory / "paired_comparisons.csv", comp_csv_rows)

    dwell_audit = audit_dwell_and_gates(banks)
    write(directory / "dwell_audit.json", dwell_audit)
    sens = summarize_sensitivity(directory)
    cal_split, plat_summary = evaluate_calibration_and_platforms(directory)
    make_quality_time_plot(arm_summaries, directory / "quality_time.png")

    # Determine honest pre-declared G2 / Path B verdict
    bc_vs_cheap = comparisons["cheap_plat_async__vs__bounded_cert_plat_async"]
    bc_vs_rec = comparisons["recursive_plat_async__vs__bounded_cert_plat_async"]
    bc_vs_csa = comparisons["snapshot_csa_plat_async__vs__bounded_cert_plat_async"]
    rec_vs_cheap = comparisons["cheap_plat_async__vs__recursive_plat_async"]
    bc_cpu_ok = arm_summaries["bounded_cert_plat_async"]["mean_cpu_seconds"] <= 2.0
    bc_zero_expired = (
        dwell_audit["arms"]["bounded_cert_plat_async"]["background_rollouts_expired_after_departure"]
        == 0
    )

    if (
        bc_cpu_ok
        and bc_zero_expired
        and bc_vs_cheap["date_cluster_bootstrap"]["ci95_saving_seconds"][0] > 0.0
        and bc_vs_rec["date_cluster_bootstrap"]["ci95_saving_seconds"][0] > 0.0
        and bc_vs_csa["date_cluster_bootstrap"]["ci95_saving_seconds"][0] > 0.0
    ):
        verdict = "PASS_BOUNDED_SELECTIVE"
    elif rec_vs_cheap["date_cluster_bootstrap"]["ci95_saving_seconds"][0] > 0.0:
        verdict = "PASS_ANALYTICAL_RECURSIVE"
    else:
        verdict = "CONFIRM_STOPPING_RULE_PATH_B"

    summary = dict(
        stage="bounded_gate_5d_v1",
        selected_dates=manifest["selected_dates"],
        total_roots=contract["roots"],
        original_72_additional_budget=0,
        verdict=verdict,
        arms=arm_summaries,
        comparisons=comparisons,
        dwell_summary=dwell_audit,
        tau_same_sensitivity=sens,
        calibration_c1=cal_split["metric_c1_initial_availability"],
        calibration_c2_residual=cal_split["metric_c2_conditional_onboard_calibration"][
            "date_cluster_inference"
        ]["mean_residual"],
        calibration_c2_brier_skill=cal_split["metric_c2_conditional_onboard_calibration"][
            "brier_skill_vs_climatology"
        ],
        platform_contract_resolved_fraction=plat_summary["platform_resolved_fraction"],
    )
    write(directory / "summary.json", summary)

    # Independent integrity audit receipt
    exposed_72 = set(read(directory / "exposure_inventory.json")["known_exposed_service_dates"])
    audit = dict(
        audited_at_utc=datetime.now(timezone.utc).isoformat(),
        checks=dict(
            seventy_two_prior_dates_excluded=not bool(exposed_72 & set(manifest["selected_dates"])),
            five_chronological_dates_frozen=len(manifest["selected_dates"]) == 5
            and manifest["dates_frozen"],
            dates_frozen_before_menus=manifest["frozen_at_utc"] <= selection["frozen_at_utc"],
            menus_frozen_before_rt_decode=selection["frozen_at_utc"]
            <= read(directory / "rt_decode_summary.json")["source_processing_started_at_utc"],
            menus_frozen_before_actual_filter=selection["frozen_at_utc"]
            <= read(directory / "actual_acquisition.json")["started_at_utc"],
            all_60_roots_retained=len(banks) == 60
            and all(len(b) == 8 for b in banks),
            original_72_additional_budget_zero=contract["original_72_additional_budget"] == 0,
            bounded_cert_sub_2s_mean_cpu=bc_cpu_ok,
        ),
    )
    audit["all_passed"] = all(audit["checks"].values())
    write(directory / "audit.json", audit)

    write(
        directory / "exposure_record.json",
        dict(
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            dates=manifest["selected_dates"],
            use="Final 5-day bounded confirmation/falsification experiment completed; dates now exposed.",
        ),
    )

    # Generate concise RESULTS.md
    g1_comp = comparisons["cheap_statmax_ideal__vs__cheap_plat_async"]
    f128_dwell = dwell_audit["arms"]["full128_plat_async"]
    bc_dwell = dwell_audit["arms"]["bounded_cert_plat_async"]
    lines = [
        "# Final 5-Day Bounded Experiment (`bounded_gate_5d_v1`) — Complete Results",
        "",
        f"- **Frozen Dates**: `{', '.join(manifest['selected_dates'])}` (`60` public roots, `480` paired arm executions, `0` additional budget on original 72 roots)",
        f"- **Audit Status**: `all_passed = {audit['all_passed']}` (`72` prior exposed dates excluded; chronological pre-outcome freezes verified)",
        f"- **Pre-Declared Verdict**: **`{verdict}`**",
        "",
        "## 1. Arm Quality & Computation Summary",
        "",
        "| Arm | Complete / 60 | Mean Loss (Complete, s) | All-60 Bounds (s) | Mean CPU (s) | Max CPU (s) | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lbl in labels:
        a = arm_summaries[lbl]
        lo_b, hi_b = a["all_60_loss_bounds"]
        m_loss = f"{a['complete_mean_loss_seconds']:.2f}" if a["complete_mean_loss_seconds"] is not None else "N/A"
        lines.append(
            f"| `{lbl}` | `{a['complete_roots']}/60` | `{m_loss}` | `[{lo_b:.2f}, {hi_b:.2f}]` | "
            f"`{a['mean_cpu_seconds']:.3f}` | `{a['max_cpu_seconds']:.3f}` | `{a['total_gates']}` |"
        )
    lines.extend(
        [
            "",
            "## 2. Key Paired Comparisons (Positive Saving = Subject Beats Baseline)",
            "",
            "| Baseline | Subject | Wins / Losses / Ties | Mean Saving (s) | 95% Date-Bootstrap CI (s) | Mean CPU Delta (s) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in comp_csv_rows:
        lines.append(
            f"| `{row['baseline']}` | `{row['subject']}` | `{row['wins']} / {row['losses']} / {row['ties']}` | "
            f"`{row['mean_saving_seconds']:+.2f}` | `[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}]` | `{row['mean_cpu_delta_seconds']:+.3f}` |"
        )
    lines.extend(
        [
            "",
            "## 3. G1 Contract, Physical Dwell Expiration & Conditional Calibration",
            "",
            f"- **G1 Platform-Pair MCT Effect (`cheap_plat_async` vs `cheap_statmax_ideal`)**: mean saving `{g1_comp['complete_mean_saving_seconds']:+.2f}s` (`{g1_comp['wins_count']}` wins, `{g1_comp['losses_count']}` losses). Static + same-platform coverage = `{100.0 * plat_summary['platform_resolved_fraction']:.1f}%`; GTFS-RT platform agreement = `{100.0 * (plat_summary['gtfs_rt_platform_announcements']['static_agreement_fraction'] or 0.0):.2f}%`.",
            f"- **Physical Dwell Reality (`ist-daten` `[AN_PROGNOSE, AB_PROGNOSE]`)**: median continuation station dwell = `{dwell_audit['dwell_quantiles_seconds']['median']:.1f}s` (`p10 = {dwell_audit['dwell_quantiles_seconds']['p10']:.1f}s`, `p90 = {dwell_audit['dwell_quantiles_seconds']['p90']:.1f}s`).",
            f"  - `full128_plat_async`: `{f128_dwell['background_rollouts_expired_after_departure']} / {f128_dwell['background_rollouts_submitted']}` (`{100.0 * f128_dwell['expiration_rate']:.1f}%`) rollout gates expired AFTER physical train departure!",
            f"  - `bounded_cert_plat_async`: `{bc_dwell['background_rollouts_expired_after_departure']} / {bc_dwell['background_rollouts_submitted']}` (`{100.0 * bc_dwell['expiration_rate']:.1f}%`) expired (`mean CPU = {arm_summaries['bounded_cert_plat_async']['mean_cpu_seconds']:.3f}s <= 2.0s`).",
            f"- **Calibration Split (`C1` vs `C2`)**: `C1` reached `{cal_split['metric_c2_conditional_onboard_calibration']['reached_resolved_opportunities']} / {cal_split['metric_c1_initial_availability']['registered_opportunities']}` opportunities (`{cal_split['metric_c1_initial_availability']['pre_onboard_canceled_roots']}` canceled initial roots); `C2` conditional `P10` residual = `{cal_split['metric_c2_conditional_onboard_calibration']['date_cluster_inference']['mean_residual']['10']:+.4f}`, 4-class Brier = `{cal_split['metric_c2_conditional_onboard_calibration']['mean_brier_4class']:.4f}` (Brier skill = `{100.0 * cal_split['metric_c2_conditional_onboard_calibration']['brier_skill_vs_climatology']:+.1f}%` vs climatology).",
            "",
        ]
    )
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Summary & RESULTS.md generated. Verdict:", verdict, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE)
    summarize(parser.parse_args().directory)
