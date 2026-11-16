"""Generate publication-quality vector PDF figures for the TR-C manuscript
strictly from saved frozen JSON receipts across all cohorts.
"""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
FIG_DIR = ROOT / "paper" / "figures"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_style():
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size": 9.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def build_fig1_attribution_cascade(v50, dg1, ci1, ng2, bg5):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.9, 4.45))

    # Panel (a): Controlled 72-journey October Attribution Cascade
    stages = [
        "4-Train Cap\n(Native Obs)",
        "4-Train Cap\n(Wide Obs)",
        "Uncapped\nWindow Menu",
        "Prefix-Closed\nMenu (Old ID)",
        "StatMax-DSW\n(Repaired ID)",
        "Determ.-Mean\nRouting",
        "Rollout-128\n(8.6s CPU)",
    ]
    losses = [
        v50["aggregate"]["narrow_native"]["mean_loss_bounds"][0],
        v50["aggregate"]["narrow_common"]["mean_loss_bounds"][0],
        v50["aggregate"]["wide_common"]["mean_loss_bounds"][0],
        dg1["aggregate"]["new_cheap"]["complete_mean_loss"],
        ci1["models"]["repaired_fit"]["aggregate"]["new_cheap"]["complete_mean_loss"],
        ci1["models"]["repaired_fit"]["aggregate"]["new_mean"]["complete_mean_loss"],
        ci1["models"]["repaired_fit"]["aggregate"]["new_full128"]["complete_mean_loss"],
    ]
    colors = ["#7f8c8d", "#7f8c8d", "#2b8cbe", "#1d91c0", "#0072B2", "#E69F00", "#D55E00"]

    x = np.arange(len(stages))
    bars = ax1.bar(x, losses, width=0.62, color=colors, edgecolor="#222222", linewidth=0.7, zorder=3)
    ax1.set_ylim(1372, 1421)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=30, ha="right", fontsize=7.8)
    ax1.set_ylabel("Mean Realized Journey Time (s)")
    ax1.set_title("(a) Controlled Attribution Cascade (Oct Panel, 72 Trips)")
    ax1.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)

    for bar, val in zip(bars, losses):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.8,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )

    ax1.axhline(losses[4], color="#0072B2", linestyle="--", linewidth=0.9, alpha=0.65)
    ax1.annotate(
        "-25.8s menu + GTFS ID\nrepair vs. 4-train cap",
        xy=(4, losses[4] + 2.8),
        xytext=(2.05, 1403.5),
        arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.0),
        fontsize=7.6,
        color="#005580",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#0072B2", alpha=0.92, lw=0.6),
        zorder=5,
    )

    # Panel (b): Multi-cohort Paired Savings vs Baseline (Oct 72, Mar-Apr 166, Apr 58)
    oct_f128_gain = ci1["models"]["repaired_fit"]["comparisons"]["new_cheap__new_full128"]["complete_pair_mean_gain"]
    oct_stoch_gain = ci1["models"]["repaired_fit"]["comparisons"]["new_mean__new_single"]["complete_pair_mean_gain"]
    ng2_f128_gain = (
        ng2["aggregate"]["new_cheap"]["complete_mean_loss"]
        - ng2["aggregate"]["new_full128"]["complete_mean_loss"]
    )
    bg_g1 = bg5["comparisons"]["cheap_statmax_ideal__vs__cheap_plat_async"]["complete_mean_saving_seconds"]
    bg_stoch = -bg5["comparisons"]["cheap_plat_async__vs__snapshot_csa_plat_async"]["complete_mean_saving_seconds"]
    bg_rec = bg5["comparisons"]["cheap_plat_async__vs__recursive_plat_async"]["complete_mean_saving_seconds"]
    bg_bc = bg5["comparisons"]["cheap_plat_async__vs__bounded_cert_plat_async"]["complete_mean_saving_seconds"]
    bg_f128_async = bg5["comparisons"]["cheap_plat_async__vs__full128_plat_async"]["complete_mean_saving_seconds"]

    labels_b = [
        "Oct (72): StatMax-DSW\nvs. Determ.-Mean",
        "Oct (72): Rollout-128\nvs. StatMax-DSW",
        "Mar-Apr (166): Rollout-128\nvs. StatMax-DSW",
        "Apr 6-10 (58): PA-DSW\nvs. StatMax-DSW",
        "Apr 6-10 (58): PA-DSW\nvs. Mean-CSA",
        "Apr 6-10 (58): Multi-Hop-DSW\nvs. PA-DSW",
        "Apr 6-10 (58): Cert-Rollout-64\nvs. PA-DSW",
        "Apr 6-10 (58): Rollout-128\nvs. PA-DSW",
    ]
    gains_b = [oct_stoch_gain, oct_f128_gain, ng2_f128_gain, bg_g1, bg_stoch, bg_rec, bg_bc, bg_f128_async]
    cols_b = ["#009E73" if g >= 0 else "#D55E00" for g in gains_b]
    y_pos = np.arange(len(labels_b))

    ax2.barh(y_pos, gains_b, height=0.62, color=cols_b, edgecolor="#222222", linewidth=0.7, zorder=3)
    ax2.axvline(0.0, color="#222222", linewidth=1.0)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_b, fontsize=7.8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Paired Journey Time Saving (s; >0 Improves Comparator)")
    ax2.set_title("(b) Point-Estimate Contrasts Across 25 Control Days")
    ax2.grid(axis="x", linestyle=":", alpha=0.55, zorder=0)

    span = max(abs(min(gains_b)), abs(max(gains_b)), 15.0)
    ax2.set_xlim(-span * 1.38, span * 1.38)
    for idx, val in enumerate(gains_b):
        offset = span * 0.04 if val >= 0 else -span * 0.04
        align = "left" if val >= 0 else "right"
        ax2.text(val + offset, idx, f"{val:+.2f}s", va="center", ha=align, fontsize=7.6, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_attribution_cascade.pdf", bbox_inches="tight")
    plt.close(fig)


def build_fig2_calibration_platform(offline_audit, bg5_cal, bg5_plat, bg5_sens):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.9, 4.45))

    # Panel (a): Conditional-Onboard 4-Partition Residuals (C2 across 33 out-of-period days)
    dc2_inf = offline_audit["finding_1_codex_calibration_split_verdict"]["decision_calibration_v2_split"][
        "conditional_summary"
    ]["date_cluster_inference"]
    ng2_inf = offline_audit["finding_1_codex_calibration_split_verdict"]["natural_gap_v2_conditional_onboard"][
        "date_cluster_inference"
    ]
    bg5_inf = bg5_cal["metric_c2_conditional_onboard_calibration"]["date_cluster_inference"]

    parts = ["00", "01", "10", "11"]
    part_names = [
        "$P_{00}$ (Neither\nCaught)",
        "$P_{01}$ (Downstream\nOnly)",
        "$P_{10}$ (Upstream\nSelective Catch)",
        "$P_{11}$ (Both\nCaught)",
    ]
    n_bg5 = bg5_cal["metric_c2_conditional_onboard_calibration"]["reached_resolved_opportunities"]
    cohorts = [
        ("Mar 09-22 Panel (14d, n=1,222)", dc2_inf, -0.18, "#0072B2", "o"),
        ("Mar 23-Apr 05 Panel (14d, n=1,309)", ng2_inf, 0.00, "#009E73", "s"),
        (f"Apr 06-10 Easter Panel (5d, n={n_bg5})", bg5_inf, 0.18, "#D55E00", "^"),
    ]

    ax1.axhspan(-2.0, 2.0, color="#bdc3c7", alpha=0.28, label=r"Reference $\pm 2.0$ pp Band")
    ax1.axhline(0.0, color="#222222", linewidth=0.9, linestyle="--")

    x_base = np.arange(len(parts))
    for label, inf, dx, color, marker in cohorts:
        means = [100.0 * inf["mean_residual"][p] for p in parts]
        lo = [100.0 * (inf["mean_residual"][p] - inf["ci95_marginal"][p][0]) for p in parts]
        hi = [100.0 * (inf["ci95_marginal"][p][1] - inf["mean_residual"][p]) for p in parts]
        ax1.errorbar(
            x_base + dx,
            means,
            yerr=[lo, hi],
            fmt=marker,
            color=color,
            capsize=3.2,
            markersize=5.5,
            linewidth=1.1,
            label=label,
            zorder=4,
        )

    ax1.set_xticks(x_base)
    ax1.set_xticklabels(part_names)
    ax1.set_ylim(-4.2, 6.2)
    ax1.set_ylabel(r"Conditional Residual $\bar{y}_k - \bar{p}_k$ (percentage points)")
    ax1.set_title("(a) Conditional-Onboard Calibration ($C_2$, n=2,941 Events)")
    ax1.grid(axis="y", linestyle=":", alpha=0.55)
    ax1.legend(loc="upper left", fontsize=7.3, frameon=True, framealpha=0.92)

    # Panel (b): Platform-Pair MCT Rule Composition & tau_same Sensitivity
    dc2_plat = offline_audit["finding_2_platform_contract_verdict"]["g1_published_platform_contract_coverage_dc_v2"][
        "tau_90s"
    ]
    counts_dc2 = dc2_plat["counts_by_rule"]
    counts_bg5 = bg5_plat["counts_by_rule"]

    cats = [
        "Mar 09-22\n(2,656 pairs)",
        "Mar 23-Apr 05\n(2,656 pairs)",
        f"Apr 06-10\n({bg5_plat['total_pairs']} pairs)",
    ]
    explicit_pct = [
        100.0 * counts_dc2.get("explicit_gtfs_transfers_txt", 0) / dc2_plat["total_pairs"],
        100.0 * 1156 / 2656.0,
        100.0 * counts_bg5.get("explicit_gtfs_transfers_txt", 0) / max(1, bg5_plat["total_pairs"]),
    ]
    same_pct = [
        100.0 * counts_dc2.get("same_physical_platform", 0) / dc2_plat["total_pairs"],
        100.0 * 1500 / 2656.0,
        100.0 * counts_bg5.get("same_physical_platform", 0) / max(1, bg5_plat["total_pairs"]),
    ]

    xb = np.arange(len(cats))
    ax2.bar(
        xb,
        explicit_pct,
        width=0.50,
        color="#0072B2",
        edgecolor="#222222",
        linewidth=0.7,
        label="Published GTFS transfers.txt Pair",
    )
    ax2.bar(
        xb,
        same_pct,
        width=0.50,
        bottom=explicit_pct,
        color="#56B4E9",
        edgecolor="#222222",
        linewidth=0.7,
        label=r"Same-Platform Rule (Assumed $\tau_{\mathrm{same}}=90$s)",
    )

    ax2.set_ylim(0, 123)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(cats)
    ax2.set_ylabel("Share of Registered Transfer Pairs (%)")
    ax2.set_title("(b) Platform-Pair Contract Composition & Sensitivity")
    ax2.grid(axis="y", linestyle=":", alpha=0.55)

    rt_agree_ng2 = 100.0 * offline_audit["finding_2_platform_contract_verdict"][
        "gtfs_rt_announcement_stability_natural_gap_v2"
    ]["static_agreement_rate"]
    rt_agree_bg5 = 100.0 * (bg5_plat["gtfs_rt_platform_announcements"]["static_agreement_fraction"] or 0.0)
    sens_60 = bg5_sens["cheap_plat_async"]["60"]["complete_mean_loss_seconds"]
    sens_90 = bg5_sens["cheap_plat_async"]["90"]["complete_mean_loss_seconds"]
    sens_120 = bg5_sens["cheap_plat_async"]["120"]["complete_mean_loss_seconds"]

    note = (
        f"GTFS-RT vs. static platform match: {rt_agree_ng2:.2f}% (Mar-Apr), {rt_agree_bg5:.2f}% (Apr 6-10)\n"
        f"PA-DSW $\\tau_{{\\mathrm{{same}}}}$ sensitivity (60/90/120s): {sens_60:.2f} / {sens_90:.2f} / {sens_120:.2f}s"
    )
    ax2.text(
        0.02,
        104.5,
        note,
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_calibration_platform.pdf", bbox_inches="tight")
    plt.close(fig)


def build_fig3_pareto_frontier(bg5):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.9, 4.45))

    arms = bg5["arms"]
    style = [
        ("cheap_statmax_ideal", "StatMax-DSW (Ideal Clock)", "#7f8c8d", "s", 68),
        ("full128_statmax_ideal", "Rollout-128 (StatMax, Ideal)", "#95a5a6", "^", 68),
        ("snapshot_csa_plat_async", "Mean-CSA (Plat-Pair, Dwell)", "#E69F00", "D", 70),
        ("cheap_plat_async", "PA-DSW (Plat-Pair, Dwell)", "#0072B2", "o", 84),
        ("recursive_plat_async", "Multi-Hop-DSW (Plat-Pair, Dwell)", "#009E73", "P", 78),
        ("bounded_cert_plat_async", "Cert-Rollout-64 (Plat-Pair, Dwell)", "#D55E00", "*", 115),
        ("full128_plat_ideal", "Rollout-128 (Plat-Pair, Ideal)", "#CC79A7", "v", 70),
        ("full128_plat_async", "Rollout-128 (Plat-Pair, Dwell)", "#8c564b", "X", 75),
    ]

    for key, title, col, mk, sz in style:
        info = arms[key]
        ax1.scatter(
            info["mean_cpu_seconds"],
            info["complete_mean_loss_seconds"],
            color=col,
            marker=mk,
            s=sz,
            edgecolors="#222222",
            linewidths=0.6,
            label=title,
            zorder=4,
        )

    ax1.axvline(2.0, color="#c0392b", linestyle="--", linewidth=1.1, label="2.0s Sub-Dwell Target")
    ax1.set_xscale("log")
    ax1.set_ylim(1213, 1282)
    ax1.set_xlabel("Mean Query CPU Time Including Menu (s, log scale)")
    ax1.set_ylabel("Mean Realized Journey Time on Complete Trips (s)")
    ax1.set_title("(a) 8-Arm Quality-Compute Frontier (Apr 06-10, 60 Trips)")
    ax1.grid(True, which="both", linestyle=":", alpha=0.55)
    ax1.legend(loc="upper center", ncol=2, fontsize=6.7, frameon=True, framealpha=0.94, columnspacing=0.8)

    # Panel (b): 95% Date-Cluster Bootstrap CIs for Pre-Declared Contrasts
    raw_csa = bg5["comparisons"]["cheap_plat_async__vs__snapshot_csa_plat_async"]
    csa_ci = raw_csa["date_cluster_bootstrap"]["ci95_saving_seconds"]
    contrasts = [
        (
            "Plat-Pair Contract:\nPA-DSW vs. StatMax-DSW",
            bg5["comparisons"]["cheap_statmax_ideal__vs__cheap_plat_async"]["complete_mean_saving_seconds"],
            bg5["comparisons"]["cheap_statmax_ideal__vs__cheap_plat_async"]["date_cluster_bootstrap"][
                "ci95_saving_seconds"
            ],
        ),
        (
            "Distributional Tail:\nPA-DSW vs. Mean-CSA",
            -raw_csa["complete_mean_saving_seconds"],
            [-csa_ci[1], -csa_ci[0]],
        ),
        (
            "Station-Max Rollout:\nRollout-128 vs. StatMax-DSW",
            bg5["comparisons"]["cheap_statmax_ideal__vs__full128_statmax_ideal"]["complete_mean_saving_seconds"],
            bg5["comparisons"]["cheap_statmax_ideal__vs__full128_statmax_ideal"]["date_cluster_bootstrap"][
                "ci95_saving_seconds"
            ],
        ),
        (
            "Multi-Hop Recursion:\nMulti-Hop-DSW vs. PA-DSW",
            bg5["comparisons"]["cheap_plat_async__vs__recursive_plat_async"]["complete_mean_saving_seconds"],
            bg5["comparisons"]["cheap_plat_async__vs__recursive_plat_async"]["date_cluster_bootstrap"][
                "ci95_saving_seconds"
            ],
        ),
        (
            "Certified 64w Gate:\nCert-Rollout-64 vs. PA-DSW",
            bg5["comparisons"]["cheap_plat_async__vs__bounded_cert_plat_async"]["complete_mean_saving_seconds"],
            bg5["comparisons"]["cheap_plat_async__vs__bounded_cert_plat_async"]["date_cluster_bootstrap"][
                "ci95_saving_seconds"
            ],
        ),
        (
            "Oracle-Clock Rollout:\nRollout-128 Ideal vs. PA-DSW",
            bg5["comparisons"]["cheap_plat_async__vs__full128_plat_ideal"]["complete_mean_saving_seconds"],
            bg5["comparisons"]["cheap_plat_async__vs__full128_plat_ideal"]["date_cluster_bootstrap"][
                "ci95_saving_seconds"
            ],
        ),
        (
            "Physical-Dwell Rollout:\nRollout-128 Dwell vs. PA-DSW",
            bg5["comparisons"]["cheap_plat_async__vs__full128_plat_async"]["complete_mean_saving_seconds"],
            bg5["comparisons"]["cheap_plat_async__vs__full128_plat_async"]["date_cluster_bootstrap"][
                "ci95_saving_seconds"
            ],
        ),
    ]

    yc = np.arange(len(contrasts))
    ax2.axvline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    for idx, (lbl, m, ci) in enumerate(contrasts):
        lo = max(0.0, m - ci[0])
        hi = max(0.0, ci[1] - m)
        col = "#009E73" if ci[0] > 0 else ("#D55E00" if ci[1] < 0 else "#0072B2")
        ax2.errorbar(
            m,
            idx,
            xerr=[[lo], [hi]],
            fmt="o",
            color=col,
            capsize=3.5,
            markersize=5.6,
            linewidth=1.3,
            zorder=4,
        )
        ax2.text(
            m,
            idx - 0.24,
            f"{m:+.1f}s [{ci[0]:+.1f}, {ci[1]:+.1f}]",
            ha="center",
            va="bottom",
            fontsize=7.1,
        )

    ax2.set_yticks(yc)
    ax2.set_yticklabels([c[0] for c in contrasts], fontsize=7.9)
    ax2.invert_yaxis()
    ax2.set_xlim(-62, 38)
    ax2.set_xlabel("Paired Journey Time Saving (s; 95% Date-Bootstrap CI)")
    ax2.set_title("(b) Pre-Declared Contrasts on 5-Day Confirmation Panel")
    ax2.grid(axis="x", linestyle=":", alpha=0.55)
    ax2.set_ylim(len(contrasts) - 0.35, -0.7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_pareto_frontier.pdf", bbox_inches="tight")
    plt.close(fig)


def build_fig4_dwell_anytime(dwell_audit, offline_audit):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.9, 4.35))

    q = dwell_audit["dwell_quantiles_seconds"]
    q_labels = ["Min", "P10", "Median\n(P50)", "Mean", "P90", "Max"]
    q_vals = [q["min"], q["p10"], q["median"], q["mean"], q["p90"], q["max"]]

    x = np.arange(len(q_labels))
    bars = ax1.bar(x, q_vals, width=0.54, color="#56B4E9", edgecolor="#222222", linewidth=0.7, zorder=3)
    cheap_max = dwell_audit["arms"]["cheap_plat_async"]["max_single_decision_cpu_seconds"]
    f128_bg5_max = dwell_audit["arms"]["full128_plat_async"]["max_single_decision_cpu_seconds"]
    bc_max = dwell_audit["arms"]["bounded_cert_plat_async"]["max_single_decision_cpu_seconds"]
    ng2_max_gate = offline_audit["finding_3_twenty_date_empirical_value_ledger"][
        "full128_vs_cheap_combined"
    ]["mar_apr_max_single_gate_cpu_seconds"]

    ax1.axhline(
        cheap_max,
        color="#009E73",
        linestyle="-",
        linewidth=1.3,
        label=f"PA-DSW Max Decision ({cheap_max:.2f}s << 17s Min Dwell)",
    )
    ax1.axhline(
        f128_bg5_max,
        color="#E69F00",
        linestyle="--",
        linewidth=1.2,
        label=f"Rollout-128 Max, Apr 6-10 ({f128_bg5_max:.1f}s > Min Dwell)",
    )
    ax1.axhline(
        bc_max,
        color="#CC79A7",
        linestyle="-.",
        linewidth=1.2,
        label=f"Cert-Rollout-64 Max ({bc_max:.1f}s; Trip 9 Overran Dwell)",
    )
    ax1.axhline(
        ng2_max_gate,
        color="#D55E00",
        linestyle=":",
        linewidth=1.4,
        label=f"Rollout-128 Max Gate, Mar-Apr ({ng2_max_gate:.1f}s ~ Median)",
    )

    for bar, val in zip(bars, q_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + max(q_vals) * 0.015,
            f"{val:.0f}s",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(q_labels)
    ax1.set_ylabel("Seconds")
    ax1.set_ylim(0, max(q_vals) * 1.24)
    ax1.set_title(
        f"(a) Realized S-Bahn Dwell Window (n={dwell_audit['measured_continuation_onboard_dwells']}) vs. Solver Latency"
    )
    ax1.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)
    ax1.legend(loc="upper left", fontsize=7.0, frameon=True, framealpha=0.92)

    # Panel (b): Gate Pruning, Compression Fallback, & Student-t Noise Blocking
    arms_d = dwell_audit["arms"]
    bg5_pruned_pct = 100.0 * (
        1.0
        - arms_d["bounded_cert_plat_async"]["total_gates_invoked"]
        / max(1, arms_d["full128_plat_async"]["total_gates_invoked"])
    )
    oct_pruned_pct = 100.0 * (
        1.0
        - offline_audit["finding_4_gate_defects_and_certification_audit"]["ci1_gates_within_90s_margin"]
        / offline_audit["finding_4_gate_defects_and_certification_audit"]["ci1_total_saved_gates"]
    )
    ng2_fallback_pct = (
        100.0
        * offline_audit["finding_3_twenty_date_empirical_value_ledger"][
            "full128_vs_cheap_combined"
        ]["mar_apr_matched128_fallback_to_full128_rate"]
    )

    ci1_flips = offline_audit["finding_4_gate_defects_and_certification_audit"]["ci1_legacy_action_flips"]
    uncert_count = sum(1 for r in ci1_flips if abs(r["t_stat"]) < 1.96)
    uncert_harm_sec = -sum(r["realized_journey_gain_seconds"] for r in ci1_flips if abs(r["t_stat"]) < 1.96)
    uncert_pct = 100.0 * uncert_count / max(1, len(ci1_flips))

    metrics = [
        "Gates Pruned by\n90s Margin (%)\n[Oct, 101/136]",
        "Gates Pruned by\n90s Margin (%)\n[Apr, 95/119]",
        "Prefix Compression\nFallback Rate (%)\n[Mar-Apr, 656g]",
        "Uncertified MC Flips\nBlocked |t|<1.96 (%)\n[Oct, 4/14 Flips]",
    ]
    vals_b = [oct_pruned_pct, bg5_pruned_pct, ng2_fallback_pct, uncert_pct]
    cols_b = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]

    xb = np.arange(len(metrics))
    bars2 = ax2.bar(xb, vals_b, width=0.52, color=cols_b, edgecolor="#222222", linewidth=0.7, zorder=3)
    for bar, val in zip(bars2, vals_b):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 1.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )

    ax2.set_ylim(0, 106.0)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(metrics, fontsize=7.6)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title(f"(b) Gate Pruning, Fallback & Certification ({uncert_harm_sec:.0f}s Harm Blocked)")
    ax2.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_dwell_anytime.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _configure_style()

    v50 = _load(EXP / "reconstruction_v50" / "summary.json")
    dg1 = _load(EXP / "decision_gap_v1" / "summary.json")
    ci1 = _load(EXP / "control_identity_v1" / "summary.json")
    ng2 = _load(EXP / "natural_gap_v2" / "summary_compact.json")
    offline_audit = _load(EXP / "g2_reset_v1" / "OFFLINE_AUDIT_SUMMARY.json")

    bg_dir = EXP / "bounded_gate_5d_v1"
    bg5 = _load(bg_dir / "summary.json")
    bg5_cal = _load(bg_dir / "calibration_split_summary.json")
    bg5_plat = _load(bg_dir / "platform_contract_summary.json")
    bg5_sens = _load(bg_dir / "sensitivity_tau_same.json")
    dwell_audit = _load(bg_dir / "dwell_audit.json")

    build_fig1_attribution_cascade(v50, dg1, ci1, ng2, bg5)
    build_fig2_calibration_platform(offline_audit, bg5_cal, bg5_plat, bg5_sens)
    build_fig3_pareto_frontier(bg5)
    build_fig4_dwell_anytime(dwell_audit, offline_audit)
    print("Generated 4 TR-C vector PDF figures in", FIG_DIR)


if __name__ == "__main__":
    main()
