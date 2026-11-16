"""Separate pre-onboard initial-ride cancellations from conditional onboard calibration,
and classify static/announced platform-pair MCT contracts without unpassable 2023 track gates.
"""

from collections import Counter
import numpy as np

PARTITIONS = ("00", "01", "10", "11")
DEFAULT_SEED = 20261115
DEFAULT_REPLICATES = 20000
LEGACY_EQUIVALENCE_MARGIN = 0.02
BONFERRONI_LOWER_Q = 0.00625
BONFERRONI_UPPER_Q = 0.99375


def _whole_date_bootstrap(day_residuals, *, seed=DEFAULT_SEED, replicates=DEFAULT_REPLICATES):
    """Bootstrap date-cluster means (equal weight per operating day, matching v2 convention)."""
    matrix = np.asarray(day_residuals, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 4 or len(matrix) == 0:
        raise ValueError("day_residuals must have shape (n_days, 4)")
    n_days = len(matrix)
    point = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_days, size=(replicates, n_days))
    draws = matrix[picks].mean(axis=1)
    ci95 = np.quantile(draws, [0.025, 0.975], axis=0)
    bonf95 = np.quantile(draws, [BONFERRONI_LOWER_Q, BONFERRONI_UPPER_Q], axis=0)
    se = draws.std(axis=0, ddof=1)
    legacy_within_margin = bool(
        np.all(bonf95[0] >= -LEGACY_EQUIVALENCE_MARGIN)
        and np.all(bonf95[1] <= LEGACY_EQUIVALENCE_MARGIN)
    )
    legacy_point_exceeds = bool(np.any(np.abs(point) > LEGACY_EQUIVALENCE_MARGIN))
    return {
        "n_days": int(n_days),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_residual": {k: float(point[i]) for i, k in enumerate(PARTITIONS)},
        "bootstrap_se": {k: float(se[i]) for i, k in enumerate(PARTITIONS)},
        "ci95_marginal": {
            k: [float(ci95[0, i]), float(ci95[1, i])] for i, k in enumerate(PARTITIONS)
        },
        "ci95_bonferroni_4class": {
            k: [float(bonf95[0, i]), float(bonf95[1, i])] for i, k in enumerate(PARTITIONS)
        },
        "positive_days_by_partition": {
            k: int(np.sum(matrix[:, i] > 0)) for i, k in enumerate(PARTITIONS)
        },
        "legacy_0p02_gate_would_pass": legacy_within_margin,
        "legacy_0p02_point_estimate_violated": legacy_point_exceeds,
    }


def split_event_calibration(events, *, seed=DEFAULT_SEED, replicates=DEFAULT_REPLICATES):
    """Separate pre-onboard initial-vehicle cancellations from conditional-onboard events."""
    pre_onboard_canceled = []
    conditional_resolved = []
    conditional_unresolved = []

    for row in events:
        init_status = row.get("initial_state", {}).get("status")
        model = row.get("model", {})
        actual = row.get("actual", {})
        if init_status != "observed" or model.get("reason") == "initial_history_unavailable":
            pre_onboard_canceled.append(row)
            continue
        part = actual.get("actual_partition")
        if model.get("status") == "resolved" and part in PARTITIONS:
            conditional_resolved.append(row)
        else:
            conditional_unresolved.append(row)

    by_day = {}
    briers = []
    obs_vectors = []
    for row in conditional_resolved:
        p = np.array([float(row["model"]["partitions"][k]) for k in PARTITIONS])
        y = np.array([1.0 if row["actual"]["actual_partition"] == k else 0.0 for k in PARTITIONS])
        if abs(p.sum() - 1.0) > 1e-8:
            raise ValueError("Model partition probabilities must sum to 1")
        res = y - p
        by_day.setdefault(row["day"], []).append(res)
        briers.append(float(np.sum((p - y) ** 2)))
        obs_vectors.append(y)

    days_sorted = sorted(by_day)
    day_means = [np.mean(by_day[d], axis=0) for d in days_sorted]
    bootstrap = _whole_date_bootstrap(day_means, seed=seed, replicates=replicates)

    obs_mat = np.asarray(obs_vectors, dtype=float)
    clim = obs_mat.mean(axis=0)
    clim_brier = float(np.mean(np.sum((obs_mat - clim[None, :]) ** 2, axis=1)))
    mean_brier = float(np.mean(briers))
    brier_skill = 1.0 - (mean_brier / clim_brier) if clim_brier > 0 else 0.0

    roots_total = len({(r["day"], r["root_index"]) for r in events})
    roots_canceled = len({(r["day"], r["root_index"]) for r in pre_onboard_canceled})

    return {
        "service_availability": {
            "total_roots": roots_total,
            "pre_onboard_canceled_roots": roots_canceled,
            "pre_onboard_cancellation_rate": (
                roots_canceled / roots_total if roots_total else 0.0
            ),
            "unreached_opportunities": len(pre_onboard_canceled),
        },
        "conditional_onboard": {
            "reached_resolved_opportunities": len(conditional_resolved),
            "reached_unresolved_opportunities": len(conditional_unresolved),
            "empirical_rates": {k: float(clim[i]) for i, k in enumerate(PARTITIONS)},
            "mean_brier_4class": mean_brier,
            "climatology_brier_4class": clim_brier,
            "brier_skill_vs_climatology": brier_skill,
            "date_cluster_inference": bootstrap,
            "per_day_mean_residual": {
                d: {k: float(day_means[idx][i]) for i, k in enumerate(PARTITIONS)}
                for idx, d in enumerate(days_sorted)
            },
        },
    }


def extract_conditional_from_compact_summary(
    summary, *, seed=DEFAULT_SEED, replicates=DEFAULT_REPLICATES
):
    """Exact algebraic recovery of conditional-onboard residuals from compact incident bounds.

    When unresolved events arise purely from unreached initial rides (symmetric [-1, +1]
    bounds on every partition), each incident's resolved residual sum is S_k = N * (lo_k + hi_k) / 2
    with exact verification hi_k - lo_k == 2 * (N - R) / N and sum_k S_k == 0.
    """
    day_means = []
    days_sorted = []
    total_opp = 0
    total_resolved = 0
    total_unreached = 0
    weighted_sum = np.zeros(4, dtype=float)
    brier_weighted = 0.0

    for day_row in sorted(summary["days"], key=lambda r: r["day"]):
        day = day_row["day"]
        day_sum = np.zeros(4, dtype=float)
        day_res = 0
        day_u = 0
        for inc in day_row["incidents"]:
            n = len(inc["event_row_indices"])
            lo0, hi0 = inc["bias_bounds"]["00"]
            u = int(round(n * (hi0 - lo0) / 2.0))
            r_mod = n - u
            expected_width = 2.0 * u / n
            inc_sum = np.zeros(4, dtype=float)
            for i, k in enumerate(PARTITIONS):
                lo, hi = inc["bias_bounds"][k]
                if abs((hi - lo) - expected_width) > 1e-11:
                    raise ValueError(
                        f"Incident {day}:{inc['index']} partition {k} does not have symmetric [-1,1] unreached bounds"
                    )
                p_lo, p_hi = inc["predicted_probability_bounds"][k]
                o_lo, o_hi = inc["observed_frequency_bounds"][k]
                if abs((p_hi - p_lo) - (u / n)) > 1e-11 or abs((o_hi - o_lo) - (u / n)) > 1e-11:
                    raise ValueError(
                        f"Incident {day}:{inc['index']} probability/frequency widths do not match unreached count"
                    )
                inc_sum[i] = n * 0.5 * (lo + hi)
            if r_mod > 0 and abs(inc_sum.sum()) > 1e-8:
                raise ValueError(f"Partition residual sum is nonzero: {inc_sum.sum()}")
            b_lo, b_hi = inc["four_class_brier_bounds"]
            if abs((b_hi - b_lo) - (2.0 * u / n)) > 1e-11:
                raise ValueError("Incident Brier bounds do not match [0, 2] unreached width")
            brier_weighted += n * b_lo

            day_sum += inc_sum
            day_res += r_mod
            day_u += u
            total_opp += n
            total_resolved += r_mod
            total_unreached += u

        if (
            day_res != int(day_row["model_resolved"])
            or day_res != int(day_row["label_resolved"])
            or (day_res + day_u) != int(day_row["represented_event_rows"])
        ):
            raise ValueError(f"Day {day} incident counts do not match day header totals")
        if day_res == 0:
            raise ValueError(f"Day {day} has zero reached onboard opportunities")
        days_sorted.append(day)
        day_means.append(day_sum / day_res)
        weighted_sum += day_sum

    bootstrap = _whole_date_bootstrap(day_means, seed=seed, replicates=replicates)
    return {
        "total_registered_opportunities": total_opp,
        "unreached_initial_cancellation_opportunities": total_unreached,
        "conditional_reached_opportunities": total_resolved,
        "opportunity_weighted_mean_residual": {
            k: float(weighted_sum[i] / total_resolved) for i, k in enumerate(PARTITIONS)
        },
        "conditional_mean_brier_4class": float(brier_weighted / total_resolved),
        "date_cluster_inference": bootstrap,
        "per_day_mean_residual": {
            d: {k: float(day_means[idx][i]) for i, k in enumerate(PARTITIONS)}
            for idx, d in enumerate(days_sorted)
        },
    }


def classify_platform_pair_contract(pair, *, tau_same_platform=90):
    """Assign explicit G1 transfer time under published/announced platform-pair contract."""
    arr_stop = pair.get("arrival", {}).get("stop_id")
    dep_stop = pair.get("departure", {}).get("stop_id")
    arr_plat = bool(pair.get("arrival", {}).get("platform_specific"))
    dep_plat = bool(pair.get("departure", {}).get("platform_specific"))
    stated = pair.get("stated_pair_seconds")
    station_fallback = int(pair["fixed_mct_seconds"])

    if stated is not None:
        return {
            "rule": "explicit_gtfs_transfers_txt",
            "contract_mct_seconds": int(stated),
            "legacy_mct_seconds": station_fallback,
            "mct_delta_seconds": int(stated) - station_fallback,
        }
    if arr_plat and dep_plat and arr_stop is not None and arr_stop == dep_stop:
        return {
            "rule": "same_physical_platform",
            "contract_mct_seconds": int(tau_same_platform),
            "legacy_mct_seconds": station_fallback,
            "mct_delta_seconds": int(tau_same_platform) - station_fallback,
        }
    return {
        "rule": "parent_station_fallback",
        "contract_mct_seconds": station_fallback,
        "legacy_mct_seconds": station_fallback,
        "mct_delta_seconds": 0,
    }


def summarize_platform_contract(pairs, *, tau_same_platform=90):
    """Audit G1 platform-pair coverage and artificial station-max inflation."""
    by_rule = Counter()
    by_role_rule = Counter()
    deltas = []
    for pair in pairs:
        res = classify_platform_pair_contract(pair, tau_same_platform=tau_same_platform)
        by_rule[res["rule"]] += 1
        by_role_rule[(pair["role"], res["rule"])] += 1
        deltas.append(res["mct_delta_seconds"])
    total = len(pairs)
    explicit_or_same = by_rule["explicit_gtfs_transfers_txt"] + by_rule["same_physical_platform"]
    return {
        "total_pairs": total,
        "tau_same_platform_seconds": int(tau_same_platform),
        "counts_by_rule": dict(by_rule),
        "counts_by_role_and_rule": {f"{r}:{k}": v for (r, k), v in sorted(by_role_rule.items())},
        "platform_resolved_fraction": explicit_or_same / total if total else 0.0,
        "mean_mct_change_vs_station_max_seconds": float(np.mean(deltas)) if deltas else 0.0,
    }
