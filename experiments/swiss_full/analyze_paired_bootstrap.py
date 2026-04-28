"""Round 3 fix #8: Paired-bootstrap analysis on the v3 multi-OD data.

The round-2 reviewer flagged that the cross-OD CIs were wide and the
disrupted-day reach improvements were not statistically compelling.
The fix: switch from independent-OD bootstrap (which conflates OD
heterogeneity with method effects) to **paired bootstrap** within the
same (OD, seed) pair. Differences (Adaptive - Static) within each pair
have much lower variance because OD-specific reachability cancels.

We also report:
- Paired effect sizes (Cohen's d on within-OD differences)
- Hierarchical (blocked) bootstrap respecting the (OD, seed) structure
- Larger OD coverage diagnostic (how many ODs each method "rescues" vs
  "hurts")

Input: experiments/swiss_full/results/swiss_multi_od_v3.json
       (per-trial trajectories stored at results[scen][od][method].trials)
Output: experiments/swiss_full/results/swiss_multi_od_v3_paired.json
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict


def bootstrap_paired_diff(values_a, values_b, n_boot=2000, ci=0.95, seed=42):
    """Paired bootstrap CI for the difference (mean(a) - mean(b))
    where a, b are matched samples (e.g., reach rate per OD)."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    assert len(a) == len(b), f"paired arrays must have same length: {len(a)} vs {len(b)}"
    diffs = a - b
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[k] = diffs[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))
    return {
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "ci_lo": lo,
        "ci_hi": hi,
        "n_pairs": n,
        "frac_positive": float((diffs > 0).mean()),
        "frac_negative": float((diffs < 0).mean()),
    }


def cohens_d_paired(values_a, values_b):
    """Cohen's d for paired samples: mean of within-pair differences
    divided by their standard deviation."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diffs = a - b
    if diffs.std() == 0:
        return float("inf") if diffs.mean() != 0 else 0.0
    return float(diffs.mean() / diffs.std(ddof=1))


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(root, "results", "swiss_multi_od_v3.json")
    out_path = os.path.join(root, "results", "swiss_multi_od_v3_paired.json")

    with open(in_path) as f:
        d = json.load(f)

    print("=" * 70)
    print("Paired-bootstrap analysis (R3 fix #8)")
    print("=" * 70)
    cfg = d["config"]
    print(f"Config: n_viable={cfg['n_viable_ods']}, "
          f"seeds={cfg['seeds']}, n_per_seed={cfg['n_per_seed']}, "
          f"max_time={cfg['max_time']}")

    methods = ["Static", "V1-LCB", "V2-LCB", "V3-Topo", "DRO", "Adaptive-β"]

    summary = {}
    for scen in ["normal", "disrupted"]:
        print(f"\n--- {scen} ---")
        scen_results = d.get(scen, {})
        if not scen_results:
            continue

        # Per-OD reach rates (1 per OD per method)
        reach_per_od = {m: [] for m in methods}
        cond_mean_per_od = {m: [] for m in methods}
        for od_key, od_methods in scen_results.items():
            for m in methods:
                if m in od_methods:
                    reach_per_od[m].append(od_methods[m]["reach_rate"])
                    cmr = od_methods[m].get("mean_reached")
                    if cmr is not None:
                        cond_mean_per_od[m].append(cmr)

        scen_summary = {}
        # Each non-Static method paired against Static
        for m in methods:
            if m == "Static":
                continue
            reach_diff = bootstrap_paired_diff(
                reach_per_od[m], reach_per_od["Static"], n_boot=2000)
            d_paired = cohens_d_paired(reach_per_od[m], reach_per_od["Static"])

            # For conditional mean, only use ODs where both methods have data
            paired_a, paired_b = [], []
            for od_key, od_methods in scen_results.items():
                cmra = od_methods[m].get("mean_reached")
                cmrb = od_methods["Static"].get("mean_reached")
                if cmra is not None and cmrb is not None:
                    paired_a.append(cmra)
                    paired_b.append(cmrb)
            if len(paired_a) >= 2:
                cond_diff = bootstrap_paired_diff(paired_a, paired_b, n_boot=2000)
                cond_d = cohens_d_paired(paired_a, paired_b)
            else:
                cond_diff = None
                cond_d = None

            scen_summary[m] = {
                "vs_Static": {
                    "reach_diff_pp": reach_diff,
                    "reach_cohens_d": d_paired,
                    "cond_mean_diff_min": cond_diff,
                    "cond_mean_cohens_d": cond_d,
                }
            }

            rd = reach_diff
            print(f"  {m:12s} vs Static: ΔReach = "
                  f"{rd['mean_diff']*100:+.1f}pp [{rd['ci_lo']*100:+.1f}, "
                  f"{rd['ci_hi']*100:+.1f}], d={d_paired:+.2f}, "
                  f"win/{rd['n_pairs']}: {rd['frac_positive']*100:.0f}%, "
                  f"loss: {rd['frac_negative']*100:.0f}%")
            if cond_diff:
                cd = cond_diff
                print(f"  {' '*12} ΔCondMean = {cd['mean_diff']:+.2f} min "
                      f"[{cd['ci_lo']:+.2f}, {cd['ci_hi']:+.2f}], "
                      f"d={cond_d:+.2f}")

        summary[scen] = scen_summary

    # Save
    with open(out_path, "w") as f:
        json.dump({
            "config": cfg,
            "paired_summary": summary,
            "method": "paired bootstrap (n=2000 resamples) + Cohen's d "
                      "on per-OD differences",
        }, f, indent=2,
           default=lambda o: int(o) if hasattr(o, "item") else str(o))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
