"""Multi-OD Swiss experiment v3 (round-1 review fix #8).

Improvements over v2:
1. Stores per-trial travel times so bootstrap CIs can be computed
   post-hoc.
2. Runs each cell with multiple base seeds (default 5) → 5 × N
   journeys per cell, treated as a single concatenated sample for
   non-parametric bootstrap.
3. Reports 95% bootstrap CIs for both reach rate and conditional
   mean travel time.

Output: results/swiss_multi_od_v3.json
"""

import sys, os, json, time, pickle, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from collections import defaultdict

from src.pmf import PMF
from src.bandit_router import BanditRouter
from src.bandit_router_v2 import BanditRouterV2
from src.bandit_router_v3 import BanditRouterV3
from src.dro_router import DRORouter
from src.adaptive_bandit_router import AdaptiveBetaBanditRouter
from src.router import StaticRouter
from src.simulate_bandit import simulate_bandit_journey
from src.gtfs_parser import load_routes
from src.simulator import (RegimeSchedule, set_regime_dist_fn,
                           _regime_dist_cache)

sys.path.insert(0, os.path.dirname(__file__))
from run_multi_od import find_od_pairs  # type: ignore


def build_regime_fn(normal_by_name, disrupted_by_name):
    def real_day_regime(name):
        src = normal_by_name if name == 'normal' else disrupted_by_name
        result = {}
        for rname, d in src.items():
            delays = np.arange(-5, 65)
            mean, std = d['mean'], max(d['std'], 0.5)
            probs = np.exp(-0.5 * ((delays - mean) / std) ** 2)
            probs /= probs.sum()
            cancel = d.get('cancel_rate', 0)
            if cancel > 0:
                probs *= (1 - cancel)
            info = {'delay_probs': probs, 'delay_offset': -5}
            if cancel > 0.01:
                info['cancel_prob'] = cancel
            result[rname] = info
        return result
    return real_day_regime


def bootstrap_ci(values, n_boot=1000, ci=0.95, statistic=np.mean, seed=42):
    """Non-parametric bootstrap CI on the given statistic."""
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    n = len(arr)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = statistic(arr[idx])
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1 - alpha))
    return [lo, hi]


def run_od_multi_seed(g, s1, s2, sched, methods, N_per_seed, max_time,
                      seeds):
    """Run each method for `N_per_seed` journeys at each seed in `seeds`,
    return per-trial data so we can bootstrap later."""
    out = {}
    for name, make in methods.items():
        all_tts = []
        for seed in seeds:
            _regime_dist_cache.clear()
            for i in range(N_per_seed):
                ri = make(copy.deepcopy(g))
                if isinstance(ri, AdaptiveBetaBanditRouter):
                    ri.route(s1, s2, 490)
                jrng = np.random.default_rng(seed + i)
                t_dep = 490 + jrng.integers(0, 10)
                res = simulate_bandit_journey(
                    ri.graph, ri, s1, s2, int(t_dep), sched, jrng, max_time)
                tt = res.arrival_time - res.departure_time
                all_tts.append(float(tt))
        arr = np.array(all_tts)
        reached = arr < max_time
        # Conditional mean (only over completions)
        mean_reached = float(arr[reached].mean()) if reached.any() else None
        # Bootstrap CIs
        reach_ci = bootstrap_ci(reached.astype(float), n_boot=1000)
        mean_all_ci = bootstrap_ci(arr.tolist(), n_boot=1000)
        if reached.any():
            mean_reached_ci = bootstrap_ci(arr[reached].tolist(), n_boot=1000)
        else:
            mean_reached_ci = None
        out[name] = {
            "n_total": len(arr),
            "reach_rate": float(reached.mean()),
            "reach_rate_ci95": reach_ci,
            "mean_all": float(arr.mean()),
            "mean_all_ci95": mean_all_ci,
            "mean_reached": mean_reached,
            "mean_reached_ci95": mean_reached_ci,
            "p95": float(np.percentile(arr, 95)),
            "trials": all_tts,  # store for downstream re-bootstrap
        }
    return out


if __name__ == "__main__":
    print("=" * 70)
    print("Swiss Multi-OD v3 (per-trial data + bootstrap CIs)")
    print("=" * 70)

    with open('data/zurich_wide.pkl', 'rb') as f:
        g = pickle.load(f)
    with open('data/day_distributions.pkl', 'rb') as f:
        day_dists = pickle.load(f)
    route_names_map = load_routes('data/swiss_gtfs')

    normal_by_name, disrupted_by_name = {}, {}
    for rid, d in day_dists['normal_day'].items():
        name = route_names_map.get(rid)
        if name:
            normal_by_name[name] = d
    for rid, d in day_dists['disrupted_day'].items():
        name = route_names_map.get(rid)
        if name and (name not in disrupted_by_name or
                     d['mean'] > disrupted_by_name[name]['mean']):
            disrupted_by_name[name] = d

    for c in g.connections:
        d = normal_by_name.get(c.route)
        mean = d['mean'] if d else 0.5
        std = max(d['std'] if d else 1.5, 0.5)
        delays = np.arange(-5, 20)
        probs = np.exp(-0.5 * ((delays - mean) / std) ** 2)
        probs /= probs.sum()
        c.dep_distribution = PMF.from_delays(c.dep_time, probs, -5)
        c.arr_distribution = PMF.from_delays(c.arr_time, probs, -5)

    set_regime_dist_fn(build_regime_fn(normal_by_name, disrupted_by_name))

    print("\nFinding OD pairs...")
    od_pairs = find_od_pairs(g, disrupted_by_name, n_pairs=20,
                             max_search_time=120)
    print(f"Found {len(od_pairs)} candidate OD pairs")

    methods = {
        "Static":     lambda g: StaticRouter(g),
        "V1-LCB":     lambda g: BanditRouter(g),
        "V2-LCB":     lambda g: BanditRouterV2(g, n_estimators=5,
                                               beta_base=1.0, beta_ood=1.0,
                                               seed=42),
        "V3-Topo":    lambda g: BanditRouterV3(g),
        "DRO":        lambda g: DRORouter(g, beta=1.5, gamma=60.0),
        "Adaptive-β": lambda g: AdaptiveBetaBanditRouter(g),
    }

    SEEDS = [42, 137, 271, 314, 577]  # 5 base seeds
    N_PER_SEED = 30
    MAX_TIME = 120

    # PASS 1: viability screening on Oct 2 normal day
    print(f"\n=== PASS 1: normal-day viability screening "
          f"({len(SEEDS)} seeds × {N_PER_SEED} = "
          f"{len(SEEDS)*N_PER_SEED} journeys per cell) ===")
    sched_normal = RegimeSchedule(shifts=[(0, 'normal')])
    sched_disrupted = RegimeSchedule(shifts=[(0, 'disrupted')])

    viable_ods = []
    normal_results = {}
    for i, od in enumerate(od_pairs):
        s1, s2 = od['s1'], od['s2']
        print(f"\n[{i+1}/{len(od_pairs)}] {od['s1_name']} → {od['s2_name']}")
        out = run_od_multi_seed(g, s1, s2, sched_normal, methods,
                                N_PER_SEED, MAX_TIME, SEEDS)
        static_reach = out["Static"]["reach_rate"]
        print(f"  Static normal reach={static_reach:.0%}")
        if static_reach >= 0.8:
            viable_ods.append(od)
            normal_results[f"{od['s1_name']}→{od['s2_name']}"] = out
        else:
            print(f"  FILTERED (normal reach < 80%)")

    print(f"\n=== {len(viable_ods)} of {len(od_pairs)} ODs viable on normal day ===")

    # PASS 2: disrupted-day comparison
    print("\n=== PASS 2: disrupted-day comparison on viable ODs ===")
    disrupted_results = {}
    for i, od in enumerate(viable_ods):
        s1, s2 = od['s1'], od['s2']
        print(f"\n[{i+1}/{len(viable_ods)}] {od['s1_name']} → {od['s2_name']}")
        out = run_od_multi_seed(g, s1, s2, sched_disrupted, methods,
                                N_PER_SEED, MAX_TIME, SEEDS)
        disrupted_results[f"{od['s1_name']}→{od['s2_name']}"] = out
        for m, s in out.items():
            mr = s['mean_reached'] if s['mean_reached'] is not None else 0
            mr_ci = s.get('mean_reached_ci95')
            mr_ci_str = f" CI=[{mr_ci[0]:.1f},{mr_ci[1]:.1f}]" if mr_ci else ""
            print(f"  {m:12s} reach={s['reach_rate']:.0%} "
                  f"CI=[{s['reach_rate_ci95'][0]:.0%},{s['reach_rate_ci95'][1]:.0%}] "
                  f"mean_reached={mr:.1f}{mr_ci_str}")

    # AGGREGATE with bootstrap CI on across-OD averages
    print("\n" + "=" * 70)
    print("AGGREGATE (across viable ODs, bootstrap CI on cross-OD averages)")
    print("=" * 70)

    summary = {}
    for scen_name, scen_results in [("normal", normal_results),
                                     ("disrupted", disrupted_results)]:
        summary[scen_name] = {}
        if not scen_results:
            continue
        print(f"\n--- {scen_name} ---")
        for m in methods:
            reach_rates = [r[m]["reach_rate"] for r in scen_results.values()]
            mean_reacheds = [r[m]["mean_reached"] for r in scen_results.values()
                             if r[m]["mean_reached"] is not None]
            agg_reach_ci = bootstrap_ci(reach_rates, n_boot=1000)
            agg_mr_ci = bootstrap_ci(mean_reacheds, n_boot=1000) \
                if mean_reacheds else None
            summary[scen_name][m] = {
                "avg_reach_rate": float(np.mean(reach_rates)),
                "avg_reach_rate_ci95": agg_reach_ci,
                "avg_mean_reached": float(np.mean(mean_reacheds))
                                    if mean_reacheds else None,
                "avg_mean_reached_ci95": agg_mr_ci,
                "n_ods": len(reach_rates),
            }
            mr_str = (f" (CI [{agg_mr_ci[0]:.1f},{agg_mr_ci[1]:.1f}])"
                      if agg_mr_ci else "")
            mr_val = (f"mean_reached={np.mean(mean_reacheds):.1f}"
                      if mean_reacheds else "(all timeouts)")
            print(f"  {m:12s} reach={np.mean(reach_rates):.0%} "
                  f"(CI [{agg_reach_ci[0]:.0%},{agg_reach_ci[1]:.0%}]) "
                  f"{mr_val}{mr_str}")

    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    out_path = 'experiments/swiss_full/results/swiss_multi_od_v3.json'
    with open(out_path, 'w') as f:
        json.dump({
            "config": {"n_od_candidates": len(od_pairs),
                       "n_viable_ods": len(viable_ods),
                       "viability_threshold": 0.8,
                       "seeds": SEEDS,
                       "n_per_seed": N_PER_SEED,
                       "n_total_per_cell": len(SEEDS) * N_PER_SEED,
                       "max_time": MAX_TIME,
                       "bootstrap_iters": 1000,
                       "ci_level": 0.95},
            "viable_ods": [{"s1_name": od["s1_name"], "s2_name": od["s2_name"],
                            "disrupted_routes": od["disrupted_routes"],
                            "safe_routes": od["safe_routes"]}
                           for od in viable_ods],
            "normal": normal_results,
            "disrupted": disrupted_results,
            "summary": summary,
        }, f, indent=2,
           default=lambda o: int(o) if hasattr(o, "item") else str(o))
    print(f"\nSaved to {out_path}")
