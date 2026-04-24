"""Multi-OD Swiss experiment v2: filters OD pairs by reachability + adds
reach-rate metric (not just travel time).

Fixes two issues from the original run_multi_od.py:
1. Many ODs were unreachable on disrupted day (30/30 timeouts), dragging
   averages to the 120-min cap and washing out method differences.
2. No reach-rate metric — but "ability to reach at all" is what matters
   under catastrophic disruption.

New output per scenario × method:
  - mean_travel (conditional on completion, i.e. excluding timeouts)
  - reach_rate (fraction of journeys that reach before t_max)
  - p95_travel (capped)
  - ODs_reachable (num ODs with reach_rate > 50% for at least one method)

Output: results/swiss_multi_od_v2.json
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

# re-use OD-finding logic from the v1 script
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


def run_od(g, s1, s2, scen_name, sched, methods, N, max_time, seed):
    _regime_dist_cache.clear()
    out = {}
    for name, make in methods.items():
        tts, timeouts = [], 0
        for i in range(N):
            ri = make(copy.deepcopy(g))
            if isinstance(ri, AdaptiveBetaBanditRouter):
                ri.route(s1, s2, 490)
            jrng = np.random.default_rng(seed + i)
            t_dep = 490 + jrng.integers(0, 10)
            res = simulate_bandit_journey(
                ri.graph, ri, s1, s2, int(t_dep), sched, jrng, max_time)
            tt = res.arrival_time - res.departure_time
            tts.append(tt)
            if tt >= max_time:
                timeouts += 1
        arr = np.array(tts)
        reached = arr < max_time
        out[name] = {
            "mean_all": float(arr.mean()),
            "mean_reached": float(arr[reached].mean()) if reached.any() else None,
            "p95": float(np.percentile(arr, 95)),
            "reach_rate": float(reached.mean()),
            "timeouts": int(timeouts),
            "n": N,
        }
    return out


if __name__ == "__main__":
    print("=" * 70)
    print("Swiss Multi-OD v2 (OD filtering + reach-rate metric)")
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

    print("\nFinding OD pairs with disrupted/safe route mix...")
    od_pairs = find_od_pairs(g, disrupted_by_name, n_pairs=20, max_search_time=120)
    print(f"Found {len(od_pairs)} candidate OD pairs")

    methods = {
        "Static":     lambda g: StaticRouter(g),
        "V1-LCB":     lambda g: BanditRouter(g),
        "V2-LCB":     lambda g: BanditRouterV2(g, n_estimators=5,
                                               beta_base=1.0, beta_ood=1.0, seed=42),
        "V3-Topo":    lambda g: BanditRouterV3(g),
        "DRO":        lambda g: DRORouter(g, beta=1.5, gamma=60.0),
        "Adaptive-β": lambda g: AdaptiveBetaBanditRouter(g),
    }

    scenarios = {
        "normal":    RegimeSchedule(shifts=[(0, 'normal')]),
        "disrupted": RegimeSchedule(shifts=[(0, 'disrupted')]),
    }

    # Pass 1: run normal on all ODs to identify "always-reachable" pairs
    print("\n=== PASS 1: normal-day viability screening ===")
    viable_ods = []
    normal_results = {}
    for i, od in enumerate(od_pairs):
        s1, s2 = od['s1'], od['s2']
        print(f"\n[{i+1}/{len(od_pairs)}] {od['s1_name']} → {od['s2_name']}")
        out = run_od(g, s1, s2, 'normal', scenarios['normal'], methods,
                     N=20, max_time=120, seed=42)
        # require Static to reach in at least 80% of normal-day journeys
        static_reach = out["Static"]["reach_rate"]
        print(f"  Static normal reach_rate={static_reach:.0%} "
              f"(mean_all={out['Static']['mean_all']:.1f})")
        if static_reach >= 0.8:
            viable_ods.append(od)
            normal_results[f"{od['s1_name']}→{od['s2_name']}"] = out
        else:
            print(f"  FILTERED (normal reach_rate < 80%)")

    print(f"\n=== {len(viable_ods)} of {len(od_pairs)} ODs viable on normal day ===")

    # Pass 2: run disrupted on viable ODs
    print("\n=== PASS 2: disrupted-day comparison on viable ODs ===")
    disrupted_results = {}
    for i, od in enumerate(viable_ods):
        s1, s2 = od['s1'], od['s2']
        print(f"\n[{i+1}/{len(viable_ods)}] {od['s1_name']} → {od['s2_name']}")
        out = run_od(g, s1, s2, 'disrupted', scenarios['disrupted'], methods,
                     N=30, max_time=120, seed=42)
        disrupted_results[f"{od['s1_name']}→{od['s2_name']}"] = out
        for m, s in out.items():
            print(f"  {m:12s} reach={s['reach_rate']:.0%} "
                  f"mean_all={s['mean_all']:.1f} "
                  f"mean_reached={s['mean_reached'] if s['mean_reached'] else 0:.1f}")

    # Aggregate across viable ODs
    print("\n" + "=" * 70)
    print("AGGREGATE (across viable ODs)")
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
            summary[scen_name][m] = {
                "avg_reach_rate": float(np.mean(reach_rates)),
                "avg_mean_reached": float(np.mean(mean_reacheds)) if mean_reacheds else None,
                "n_ods": len(reach_rates),
            }
            if mean_reacheds:
                print(f"  {m:12s} reach={np.mean(reach_rates):.0%} "
                      f"mean_reached={np.mean(mean_reacheds):.1f} (n={len(reach_rates)} ODs)")
            else:
                print(f"  {m:12s} reach={np.mean(reach_rates):.0%} "
                      f"(all timeouts)")

    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    out_path = 'experiments/swiss_full/results/swiss_multi_od_v2.json'
    with open(out_path, 'w') as f:
        json.dump({
            "config": {"n_od_candidates": len(od_pairs),
                       "n_viable_ods": len(viable_ods),
                       "viability_threshold": 0.8, "N_per_cell": 30,
                       "max_time": 120},
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
