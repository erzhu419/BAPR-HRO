"""Reproduce historical Swiss normal-day result: DRO -19.2% on
Sternen Oerlikon → Sihlpost/HB (commit 2914c78).

This OD was selected because its hyperpath at the source stop contains
BOTH disrupted routes (route 7, affected on Oct 29) AND safe routes
(route 14). So even under Oct 2 normal conditions, delay heterogeneity
across routes gives LCB/DRO room to differentiate.

Output: results/swiss_normal_focused.json
"""

import sys, os, json, time, pickle, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

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


def run_one_scenario(g, s1, s2, sched, methods, N=30, seed=42, max_time=120):
    _regime_dist_cache.clear()
    out = {}
    for name, make_router in methods.items():
        tts, timeouts = [], 0
        for i in range(N):
            ri = make_router(copy.deepcopy(g))
            if isinstance(ri, AdaptiveBetaBanditRouter):
                ri.route(s1, s2, 490)
            jrng = np.random.default_rng(seed + i)
            res = simulate_bandit_journey(
                ri.graph, ri, s1, s2, 490, sched, jrng, max_time)
            tt = res.arrival_time - res.departure_time
            tts.append(tt)
            if tt >= max_time:
                timeouts += 1
        arr = np.array(tts)
        out[name] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "std": float(arr.std()),
            "timeouts": int(timeouts),
            "n": N,
        }
        print(f"  {name:15s} mean={arr.mean():.1f} p95={np.percentile(arr,95):.1f} "
              f"timeouts={timeouts}/{N}")

    base = out["Static"]["mean"]
    for n in out:
        out[n]["improvement_pct"] = 0.0 if n == "Static" else \
            (base - out[n]["mean"]) / base * 100
    return out


if __name__ == "__main__":
    print("=" * 70)
    print("Swiss Real Data: Sternen Oerlikon → Sihlpost/HB")
    print("Target: reproduce DRO -19.2% on normal day (commit 2914c78)")
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

    # Calibrate connection-level distributions from normal-day stats
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

    s1, s2 = 67001060, 201257157
    print(f"OD: {g.stops[s1].name} → {g.stops[s2].name}")

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

    results = {}
    for scen_name, sched in scenarios.items():
        print(f"\n--- {scen_name} ---")
        t0 = time.time()
        results[scen_name] = run_one_scenario(
            g, s1, s2, sched, methods, N=30, seed=42, max_time=120)
        print(f"  [{time.time()-t0:.1f}s]")

    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    out_path = 'experiments/swiss_full/results/swiss_normal_focused.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, "item") else str(o))
    print(f"\nSaved to {out_path}")

    # summary
    print("\n=== SUMMARY ===")
    for scen, methods_res in results.items():
        print(f"\n{scen}:")
        base = methods_res["Static"]["mean"]
        for m, s in methods_res.items():
            marker = "" if m == "Static" else f"  ({s['improvement_pct']:+.1f}%)"
            print(f"  {m:15s} mean={s['mean']:6.1f}{marker}")
