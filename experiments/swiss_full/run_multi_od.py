"""Experiment 1: Swiss multi-OD comparison (10+ pairs, N=30, bootstrap CI).

This is the most critical missing experiment. Runs Static/LCB/V2/V3/DRO
on multiple OD pairs from real Swiss GTFS data to establish statistical
significance of the DRO advantage.

Output: results/swiss_multi_od.json
"""

import sys, os, json, time, pickle, copy
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.durner.topocsa import topocsa
from src.gtfs_parser import load_routes
from src.gtfs_rt_parser import load_distributions
from src.pmf import PMF
from src.bandit_router import BanditRouter
from src.bandit_router_v2 import BanditRouterV2
from src.bandit_router_v3 import BanditRouterV3
from src.dro_router import DRORouter
from src.router import StaticRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import RegimeSchedule, set_regime_dist_fn, _regime_dist_cache
from src.transit_graph import TransitGraph


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95, rng=None):
    """Compute bootstrap confidence interval."""
    if rng is None:
        rng = np.random.default_rng(0)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def find_od_pairs(g, disrupted_by_name, n_pairs=15, max_search_time=120):
    """Find OD pairs where hyperpath includes both disrupted and safe routes."""
    transfers = g.get_transfer_stops()
    disrupted_routes = set()
    for rname, d in disrupted_by_name.items():
        if d.get('cancel_rate', 0) > 0.05 or d['mean'] > 5:
            disrupted_routes.add(rname)

    # Find stops with mixed routes
    stop_routes = defaultdict(set)
    for c in g.connections:
        stop_routes[c.dep_stop].add(c.route)

    mixed_stops = [sid for sid, routes in stop_routes.items()
                   if (routes & disrupted_routes) and (routes - disrupted_routes)]

    # Find connected pairs
    pairs = []
    t0 = time.time()
    for s1 in mixed_stops:
        if time.time() - t0 > max_search_time:
            break
        for s2 in transfers:
            if s1 == s2:
                continue
            if time.time() - t0 > max_search_time:
                break
            try:
                r = topocsa(g, s1, s2, 490)
                if r.mean_arrival < float('inf') and r.mean_arrival < 540:
                    labels = r.stop_labels.get(s1, [])
                    routes = set(g.connections[l.connection_id].route for l in labels)
                    d_routes = routes & disrupted_routes
                    s_routes = routes - disrupted_routes
                    if d_routes and s_routes:
                        pairs.append({
                            "s1": s1, "s2": s2,
                            "s1_name": g.stops[s1].name,
                            "s2_name": g.stops[s2].name,
                            "mean_arrival": r.mean_arrival,
                            "disrupted_routes": list(d_routes),
                            "safe_routes": list(s_routes),
                        })
                        if len(pairs) >= n_pairs:
                            return pairs
            except Exception:
                continue

    # If not enough mixed pairs, add any connected pairs
    if len(pairs) < n_pairs:
        for s1 in transfers[:30]:
            if time.time() - t0 > max_search_time:
                break
            for s2 in transfers:
                if s1 == s2:
                    continue
                if (s1, s2) in [(p["s1"], p["s2"]) for p in pairs]:
                    continue
                try:
                    r = topocsa(g, s1, s2, 490)
                    if r.mean_arrival < float('inf') and r.mean_arrival < 540:
                        pairs.append({
                            "s1": s1, "s2": s2,
                            "s1_name": g.stops[s1].name,
                            "s2_name": g.stops[s2].name,
                            "mean_arrival": r.mean_arrival,
                            "disrupted_routes": [],
                            "safe_routes": [],
                        })
                        if len(pairs) >= n_pairs:
                            return pairs
                except Exception:
                    continue

    return pairs


def run_experiment(
    g, od_pairs, normal_by_name, disrupted_by_name,
    N=30, seed=42, max_time=120,
):
    """Run the full multi-OD experiment."""

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

    set_regime_dist_fn(real_day_regime)

    methods = {
        'Static': StaticRouter,
        'LCB': BanditRouter,
        'V2': BanditRouterV2,
        'V3': BanditRouterV3,
        'DRO': DRORouter,
    }

    scenarios = {
        'normal': RegimeSchedule(shifts=[(0, 'normal')]),
        'disrupted': RegimeSchedule(shifts=[(0, 'disrupted')]),
        'shift': RegimeSchedule(shifts=[(0, 'normal'), (495, 'disrupted')]),
    }

    results = []
    rng_ci = np.random.default_rng(0)

    for pair_idx, pair in enumerate(od_pairs):
        s1, s2 = pair["s1"], pair["s2"]
        print(f"\n[{pair_idx+1}/{len(od_pairs)}] {pair['s1_name']} -> {pair['s2_name']}")

        pair_result = {
            "od": pair,
            "scenarios": {},
        }

        for scen_name, sched in scenarios.items():
            scen_result = {}
            for mname, Cls in methods.items():
                _regime_dist_cache.clear()
                times = []
                for i in range(N):
                    ri = Cls(copy.deepcopy(g))
                    rng = np.random.default_rng(seed + i + pair_idx * 1000)
                    t_dep = 490 + rng.integers(0, 10)
                    res = simulate_bandit_journey(
                        ri.graph, ri, s1, s2, t_dep, sched, rng, max_time)
                    times.append(res.arrival_time - res.departure_time)

                arr = np.array(times)
                ci_lo, ci_hi = bootstrap_ci(arr, rng=rng_ci)
                scen_result[mname] = {
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": float(arr.std()),
                    "p95": float(np.percentile(arr, 95)),
                    "ci_95": [ci_lo, ci_hi],
                    "timeouts": int((arr >= max_time).sum()),
                }

            # Compute improvement over static
            static_mean = scen_result['Static']['mean']
            for mname in methods:
                m = scen_result[mname]['mean']
                scen_result[mname]['improvement_pct'] = \
                    float((static_mean - m) / static_mean * 100) if static_mean > 0 else 0

            pair_result["scenarios"][scen_name] = scen_result
            print(f"  {scen_name}: " + " | ".join(
                f"{m}={r['mean']:.1f}({r['improvement_pct']:+.1f}%)"
                for m, r in scen_result.items()))

        results.append(pair_result)

    return results


def aggregate_results(results):
    """Aggregate across OD pairs."""
    methods = ['Static', 'LCB', 'V2', 'V3', 'DRO']
    scenarios = ['normal', 'disrupted', 'shift']

    summary = {}
    for scen in scenarios:
        summary[scen] = {}
        for method in methods:
            improvements = []
            means = []
            for pair_result in results:
                if scen in pair_result['scenarios']:
                    r = pair_result['scenarios'][scen][method]
                    improvements.append(r['improvement_pct'])
                    means.append(r['mean'])

            if improvements:
                summary[scen][method] = {
                    "mean_improvement": float(np.mean(improvements)),
                    "std_improvement": float(np.std(improvements)),
                    "mean_travel": float(np.mean(means)),
                    "n_pairs": len(improvements),
                }

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("Swiss Multi-OD Experiment")
    print("=" * 60)

    # Load data
    with open('data/zurich_wide.pkl', 'rb') as f:
        g = pickle.load(f)
    with open('data/day_distributions.pkl', 'rb') as f:
        day_dists = pickle.load(f)
    route_names_map = load_routes('data/swiss_gtfs')

    normal_by_name = {}
    disrupted_by_name = {}
    for rid, d in day_dists['normal_day'].items():
        name = route_names_map.get(rid)
        if name:
            normal_by_name[name] = d
    for rid, d in day_dists['disrupted_day'].items():
        name = route_names_map.get(rid)
        if name and (name not in disrupted_by_name or
                     d['mean'] > disrupted_by_name[name]['mean']):
            disrupted_by_name[name] = d

    # Apply normal delays to graph
    for c in g.connections:
        d = normal_by_name.get(c.route)
        mean = d['mean'] if d else 0.5
        std = max(d['std'] if d else 1.5, 0.5)
        delays = np.arange(-5, 20)
        probs = np.exp(-0.5 * ((delays - mean) / std) ** 2)
        probs /= probs.sum()
        c.dep_distribution = PMF.from_delays(c.dep_time, probs, -5)
        c.arr_distribution = PMF.from_delays(c.arr_time, probs, -5)

    print(f"\nGraph: {g.summary()}")
    print(f"Normal routes: {len(normal_by_name)}")
    print(f"Disrupted routes: {len(disrupted_by_name)}")

    # Find OD pairs
    print("\nFinding OD pairs...")
    od_pairs = find_od_pairs(g, disrupted_by_name, n_pairs=10, max_search_time=180)
    print(f"Found {len(od_pairs)} OD pairs")

    # Run experiment
    results = run_experiment(
        g, od_pairs, normal_by_name, disrupted_by_name,
        N=30, seed=42, max_time=120,
    )

    # Aggregate
    summary = aggregate_results(results)
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    for scen, methods in summary.items():
        print(f"\n{scen}:")
        for method, stats in methods.items():
            print(f"  {method:8s}: mean_travel={stats['mean_travel']:.1f} "
                  f"improvement={stats['mean_improvement']:+.1f}% "
                  f"±{stats['std_improvement']:.1f}% "
                  f"(n={stats['n_pairs']})")

    # Save
    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    output = {
        "od_pairs": od_pairs,
        "results": results,
        "summary": summary,
        "config": {"N": 30, "seed": 42, "max_time": 120},
    }
    with open('experiments/swiss_full/results/swiss_multi_od.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("\nSaved to experiments/swiss_full/results/swiss_multi_od.json")
