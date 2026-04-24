"""Experiment 4: Ablation study on Swiss data.

Systematically varies key hyperparameters:
  - β (pessimism): [-1, 0, 0.5, 1, 1.5, 2, 3]
  - γ (cancel penalty): [0, 10, 30, 60, 120]
  - K (ensemble size, V2): [1, 3, 5, 10]
  - Observation window: [5, 10, 15, 25 min]

Output: results/ablation.json
"""

import sys, os, json, time, pickle, copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.gtfs_parser import load_routes
from src.gtfs_rt_parser import load_distributions
from src.pmf import PMF
from src.bandit_router import BanditRouter
from src.bandit_router_v2 import BanditRouterV2
from src.dro_router import DRORouter
from src.router import StaticRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import RegimeSchedule, set_regime_dist_fn, _regime_dist_cache


def run_ablation(g, s1, s2, normal_by_name, disrupted_by_name, N=20, seed=42):
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

    sched_normal = RegimeSchedule(shifts=[(0, 'normal')])
    sched_disrupt = RegimeSchedule(shifts=[(0, 'disrupted')])

    results = {}

    # Ablation 1: β sensitivity (DRO)
    print("\n--- β sensitivity ---")
    beta_results = {}
    for beta in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for scen_name, sched in [('normal', sched_normal), ('disrupted', sched_disrupt)]:
            _regime_dist_cache.clear()
            times = []
            for i in range(N):
                ri = DRORouter(copy.deepcopy(g), beta=beta, gamma=60.0)
                rng = np.random.default_rng(seed + i)
                res = simulate_bandit_journey(
                    ri.graph, ri, s1, s2, 490, sched, rng, 120)
                times.append(res.arrival_time - res.departure_time)
            arr = np.array(times)
            key = f"beta={beta}_{scen_name}"
            beta_results[key] = {
                "beta": beta, "scenario": scen_name,
                "mean": float(arr.mean()), "std": float(arr.std()),
                "p95": float(np.percentile(arr, 95)),
            }
            print(f"  β={beta:+.1f} {scen_name}: mean={arr.mean():.1f}")
    results["beta_sensitivity"] = beta_results

    # Ablation 2: γ (cancel penalty weight)
    print("\n--- γ sensitivity ---")
    gamma_results = {}
    for gamma in [0, 10, 30, 60, 120, 300]:
        for scen_name, sched in [('normal', sched_normal), ('disrupted', sched_disrupt)]:
            _regime_dist_cache.clear()
            times = []
            for i in range(N):
                ri = DRORouter(copy.deepcopy(g), beta=1.5, gamma=gamma)
                rng = np.random.default_rng(seed + i)
                res = simulate_bandit_journey(
                    ri.graph, ri, s1, s2, 490, sched, rng, 120)
                times.append(res.arrival_time - res.departure_time)
            arr = np.array(times)
            key = f"gamma={gamma}_{scen_name}"
            gamma_results[key] = {
                "gamma": gamma, "scenario": scen_name,
                "mean": float(arr.mean()), "std": float(arr.std()),
            }
            print(f"  γ={gamma:>3d} {scen_name}: mean={arr.mean():.1f}")
    results["gamma_sensitivity"] = gamma_results

    # Ablation 3: K (ensemble size, V2)
    print("\n--- K (ensemble size) ---")
    k_results = {}
    for K in [1, 3, 5, 10, 20]:
        for scen_name, sched in [('normal', sched_normal), ('disrupted', sched_disrupt)]:
            _regime_dist_cache.clear()
            times = []
            for i in range(N):
                ri = BanditRouterV2(copy.deepcopy(g), n_estimators=K)
                rng = np.random.default_rng(seed + i)
                res = simulate_bandit_journey(
                    ri.graph, ri, s1, s2, 490, sched, rng, 120)
                times.append(res.arrival_time - res.departure_time)
            arr = np.array(times)
            key = f"K={K}_{scen_name}"
            k_results[key] = {
                "K": K, "scenario": scen_name,
                "mean": float(arr.mean()), "std": float(arr.std()),
            }
            print(f"  K={K:>2d} {scen_name}: mean={arr.mean():.1f}")
    results["ensemble_size"] = k_results

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Ablation Study")
    print("=" * 60)

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

    for c in g.connections:
        d = normal_by_name.get(c.route)
        mean = d['mean'] if d else 0.5
        std = max(d['std'] if d else 1.5, 0.5)
        delays = np.arange(-5, 20)
        probs = np.exp(-0.5 * ((delays - mean) / std) ** 2)
        probs /= probs.sum()
        c.dep_distribution = PMF.from_delays(c.dep_time, probs, -5)
        c.arr_distribution = PMF.from_delays(c.arr_time, probs, -5)

    s1, s2 = 67001060, 201257157
    print(f"OD: {g.stops[s1].name} -> {g.stops[s2].name}")

    results = run_ablation(g, s1, s2, normal_by_name, disrupted_by_name, N=20)

    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    with open('experiments/swiss_full/results/ablation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiments/swiss_full/results/ablation.json")
