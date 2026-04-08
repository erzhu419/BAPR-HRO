"""Experiment 3: Multi-journey Adapt-β convergence.

Shows that EXP3 meta-bandit learns the optimal β across multiple
journeys. This is the key experiment for the meta-learning claim.

Protocol:
- Run 50 journeys sequentially on the same OD pair
- After each journey, update EXP3 weights
- Track: β value, travel time, β distribution entropy

Output: results/adapt_beta_convergence.json
"""

import sys, os, json, time, pickle, copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.gtfs_parser import load_routes
from src.gtfs_rt_parser import load_distributions
from src.pmf import PMF
from src.adaptive_bandit_router import AdaptiveBetaBanditRouter
from src.bandit_router import BanditRouter
from src.router import StaticRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import RegimeSchedule, set_regime_dist_fn, _regime_dist_cache


def run_convergence(g, s1, s2, normal_by_name, disrupted_by_name,
                    n_journeys=50, seed=42):
    """Run sequential journeys and track β convergence."""

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

    scenarios = {
        'normal': RegimeSchedule(shifts=[(0, 'normal')]),
        'disrupted': RegimeSchedule(shifts=[(0, 'disrupted')]),
        'alternating': None,  # alternates normal/disrupted every 10 journeys
    }

    results = {}
    for scen_name, sched in scenarios.items():
        print(f"\n--- {scen_name} ---")

        # Adaptive-β: persistent across journeys
        adapt_router = AdaptiveBetaBanditRouter(copy.deepcopy(g))
        adapt_router.route(s1, s2, 490)

        # Static baseline (no learning)
        static_times = []
        adapt_times = []
        beta_history = []
        beta_probs_history = []

        for j in range(n_journeys):
            # Determine regime
            if scen_name == 'alternating':
                if (j // 10) % 2 == 0:
                    sched_j = RegimeSchedule(shifts=[(0, 'normal')])
                else:
                    sched_j = RegimeSchedule(shifts=[(0, 'disrupted')])
            else:
                sched_j = sched

            _regime_dist_cache.clear()
            rng = np.random.default_rng(seed + j)

            # Adaptive-β journey
            adapt_router.begin_journey()
            current_beta = adapt_router.current_beta
            res_adapt = simulate_bandit_journey(
                adapt_router.graph, adapt_router, s1, s2,
                490 + rng.integers(0, 10), sched_j, rng, 50)
            travel_adapt = res_adapt.arrival_time - res_adapt.departure_time
            adapt_router.end_journey(travel_adapt)

            # Static journey (same seed)
            rng2 = np.random.default_rng(seed + j)
            static_r = StaticRouter(copy.deepcopy(g))
            res_static = simulate_bandit_journey(
                static_r.graph, static_r, s1, s2,
                490 + rng2.integers(0, 10), sched_j, rng2, 50)
            travel_static = res_static.arrival_time - res_static.departure_time

            adapt_times.append(travel_adapt)
            static_times.append(travel_static)
            beta_history.append(current_beta)
            beta_probs_history.append(adapt_router.beta_probs.tolist())

            if (j + 1) % 10 == 0:
                recent_adapt = np.mean(adapt_times[-10:])
                recent_static = np.mean(static_times[-10:])
                print(f"  Journey {j+1}: β={current_beta:.2f} "
                      f"E[β]={adapt_router.expected_beta:.2f} "
                      f"adapt={recent_adapt:.1f} static={recent_static:.1f}")

        results[scen_name] = {
            "adapt_times": adapt_times,
            "static_times": static_times,
            "beta_history": beta_history,
            "beta_probs_history": beta_probs_history,
            "beta_grid": adapt_router.beta_grid,
            "final_expected_beta": adapt_router.expected_beta,
            "final_beta_probs": adapt_router.beta_probs.tolist(),
        }

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Adapt-β Convergence Experiment")
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

    # Use the known good OD pair
    s1, s2 = 67001060, 201257157
    print(f"OD: {g.stops[s1].name} -> {g.stops[s2].name}")

    results = run_convergence(
        g, s1, s2, normal_by_name, disrupted_by_name,
        n_journeys=50, seed=42)

    # Save
    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    with open('experiments/swiss_full/results/adapt_beta_convergence.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiments/swiss_full/results/adapt_beta_convergence.json")
