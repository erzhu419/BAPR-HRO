"""R15.3 re-run of fig_adapt_beta convergence with A8 cross-journey
persistent EXP3 state.

A8 made AdaptiveBetaBanditRouter share weights at the class level by
default. For this experiment we want each scenario's curve to start
from a flat prior, so we explicitly reset the shared state at the
boundary between scenarios.

Output: experiments/swiss_full/results/adapt_beta_convergence.json
"""

import sys, os, json, copy, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np

from src.gtfs_parser import load_routes
from src.pmf import PMF
from src.adaptive_bandit_router import AdaptiveBetaBanditRouter
from src.router import StaticRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import RegimeSchedule, set_regime_dist_fn, _regime_dist_cache


def run_convergence(g, s1, s2, normal_by_name, disrupted_by_name,
                    n_journeys=50, seed=42):
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

    scenarios = ('normal', 'disrupted', 'alternating')
    results = {}
    for scen_name in scenarios:
        print(f'\n--- {scen_name} ---')
        # A8: reset class-level shared EXP3 state so each scenario's
        # curve starts from a flat prior (not the residue of previous one).
        AdaptiveBetaBanditRouter.reset_shared_state(
            len(AdaptiveBetaBanditRouter(copy.deepcopy(g)).beta_grid))

        adapt_router = AdaptiveBetaBanditRouter(copy.deepcopy(g))
        adapt_router.route(s1, s2, 490)

        static_times, adapt_times, beta_history, beta_probs_history = [], [], [], []
        for j in range(n_journeys):
            if scen_name == 'alternating':
                regime = 'normal' if (j // 10) % 2 == 0 else 'disrupted'
                sched_j = RegimeSchedule(shifts=[(0, regime)])
            else:
                sched_j = RegimeSchedule(shifts=[(0, scen_name)])

            _regime_dist_cache.clear()
            rng = np.random.default_rng(seed + j)
            # P0 #4 R3 review: simulate_bandit_journey() already
            # calls begin_journey() / end_journey() on
            # AdaptiveBetaBanditRouter. Calling them again here
            # caused (i) the recorded current_beta to be the *first*
            # sample, while the simulator drew a fresh one for the
            # actual run, and (ii) the EXP3 weights to be updated
            # twice per journey. Read current_beta from the router
            # *after* the simulator's begin_journey by capturing it
            # from the post-simulation state (the simulator does not
            # resample after end_journey).
            res_a = simulate_bandit_journey(
                adapt_router.graph, adapt_router, s1, s2,
                490 + int(rng.integers(0, 10)), sched_j, rng, 120)
            current_beta = adapt_router.current_beta
            tt_a = res_a.arrival_time - res_a.departure_time

            rng2 = np.random.default_rng(seed + j)
            static_r = StaticRouter(copy.deepcopy(g))
            res_s = simulate_bandit_journey(
                static_r.graph, static_r, s1, s2,
                490 + int(rng2.integers(0, 10)), sched_j, rng2, 120)
            tt_s = res_s.arrival_time - res_s.departure_time

            adapt_times.append(tt_a)
            static_times.append(tt_s)
            beta_history.append(current_beta)
            beta_probs_history.append(adapt_router.beta_probs.tolist())

            if (j + 1) % 10 == 0:
                ra = float(np.mean(adapt_times[-10:]))
                rs = float(np.mean(static_times[-10:]))
                print(f'  j={j+1}: β={current_beta:.2f} '
                      f'E[β]={adapt_router.expected_beta:.2f} '
                      f'adapt={ra:.1f} static={rs:.1f}')

        results[scen_name] = {
            'adapt_times': adapt_times,
            'static_times': static_times,
            'beta_history': beta_history,
            'beta_probs_history': beta_probs_history,
            'beta_grid': adapt_router.beta_grid,
            'final_expected_beta': adapt_router.expected_beta,
            'final_beta_probs': adapt_router.beta_probs.tolist(),
        }
    return results


if __name__ == '__main__':
    print('=' * 60)
    print('Adapt-β Convergence R15 Re-run (A8 shared-state isolated)')
    print('=' * 60)

    with open('data/zurich_wide.pkl', 'rb') as f:
        g = pickle.load(f)
    with open('data/day_distributions.pkl', 'rb') as f:
        day_dists = pickle.load(f)
    route_names_map = load_routes('data/swiss_gtfs')

    normal_by_name, disrupted_by_name = {}, {}
    for rid, d in day_dists['normal_day'].items():
        name = route_names_map.get(rid)
        if name: normal_by_name[name] = d
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
    print(f'OD: {g.stops[s1].name} -> {g.stops[s2].name}')

    results = run_convergence(g, s1, s2, normal_by_name, disrupted_by_name,
                              n_journeys=50, seed=42)

    out_path = 'experiments/swiss_full/results/adapt_beta_convergence.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else str(o))
    print(f'\nSaved → {out_path}')
