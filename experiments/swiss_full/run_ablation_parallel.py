"""R15 re-run of β/γ/K sensitivity (Table 8 + Fig 5) using 8-worker pool.

Output: experiments/swiss_full/results/ablation.json
"""

import sys, os, json, time, pickle, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from multiprocessing import Pool

import numpy as np

from src.gtfs_parser import load_routes
from src.pmf import PMF
from src.bandit_router_v2 import BanditRouterV2
from src.dro_router import DRORouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import RegimeSchedule, set_regime_dist_fn, _regime_dist_cache


_G = None
_NORMAL = None
_DISRUPTED = None
_S1 = 67001060
_S2 = 201257157
_T_DEP = 490
_MAX_TIME = 120


def _build_regime_fn(normal_by_name, disrupted_by_name):
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


def _init_worker(g, normal_by_name, disrupted_by_name):
    global _G, _NORMAL, _DISRUPTED
    _G = g
    _NORMAL = normal_by_name
    _DISRUPTED = disrupted_by_name
    set_regime_dist_fn(_build_regime_fn(normal_by_name, disrupted_by_name))


def _run_trial(args):
    sweep, param_val, scen_name, trial_idx, seed = args
    _regime_dist_cache.clear()
    sched = (RegimeSchedule(shifts=[(0, 'normal')]) if scen_name == 'normal'
             else RegimeSchedule(shifts=[(0, 'disrupted')]))
    g = copy.deepcopy(_G)
    if sweep == 'beta':
        ri = DRORouter(g, beta=param_val, gamma=60.0)
    elif sweep == 'gamma':
        ri = DRORouter(g, beta=1.5, gamma=param_val)
    elif sweep == 'K':
        ri = BanditRouterV2(g, n_estimators=int(param_val))
    else:
        raise ValueError(sweep)
    rng = np.random.default_rng(seed + trial_idx)
    res = simulate_bandit_journey(g, ri, _S1, _S2, _T_DEP, sched, rng, _MAX_TIME)
    return sweep, param_val, scen_name, trial_idx, float(res.arrival_time - res.departure_time)


def main(N=20, n_workers=8, seed=42):
    print('=' * 70)
    print(f'Ablation Study (R15 RE-RUN, N={N}, {n_workers} workers)')
    print('=' * 70)

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

    sweeps = {
        'beta':  [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        'gamma': [0, 10, 30, 60, 120, 300],
        'K':     [1, 3, 5, 10, 20],
    }
    tasks = [(sw, val, scen, i, seed)
             for sw, vals in sweeps.items()
             for val in vals
             for scen in ('normal', 'disrupted')
             for i in range(N)]
    print(f'Total trials: {len(tasks)}')

    t0 = time.time()
    with Pool(n_workers, initializer=_init_worker,
              initargs=(g, normal_by_name, disrupted_by_name)) as pool:
        rows = pool.map(_run_trial, tasks, chunksize=4)
    print(f'[{time.time()-t0:.1f}s]')

    out = {'beta_sensitivity': {}, 'gamma_sensitivity': {}, 'ensemble_size': {}}
    sweep_to_key = {'beta': 'beta_sensitivity', 'gamma': 'gamma_sensitivity', 'K': 'ensemble_size'}
    bucket = {(sw, val, scen): [] for sw in sweeps for val in sweeps[sw]
              for scen in ('normal', 'disrupted')}
    for sw, val, scen, _i, tt in rows:
        bucket[(sw, val, scen)].append(tt)
    for (sw, val, scen), times in bucket.items():
        arr = np.asarray(times)
        key = sweep_to_key[sw]
        slot = f'{sw}={val}_{scen}'
        out[key][slot] = {
            sw: val, 'scenario': scen,
            'mean': float(arr.mean()),
            'std':  float(arr.std()),
            'p95':  float(np.percentile(arr, 95)),
        }

    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    out_path = 'experiments/swiss_full/results/ablation.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved → {out_path}')

    print('\n=== SUMMARY ===')
    for sw_key, label in [('beta_sensitivity', 'β'), ('gamma_sensitivity', 'γ'), ('ensemble_size', 'K')]:
        print(f'\n[{label}]')
        for slot, d in out[sw_key].items():
            print(f'  {slot:<28} mean={d["mean"]:6.1f} p95={d["p95"]:6.1f}')


if __name__ == '__main__':
    main()
