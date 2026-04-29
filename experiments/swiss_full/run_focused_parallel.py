"""R15 re-run of focused OD (Sternen Oerlikon -> Sihlpost/HB) using
14-worker pool. Replaces serial run_swiss_normal_focused.py and
exposes the V2 cold-start fix + A1-A10 effects on Table 6.

Output: experiments/swiss_full/results/swiss_normal_focused.json
"""

import os, sys, json, time, pickle, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from multiprocessing import Pool

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


_G = None
_NORMAL = None
_DISRUPTED = None
_SCHEDULES = None
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


def _make_router(name, g):
    if name == 'Static':       return StaticRouter(g)
    if name == 'V1-LCB':       return BanditRouter(g)
    if name == 'V2-LCB':       return BanditRouterV2(g, n_estimators=5,
                                                     beta_base=1.0,
                                                     beta_ood=1.0, seed=42)
    if name == 'V3-Topo':      return BanditRouterV3(g)
    if name == 'DRO':          return DRORouter(g, beta=1.5, gamma=60.0)
    if name == 'Adaptive-β':   return AdaptiveBetaBanditRouter(g)
    raise ValueError(name)


def _init_worker(g, normal_by_name, disrupted_by_name, schedules):
    global _G, _NORMAL, _DISRUPTED, _SCHEDULES
    _G = g
    _NORMAL = normal_by_name
    _DISRUPTED = disrupted_by_name
    _SCHEDULES = schedules
    set_regime_dist_fn(_build_regime_fn(normal_by_name, disrupted_by_name))


def _run_trial(args):
    """One (method, scenario, trial_idx) → travel time + timeout flag."""
    method, scen_name, trial_idx, seed_base = args
    _regime_dist_cache.clear()
    sched = _SCHEDULES[scen_name]
    g_local = copy.deepcopy(_G)
    ri = _make_router(method, g_local)
    if isinstance(ri, AdaptiveBetaBanditRouter):
        ri.route(_S1, _S2, _T_DEP)
    jrng = np.random.default_rng(seed_base + trial_idx)
    res = simulate_bandit_journey(
        g_local, ri, _S1, _S2, _T_DEP, sched, jrng, _MAX_TIME)
    tt = res.arrival_time - res.departure_time
    return method, scen_name, trial_idx, float(tt)


def main(N: int = 30, n_workers: int = 14, seed_base: int = 42):
    print('=' * 70)
    print('Swiss Real Data: Sternen Oerlikon → Sihlpost/HB (R15 RE-RUN)')
    print(f'N={N} per cell, parallel {n_workers}-worker pool')
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

    schedules = {
        'normal':    RegimeSchedule(shifts=[(0, 'normal')]),
        'disrupted': RegimeSchedule(shifts=[(0, 'disrupted')]),
    }
    methods = ['Static', 'V1-LCB', 'V2-LCB', 'V3-Topo', 'DRO', 'Adaptive-β']
    tasks = [(m, s, i, seed_base)
             for s in schedules
             for m in methods
             for i in range(N)]
    print(f'Total trials: {len(tasks)} → {n_workers} workers')

    t0 = time.time()
    with Pool(n_workers, initializer=_init_worker,
              initargs=(g, normal_by_name, disrupted_by_name, schedules)) as pool:
        rows = pool.map(_run_trial, tasks, chunksize=4)
    print(f'[{time.time()-t0:.1f}s elapsed]')

    # Aggregate
    out = {s: {m: {'trials': []} for m in methods} for s in schedules}
    for m, s, _i, tt in rows:
        out[s][m]['trials'].append(tt)
    for s in schedules:
        base = float(np.mean(out[s]['Static']['trials']))
        for m in methods:
            arr = np.asarray(out[s][m]['trials'])
            out[s][m].update({
                'mean':    float(arr.mean()),
                'median':  float(np.median(arr)),
                'p95':     float(np.percentile(arr, 95)),
                'std':     float(arr.std()),
                'timeouts': int((arr >= _MAX_TIME).sum()),
                'n':        N,
                'improvement_pct': 0.0 if m == 'Static'
                                   else (base - arr.mean()) / base * 100,
            })

    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    out_path = 'experiments/swiss_full/results/swiss_normal_focused.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else str(o))
    print(f'Saved → {out_path}\n')

    print('=== SUMMARY ===')
    for s in schedules:
        print(f'\n[{s}]')
        for m in methods:
            d = out[s][m]
            tag = '' if m == 'Static' else f"  ({d['improvement_pct']:+.1f}%)"
            print(f'  {m:<14} mean={d["mean"]:6.1f} '
                  f'p95={d["p95"]:6.1f} timeouts={d["timeouts"]:>2}/{N}{tag}')


if __name__ == '__main__':
    main()
