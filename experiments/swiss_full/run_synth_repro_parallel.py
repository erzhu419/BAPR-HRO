"""R15 re-run of synthetic reproduction (Table 5) using 14-worker pool.

Output: experiments/swiss_full/results/synthetic_reproduction.json
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from multiprocessing import Pool

import numpy as np

from src.synthetic_network import create_bus_story_network
from src.router import StaticRouter
from src.bandit_router import BanditRouter
from src.bandit_router_v2 import BanditRouterV2
from src.dro_router import DRORouter
from src.adaptive_bandit_router import AdaptiveBetaBanditRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import RegimeSchedule

_SCHEDULES = None
_MAX_TIME = 180


def _make_router(name, g, seed=42):
    if name == 'Static':       return StaticRouter(g)
    if name == 'V1-LCB':       return BanditRouter(g)
    if name == 'V2-LCB':       return BanditRouterV2(g, n_estimators=5,
                                                     beta_base=1.0,
                                                     beta_ood=1.0, seed=seed)
    if name == 'DRO':          return DRORouter(g, beta=1.5, gamma=60.0)
    if name == 'Adaptive-β':   return AdaptiveBetaBanditRouter(g)
    raise ValueError(name)


def _init_worker(schedules):
    global _SCHEDULES
    _SCHEDULES = schedules


def _run_trial(args):
    method, scen_name, trial_idx, seed = args
    sched = _SCHEDULES[scen_name]
    g = create_bus_story_network()
    ri = _make_router(method, g, seed=seed)
    if isinstance(ri, AdaptiveBetaBanditRouter):
        ri.route(0, 9, 490)
    rng = np.random.default_rng(seed)
    jrng = np.random.default_rng(seed + trial_idx)
    t_dep = 480 + rng.integers(0, 20)
    res = simulate_bandit_journey(
        g, ri, 0, 9, int(t_dep), sched, jrng, _MAX_TIME)
    return method, scen_name, trial_idx, float(res.arrival_time - res.departure_time)


def main(N=100, n_workers=12, seed=42):
    print('=' * 70)
    print(f'Synthetic Reproduction (R15 RE-RUN, N={N}, {n_workers} workers)')
    print('=' * 70)

    schedules = {
        'no_disruption': RegimeSchedule(shifts=[(0, 'normal')]),
        'disrupted_402': RegimeSchedule(shifts=[
            (0, 'normal'), (490, 'disrupted_402'), (540, 'normal')]),
        'rush_hour': RegimeSchedule(shifts=[
            (0, 'normal'), (480, 'rush_hour'), (570, 'normal')]),
        'multi_shift': RegimeSchedule(shifts=[
            (0, 'normal'), (485, 'rush_hour'),
            (510, 'disrupted_402'), (540, 'normal')]),
    }
    methods = ['Static', 'V1-LCB', 'V2-LCB', 'DRO', 'Adaptive-β']
    tasks = [(m, s, i, seed)
             for s in schedules
             for m in methods
             for i in range(N)]
    print(f'Total trials: {len(tasks)}')

    t0 = time.time()
    with Pool(n_workers, initializer=_init_worker,
              initargs=(schedules,)) as pool:
        rows = pool.map(_run_trial, tasks, chunksize=8)
    print(f'[{time.time()-t0:.1f}s]')

    out = {s: {m: {'trials': []} for m in methods} for s in schedules}
    for m, s, _i, tt in rows:
        out[s][m]['trials'].append(tt)
    for s in schedules:
        base = float(np.mean(out[s]['Static']['trials']))
        for m in methods:
            arr = np.asarray(out[s][m]['trials'])
            out[s][m].update({
                'mean': float(arr.mean()),
                'median': float(np.median(arr)),
                'p95': float(np.percentile(arr, 95)),
                'std': float(arr.std()),
                'timeouts': int((arr >= _MAX_TIME).sum()),
                'n': N,
                'improvement_pct': 0.0 if m == 'Static'
                                   else (base - arr.mean()) / base * 100,
            })

    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    out_path = 'experiments/swiss_full/results/synthetic_reproduction.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else str(o))
    print(f'Saved → {out_path}')

    print('\n=== SUMMARY ===')
    for s in schedules:
        base = out[s]['Static']['mean']
        print(f'\n[{s}] Static mean={base:.1f}')
        for m in methods:
            d = out[s][m]
            tag = '' if m == 'Static' else f"  ({d['improvement_pct']:+.1f}%)"
            print(f'  {m:<12} mean={d["mean"]:6.1f} p95={d["p95"]:6.1f} '
                  f'timeouts={d["timeouts"]:>3}/{N}{tag}')


if __name__ == '__main__':
    main()
