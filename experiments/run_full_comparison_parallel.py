"""R15.3 parallel re-run of tab:main + tab:extended (synthetic main comparison).

Includes V3-Topo and Adaptive-β so the table is consistent with the
contribution list. Uses 12-worker pool.

Output: experiments/results/full_comparison_R15.json
"""

from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multiprocessing import Pool

import numpy as np

from src.synthetic_network import create_bus_story_network, create_regime_distributions
from src.large_network import create_grid_network, create_grid_regime_distributions
from src.router import StaticRouter
from src.bandit_router import BanditRouter
from src.bandit_router_v2 import BanditRouterV2
from src.bandit_router_v3 import BanditRouterV3
from src.adaptive_bandit_router import AdaptiveBetaBanditRouter
from src.ssp_mdp import PosteriorSamplingRouter
from src.bamcp_router import BAMCPRouter
from src.dro_router import DRORouter
from src.sw_lcb_router import SWLCBRouter
from src.exp3_router import EXP3Router
from src.oracle_router import OracleRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import RegimeSchedule, set_regime_dist_fn, _regime_dist_cache


SMALL_SCHEDULES = {
    'normal':       RegimeSchedule(shifts=[(0, 'normal')]),
    'disrupted':    RegimeSchedule(shifts=[(0, 'normal'), (490, 'disrupted_402'), (540, 'normal')]),
    'rush_hour':    RegimeSchedule(shifts=[(0, 'normal'), (480, 'rush_hour'), (570, 'normal')]),
    'multi':        RegimeSchedule(shifts=[(0, 'normal'), (485, 'rush_hour'),
                                           (510, 'disrupted_402'), (540, 'normal')]),
}
LARGE_SCHEDULES = {
    'normal':       RegimeSchedule(shifts=[(0, 'normal')]),
    'disrupted':    RegimeSchedule(shifts=[(0, 'normal'), (490, 'central_disruption'), (560, 'normal')]),
    'full_chaos':   RegimeSchedule(shifts=[(0, 'normal'), (485, 'central_disruption'),
                                           (520, 'full_chaos'), (560, 'normal')]),
}

METHODS = ['Static', 'LCB-V1', 'LCB-V2', 'V3-Topo', 'DRO', 'Adaptive-β',
           'PS-SSP', 'BAMCP-60', 'SW-LCB', 'EXP3', 'Oracle']

NET_CFG = {
    'small': {'graph': create_bus_story_network, 'regime': create_regime_distributions,
              'sched': SMALL_SCHEDULES, 'src': 0, 'dst': 9, 'max_time': 120},
    'large': {'graph': create_grid_network, 'regime': create_grid_regime_distributions,
              'sched': LARGE_SCHEDULES, 'src': 0, 'dst': 48, 'max_time': 180},
}


def _make_router(name, graph, seed, sched, regime_fn):
    if name == 'Static':       return StaticRouter(graph)
    if name == 'LCB-V1':       return BanditRouter(graph)
    if name == 'LCB-V2':       return BanditRouterV2(graph, n_estimators=5,
                                                     beta_base=1.0, beta_ood=1.0, seed=seed)
    if name == 'V3-Topo':      return BanditRouterV3(graph, n_estimators=5,
                                                     beta_base=1.0, beta_ood=1.0, seed=seed)
    if name == 'DRO':          return DRORouter(graph, beta=1.5, gamma=60.0)
    if name == 'Adaptive-β':   return AdaptiveBetaBanditRouter(graph)
    if name == 'PS-SSP':       return PosteriorSamplingRouter(graph)
    if name == 'BAMCP-60':     return BAMCPRouter(graph, n_simulations=60)
    if name == 'SW-LCB':       return SWLCBRouter(graph, window_size=20, beta=1.5, gamma=60.0)
    if name == 'EXP3':         return EXP3Router(graph, gamma=0.1, eta=0.05, cancel_cost=60.0)
    if name == 'Oracle':       return OracleRouter(graph, regime_dist_fn=regime_fn,
                                                    regime_schedule_fn=sched.get_regime,
                                                    cancel_cost=60.0)
    raise ValueError(name)


def _init_worker():
    pass


def _run_one(args):
    net, scenario, method, i, seed_base = args
    cfg = NET_CFG[net]
    sched = cfg['sched'][scenario]
    set_regime_dist_fn(cfg['regime'])
    _regime_dist_cache.clear()

    graph = cfg['graph']()
    router = _make_router(method, graph, seed_base + i, sched, cfg['regime'])
    if isinstance(router, AdaptiveBetaBanditRouter):
        router.route(cfg['src'], cfg['dst'], 490)
    rng_t = np.random.default_rng(seed_base + i)
    t_dep = 480 + int(rng_t.integers(0, 20))
    journey_rng = np.random.default_rng(seed_base + i * 31337)
    res = simulate_bandit_journey(
        graph=graph, router=router,
        s_source=cfg['src'], s_dest=cfg['dst'],
        t_depart=t_dep, regime_schedule=sched,
        rng=journey_rng, max_time=cfg['max_time'])
    tt = res.arrival_time - res.departure_time
    return net, scenario, method, i, float(tt), int(res.n_replans)


def main(n_workers=12, n_small=100, n_large=50, seed=42):
    print('=' * 70)
    print(f'R15 Full Comparison Re-run (parallel, {n_workers} workers)')
    print(f'  Small: N={n_small} per cell × 4 scenarios')
    print(f'  Large: N={n_large} per cell × 3 scenarios')
    print(f'  Methods: {len(METHODS)}')
    print('=' * 70)

    tasks = []
    for scen in SMALL_SCHEDULES:
        for m in METHODS:
            for i in range(n_small):
                tasks.append(('small', scen, m, i, seed))
    for scen in LARGE_SCHEDULES:
        for m in METHODS:
            for i in range(n_large):
                tasks.append(('large', scen, m, i, seed))
    print(f'Total trials: {len(tasks)}')

    t0 = time.time()
    with Pool(n_workers, initializer=_init_worker) as pool:
        rows = pool.map(_run_one, tasks, chunksize=4)
    print(f'[{time.time() - t0:.1f}s elapsed]')

    out = {}
    for net, scen, m, i, tt, replans in rows:
        out.setdefault(net, {}).setdefault(scen, {}).setdefault(m, {'trials': [], 'replans': []})
        out[net][scen][m]['trials'].append(tt)
        out[net][scen][m]['replans'].append(replans)

    for net in out:
        for scen in out[net]:
            base = float(np.mean(out[net][scen]['Static']['trials']))
            mt = NET_CFG[net]['max_time']
            for m in METHODS:
                if m not in out[net][scen]: continue
                arr = np.asarray(out[net][scen][m]['trials'])
                out[net][scen][m].update({
                    'mean':        float(arr.mean()),
                    'median':      float(np.median(arr)),
                    'p95':         float(np.percentile(arr, 95)),
                    'std':         float(arr.std()),
                    'timeout_pct': float((arr >= mt).mean() * 100),
                    'n':           len(arr),
                    'improvement_pct': 0.0 if m == 'Static'
                                       else (base - arr.mean()) / base * 100,
                })

    os.makedirs('experiments/results', exist_ok=True)
    out_path = 'experiments/results/full_comparison_R15.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else str(o))
    print(f'Saved → {out_path}\n')

    print('=== SUMMARY (mean, P95, TO%) ===')
    for net in ('small', 'large'):
        if net not in out: continue
        print(f'\n[{net.upper()} NETWORK]')
        scens = list(out[net].keys())
        print(f'{"":12}', end='')
        for s in scens: print(f'{s:>22}', end='')
        print()
        for m in METHODS:
            if m not in out[net][scens[0]]: continue
            print(f'{m:<12}', end='')
            for s in scens:
                d = out[net][s][m]
                print(f' {d["mean"]:>6.1f}/{d["p95"]:>4.0f}/{d["timeout_pct"]:>4.1f}%', end='')
            print()


if __name__ == '__main__':
    main()
