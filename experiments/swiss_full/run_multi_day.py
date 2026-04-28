"""Multi-day Swiss real-data evaluation. PARALLEL via Pool.

For each date in per_day_distributions.pkl, run multi-OD reach-rate +
conditional-mean comparison across Static / V1-LCB / V2-LCB / V3-Topo
/ DRO / Adaptive-β. Aggregate cross-day statistics.

Output: results/swiss_multi_day.json
"""

import sys, os, json, time, pickle, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from collections import defaultdict
from multiprocessing import Pool

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

sys.path.insert(0, os.path.dirname(__file__))
from run_multi_od import find_od_pairs  # type: ignore


# ────────────────────────────────────────────────────────────────────
# Worker globals (set per process via _init_worker)
# ────────────────────────────────────────────────────────────────────
_GRAPH = None
_ODS = None


def _init_worker(graph, ods):
    global _GRAPH, _ODS
    _GRAPH = graph
    _ODS = ods


def _build_regime_fn(day_dist):
    def real_day_regime(_regime_name):
        result = {}
        for rname, d in day_dist.items():
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


def _make_methods(seed=42):
    return {
        'Static':     lambda g: StaticRouter(g),
        'V1-LCB':     lambda g: BanditRouter(g),
        'V2-LCB':     lambda g: BanditRouterV2(g, n_estimators=5,
                                               beta_base=1.0,
                                               beta_ood=1.0,
                                               seed=seed),
        'V3-Topo':    lambda g: BanditRouterV3(g, n_estimators=5,
                                               beta_base=1.0,
                                               beta_ood=1.0,
                                               seed=seed),
        'DRO':        lambda g: DRORouter(g, beta=1.5, gamma=60.0),
        'Adaptive-β': lambda g: AdaptiveBetaBanditRouter(g),
    }


def _categorize_day(day_dist):
    if not day_dist:
        return 'empty'
    cancels = [v['cancel_rate'] for v in day_dist.values()]
    cr = float(np.mean(cancels))
    if cr < 0.005:
        return 'normal'
    elif cr < 0.02:
        return 'mild'
    else:
        return 'severe'


def _run_one_day(args):
    """Worker: run ALL methods × ALL ODs on one day. Returns
    (date, category, results_dict_or_error)."""
    date, dist, n_per_seed, seeds, max_time = args
    if not dist:
        return date, 'empty', {}, None

    cat = _categorize_day(dist)
    _regime_dist_cache.clear()
    regime_fn = _build_regime_fn(dist)
    set_regime_dist_fn(regime_fn)
    sched = RegimeSchedule(shifts=[(0, 'normal')])

    methods = _make_methods(seed=42)
    out = {}
    for src_name, src_id, dst_name, dst_id in _ODS:
        od_key = f"{src_name} → {dst_name}"
        out[od_key] = {}
        for mname, make_router in methods.items():
            tts, timeouts = [], 0
            for s in seeds:
                for i in range(n_per_seed):
                    ri = make_router(copy.deepcopy(_GRAPH))
                    if isinstance(ri, AdaptiveBetaBanditRouter):
                        ri.route(src_id, dst_id, 490)
                    jrng = np.random.default_rng(s * 100 + i)
                    res = simulate_bandit_journey(
                        copy.deepcopy(_GRAPH), ri, src_id, dst_id, 490,
                        sched, jrng, max_time=max_time)
                    if res.arrival_time >= 99999 or res.arrival_time is None:
                        timeouts += 1
                        tts.append(max_time)
                    else:
                        tt = (res.arrival_time - 490) / 1.0  # in minutes
                        if tt >= max_time:
                            timeouts += 1
                            tts.append(max_time)
                        else:
                            tts.append(tt)

            n = len(tts)
            tts_arr = np.asarray(tts)
            comp_mask = tts_arr < max_time
            n_comp = int(comp_mask.sum())
            cond_mean = float(tts_arr[comp_mask].mean()) if n_comp > 0 else max_time
            out[od_key][mname] = {
                'mean': float(tts_arr.mean()),
                'reach_rate': float(n_comp) / n,
                'cond_mean': cond_mean,
                'timeouts': int(timeouts),
                'n': n,
                'trials': [float(x) for x in tts],
            }
    return date, cat, out, None


def main(per_day_path: str = 'data/per_day_distributions.pkl',
         out_path: str = 'experiments/swiss_full/results/swiss_multi_day.json',
         n_per_seed: int = 15,
         seeds=(0, 1, 2),
         dates_subset=None,
         n_workers: int = 18):
    print('Loading graph and per-day distributions...', flush=True)
    g = pickle.load(open('data/zurich_wide.pkl', 'rb'))
    per_day = pickle.load(open(per_day_path, 'rb'))

    if dates_subset is not None:
        per_day = {d: per_day[d] for d in dates_subset if d in per_day}
    print(f'Days to evaluate: {len(per_day)}', flush=True)

    # Reuse the 18 viable OD pairs already screened by run_swiss_multi_od_v3.
    print('Loading viable OD pairs from v3 result...', flush=True)
    v3 = json.load(open('experiments/swiss_full/results/swiss_multi_od_v3.json'))
    viable_ods = v3['viable_ods']

    # Build name → stop_id lookup once. g.stops is a dict {sid → Stop}.
    name_to_id = {}
    for sid, stop in g.stops.items():
        name_to_id.setdefault(stop.name, sid)

    ods = []
    missing = 0
    for od in viable_ods:
        s1 = name_to_id.get(od['s1_name'])
        s2 = name_to_id.get(od['s2_name'])
        if s1 is None or s2 is None:
            missing += 1
            continue
        ods.append((od['s1_name'], s1, od['s2_name'], s2))
    print(f'Resolved {len(ods)}/{len(viable_ods)} OD pairs '
          f'({missing} missing in graph).', flush=True)
    print(f'Pool size = {n_workers}.', flush=True)

    args_list = [
        (date, dist, n_per_seed, list(seeds), 120)
        for date, dist in sorted(per_day.items())
    ]

    out = {'config': {'n_per_seed': n_per_seed, 'seeds': list(seeds),
                       'n_per_cell': n_per_seed * len(seeds),
                       'n_ods': len(ods),
                       'n_days': len(per_day)},
           'per_day': {}, 'summary': {}}

    t0 = time.time()
    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(g, ods)) as pool:
        for i, (date, cat, day_out, err) in enumerate(
                pool.imap_unordered(_run_one_day, args_list)):
            if err:
                print(f'  [{i+1}/{len(args_list)}] {date} ({cat}) FAILED: {err}',
                      flush=True)
                out['per_day'][date] = {'category': cat, 'error': err}
            elif not day_out:
                print(f'  [{i+1}/{len(args_list)}] {date} (empty distribution)',
                      flush=True)
                out['per_day'][date] = {'category': cat}
            else:
                # Quick summary
                static_reach = np.mean([v['Static']['reach_rate']
                                         for v in day_out.values()])
                v1_reach = np.mean([v['V1-LCB']['reach_rate']
                                     for v in day_out.values()])
                print(f'  [{i+1}/{len(args_list)}] {date} ({cat}) '
                      f'Static={static_reach*100:.0f}% V1={v1_reach*100:.0f}% '
                      f'[{(time.time()-t0):.0f}s]', flush=True)
                out['per_day'][date] = {'category': cat, 'results': day_out}

    elapsed = time.time() - t0
    print(f'\nWall time: {elapsed:.0f}s', flush=True)

    # Aggregate across days
    summary = defaultdict(lambda: defaultdict(list))
    for date, d in out['per_day'].items():
        if 'results' not in d:
            continue
        for od, by_method in d['results'].items():
            for m, r in by_method.items():
                summary[m]['reach_rate'].append(r['reach_rate'])
                summary[m]['cond_mean'].append(r['cond_mean'])

    out['summary'] = {m: {'mean_reach': float(np.mean(v['reach_rate'])),
                           'std_reach':  float(np.std(v['reach_rate'])),
                           'mean_cond':  float(np.mean(v['cond_mean'])),
                           'n_obs':      len(v['reach_rate'])}
                       for m, v in summary.items()}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {out_path}', flush=True)

    print('\n=== Cross-day summary ===', flush=True)
    print(f'{"Method":<14} {"reach":>9} {"cond":>9}')
    for m, v in out['summary'].items():
        print(f'{m:<14} {v["mean_reach"]*100:>7.1f}% {v["mean_cond"]:>8.1f}min',
              flush=True)


if __name__ == '__main__':
    main()
