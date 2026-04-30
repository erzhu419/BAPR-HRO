"""BOCD + topocsa hyperpath-recompute baseline on Swiss real data.

Runs RecomputeRouter (BOCD-style: re-run topocsa with disrupted-day
distributions on cancellation) and compares it against Static and
V1-LCB on the 35-day x 18-OD panel. The result quantifies the
``hyperpath recomputation fails'' claim: under the same simulator
that the main benchmark uses, a recompute strategy is essentially
indistinguishable from the static hyperpath, while the proposed
re-ranking strategy improves both expected travel time and reach
rate.

Output:
    experiments/swiss_full/results/swiss_recompute_baseline.json
"""

import sys, os, json, time, pickle, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from collections import defaultdict
from multiprocessing import Pool

from src.bandit_router import BanditRouter
from src.recompute_router import RecomputeRouter
from src.router import StaticRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import (RegimeSchedule, set_regime_dist_fn,
                           _regime_dist_cache)

sys.path.insert(0, os.path.dirname(__file__))
from run_multi_day import _build_regime_fn, _categorize_day  # type: ignore


_GRAPH = None
_ODS = None
_PER_DAY = None


def _init_worker(graph, ods, per_day):
    global _GRAPH, _ODS, _PER_DAY
    _GRAPH = graph
    _ODS = ods
    _PER_DAY = per_day


def _normal_dist_for_day(date, per_day):
    """The 'normal' baseline distribution that BOCD assumes before any
    cancellation is seen: take the average of all normal days in the
    same archive (categorized by mean cancel rate < 0.5%)."""
    normal_days = []
    for d, dist in per_day.items():
        if not dist:
            continue
        cr = float(np.mean([v.get('cancel_rate', 0) for v in dist.values()]))
        if cr < 0.005:
            normal_days.append(dist)
    if not normal_days:
        return per_day[date]
    routes = set()
    for d in normal_days:
        routes.update(d.keys())
    avg = {}
    for r in routes:
        means = [d[r]['mean'] for d in normal_days if r in d]
        stds = [d[r]['std'] for d in normal_days if r in d]
        cxls = [d[r].get('cancel_rate', 0.0) for d in normal_days if r in d]
        avg[r] = {'mean': float(np.mean(means)),
                   'std': float(np.mean(stds)),
                   'cancel_rate': float(np.mean(cxls))}
    return avg


def _make_methods(seed=42, normal_dist=None, disrupted_dist=None):
    return {
        'Static':            lambda g: StaticRouter(g),
        'BOCD-recompute':    lambda g: RecomputeRouter(g, normal_dist=normal_dist,
                                                        disrupted_dist=disrupted_dist),
        'V1-LCB (R16)':      lambda g: BanditRouter(g),
    }


def _run_one_day(args):
    date, dist, n_per_seed, seeds, max_time = args
    if not dist:
        return date, 'empty', {}, None

    cat = _categorize_day(dist)
    _regime_dist_cache.clear()
    set_regime_dist_fn(_build_regime_fn(dist))
    sched = RegimeSchedule(shifts=[(0, 'normal')])

    normal_dist = _normal_dist_for_day(date, _PER_DAY)
    methods = _make_methods(seed=42, normal_dist=normal_dist,
                            disrupted_dist=dist)

    out = {}
    for src_name, src_id, dst_name, dst_id in _ODS:
        od_key = f"{src_name} → {dst_name}"
        out[od_key] = {}
        for mname, make_router in methods.items():
            tts, timeouts = [], 0
            for s in seeds:
                for i in range(n_per_seed):
                    ri = make_router(copy.deepcopy(_GRAPH))
                    jrng = np.random.default_rng(s * 100 + i)
                    res = simulate_bandit_journey(
                        copy.deepcopy(_GRAPH), ri, src_id, dst_id, 490,
                        sched, jrng, max_time=max_time)
                    if res.arrival_time >= 99999 or res.arrival_time is None:
                        timeouts += 1
                        tts.append(max_time)
                    else:
                        tt = (res.arrival_time - 490) / 1.0
                        if tt >= max_time:
                            timeouts += 1
                            tts.append(max_time)
                        else:
                            tts.append(tt)
            tts_arr = np.asarray(tts)
            comp_mask = tts_arr < max_time
            n_comp = int(comp_mask.sum())
            cond = float(tts_arr[comp_mask].mean()) if n_comp else max_time
            out[od_key][mname] = {
                'mean': float(tts_arr.mean()),
                'reach_rate': float(n_comp) / len(tts),
                'cond_mean': cond,
                'timeouts': int(timeouts),
                'n': len(tts),
            }
    return date, cat, out, None


def main(per_day_path: str = 'data/per_day_distributions.pkl',
         out_path: str = 'experiments/swiss_full/results/swiss_recompute_baseline.json',
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

    print('Loading viable OD pairs from v3 result...', flush=True)
    v3 = json.load(open('experiments/swiss_full/results/swiss_multi_od_v3.json'))
    viable_ods = v3['viable_ods']
    name_to_id = {}
    for sid, stop in g.stops.items():
        name_to_id.setdefault(stop.name, sid)
    ods = []
    for od in viable_ods:
        s1 = name_to_id.get(od['s1_name'])
        s2 = name_to_id.get(od['s2_name'])
        if s1 is None or s2 is None:
            continue
        ods.append((od['s1_name'], s1, od['s2_name'], s2))
    print(f'Resolved {len(ods)} OD pairs.  Pool size = {n_workers}.', flush=True)

    args_list = [(date, dist, n_per_seed, list(seeds), 120)
                 for date, dist in sorted(per_day.items())]

    out = {'config': {'n_per_seed': n_per_seed, 'seeds': list(seeds),
                       'n_per_cell': n_per_seed * len(seeds),
                       'n_ods': len(ods),
                       'n_days': len(per_day)},
           'per_day': {}}

    t0 = time.time()
    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(g, ods, per_day)) as pool:
        for i, (date, cat, day_out, err) in enumerate(
                pool.imap_unordered(_run_one_day, args_list)):
            if err:
                print(f'  [{i+1}/{len(args_list)}] {date} FAILED: {err}', flush=True)
                out['per_day'][date] = {'category': cat, 'error': err}
            elif not day_out:
                out['per_day'][date] = {'category': cat}
            else:
                out['per_day'][date] = {'category': cat, 'methods': day_out}
                static_means = [v['Static']['mean'] for v in day_out.values()]
                rec_means = [v['BOCD-recompute']['mean'] for v in day_out.values()]
                v1_means = [v['V1-LCB (R16)']['mean'] for v in day_out.values()]
                print(f'  [{i+1}/{len(args_list)}] {date} ({cat}) '
                      f'Static={np.mean(static_means):.2f} '
                      f'BOCD={np.mean(rec_means):.2f} '
                      f'V1={np.mean(v1_means):.2f}', flush=True)

    summary = {}
    for m in ('Static', 'BOCD-recompute', 'V1-LCB (R16)'):
        cell_means, cell_reach = [], []
        for day_data in out['per_day'].values():
            if 'methods' not in day_data:
                continue
            for od_key, methods in day_data['methods'].items():
                if m in methods:
                    cell_means.append(methods[m]['mean'])
                    cell_reach.append(methods[m]['reach_rate'])
        summary[m] = {
            'mean_E_total': float(np.mean(cell_means)) if cell_means else None,
            'mean_reach': float(np.mean(cell_reach)) if cell_reach else None,
        }
    out['summary'] = summary

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    elapsed = time.time() - t0
    print(f'\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'Output: {out_path}')
    print('\n=== Cross-day cell-mean E[total] (min) and reach rate ===')
    for m in ('Static', 'BOCD-recompute', 'V1-LCB (R16)'):
        s = summary[m]
        if s['mean_E_total'] is not None:
            print(f'  {m:18}: E[total]={s["mean_E_total"]:.2f}  reach={s["mean_reach"]*100:.1f}%')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-per-seed', type=int, default=15)
    p.add_argument('--n-workers', type=int, default=18)
    p.add_argument('--seeds', type=str, default='0',
                   help='Comma-separated seed list. Default "0" reproduces '
                        'the audit-budget table; use "0,1,2" for the '
                        'full 45-trial-per-cell main-paper protocol.')
    p.add_argument('--smoke', action='store_true',
                   help='Smoke test: 1 day (Oct 29), 1 seed, 3 trials per cell.')
    p.add_argument('--out', default='experiments/swiss_full/results/swiss_recompute_baseline.json')
    args = p.parse_args()
    if args.smoke:
        main(n_per_seed=3, seeds=(0,),
             dates_subset=['2023-10-29'],
             n_workers=4,
             out_path=args.out + '.smoke.json')
    else:
        seeds = tuple(int(x) for x in args.seeds.split(','))
        main(n_per_seed=args.n_per_seed, seeds=seeds,
             n_workers=args.n_workers,
             out_path=args.out)
