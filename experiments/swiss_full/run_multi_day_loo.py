"""Multi-day Swiss real-data evaluation with LEAVE-ONE-DAY-OUT historical priors.

For each evaluation date d, the per-route historical prior (A4) is built
from all normal days *except* d itself; this makes the 35-day Swiss
benchmark a strict out-of-sample evaluation of A4, addressing the
data-leakage concern raised by GPT-5.5 round-6 review.

Output: results/swiss_multi_day_loo.json
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
from src.simulator import (RegimeSchedule, set_regime_dist_fn,
                           _regime_dist_cache)


_GRAPH = None
_ODS = None
_LOO_PRIORS = None  # {date: {route_name: {mean,std,cancel_rate,n_days}}}


def _init_worker(graph, ods, loo_priors):
    global _GRAPH, _ODS, _LOO_PRIORS
    _GRAPH = graph
    _ODS = ods
    _LOO_PRIORS = loo_priors


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


def _make_methods(seed, prior_override):
    return {
        'Static':     lambda g: StaticRouter(g),
        'V1-LCB':     lambda g: BanditRouter(g,
                                             route_priors_override=prior_override),
        'V2-LCB':     lambda g: BanditRouterV2(g, n_estimators=5,
                                               beta_base=1.0, beta_ood=1.0,
                                               seed=seed,
                                               route_priors_override=prior_override),
        'V3-Topo':    lambda g: BanditRouterV3(g, n_estimators=5,
                                               beta_base=1.0, beta_ood=1.0,
                                               seed=seed,
                                               route_priors_override=prior_override),
        'DRO':        lambda g: DRORouter(g, beta=1.5, gamma=60.0),
        # P0 #3 R3 review: per-cell scope. share_meta_state=False
        # gives each (day, OD) cell its own EXP3 state so the result
        # is reproducible across worker schedules.
        'Adaptive-β': lambda g: AdaptiveBetaBanditRouter(g,
                                                         route_priors_override=prior_override,
                                                         share_meta_state=False),
    }


def _categorize_day(day_dist):
    if not day_dist:
        return 'empty'
    cancels = [v['cancel_rate'] for v in day_dist.values()]
    cr = float(np.mean(cancels))
    return 'normal' if cr < 0.005 else ('mild' if cr < 0.02 else 'severe')


def _run_one_day(args):
    date, dist, n_per_seed, seeds, max_time = args
    if not dist:
        return date, 'empty', {}, None

    cat = _categorize_day(dist)
    _regime_dist_cache.clear()
    set_regime_dist_fn(_build_regime_fn(dist))
    sched = RegimeSchedule(shifts=[(0, 'normal')])

    prior_override = _LOO_PRIORS.get(date, {})  # priors EXCLUDING this day

    methods = _make_methods(seed=42, prior_override=prior_override)
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
                        tt = (res.arrival_time - 490) / 1.0
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
         loo_path: str = 'data/route_priors_loo.pkl',
         out_path: str = 'experiments/swiss_full/results/swiss_multi_day_loo.json',
         n_per_seed: int = 15,
         seeds=(0, 1, 2),
         n_workers: int = 18):
    print('Loading graph + per-day distributions + LOO priors...', flush=True)
    g = pickle.load(open('data/zurich_wide.pkl', 'rb'))
    per_day = pickle.load(open(per_day_path, 'rb'))
    loo_priors = pickle.load(open(loo_path, 'rb'))
    print(f'Days={len(per_day)}, LOO prior dates={len(loo_priors)}', flush=True)

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
        if s1 is not None and s2 is not None:
            ods.append((od['s1_name'], s1, od['s2_name'], s2))
    print(f'Resolved {len(ods)}/{len(viable_ods)} OD pairs', flush=True)
    print(f'Pool size = {n_workers}.', flush=True)

    args_list = [(date, dist, n_per_seed, list(seeds), 120)
                 for date, dist in sorted(per_day.items())]

    out = {'config': {'loo': True,
                      'n_per_seed': n_per_seed, 'seeds': list(seeds),
                      'n_per_cell': n_per_seed * len(seeds),
                      'n_ods': len(ods), 'n_days': len(per_day)},
           'per_day': {}, 'summary': {}}

    t0 = time.time()
    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(g, ods, loo_priors)) as pool:
        for i, (date, cat, day_out, err) in enumerate(
                pool.imap_unordered(_run_one_day, args_list)):
            if err:
                print(f'  [{i+1}/{len(args_list)}] {date} ({cat}) FAIL: {err}', flush=True)
                out['per_day'][date] = {'category': cat, 'error': err}
            elif not day_out:
                out['per_day'][date] = {'category': cat}
            else:
                out['per_day'][date] = {'category': cat, 'results': day_out}
                static = day_out[next(iter(day_out))].get('Static', {})
                v1 = day_out[next(iter(day_out))].get('V1-LCB', {})
                print(f'  [{i+1}/{len(args_list)}] {date} {cat} '
                      f'Static={static.get("reach_rate", 0)*100:.0f}% '
                      f'V1={v1.get("reach_rate", 0)*100:.0f}% '
                      f'[{time.time()-t0:.0f}s]', flush=True)

    print(f'\nTotal wall time: {time.time()-t0:.0f}s', flush=True)

    # Cross-day summary
    methods = ['Static', 'V1-LCB', 'V2-LCB', 'V3-Topo', 'DRO', 'Adaptive-β']
    summary = {m: {'reach_rate': [], 'cond_mean': []} for m in methods}
    for date, day in out['per_day'].items():
        if 'results' not in day: continue
        for od, res in day['results'].items():
            for m in methods:
                r = res.get(m)
                if r is None: continue
                summary[m]['reach_rate'].append(r['reach_rate'])
                summary[m]['cond_mean'].append(r['cond_mean'])
    out['summary'] = {m: {'mean_reach': float(np.mean(v['reach_rate'])),
                          'std_reach':  float(np.std(v['reach_rate'])),
                          'mean_cond':  float(np.mean(v['cond_mean'])),
                          'n_obs':      len(v['reach_rate'])}
                      for m, v in summary.items()}

    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else str(o))
    print(f'Saved → {out_path}')

    print('\n=== Cross-day summary (LOO priors, A4 strict OOS) ===')
    for m in methods:
        s = out['summary'][m]
        print(f'  {m:<14} reach={s["mean_reach"]*100:>5.1f}% '
              f'cond={s["mean_cond"]:>5.1f}min n_obs={s["n_obs"]}')


if __name__ == '__main__':
    main()
