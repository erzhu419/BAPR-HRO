"""Component ablation A0/A1/A2/A3 evaluated under the same R16 simulator.

Each ablation row is run under the current (R16) typed-cancellation
simulator and the current A7 absolute-deadline computation; only the
router-side toggles vary across rows. This makes the causal
attribution of A2 (the layered hyperpath-risk score) as the dominant
fix independent of simulator provenance.

Output: experiments/swiss_full/results/swiss_component_ablation_R16.json

Configurations:
  A0  legacy_cold_start, no A7, no A4, no gate, prior_var=25, Beta(1,9)
  A1  posterior cold-start, no A7, no A4, no gate, prior_var=2, Beta(1,99)
  A2  posterior cold-start, A7 on,  no A4, no gate, prior_var=2, Beta(1,99)
  A3  full default (= deployed R16)

All four configs use:
  - the R16 simulator (typed cancellations: late_no_show != true_cancel,
    delay posterior is not contaminated by patience timeouts)
  - the R16 A7 deadline computation (absolute clock-time deadline)
  - the same multi-day Swiss panel (35 days x 18 ODs x 45 trials per cell)
"""

import sys, os, json, time, pickle, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from collections import defaultdict
from multiprocessing import Pool

from src.pmf import PMF
from src.bandit_router import BanditRouter, RouteBeliefState
from src.bandit_router_v2 import BanditRouterV2
from src.bandit_router_v3 import BanditRouterV3
from src.dro_router import DRORouter
from src.adaptive_bandit_router import AdaptiveBetaBanditRouter
from src.router import StaticRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import (RegimeSchedule, set_regime_dist_fn,
                           _regime_dist_cache)

sys.path.insert(0, os.path.dirname(__file__))
from run_multi_day import _build_regime_fn, _categorize_day  # type: ignore


# ────────────────────────────────────────────────────────────────────
# Worker globals
# ────────────────────────────────────────────────────────────────────
_GRAPH = None
_ODS = None


def _init_worker(graph, ods):
    global _GRAPH, _ODS
    _GRAPH = graph
    _ODS = ods


# ────────────────────────────────────────────────────────────────────
# A0/A1/A2/A3 router factories
# ────────────────────────────────────────────────────────────────────
# We patch the shared RouteBeliefState defaults inside each worker
# call by passing a `belief_factory` to BanditRouter via a tiny
# subclass; for V2 we set ensemble prior parameters via the
# RouteEnsembleBelief constructor's prior_var/cancel_alpha/cancel_beta
# args, and toggle cold-start via use_legacy_cold_start.

def _v1_with_priors(prior_var, cancel_alpha, cancel_beta):
    """Return a BanditRouter subclass that creates RouteBeliefState
    with the given priors instead of the module defaults."""
    class _V1(BanditRouter):
        def _get_belief(self, route):
            if route not in self.route_beliefs:
                p = (self._route_priors.get(route)
                     if self.use_hierarchical_prior else None)
                if p is None:
                    self.route_beliefs[route] = RouteBeliefState(
                        prior_var=prior_var,
                        cancel_alpha=cancel_alpha,
                        cancel_beta=cancel_beta,
                    )
                else:
                    hist_var = max(p['std'] ** 2, 0.5)
                    p_cancel = max(min(p['cancel_rate'], 0.5), 1e-4)
                    pseudo_n = 100.0
                    ca = max(p_cancel * pseudo_n, 1.0)
                    cb = max(pseudo_n - ca, 1.0)
                    self.route_beliefs[route] = RouteBeliefState(
                        prior_mean=p['mean'],
                        prior_var=float(hist_var),
                        cancel_alpha=ca,
                        cancel_beta=cb,
                    )
            return self.route_beliefs[route]
    return _V1


def _config_routers(name, seed=42):
    """Build the 5-method dict (V1, V2, V3, DRO, Adaptive-β) for one
    ablation row. Returns dict of (method_name -> factory function)."""
    if name == 'A0':
        # Pre-fix: σ_0^2=25, Beta(1,9), no A7, no A4, no gate, legacy V2.
        v1cls = _v1_with_priors(prior_var=25.0, cancel_alpha=1.0, cancel_beta=9.0)
        return {
            'Static':    lambda g: StaticRouter(g),
            'V1-LCB':    lambda g: v1cls(g, infeasibility_weight=0.0,
                                          timeout_weight=0.0,
                                          use_hierarchical_prior=False,
                                          disruption_gate=False),
            'V2-LCB':    lambda g: BanditRouterV2(g, n_estimators=5,
                                                   beta_base=1.0,
                                                   beta_ood=1.0,
                                                   seed=seed,
                                                   infeasibility_weight=0.0,
                                                   timeout_weight=0.0,
                                                   use_hierarchical_prior=False,
                                                   use_legacy_cold_start=True),
            'V3-Topo':   lambda g: BanditRouterV3(g, n_estimators=5,
                                                    beta_base=1.0,
                                                    beta_ood=1.0,
                                                    seed=seed),
            'DRO':       lambda g: DRORouter(g, beta=1.5, gamma=60.0),
            'Adaptive-β': lambda g: AdaptiveBetaBanditRouter(g,
                                                              share_meta_state=False),
        }
    if name == 'A1':
        # Tightened priors only (current defaults), still no A7/A4/gate,
        # still legacy V2 cold-start so A1 isolates only the prior fix.
        return {
            'Static':    lambda g: StaticRouter(g),
            'V1-LCB':    lambda g: BanditRouter(g, infeasibility_weight=0.0,
                                                 timeout_weight=0.0,
                                                 use_hierarchical_prior=False,
                                                 disruption_gate=False),
            'V2-LCB':    lambda g: BanditRouterV2(g, n_estimators=5,
                                                   beta_base=1.0,
                                                   beta_ood=1.0,
                                                   seed=seed,
                                                   infeasibility_weight=0.0,
                                                   timeout_weight=0.0,
                                                   use_hierarchical_prior=False,
                                                   use_legacy_cold_start=True),
            'V3-Topo':   lambda g: BanditRouterV3(g, n_estimators=5,
                                                    beta_base=1.0,
                                                    beta_ood=1.0,
                                                    seed=seed),
            'DRO':       lambda g: DRORouter(g, beta=1.5, gamma=60.0),
            'Adaptive-β': lambda g: AdaptiveBetaBanditRouter(g,
                                                              share_meta_state=False),
        }
    if name == 'A2':
        # A1 + A7 layered penalties + V2 cold-start fix. Still no A4/gate.
        return {
            'Static':    lambda g: StaticRouter(g),
            'V1-LCB':    lambda g: BanditRouter(g, use_hierarchical_prior=False,
                                                 disruption_gate=False),
            'V2-LCB':    lambda g: BanditRouterV2(g, n_estimators=5,
                                                   beta_base=1.0,
                                                   beta_ood=1.0,
                                                   seed=seed,
                                                   use_hierarchical_prior=False,
                                                   use_legacy_cold_start=False),
            'V3-Topo':   lambda g: BanditRouterV3(g, n_estimators=5,
                                                    beta_base=1.0,
                                                    beta_ood=1.0,
                                                    seed=seed),
            'DRO':       lambda g: DRORouter(g, beta=1.5, gamma=60.0),
            'Adaptive-β': lambda g: AdaptiveBetaBanditRouter(g,
                                                              share_meta_state=False),
        }
    if name == 'A3':
        # Full deployed R16 (= current default).
        return {
            'Static':    lambda g: StaticRouter(g),
            'V1-LCB':    lambda g: BanditRouter(g),
            'V2-LCB':    lambda g: BanditRouterV2(g, n_estimators=5,
                                                   beta_base=1.0,
                                                   beta_ood=1.0,
                                                   seed=seed),
            'V3-Topo':   lambda g: BanditRouterV3(g, n_estimators=5,
                                                    beta_base=1.0,
                                                    beta_ood=1.0,
                                                    seed=seed),
            'DRO':       lambda g: DRORouter(g, beta=1.5, gamma=60.0),
            'Adaptive-β': lambda g: AdaptiveBetaBanditRouter(g,
                                                              share_meta_state=False),
        }
    raise ValueError(f"unknown config: {name}")


def _run_one_day(args):
    """Worker: run all four ablation rows × all methods × all ODs on one day."""
    date, dist, n_per_seed, seeds, max_time = args
    if not dist:
        return date, 'empty', {}, None

    cat = _categorize_day(dist)
    _regime_dist_cache.clear()
    set_regime_dist_fn(_build_regime_fn(dist))
    sched = RegimeSchedule(shifts=[(0, 'normal')])

    out = {row: {} for row in ('A0', 'A1', 'A2', 'A3')}
    for row in ('A0', 'A1', 'A2', 'A3'):
        methods = _config_routers(row)
        for src_name, src_id, dst_name, dst_id in _ODS:
            od_key = f"{src_name} → {dst_name}"
            out[row][od_key] = {}
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
                tts_arr = np.asarray(tts)
                comp_mask = tts_arr < max_time
                n_comp = int(comp_mask.sum())
                cond = float(tts_arr[comp_mask].mean()) if n_comp else max_time
                out[row][od_key][mname] = {
                    'mean': float(tts_arr.mean()),
                    'reach_rate': float(n_comp) / len(tts),
                    'cond_mean': cond,
                    'timeouts': int(timeouts),
                    'n': len(tts),
                }
    return date, cat, out, None


def main(per_day_path: str = 'data/per_day_distributions.pkl',
         out_path: str = 'experiments/swiss_full/results/swiss_component_ablation_R16.json',
         n_per_seed: int = 15,
         seeds=(0, 1, 2),
         dates_subset=None,
         n_workers: int = 14):
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
                       'n_days': len(per_day),
                       'rows': ['A0', 'A1', 'A2', 'A3']},
           'per_day': {}}

    t0 = time.time()
    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(g, ods)) as pool:
        for i, (date, cat, day_out, err) in enumerate(
                pool.imap_unordered(_run_one_day, args_list)):
            if err:
                print(f'  [{i+1}/{len(args_list)}] {date} FAILED: {err}', flush=True)
                out['per_day'][date] = {'category': cat, 'error': err}
            elif not day_out:
                out['per_day'][date] = {'category': cat}
            else:
                out['per_day'][date] = {'category': cat, 'rows': day_out}
                # quick feedback
                static_means = [v['Static']['mean']
                                 for v in day_out['A1'].values()]
                v1_a1 = [v['V1-LCB']['mean'] for v in day_out['A1'].values()]
                v1_a3 = [v['V1-LCB']['mean'] for v in day_out['A3'].values()]
                print(f'  [{i+1}/{len(args_list)}] {date} ({cat}) '
                      f'Static={np.mean(static_means):.2f} '
                      f'V1@A1={np.mean(v1_a1):.2f} '
                      f'V1@A3={np.mean(v1_a3):.2f}', flush=True)

    # Cross-day cell-mean E[total] per (row, method).
    summary = {}
    for row in ('A0', 'A1', 'A2', 'A3'):
        summary[row] = {}
        method_names = list(_config_routers(row).keys())
        for m in method_names:
            cell_means = []
            for day_data in out['per_day'].values():
                if 'rows' not in day_data:
                    continue
                for od_key, methods in day_data['rows'][row].items():
                    if m in methods:
                        cell_means.append(methods[m]['mean'])
            summary[row][m] = float(np.mean(cell_means)) if cell_means else None
    out['summary'] = summary

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)

    elapsed = time.time() - t0
    print(f'\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'Output: {out_path}')
    print('\n=== Cross-day cell-mean E[total] (min) under R16 simulator ===')
    for row in ('A0', 'A1', 'A2', 'A3'):
        line = f'{row}:'
        for m, v in summary[row].items():
            if v is not None:
                line += f'  {m}={v:.2f}'
        print(line)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-per-seed', type=int, default=15)
    p.add_argument('--n-workers', type=int, default=14)
    p.add_argument('--seeds', type=str, default='0',
                   help='Comma-separated seed list. Default "0" reproduces '
                        'the audit-budget table; use "0,1,2" for the '
                        'full 45-trial-per-cell main-paper protocol.')
    p.add_argument('--smoke', action='store_true',
                   help='Smoke test: 1 day, 1 seed, 3 trials per cell.')
    p.add_argument('--out', default='experiments/swiss_full/results/swiss_component_ablation_R16.json')
    args = p.parse_args()
    if args.smoke:
        main(n_per_seed=3, seeds=(0,),
             dates_subset=['2023-10-29'],
             n_workers=args.n_workers,
             out_path=args.out + '.smoke.json')
    else:
        seeds = tuple(int(x) for x in args.seeds.split(','))
        main(n_per_seed=args.n_per_seed,
             seeds=seeds,
             n_workers=args.n_workers,
             out_path=args.out)
