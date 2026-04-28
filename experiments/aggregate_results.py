"""Post-wave aggregator.

Reads:
  experiments_log/multiseed/{vrp,uc,sdn}_seed{0..7}.json
  experiments_log/bamcp_sweep/{small,large}_{normal,disrupted}_b{60,120,240}.json
  experiments/swiss_full/results/swiss_multi_day.json (if present)

Writes:
  experiments_log/aggregate_summary.json
  experiments_log/aggregate_summary.txt
"""

import os, json, glob, sys
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_multiseed(domain):
    files = sorted(glob.glob(os.path.join(BASE, 'experiments_log',
                                          'multiseed', f'{domain}_seed*.json')))
    runs = [json.load(open(f)) for f in files]
    if not runs:
        return None

    methods = set()
    for r in runs:
        methods.update(r.get('results', {}).keys())

    out = {}
    for m in sorted(methods):
        means = [r['results'][m]['mean'] for r in runs if m in r.get('results', {})]
        if not means:
            continue
        out[m] = {'n_seeds': len(means),
                  'mean': float(np.mean(means)),
                  'std': float(np.std(means)),
                  'ci95_lo': float(np.percentile(means, 2.5)),
                  'ci95_hi': float(np.percentile(means, 97.5))}
    return out


def load_bamcp_sweep():
    files = sorted(glob.glob(os.path.join(BASE, 'experiments_log',
                                          'bamcp_sweep', '*.json')))
    by_cell: dict[tuple, dict] = defaultdict(dict)
    for f in files:
        d = json.load(open(f))
        cell = (d['net'], d['scenario'])
        for m, r in d['results'].items():
            by_cell[cell][m] = {k: r[k]
                                for k in ('mean_time', 'reach_rate',
                                          'mean', 'wall_s')
                                if k in r}
    return {f'{net}_{scen}': by_cell[(net, scen)]
            for (net, scen) in by_cell}


def load_multi_day():
    p = os.path.join(BASE, 'experiments/swiss_full/results',
                     'swiss_multi_day.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    out = {}

    print('=== Cross-domain multi-seed ===')
    for d in ['vrp', 'uc', 'sdn']:
        agg = load_multiseed(d)
        out[f'cross_{d}'] = agg
        if not agg:
            continue
        print(f'-- {d.upper()} ({agg[list(agg)[0]]["n_seeds"]} seeds) --')
        print(f'{"Method":<16} {"mean":>10} {"std":>8} {"95%CI":>22}')
        for m, s in agg.items():
            print(f'{m:<16} {s["mean"]:>10.2f} {s["std"]:>8.2f} '
                  f'[{s["ci95_lo"]:>7.2f}, {s["ci95_hi"]:>7.2f}]')

    print('\n=== BAMCP rollout sweep ===')
    bamcp = load_bamcp_sweep()
    out['bamcp_sweep'] = bamcp
    for cell, ms in bamcp.items():
        print(f'-- {cell} --')
        for m in sorted(ms):
            r = ms[m]
            print(f'  {m:<14} mean={r.get("mean_time", r.get("mean", "N/A"))} '
                  f'reach={r.get("reach_rate", "N/A")}')

    print('\n=== Multi-day Swiss ===')
    md = load_multi_day()
    if md:
        out['multi_day'] = md.get('summary', md)
        for m, s in md.get('summary', {}).items():
            print(f'  {m:<14} reach={s["mean_reach"]*100:.1f}%  '
                  f'cond={s["mean_cond"]:.1f}min  n={s["n_obs"]}')

    out_path = os.path.join(BASE, 'experiments_log', 'aggregate_summary.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
