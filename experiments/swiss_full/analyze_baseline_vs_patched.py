"""Compare baseline (un-patched) vs patched multi-day Swiss results.

Inputs:
  results/swiss_multi_day_baseline.json  (un-patched V1/V2/V3)
  results/swiss_multi_day.json           (patched V1/V2/V3)

Outputs aggregate + per-day delta + paired stats.
"""

import json, os, sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    return json.load(open(path))


def aggregate(d):
    """Return {method: {'reach': [list per (day, OD)], 'cond': [list]}}."""
    out = defaultdict(lambda: {'reach': [], 'cond': []})
    for date, dd in d.get('per_day', {}).items():
        if 'results' not in dd:
            continue
        for od, by_m in dd['results'].items():
            for m, r in by_m.items():
                out[m]['reach'].append(r['reach_rate'])
                out[m]['cond'].append(r['cond_mean'])
    return out


def summary(agg, name):
    print(f'\n=== {name} ===')
    print(f'{"Method":<14} {"reach":>9} {"std":>7} {"cond":>9}')
    for m in ['Static', 'V1-LCB', 'V2-LCB', 'V3-Topo', 'DRO', 'Adaptive-β']:
        if m not in agg:
            continue
        r = agg[m]
        rmean = np.mean(r['reach'])
        rstd = np.std(r['reach'])
        cmean = np.mean(r['cond'])
        print(f'{m:<14} {rmean*100:>7.1f}% {rstd*100:>5.1f}% {cmean:>8.1f}min')


def paired_diff(base, pat, key='reach'):
    """Compute paired (per-cell) diff between patched and baseline."""
    print(f'\n=== Per-cell diff: patched - baseline ({key}) ===')
    print(f'{"Method":<14} {"mean_diff":>10} {"%cells_better":>13} {"%cells_worse":>13}')
    for m in ['V1-LCB', 'V2-LCB', 'V3-Topo', 'DRO', 'Adaptive-β']:
        if m not in base or m not in pat:
            continue
        # Pair by index assuming same order
        b = np.asarray(base[m][key])
        p = np.asarray(pat[m][key])
        if len(b) != len(p):
            print(f'{m:<14} length mismatch: {len(b)} vs {len(p)}')
            continue
        diff = p - b
        better = float((diff > 0.001).mean())
        worse = float((diff < -0.001).mean())
        scale = 100 if key == 'reach' else 1
        unit = '%' if key == 'reach' else 'min'
        print(f'{m:<14} {diff.mean()*scale:>+9.2f}{unit} '
              f'{better*100:>11.1f}% {worse*100:>11.1f}%')


def per_day_disrupted(d, name):
    """Show disrupted-day rows specifically."""
    print(f'\n=== Disrupted day(s) in {name} ===')
    for date, dd in d.get('per_day', {}).items():
        if dd.get('category') != 'severe':
            continue
        if 'results' not in dd:
            continue
        # Avg across ODs
        method_reach = defaultdict(list)
        for od, by_m in dd['results'].items():
            for m, r in by_m.items():
                method_reach[m].append(r['reach_rate'])
        print(f'  {date}:', {m: f'{np.mean(v)*100:.1f}%'
                              for m, v in method_reach.items()})


def main():
    base_path = 'experiments/swiss_full/results/swiss_multi_day_baseline.json'
    pat_path = 'experiments/swiss_full/results/swiss_multi_day.json'
    if not os.path.exists(base_path) or not os.path.exists(pat_path):
        print('Need both baseline and patched files; missing one.')
        sys.exit(1)

    base = load(base_path)
    pat = load(pat_path)
    base_agg = aggregate(base)
    pat_agg = aggregate(pat)

    summary(base_agg, 'BASELINE (un-patched V1/V2/V3)')
    summary(pat_agg, 'PATCHED (tighter prior + disruption gate)')

    paired_diff(base_agg, pat_agg, 'reach')
    paired_diff(base_agg, pat_agg, 'cond')

    per_day_disrupted(base, 'baseline')
    per_day_disrupted(pat, 'patched')


if __name__ == '__main__':
    main()
