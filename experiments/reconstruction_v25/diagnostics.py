"""Describe frozen outcomes without filling missing observations or changing policy."""

import gzip
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent


def main():
    selection = json.loads((HERE/'selection.json').read_text())
    roots = {r['root_id']: r for day in selection['by_day'].values() for r in day['roots']}
    with gzip.open(HERE/'results.json.gz', 'rt') as handle:
        rows = json.load(handle)['results']
    bounds, differences = {}, []
    for kind in ('origin', 'first_transfer'):
        selected = [r for r in rows if r['root_kind'] == kind]
        bounds[kind] = {}
        for method in ('original_full_nominal', 'empirical_full_nominal'):
            lower, upper, missing = [], [], 0
            for row in selected:
                cap = roots[row['root_id']]['deadline']-roots[row['root_id']]['start']
                base, candidate = (row['outcomes'][m]['loss'] for m in ('full_first_offer', method))
                missing += base is None or candidate is None
                lower.append((0 if base is None else base)-(cap if candidate is None else candidate))
                upper.append((cap if base is None else base)-(0 if candidate is None else candidate))
            bounds[kind][method] = dict(roots=len(selected), incomplete_pairs=missing,
                mean_gain_lower=mean(lower), mean_gain_upper=mean(upper))
    for row in rows:
        old, new = (row['outcomes'][m] for m in ('original_full_nominal', 'empirical_full_nominal'))
        if old['action'] == new['action']:
            continue
        differences.append(dict(root_id=row['root_id'], root_kind=row['root_kind'],
            old_loss=old['loss'], repaired_loss=new['loss'],
            old_trace=old['trace'], repaired_trace=new['trace']))
    result = dict(scope='Post-evaluation descriptive audit; no policy or model changes.',
        missing_loss_contract='Every unresolved truncated loss is in [0, deadline-start]. '
            'Bounds cover all frozen roots and are deterministic missing-data bounds, not confidence intervals.',
        all_root_gain_bounds=bounds, changed_model_actions=differences)
    (HERE/'diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
