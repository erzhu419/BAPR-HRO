"""Reconcile all saved V24 stages without re-running any policy."""

import gzip
import json
from statistics import mean

from experiments.reconstruction_v24.run import HERE, METHODS, groups


def main():
    names = ('original60', 'grid20', 'grid10', 'empirical20', 'empirical10',
             'route_empirical20', 'route_empirical10')
    runs = {}
    for name in names:
        folder = HERE if name == 'original60' else HERE/name
        with gzip.open(folder/'results.json.gz', 'rt') as handle:
            result = json.load(handle)
        if not result['complete']:
            raise ValueError(f'{name} did not complete')
        runs[name] = {row['root_id']: row for row in result['results']}
    common = [key for key in runs['original60'] if all(
        outcome['loss'] is not None for run in runs.values()
        for outcome in run[key]['actual']['outcomes'].values())]
    tables = {}
    for name, run in runs.items():
        actual = [run[key]['actual'] for key in common]
        tables[name] = dict(common_roots=len(common),
            model_expected_loss={m: mean(row['model']['expected_loss'][m] for row in run.values()) for m in METHODS},
            actual_mean_loss={m: mean(row['outcomes'][m]['loss'] for row in actual)
                              for m in (*METHODS, 'full_first_offer')},
            actual_arrivals={m: sum(row['outcomes'][m]['arrived'] for row in actual)
                             for m in (*METHODS, 'full_first_offer')},
            actual_comparisons=groups(actual))
    differences = []
    for key, old in runs['grid20'].items():
        new = runs['route_empirical20'][key]
        a, b = (r['actual']['outcomes']['exact'] for r in (old, new))
        if a['action'] != b['action']:
            differences.append(dict(root_id=key, before=a, after=b))
    output = dict(scope='Descriptive development comparison; common actual roots only. '
                        'Model values refer to different probability laws and cannot be subtracted as policy gains.',
                  common_root_ids=common, tables=tables, original_to_repaired_action_changes=differences,
                  final_grid_action_changes=[key for key in runs['route_empirical20'] if
                      runs['route_empirical20'][key]['actual']['outcomes'] !=
                      runs['route_empirical10'][key]['actual']['outcomes']])
    (HERE/'comparison.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({name: {k: v for k, v in row.items() if k != 'actual_comparisons'}
                      for name, row in tables.items()}, indent=2))


if __name__ == '__main__':
    main()
