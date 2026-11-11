"""Assemble completed fixed-root case receipts without rerunning simulation."""
from collections import defaultdict
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['kind'] == 'replicate':
            groups[row['n'], row['method'], row['cid']].append(row)
    result = {}
    for (n, method, cid), group in sorted(groups.items()):
        conditional = {}
        for caught in (True, False):
            values = [cost for row in group for cost, event in zip(row['world_costs'], row['caught']) if event == caught]
            conditional['catch' if caught else 'fail'] = {
                'n': len(values), 'mean': float(np.mean(values)) if values else None,
                'variance': float(np.var(values, ddof=1)) if len(values) > 1 else None}
        result[f'{n}/{method}/{cid}'] = {
            'n_replicates': len(group), 'mean': float(np.mean([r['value'] for r in group])),
            'variance': float(np.var([r['value'] for r in group], ddof=1)),
            'mean_transitions': float(np.mean([r['transitions'] for r in group])),
            'max_transitions': max(r['transitions'] for r in group),
            'all_within512': all(r['within512'] for r in group),
            'mean_wall_seconds': float(np.mean([r['wall_seconds'] for r in group])),
            'mean_cpu_seconds': float(np.mean([r['cpu_seconds'] for r in group])),
            'zero_failure_replicates': sum(all(r['caught']) for r in group), 'conditional_costs': conditional}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = output.with_name(output.stem + '_protocol.json.gz')
    with gzip.open(protocol, 'rt') as handle:
        settings = json.load(handle)['settings']
    expected = [('archived', 0, 4, method, cid) for method in settings['methods'] for cid in settings['root_actions']]
    expected += [('replicate', seed[1], n, method, cid) for seed in settings['replicate_seeds']
                 for n in settings['sample_counts'] for method in settings['methods'] for cid in settings['root_actions']]
    folder = output.with_name(output.stem + '_cases')
    rows = [json.loads((folder / ('_'.join(map(str, spec)) + '.json')).read_text()) for spec in expected]
    for spec, row in zip(expected, rows):
        if tuple(row[key] for key in ('kind', 'replicate', 'n', 'method', 'cid')) != spec:
            raise ValueError('Saved case differs from frozen case identity')
        if not np.isclose(sum(row['weights']), 1.) or not np.isclose(np.dot(row['weights'], row['world_costs']), row['value']):
            raise ValueError('Saved weighted value is inconsistent')
    result = {'settings': settings, 'frozen_protocol': str(protocol.relative_to(ROOT)),
              'summary': summarize(rows), 'rows': rows,
              'summary_source': str(Path(__file__).relative_to(ROOT))}
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
