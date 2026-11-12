"""Expand all hubs of the twelve frozen roots without selecting on outcomes.

The original three-hub cap can structurally eliminate longer lookahead. This
common-menu sensitivity restores every eligible hub on the already frozen
eight-leg first-vehicle path. Roots, vehicle caps, schedules and source stay
fixed. The initial selection remains available separately.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from experiments.reconstruction_v32.screen import HERE, static_graphs, structure_roots


def expand(selection, graphs):
    protocol = dict(selection['protocol'], max_hubs=selection['protocol']['max_first_legs'])
    selected, ledger = [], []
    for old in selection['selected']:
        candidates, _, _ = structure_roots(graphs[old['day']], old['day'],
                                           old['first_cid'], protocol)
        root = next(r for r in candidates if r['goal'] == old['goal'])
        if root['first_path'] != tuple(old['first_path']):
            raise AssertionError('Expanding hubs changed the frozen first-vehicle path')
        if (root['start'], root['deadline']) != (old['start'], old['deadline']):
            raise AssertionError('Expanding hubs changed the frozen clocks')
        by_cid = {h['arrival_cid']: h for h in root['hubs']}
        for hub in old['hubs']:
            # JSON canonicalization only reconciles persisted lists with the
            # in-memory tuples; any real menu difference stops the expansion.
            if json.loads(json.dumps(by_cid[hub['arrival_cid']])) != hub:
                raise AssertionError('Expanding hubs changed a previously retained menu')
        root['selected_layer'] = old['selected_layer']
        selected.append(root)
        old_hubs = {h['arrival_cid'] for h in old['hubs']}
        ledger.append(dict(root_id=root['root_id'], capped_hubs=len(old['hubs']),
            full_hubs=len(root['hubs']),
            added_arrival_cids=[h['arrival_cid'] for h in root['hubs']
                                if h['arrival_cid'] not in old_hubs],
            capped_variants=sum(len(h['options']) for h in old['hubs']),
            full_variants=sum(len(h['options']) for h in root['hubs'])))
    result = dict(selection, selected=selected, protocol=dict(protocol,
        selection='The same twelve frozen first_cid+goal roots; restore all hubs on their frozen first path.',
        original_max_hubs=selection['protocol']['max_hubs'],
        sensitivity='Shared full-hub menu; no model, clock, root or vehicle-cap change.',
        selection_uses_outcomes=False))
    result['by_day'] = {day: dict(group,
        selected=[r for r in selected if r['day'] == day])
        for day, group in selection['by_day'].items()}
    result['expansion_ledger'] = ledger
    result['original_selection'] = 'experiments/reconstruction_v32/selection.json'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, default=HERE / 'selection.json')
    parser.add_argument('--out', type=Path, default=HERE / 'selection_full_hubs.json')
    args = parser.parse_args()
    started = perf_counter()
    selection = json.loads(args.selection.read_text())
    result = expand(selection, static_graphs())
    result['expansion_wall'] = perf_counter()-started
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(roots=len(result['selected']),
        capped_hubs=sum(r['capped_hubs'] for r in result['expansion_ledger']),
        full_hubs=sum(r['full_hubs'] for r in result['expansion_ledger']),
        ledger=result['expansion_ledger'], wall=result['expansion_wall']), indent=2))


if __name__ == '__main__':
    main()
