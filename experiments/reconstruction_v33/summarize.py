"""Separate original-law execution, actual coverage, and diagnostic references."""

from collections import Counter
import gzip
import json

import numpy as np

from experiments.reconstruction_v32.journey import paired_statistics
from experiments.reconstruction_v33.run import HERE


def actual_category(status):
    if status.startswith('unsupported_public_history'):
        return 'unsupported_public_history'
    if status.startswith('unresolved:'):
        return 'unresolved_endpoint'
    return status


def public_actions(result):
    return [(r['kind'], r.get('cid', r.get('stop')), r['clock'], r.get('choice')) for r in result['trace']]


def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    selection = json.loads((HERE.parent/'reconstruction_v32/selection_full_hubs.json').read_text())
    rows = []
    for path in sorted((HERE/'roots').glob('*.json.gz')):
        with gzip.open(path, 'rt') as handle:
            rows.append(json.load(handle))
    if len(rows) != len(selection['selected']):
        raise ValueError('Finish every frozen root before stage summarization')
    primary = protocol['primary']
    summary = {}
    for method in protocol['methods']:
        left = [[r['loss'] for r in row['sampled'][primary]] for row in rows]
        right = [[r['loss'] for r in row['sampled'][method]] for row in rows]
        model = paired_statistics(np.ravel(left), np.ravel(right))
        differences = [np.asarray(a)-b for a,b in zip(left,right)]
        se = float(np.sqrt(sum(d.var(ddof=1)/len(d) for d in differences))/len(rows))
        model.update(se=se, ci95=[model['mean_gain']-1.96*se, model['mean_gain']+1.96*se],
            mean_loss=float(np.mean(right)), uncertainty='Monte Carlo error conditional on these twelve fixed roots')
        pairs = [(row['actual'][primary]['cold'], row['actual'][method]['cold']) for row in rows]
        known = [(a,b) for a,b in pairs if a['loss'] is not None and b['loss'] is not None]
        gains = [a['loss']-b['loss'] for a,b in known]
        actual = dict(known=sum(b['loss'] is not None for a,b in pairs), paired=len(known),
            mean_loss=float(np.mean([b['loss'] for a,b in known])) if known else None,
            mean_gain=float(np.mean(gains)) if gains else None,
            wins=sum(g>0 for g in gains), harms=sum(g<0 for g in gains), ties=sum(g==0 for g in gains),
            changed_public_actions=sum(public_actions(a)!=public_actions(b) for a,b in pairs),
            categories=dict(Counter(actual_category(b['status']) for a,b in pairs)))
        summary[method] = dict(model=model, actual=actual)
    ledger = [dict(root_id=row['root_id'], root_index=row['root_index'],
        actual={m:dict(loss=value['cold']['loss'], status=value['cold']['status'])
                for m,value in row['actual'].items()}, paired_model=row['paired'],
        total_wall=row['total_wall'], endpoint_repair=row.get('endpoint_repair')) for row in rows]
    initial_wall = sum(r['total_wall'] for r in rows)
    repair_wall = sum(r.get('endpoint_repair', {}).get('actual_repair_wall', 0.) for r in rows)
    output = dict(roots=len(rows), independent_model_worlds=sum(r['evaluation_worlds'] for r in rows),
        methods=summary, actual_ledger=ledger, primary=primary,
        evidence_scope='End-to-end shared posterior and later-vehicle adaptation; fixed-history rolling scores are not the full information-tree optimum.',
        initial_root_run_wall=initial_wall, actual_endpoint_repair_wall=repair_wall,
        total_root_execution_wall=initial_wall+repair_wall,
        timing_scope='Root execution only. Joint integration, trace reconstruction, witness/reference and standalone benchmark retain separate charged receipts.')
    joint = []
    for path in sorted((HERE/'joint_roots').glob('*.json.gz')):
        with gzip.open(path, 'rt') as handle:
            row = json.load(handle)
        original = rows[row['root_index']]
        if original['root_id'] != row['root_id']:
            raise ValueError('Joint and rolling comparisons refer to different roots')
        before = [r['loss'] for r in original['sampled'][primary]]
        after = row['sampled']['one_step_rollout']
        joint.append(dict(root_id=row['root_id'], root_index=row['root_index'],
            same_G_controls=row['paired'], versus_rolling_tail_G=paired_statistics(before, after),
            compile_wall=row['compile_wall'], total_wall=row['total_wall'],
            G=[dict(stage=g['stage'], evaluated_entries=g['evaluated_entries'],
                    quadrature_worlds=g['quadrature_worlds'],
                    maximum_standard_error=max(g['standard_errors']), wall=g['wall'])
               for g in row['G']],
            actual={m:dict(loss=r['loss'], status=r['status']) for m,r in row['actual'].items()}))
    output['executed_adaptive_G'] = joint
    (HERE/'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    catalog = []
    for row, selected in zip(rows, selection['selected']):
        if row['root_id'] != selected['root_id']:
            raise ValueError('Catalogue selection and result order differ')
        catalog.append(dict(root_id=row['root_id'], root_index=row['root_index'],
            day=selected['day'], first_trip=selected['first_trip'],
            root_departure=selected['scheduled_board'], start=selected['start'],
            has_downstream_only_direct_tail=selected['has_downstream_only_direct_tail'],
            has_two_transfer_option=selected['has_two_transfer_option'],
            shared_tail_trips=selected['shared_tail_trips'],
            legal_action_contract=selected['legal_action_contract'],
            full_menu_source=f'../reconstruction_v32/selection_full_hubs.json#selected/{row["root_index"]}',
            actual=ledger[row['root_index']]['actual'],
            original_law_paired=row['paired'],
            public_histories_and_cost_receipts=f'roots/{row["root_index"]:02}.json.gz',
            differing_model_worlds={m:[i for i,(a,b) in enumerate(zip(row['sampled'][primary], values))
                                      if a['loss']!=b['loss']]
                                   for m,values in row['sampled'].items()}))
    (HERE/'decision_gap_catalog.json').write_text(json.dumps(dict(
        selection_uses_actual=False, selection_source='../reconstruction_v32/selection_full_hubs.json',
        information=protocol['information'], actions=protocol['actions'], roots=catalog,
        difference_receipts='gap_receipts.json.gz', retrospective_case='witness.json',
        original_law_references='reference_results.json',
        interpretation='Empirical disagreement frequencies are diagnostic; only the finite-law references provide exact probability-weighted Bellman contributions.'), indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
