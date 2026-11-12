"""Reproduce the diagnosed rare-history repair and public two-target control.

This reuses the supplied audit's selected root, not a newly selected example.
The full 61-root audit is retained without rerunning it.
"""

import json
from pathlib import Path
from time import perf_counter

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from, ranks_from
from experiments.reconstruction_v25.run import public_offers, execute
from experiments.reconstruction_v31.model import SegmentDurationBelief
from md.BAPR_HRO_GPT_diagnosis_v5.current.probe_small import ExactBoardCompiler
from src.full_menu_opportunity import FullMenuOpportunity
from src.plan_waiting_controls import (SingleTargetOpportunity,
    CombinedWaitingOpportunity, TwoTargetOpportunity)
from src.transfer_opportunity import TransferOpportunityScores


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    started = perf_counter()
    audit = json.loads((REPO/'md/BAPR_HRO_GPT_diagnosis_v5/current/'
        'residual_full_2023-07-17_2_31200_origin.json').read_text())
    source = HERE.parent/'reconstruction_v25'
    roots = json.loads((source/'selection.json').read_text())
    root = next(r for group in roots['by_day'].values() for r in group['roots']
                if r['root_id'] == audit['root_id'])
    fit = json.loads((source/'model_fit.json').read_text())
    segment_fit = json.loads((HERE.parent/'reconstruction_v31/segment_fit.json').read_text())
    data, _ = load_city(source/'city_transfer_endpoints.json.gz')
    graph, _, stop_ids, _, metadata, _ = dated_graph(data, root['day'], 'gtfs_station_max', True)
    belief = SegmentDurationBelief(graph, duration_fit=fit['duration_fit'], segment_fit=segment_fit,
        stop_identities={v: k for k, v in stop_ids.items()}, public_occurrences=False,
        grid_step_seconds=20, **fit['increments'])
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                  for c in graph.connections}
    options, ranks = options_from(root['full_options']), ranks_from(graph, root['full_options'])
    compiler = TransferOpportunityScores(graph, belief, parameters, root['deadline'], options)
    reference = FullMenuOpportunity(ExactBoardCompiler(compiler), ranks, root['start'],
                                     root['start']+30, root['deadline'])
    combined = CombinedWaitingOpportunity(SingleTargetOpportunity(reference))
    common_wall = perf_counter()-started
    tick = perf_counter()
    two_target = TwoTargetOpportunity(combined)
    pair_compilation_wall = perf_counter()-tick
    numeric = []
    for state in audit['numerical_audit']['worst_states']:
        value = reference.wait_score(state['clock'], state['seen'])
        error = abs(value-state['first_direct_interval_sum'])
        if error > 1e-8:
            raise AssertionError(('rare conditional law drift', error))
        numeric.append(dict(clock=state['clock'], seen=state['seen'], repaired=value,
            old_cdf_value=state['first_deployed'], reference=state['first_direct_interval_sum'], error=error))
    top = audit['top_contributions'][0]
    action, receipt = two_target.choose(top['visible'], top['seen'], top['clock'])
    if action is not None or abs(receipt['wait']-top['pair_wait']) > 1e-8:
        raise AssertionError('Two-target public-history behavior differs from the audit')
    offers = public_offers(metadata, tuple(options), root['start'], root['start']+30, root['deadline'])
    outcomes = {name: execute(graph, metadata, root, options, ranks, offers, controller)
                for name, controller in [('combined', combined), ('two_target', two_target)]}
    for name, old_name in [('combined', 'combined'), ('two_target', 'pair')]:
        previous, current = audit['actual'][old_name], outcomes[name]
        if any(previous[k] != current[k] for k in ('action', 'tail', 'loss', 'status')):
            raise AssertionError('Maintenance changed the selected actual journey')
        public_path = lambda outcome: [(row['clock'], list(row['visible']), list(row['seen']), row['choice'])
                                       for row in outcome['trace']]
        if public_path(current) != public_path(previous):
            raise AssertionError('Maintenance changed the public decision trace')
    result = dict(root_id=root['root_id'], numerical_histories=numeric,
        max_reference_error=max((r['error'] for r in numeric), default=0.),
        public_complementarity_case=dict(clock=top['clock'], visible=top['visible'],
            seen=top['seen'], action=action, receipt=receipt), actual=outcomes,
        timing=dict(common_preparation=common_wall, pair_compilation=pair_compilation_wall,
                    total=perf_counter()-started),
        interpretation='Targeted maintenance replay; timing is audit work, not a G2 speedup.')
    (HERE/'maintenance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(root_id=root['root_id'], rare_histories=len(numeric),
        max_reference_error=result['max_reference_error'],
        actual_losses={k: v['loss'] for k, v in outcomes.items()}, timing=result['timing'])))


if __name__ == '__main__':
    main()
