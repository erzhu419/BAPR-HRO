"""Twelve frozen full-integral comparisons with the agreed primary stop rule."""

from collections import Counter
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from
from experiments.reconstruction_v29.scoring import evaluate_pair
from src.empirical_duration_belief import EmpiricalDurationBelief
from src.evolving_trip_belief import EvolvingTripBelief
from src.transfer_opportunity import TransferOpportunityScores


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'
LAWS = ('original20', 'route_empirical20')


def gate(rows):
    """Unsupported comparisons cannot satisfy the user's unchanged-rank stop."""
    primary = [r for r in rows if r['law'] == 'route_empirical20']
    supported = [r for r in primary if r['scores']['status'] == 'supported']
    changed = [r['root_id'] for r in supported
               if r['scores']['nominal']['action'] != r['scores']['integrated']['action']]
    if len(primary) != 6 or len(supported) != 6:
        status = 'unresolved_support'
    elif not changed:
        status = 'stop_exact_scoring_rescue'
    else:
        status = 'ranking_change_requires_separate_policy_evidence'
    return dict(status=status, primary_law='route_empirical20', primary_cells=len(primary),
        supported_cells=len(supported), changed_roots=changed,
        interpretation='Stop is scoped to the frozen primary six histories and this scoring rescue route. '
                       'Original-law and grid-bracket sensitivity remain reported; no automatic expansion.')


def main():
    started = perf_counter()
    selection = json.loads((HERE/'selection.json').read_text())
    original = json.loads((SOURCE/'selection.json').read_text())
    roots = {r['root_id']: r for group in original['by_day'].values() for r in group['roots']}
    cases = selection['cases']
    expected = [f'2023-07-17_0_{start}_origin' for start in range(27600,32101,900)]
    gains = [c['actual_one_step']['loss']-c['actual_single']['loss'] for c in cases]
    if [c['root']['root_id'] for c in cases] != expected or (sum(g > 0 for g in gains), sum(g < 0 for g in gains)) != (2,4):
        raise AssertionError('All six original two-win/four-loss cases must be retained')
    for case in cases:
        root = case['root']
        if root != roots[root['root_id']] or case['current_cid'] not in case['visible'] or case['target_cid'] in case['seen']:
            raise AssertionError('Frozen menu or public current/target history changed')
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    graph, _, _, _, _, _ = dated_graph(data, '2023-07-17', 'gtfs_station_max', True)
    params = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    shared = perf_counter()-started
    results = []
    for law in LAWS:
        for case in cases:
            root = case['root']
            tick = perf_counter()
            kwargs = dict(public_occurrences=False, grid_step_seconds=20, **fit['increments'])
            belief = (EvolvingTripBelief(graph, **kwargs) if law == 'original20' else
                EmpiricalDurationBelief(graph, duration_fit=fit['duration_fit'], **kwargs))
            compiler = TransferOpportunityScores(graph, belief, params, root['deadline'], options_from(root['full_options']))
            setup = perf_counter()-tick
            tick = perf_counter()
            scores = evaluate_pair(compiler, case['current_cid'], case['target_cid'],
                                   case['clock'], case['entry'], case['ready'])
            evaluate_wall = perf_counter()-tick
            if law == 'route_empirical20' and scores['status'] == 'supported':
                previous = case['original_receipts']['single']
                if (abs(scores['nominal']['current']-previous['board']) > 1e-6
                        or abs(scores['nominal']['wait']-previous['wait']) > 1e-6):
                    raise AssertionError('Full integration changed the nominal V28 comparison')
            result = dict(root_id=root['root_id'], law=law, menu_size=len(root['full_options']),
                start=root['start'], actual_gain=case['actual_one_step']['loss']-case['actual_single']['loss'],
                scores=scores, wall=dict(model_and_compiler_setup=setup, pair_evaluation=evaluate_wall))
            results.append(result)
            print(law, root['root_id'], {name:scores[name] for name in ('nominal','mean_arrival','integrated')}, flush=True)
            (HERE/'results.json').write_text(json.dumps(dict(complete=False, results=results), indent=2)+'\n')
    summary = dict(shared_loading_and_graph_wall=shared, by_law={}, decision=gate(results))
    for law in LAWS:
        rows = [r for r in results if r['law'] == law]
        supported = [r for r in rows if r['scores']['status'] == 'supported']
        summary['by_law'][law] = dict(cells=len(rows), supported=len(supported),
            reasons=dict(Counter(r['scores']['reason'] for r in rows if r['scores']['status'] != 'supported')),
            changed_roots=[r['root_id'] for r in supported if r['scores']['nominal']['action'] != r['scores']['integrated']['action']],
            mean_scores={mode:{key:mean(r['scores'][mode][key] for r in supported) if supported else None
                              for key in ('current','wait','wait_minus_board')}
                         for mode in ('nominal','mean_arrival','integrated')},
            current_clock_methods=dict(Counter(r['scores']['current_clock_method'] for r in supported)),
            bracket_sensitive_roots=[r['root_id'] for r in supported if r['scores']['current_bracket']['integrated_ranking_sensitive']],
            wall={key:sum(r['wall'][key] for r in rows) for key in rows[0]['wall']})
    summary.update(total_wall=perf_counter()-started, new_worlds=0,
        nominal_primary_receipts_reproduced=sum(r['law'] == 'route_empirical20' and r['scores']['status'] == 'supported'
                                              for r in results))
    (HERE/'results.json').write_text(json.dumps(dict(complete=True, results=results, summary=summary), indent=2)+'\n')
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (HERE/'decision.json').write_text(json.dumps(summary['decision'], indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
