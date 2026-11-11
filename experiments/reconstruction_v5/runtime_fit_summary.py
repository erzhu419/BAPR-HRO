"""Audit the fixed duration candidates and summarize the matched48 rows."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_dated_replay import dated_graph


def main():
    with gzip.open(HERE / 'city_budget64_logical_results_protocol.json.gz', 'rt') as handle:
        original = json.load(handle)
    worlds = {day: dated_graph(original['data'], day, 'gtfs_station_max', True)[:2]
              for day in original['data']['holdout_dates']}
    inputs, audit, summaries, compact = [], defaultdict(int, messages=0, waits=0), {}, []
    for clock in ('logical', 'charged'):
        uct = json.loads((HERE / f'city_budget64_{clock}_results.json').read_text())
        nominal = json.loads((HERE / f'city_frontier_{clock}_results.json').read_text())
        baseline = [r for r in uct['rows'] if r['method'] == 'evolving_uct'] + [
                    r for r in nominal['rows'] if r['method'] == 'trained_nominal']
        panels = [('raw', baseline)]
        for model in ('grid60', 'cdf20'):
            path = HERE / f'city_ordered_{model}_64_{clock}_results.json'
            result = json.loads(path.read_text())
            with gzip.open(ROOT / result['frozen_protocol'], 'rt') as handle:
                protocol = json.load(handle)
            assert protocol['data'] == original['data'], path
            assert protocol['forecasts'] == original['forecasts'], path
            assert all(protocol['training']['route_priors'] == fit['route_priors']
                       for fit in original['training'].values()), path
            inputs.append({'panel': path.name, 'frozen_protocol': result['frozen_protocol'],
                           'same_full_data_forecasts_and_route_priors': True})
            assert len(result['rows']) == 8 and all(r['status'] != 'error' for r in result['rows']), path
            panels.append((model, result['rows']))
            for row in result['rows']:
                assert row['transition_budget_per_decision'] == 64 and row['samples'] == 4
                assert sum(d['model_transitions'] for d in row['decisions']) == row['planning_transitions']
                previous_time = row['od']['ready_seconds']
                for d in row['decisions']:
                    assert 0 <= d['model_transitions'] <= 64
                    assert d['clock_seconds'] >= previous_time
                    assert d['execution_clock_seconds'] >= d['clock_seconds']
                    assert d['end_seconds'] >= d['execution_clock_seconds']
                    previous_time = d['clock_seconds']
                    audit['decisions'] += 1
                    audit['waits'] += d['action'] == 'wait'
                graph, world = worlds[row['date']]
                previous_event_time = row['od']['ready_seconds']
                for event in row['events']:
                    assert event['time'] >= previous_event_time
                    previous_event_time = event['time']
                    if event['event_type'] not in ('board', 'arrive'):
                        continue
                    cid = int(event['details'].split('connection=', 1)[1].split(';', 1)[0])
                    realized = world.events[cid]
                    assert not realized.canceled
                    actual = realized.departure if event['event_type'] == 'board' else realized.arrival
                    assert event['time'] == actual
                    audit['physical_board_arrival_events'] += 1
                for poll in row['public_actual_receipts']:
                    for event_id in poll['event_ids']:
                        cid, kind = event_id.split(':')
                        cid = int(cid)
                        event = world.events[cid]
                        actual = (graph.connections[cid].dep_time if kind == 'cancel' else
                                  event.departure if kind == 'departure' else event.arrival)
                        assert actual <= poll['receipt_time']
                        audit['messages'] += 1
                    audit['polls'] += 1
                audit['new_rows'] += 1
        for model, rows in panels:
            for method in ('trained_nominal', 'evolving_uct'):
                group = [r for r in rows if r['method'] == method]
                assert len(group) == 4
                uncensored = [r['loss_seconds'] for r in group if r['loss_seconds'] is not None]
                times = [d['decision_total_ms'] for r in group for d in r['decisions']]
                summaries[f'{clock}/{model}/{method}'] = {
                    'n': 4, 'uncensored': len(uncensored),
                    'mean_loss_seconds': float(np.mean(uncensored)) if uncensored else None,
                    'mean_end_to_end_seconds': float(np.mean([r['end_to_end_ms'] for r in group]) / 1000),
                    'p95_decision_seconds': float(np.quantile(times, .95) / 1000)}
                compact.extend({'clock': clock, 'model': model, 'method': method, 'date': r['date'],
                                'query': r['query'], 'status': r['status'], 'loss_seconds': r['loss_seconds']}
                               for r in group)
    result = {'scope': 'Two past-only candidates fixed before held-out execution; these previously used '
                       'July dates/OD are a retrospective matched diagnostic, not independent confirmation.',
              'matching': inputs, 'audit': dict(audit), 'summary': summaries, 'rows': compact,
              'timing': 'Each panel uses two workers. Raw panels were run earlier; charged losses pay '
                        'their actual measured wall processing and remain sensitive to concurrent machine load.',
              'negative_results': 'Retain all losses, including regressions against raw UCT64. No holdout retuning.'}
    output = HERE / 'ordered_duration_matched_summary.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'audit': dict(audit), 'summary': summaries}, indent=2))


if __name__ == '__main__':
    main()
