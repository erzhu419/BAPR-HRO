"""Summarize and audit the fixed July-actual to October city comparison."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_dated_replay import dated_graph


def main():
    checks, comparisons, summaries, paths = defaultdict(int, messages=0), [], {}, []
    for clock in ('logical', 'charged'):
        path = HERE / f'city_past_actual_{clock}_results.json'
        new = json.loads(path.read_text())
        old = json.loads((HERE / f'city_national_{clock}_results.json').read_text())
        with gzip.open(ROOT / new['frozen_protocol'], 'rt') as handle:
            protocol = json.load(handle)
        with gzip.open(ROOT / old['frozen_protocol'], 'rt') as handle:
            previous = json.load(handle)
        data, old_data = protocol['data'], previous['data']
        day = '2023-10-25'
        assert data['dates'][day] == old_data['dates'][day]
        assert data['stops'] == old_data['stops'] and data['od_queries'] == old_data['od_queries']
        assert data['station_transfer_bounds'] == old_data['station_transfer_bounds']
        assert protocol['forecasts'] == previous['forecasts']
        graph, world, *_ = dated_graph(data, day, 'gtfs_station_max', True)
        assert len(new['rows']) == 4 and all(r['status'] != 'error' for r in new['rows'])
        lookup = {(r['query'], r['method']): r for r in old['rows']}
        for row in new['rows']:
            assert row['samples'] == 4 and row['transition_budget_per_decision'] == 64
            assert sum(d['model_transitions'] for d in row['decisions']) == row['planning_transitions']
            for d in row['decisions']:
                assert 0 <= d['model_transitions'] <= 64
                assert d['clock_seconds'] <= d['execution_clock_seconds'] <= d['end_seconds']
                checks['decisions'] += 1
            previous_time = row['od']['ready_seconds']
            for event in row['events']:
                assert event['time'] >= previous_time
                previous_time = event['time']
                if event['event_type'] not in ('board', 'arrive'):
                    continue
                cid = int(event['details'].split('connection=', 1)[1].split(';', 1)[0])
                actual = world.events[cid]
                assert event['time'] == (actual.departure if event['event_type'] == 'board' else actual.arrival)
                checks['board_arrival_events'] += 1
            for poll in row['public_actual_receipts']:
                checks['polls'] += 1
                for event_id in poll['event_ids']:
                    cid, kind = event_id.split(':')
                    cid = int(cid)
                    actual = (graph.connections[cid].dep_time if kind == 'cancel' else
                              world.events[cid].departure if kind == 'departure' else world.events[cid].arrival)
                    assert actual <= poll['receipt_time']
                    checks['messages'] += 1
            checks['new_rows'] += 1
            if row['method'] == 'trained_nominal':
                baseline = lookup[row['query'], row['method']]
                assert baseline['planning_transitions'] == row['planning_transitions'] == 0
                assert all(d['model_transitions'] == 0 for d in baseline['decisions'])
                comparisons.append({'clock': clock, 'query': row['query'],
                    'old_status': baseline['status'], 'status': row['status'],
                    'old_loss_seconds': baseline['loss_seconds'], 'loss_seconds': row['loss_seconds'],
                    'delta_seconds': None if row['loss_seconds'] is None else row['loss_seconds'] - baseline['loss_seconds'],
                    'old_labeled_budget': baseline['transition_budget_per_decision'],
                    'new_labeled_budget': 64, 'both_actual_planning_transitions': 0})
        summaries[clock] = {m: new['summary'][m] for m in ('trained_nominal', 'evolving_uct')}
        paths.append({'clock': clock, 'result': str(path.relative_to(ROOT)), 'protocol': new['frozen_protocol']})
    result = {'scope': 'Previously observed October25, two fixed OD; a prior-source diagnostic, not independent confirmation.',
              'input_audit': 'experiments/reconstruction_v5/city_past_actual_input_audit.json',
              'unchanged': 'Full October day record, graph, actual world, stop/OD mapping, station transfers, '
                           'forecast receipts, sampler count and nominal algorithm; no download.',
              'attribution': 'Four nominal pairs isolate the prior source package: old small-city-template '
                  'GTFS-RT predictions/statuses plus declared increments versus July25 complete-city actual '
                  'moments. Source cohort, prediction/realization, seasonal lag and cancellation assumptions '
                  'all belong to this package. UCT64 is reported without attributing an old UCT512 difference.',
              'cancellation': 'July performed-service source cannot identify true city cancellation rates; q=0 remains an assumption.',
              'timing': 'Two workers per panel; charged clocks pay actual processing wall time, which depends on concurrent host load.',
              'audit': dict(checks), 'outputs': paths, 'nominal_pairs': comparisons, 'summary': summaries}
    output = HERE / 'city_past_actual_summary.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'audit': dict(checks), 'nominal_pairs': comparisons,
                      'mean_loss_seconds': {clock: {m: v['uncensored_mean_loss_seconds'] for m, v in s.items()}
                                            for clock, s in summaries.items()}}, indent=2))


if __name__ == '__main__':
    main()
