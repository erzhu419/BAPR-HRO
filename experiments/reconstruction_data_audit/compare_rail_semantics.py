"""Compare preserved rail v1 inputs, corrected v2 inputs and archived selections."""
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_dated_replay import fit_training


def read(name):
    with gzip.open(HERE / name, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def operational(connection):
    return (connection['outcome'], *((connection[kind]['status'], connection[kind]['time_seconds'],
                                      connection[kind]['official_journey']) for kind in ('departure', 'arrival')))


def compare(before, after):
    changed = {}
    by_date = {}
    for day, old in before['dates'].items():
        old_connections = {row['id']: row for row in old['connections']}
        new_connections = {row['id']: row for row in after['dates'][day]['connections']}
        added = sorted(new_connections.keys() - old_connections.keys())
        removed = sorted(old_connections.keys() - new_connections.keys())
        edits = [cid for cid in old_connections.keys() & new_connections.keys()
                 if operational(old_connections[cid]) != operational(new_connections[cid])]
        changed[day] = set(edits) | set(added) | set(removed)
        conflicts = sum(row[kind]['status'] == 'CONFLICT_REAL_CANCELED'
                        for row in new_connections.values() for kind in ('departure', 'arrival'))
        by_date[day] = {'scheduled_before': len(old_connections), 'scheduled_after': len(new_connections),
                        'added_candidates': added, 'removed_candidates': removed, 'operationally_changed_ids': edits,
                        'new_conflict_endpoints': conflicts}
    return changed, by_date


def fitting(before, after, all_dates=False):
    if all_dates:
        before = {**before, 'train_dates': sorted(before['dates'])}
        after = {**after, 'train_dates': sorted(after['dates'])}
    old, new = fit_training(before), fit_training(after)
    keys = ('route_priors', 'increments', 'travel_count', 'dwell_count', 'gap_count')
    return {'equal': old == new, 'field_equality': {key: old[key] == new[key] for key in keys},
            'route_count': len(new['route_priors']),
            'changed_route_priors': {route: {'before': old['route_priors'].get(route), 'after': new['route_priors'].get(route)}
                                     for route in old['route_priors'].keys() | new['route_priors'].keys()
                                     if old['route_priors'].get(route) != new['route_priors'].get(route)}}


def main():
    old, new = read('dated_replay.json.gz'), read('dated_replay_v2.json.gz')
    changes, replay = compare(old, new)
    old_cal, new_cal = read('rail_calibration_dates.json.gz'), read('rail_calibration_dates_v2.json.gz')
    _, calibration = compare(old_cal, new_cal)
    result_path = ROOT / 'experiments/reconstruction_v4/dated_replay_gap_corrected_results.json'
    archive = json.loads(result_path.read_text())
    affected = []
    decisions = 0
    for row in archive['rows']:
        selected = [decision['source_connection'] for decision in row['decisions']
                    if decision.get('source_connection') is not None]
        decisions += len(selected)
        hits = sorted(set(selected) & changes[row['date']])
        if hits:
            affected.append({'date': row['date'], 'query': row['query'], 'method': row['method'], 'changed_selected_ids': hits})
    output = {'scope': 'Operational status, time and journey identity comparisons isolate the parser correction. '
              'The new raw status/time/cancellation fields preserve evidence independently of operational classification.',
              'replay_dates': replay, 'calibration_dates': calibration,
              'original_three_day_fit': fitting(old, new), 'eleven_day_fit': fitting(old_cal, new_cal, all_dates=True),
              'archived_replay_result': str(result_path.relative_to(ROOT)), 'archived_journeys': len(archive['rows']),
              'archived_selected_boarding_actions': decisions, 'affected_archived_journeys': affected}
    (HERE / 'rail_semantics_v1_v2_comparison.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'changed_replay_ids': sum(len(cids) for cids in changes.values()),
                      'three_day_fit_equal': output['original_three_day_fit']['equal'],
                      'eleven_day_fit_equal': output['eleven_day_fit']['equal'],
                      'affected_archived_journeys': len(affected), 'checked_boardings': decisions}))


if __name__ == '__main__':
    main()
