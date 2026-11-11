"""Complete a confirmed local stop identity using the existing small VBZ source.

The original independent-confirmation records and lock remain untouched.
The new dataset is a post-opening source-completion sensitivity with the
original July25 training record, model and scheduled candidate population.
"""
from collections import defaultdict
from copy import deepcopy
import csv
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT = ROOT / 'experiments/reconstruction_data_audit'
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_data_audit import build_vbz_replay_sample as join
from experiments.reconstruction_data_audit.build_vbz_city_replay import STOP_NAME_ALIASES


def read(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def main():
    original_path = HERE / 'city_confirmation_actuals.json.gz'
    original = read(original_path)
    output = deepcopy(original)
    stop_file = AUDIT / 'vbz_stop_mapping.csv'
    crosswalk = join.stop_crosswalk(ROOT / 'data/swiss_gtfs', stop_file.read_bytes())
    with stop_file.open() as handle:
        for row in csv.DictReader(handle):
            if row['halt_lang'] in STOP_NAME_ALIASES:
                crosswalk[row['halt_id']] = STOP_NAME_ALIASES[row['halt_lang']]
    join.DATES = ('2023-07-28', '2023-07-29')
    with gzip.open(HERE / 'city_confirmation_peak_rows.csv.gz', 'rb') as handle:
        measured, exclusions = join.measured_rows(handle.read(), crosswalk, normal_only=False)
    audit = {'scope': 'Post-opening source-completion sensitivity; no new dates or model fitting.',
             'original_dataset': str(original_path.relative_to(ROOT)),
             'alias_added': {'Zürich, Brunnenhof': '8591307'},
             'municipal_stop_id': '1525',
             'identity_evidence': 'Same provider trip and vehicle as known seq22→23 and25→26; '
                 'intervening23→24→25 follows Bucheggplatz→Brunnenhof→Berninaplatz. '
                 'Static GTFS follows Bucheggplatz→Radiostudio→Berninaplatz with identical '
                 'planned minutes, line and sequence. Both affected dates give unique matches.',
             'unchanged_training_date': '2023-07-25', 'dates': {}}
    for day in join.DATES:
        day_data = output['dates'][day]
        connections = day_data['connections']
        index = defaultdict(list)
        for c in connections:
            index[c['route_name'], c['dep_stop'], c['arr_stop'],
                  c['scheduled_departure_seconds'] // 60,
                  c['scheduled_arrival_seconds'] // 60].append(c)
        counts, matched = join.join_day(measured[day], index, {c['service_id'] for c in connections})
        by_id = {c['id']: c for c in matched}
        scheduled_by_id = {c['id']: c for c in connections}
        added, known = [], 0
        for c in connections:
            actual = by_id.get(c['id'])
            if c['outcome'] == 'MEASURED':
                provider = f"municipal:{day}:{actual['provider_trip_id']}:{actual['provider_vehicle']}" if actual else None
                if actual is None or any(
                        c[kind]['time_seconds'] != actual[f'actual_{kind}_seconds'] or
                        c[kind]['official_journey'] != provider for kind in ('departure', 'arrival')):
                    raise AssertionError(f'Alias changed an existing measured connection: {day}/{c["id"]}')
                known += 1
                continue
            if actual is None:
                continue
            provider = f"municipal:{day}:{actual['provider_trip_id']}:{actual['provider_vehicle']}"
            for kind in ('departure', 'arrival'):
                c[kind] = {'status': 'REAL', 'time_seconds': actual[f'actual_{kind}_seconds'],
                           'official_journey': provider,
                           'provider_scheduled_seconds': actual[f'provider_scheduled_{kind}_seconds']}
            c['outcome'] = 'MEASURED'
            c['provider_service_kind'] = actual['provider_service_kind']
            added.append(c['id'])
        coverage = deepcopy(day_data['coverage'])
        coverage['connection_outcome_counts']['MEASURED'] += len(added)
        coverage['connection_outcome_counts']['UNRESOLVED'] -= len(added)
        coverage['endpoint_status_counts']['REAL'] += 2 * len(added)
        coverage['endpoint_status_counts']['MISSING'] -= 2 * len(added)
        for cid in added:
            c = scheduled_by_id[cid]
            stats = coverage['by_agency'][c['agency_id']]
            stats['MEASURED'] = stats.get('MEASURED', 0) + 1
            stats['UNRESOLVED'] -= 1
        coverage['source_matching_counts'] = dict(counts)
        coverage['source_exclusions'] = {reason: n for (date_, reason), n in exclusions.items() if date_ == day}
        day_data['coverage'] = coverage
        audit['dates'][day] = {'original_coverage': original['dates'][day]['coverage'],
            'completed_coverage': coverage, 'added_measured_connections': len(added),
            'added_measured_ids': added, 'unchanged_measured_connections': known,
            'unchanged_scheduled_connections': len(connections)}
    if output['dates']['2023-07-25'] != original['dates']['2023-07-25']:
        raise AssertionError('The original training record changed')
    output['scope'] = audit['scope']
    output['source_completion'] = {key: value for key, value in audit.items() if key != 'dates'}
    output['explicit_municipal_stop_name_aliases_to_uic'] = dict(STOP_NAME_ALIASES)
    audit_path = HERE / 'city_confirmation_source_completed_audit.json'
    result_path = HERE / 'city_confirmation_source_completed.json.gz'
    with gzip.open(result_path, 'wt') as handle:
        json.dump(output, handle, ensure_ascii=False, separators=(',', ':'))
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print('DATASET', result_path, flush=True)
    print('COVERAGE', {day: {'added': value['added_measured_connections'],
        'unchanged_measured': value['unchanged_measured_connections']}
        for day, value in audit['dates'].items()}, flush=True)


if __name__ == '__main__':
    main()
