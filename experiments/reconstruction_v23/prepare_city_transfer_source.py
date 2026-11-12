"""Build the frozen July 23/24 municipal endpoint archive.

Prepare the two static days with prepare_city_schedule.py and filter the cached
municipal week with filter_vbz_week_remote.py --date 2023-07-23 --date 2023-07-24.
Pass the filtered gzip with its accompanying .audit.json to this script.
"""

import argparse
from collections import Counter, defaultdict
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


def load_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def match_day(day, schedule, rows, exclusions):
    connections = schedule['connections']
    index = defaultdict(list)
    for edge in connections:
        key = (edge['route_name'], edge['dep_stop'], edge['arr_stop'],
               edge['scheduled_departure_seconds'] // 60,
               edge['scheduled_arrival_seconds'] // 60)
        index[key].append(edge)
    counts, matched = join.join_day(rows, index, {edge['service_id'] for edge in connections})
    by_id = {edge['id']: edge for edge in matched}
    by_agency = defaultdict(Counter)
    for edge in connections:
        actual = by_id.get(edge['id'])
        if actual is None:
            for endpoint in ('departure', 'arrival'):
                edge[endpoint] = {'status': 'MISSING', 'time_seconds': None, 'official_journey': None}
            edge['outcome'] = 'UNRESOLVED'
        else:
            provider = f"municipal:{day}:{actual['provider_trip_id']}:{actual['provider_vehicle']}"
            for endpoint in ('departure', 'arrival'):
                edge[endpoint] = {
                    'status': 'REAL', 'time_seconds': actual[f'actual_{endpoint}_seconds'],
                    'official_journey': provider,
                    'provider_scheduled_seconds': actual[f'provider_scheduled_{endpoint}_seconds'],
                }
            edge['outcome'] = 'MEASURED'
            edge['provider_service_kind'] = actual['provider_service_kind']
        by_agency[edge['agency_id']][edge['outcome']] += 1
    return {
        'scheduled_connections': len(connections),
        'endpoint_status_counts': {'REAL': 2 * len(matched), 'MISSING': 2 * (len(connections) - len(matched))},
        'connection_outcome_counts': dict(Counter(edge['outcome'] for edge in connections)),
        'source_matching_counts': dict(counts),
        'source_exclusions': {reason: n for (date, reason), n in exclusions.items() if date == day},
        'by_agency': {agency: dict(values) for agency, values in by_agency.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--schedule-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'city_transfer_endpoints.json.gz')
    args = parser.parse_args()
    protocol = json.loads((HERE / 'protocol.json').read_text())
    days = tuple(protocol['dates'])
    train = protocol['training_date']
    extraction = json.loads(args.rows.with_suffix('.audit.json').read_text())
    if extraction['source'] != join.SAMPLE or set(extraction['rows_by_date']) != set(days):
        raise ValueError('Municipal extraction must match the frozen source and dates')

    stop_file = AUDIT / 'vbz_stop_mapping.csv'
    crosswalk = join.stop_crosswalk(ROOT / 'data/swiss_gtfs', stop_file.read_bytes())
    with stop_file.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['halt_lang'] in STOP_NAME_ALIASES:
                crosswalk[row['halt_id']] = STOP_NAME_ALIASES[row['halt_lang']]
    join.DATES = days
    with gzip.open(args.rows, 'rb') as handle:
        measured, exclusions = join.measured_rows(handle.read(), crosswalk, normal_only=False)

    training_archive = AUDIT / 'city_vbz_dated_replay_v2.json.gz'
    dates = {train: load_gzip(training_archive)['dates'][train]}
    coverage = {}
    for day in days:
        schedule = load_gzip(args.schedule_dir / f'city_schedule_{day}.json.gz')
        if schedule['date'] != day or schedule['time_window_seconds'] != extraction['window_seconds']:
            raise ValueError(f'Schedule date and filter window differ for {day}')
        coverage[day] = match_day(day, schedule, measured[day], exclusions)
        dates[day] = {**schedule, 'coverage': coverage[day]}

    dataset = {'scope': 'All planned city candidates with performed-service municipal endpoint measurements.',
               'train_dates': [train], 'holdout_dates': list(days), 'dates': dates,
               'source': join.SAMPLE, 'matching_table': join.STOPS,
               'planned_time_join_precision_seconds': 60}
    with gzip.open(args.output, 'wt', encoding='utf-8') as handle:
        json.dump(dataset, handle, ensure_ascii=False, separators=(',', ':'))
    audit = {'locked_at_utc': protocol['locked_at_utc'],
             'training_archive': str(training_archive.relative_to(ROOT)),
             'source_extraction': {key: value for key, value in extraction.items() if key != 'rows_by_date_line'},
             'coverage': coverage}
    audit_path = args.output.with_name(args.output.name.removesuffix('.json.gz') + '_audit.json')
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({day: values['connection_outcome_counts'] for day, values in coverage.items()}))


if __name__ == '__main__':
    main()
