"""Predeclared July28/29 city confirmation, using the retained weekly source.

Run protocol/schedules before fetching the two filtered actual-day subsets;
then match using exactly the V2 municipal identity and minute-join rules.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUDIT = ROOT / 'experiments/reconstruction_data_audit'
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_data_audit import build_vbz_replay_sample as join
from experiments.reconstruction_data_audit.build_vbz_city_replay import STOP_NAME_ALIASES
from experiments.reconstruction_v4.run_city_replay import OD_NAMES

DATES = ('2023-07-28', '2023-07-29')
PROTOCOL = HERE / 'city_confirmation_protocol.json'


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def lock():
    settings = {'train_dates': ['2023-07-25'], 'confirmation_dates': list(DATES),
                'static_od_names': [list(od) for od in OD_NAMES],
                'ready_seconds': 29100, 'deadline_seconds': 32700,
                'methods': ['trained_nominal', 'evolving_uct', 'compiled_nominal', 'analytic_recourse'],
                'max_itineraries': 8, 'uct_transition_budget': 64, 'private_worlds': 4,
                'clocks': ['logical', 'charged'], 'transfer_rule': 'gtfs_station_max',
                'matching_rule': 'Unchanged V2 stop aliases, exact service calendar and minute planned-clock join.',
                'cohort_rule': 'All calendar-active planned bus/tram connections in existing rectangle/window, including unresolved outcomes.',
                'actual_source': join.SAMPLE, 'zero_cancellation_prior': 'Performed services alone do not identify unconditional cancellation; retain existing zero assumption.',
                'no_test_selection': 'Dates, OD, methods and limits locked before opening these dates actual subsets; no source-identity retuning.'}
    if PROTOCOL.exists():
        previous = json.loads(PROTOCOL.read_text())
        if previous['settings'] != settings:
            raise ValueError('Confirmation settings differ from their pre-actual lock')
        return previous
    result = {'locked_at_utc': datetime.now(timezone.utc).isoformat(), 'settings': settings,
              'source_text': {name: (ROOT / name).read_text() for name in (
                  'experiments/reconstruction_v6/build_city_confirmation.py',
                  'experiments/reconstruction_data_audit/prepare_city_schedule.py',
                  'experiments/reconstruction_data_audit/filter_vbz_week_remote.py',
                  'experiments/reconstruction_data_audit/build_vbz_replay_sample.py',
                  'experiments/reconstruction_data_audit/build_vbz_city_replay.py',
                  'experiments/reconstruction_v6/run_itinerary_replay.py',
                  'src/risk_bounds.py', 'src/recourse_itinerary_router.py')}}
    PROTOCOL.write_text(json.dumps(result, indent=2) + '\n')
    return result


def schedules():
    lock()
    for day in DATES:
        output = HERE / f'city_confirmation_schedule_{day}.json.gz'
        if not output.exists():
            subprocess.run([sys.executable, str(AUDIT / 'prepare_city_schedule.py'),
                            '--date', day, '--output', str(output)], check=True)


def match():
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['settings']['confirmation_dates'] != list(DATES):
        raise ValueError('Date lock does not match this input builder')
    source = HERE / 'city_confirmation_peak_rows.csv.gz'
    source_audit = json.loads(source.with_suffix('.audit.json').read_text())
    if set(source_audit['rows_by_date']) != set(DATES):
        raise ValueError('Filtered source must contain exactly the declared confirmation dates')
    stop_file = AUDIT / 'vbz_stop_mapping.csv'
    crosswalk = join.stop_crosswalk(ROOT / 'data/swiss_gtfs', stop_file.read_bytes())
    with stop_file.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['halt_lang'] in STOP_NAME_ALIASES:
                crosswalk[row['halt_id']] = STOP_NAME_ALIASES[row['halt_lang']]
    join.DATES = DATES
    with gzip.open(source, 'rb') as handle:
        measured, exclusions = join.measured_rows(handle.read(), crosswalk, normal_only=False)
    original = load(AUDIT / 'city_vbz_dated_replay_v2.json.gz')
    dates = {'2023-07-25': original['dates']['2023-07-25']}
    summaries = {}
    for day in DATES:
        scheduled = load(HERE / f'city_confirmation_schedule_{day}.json.gz')
        connections = scheduled['connections']
        identity_before = [{key: value for key, value in c.items()} for c in connections]
        index = defaultdict(list)
        for c in connections:
            key = (c['route_name'], c['dep_stop'], c['arr_stop'],
                   c['scheduled_departure_seconds'] // 60, c['scheduled_arrival_seconds'] // 60)
            index[key].append(c)
        counts, matches = join.join_day(measured[day], index, {c['service_id'] for c in connections})
        by_id = {row['id']: row for row in matches}
        endpoints, by_agency = Counter(), defaultdict(Counter)
        for c in connections:
            actual = by_id.get(c['id'])
            if actual is None:
                for kind in ('departure', 'arrival'):
                    c[kind] = {'status': 'MISSING', 'time_seconds': None, 'official_journey': None}
                c['outcome'] = 'UNRESOLVED'
                endpoints['MISSING'] += 2
            else:
                provider = f"municipal:{day}:{actual['provider_trip_id']}:{actual['provider_vehicle']}"
                for kind in ('departure', 'arrival'):
                    c[kind] = {'status': 'REAL', 'time_seconds': actual[f'actual_{kind}_seconds'],
                               'official_journey': provider,
                               'provider_scheduled_seconds': actual[f'provider_scheduled_{kind}_seconds']}
                c['outcome'] = 'MEASURED'
                c['provider_service_kind'] = actual['provider_service_kind']
                endpoints['REAL'] += 2
            by_agency[c['agency_id']][c['outcome']] += 1
        # Every scheduled row and field survives the actual join unchanged.
        if any(any(c[key] != value for key, value in prior.items())
               for prior, c in zip(identity_before, connections)) or len(identity_before) != len(connections):
            raise AssertionError('The actual join changed or dropped a planned candidate')
        coverage = {'scheduled_connections': len(connections), 'endpoint_status_counts': dict(endpoints),
                    'connection_outcome_counts': dict(Counter(c['outcome'] for c in connections)),
                    'source_matching_counts': dict(counts),
                    'source_exclusions': {reason: n for (date_, reason), n in exclusions.items() if date_ == day},
                    'by_agency': {agency: dict(values) for agency, values in by_agency.items()}}
        dates[day] = {**scheduled, 'coverage': coverage}
        summaries[day] = coverage
        print(day, json.dumps(coverage), flush=True)
    if dates['2023-07-25'] != original['dates']['2023-07-25']:
        raise AssertionError('Training actual record changed')
    dataset = {'scope': 'Predeclared independent July28/29 confirmation with complete dated planned city candidates and performed-service measurements.',
               'train_dates': ['2023-07-25'], 'holdout_dates': list(DATES), 'dates': dates,
               'source': join.SAMPLE, 'catalogue': join.CATALOGUE, 'matching_table': join.STOPS,
               'planned_time_join_precision_seconds': 60,
               'explicit_municipal_stop_name_aliases_to_uic': STOP_NAME_ALIASES,
               'source_audit': source_audit, 'confirmation_protocol': str(PROTOCOL.relative_to(ROOT)),
               'locked_before_new_actuals_at_utc': protocol['locked_at_utc']}
    output = HERE / 'city_confirmation_actuals.json.gz'
    with gzip.open(output, 'wt', encoding='utf-8') as handle:
        json.dump(dataset, handle, ensure_ascii=False, separators=(',', ':'))
    audit = {'confirmation_protocol': str(PROTOCOL.relative_to(ROOT)),
             'dataset': str(output.relative_to(ROOT)), 'source_audit': source_audit,
             'unchanged_training_record': True, 'all_planned_candidates_and_fields_retained': True,
             'dates': summaries}
    (HERE / 'city_confirmation_input_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Dataset ready:', output, 'bytes', output.stat().st_size, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('lock', 'schedules', 'match'))
    stage = parser.parse_args().stage
    {'lock': lock, 'schedules': schedules, 'match': match}[stage]()


if __name__ == '__main__':
    main()
