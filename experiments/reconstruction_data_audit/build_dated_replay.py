"""Build date-specific Zurich rail replay inputs from bounded official archives."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
from datetime import date
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_data_audit.run_realized_subset import (
    ARCHIVE, DOCUMENTATION, FIELDS, PREFIX_BYTES, archive_prefixes, seconds_of_service,
)
from experiments.reconstruction_data_audit.actual_event_parser import parse_actual_endpoint
from src.gtfs_rt_parser import load_route_metadata, scheduled_services_by_date

TRAIN_DATES = ('2023-10-02', '2023-10-06', '2023-10-09')
HOLDOUT_DATES = ('2023-10-12', '2023-10-25', '2023-10-27')
AGENCIES = {'11': '85:11'}
STATIONS = {
    '8503000', '8503001', '8503003', '8503004', '8503006', '8503008',
    '8503009', '8503010', '8503011', '8503016', '8503020', '8503147',
    '8503508', '8503509', '8503200', '8503101',
}
OD_PAIRS = (
    ('8503147', '8503010'), ('8503003', '8503020'), ('8503011', '8503006'),
    ('8503006', '8503009'), ('8503509', '8503147'), ('8503001', '8503004'),
    ('8503016', '8503010'), ('8503009', '8503006'),
)
START, DEPARTURE, DEADLINE, END = 7 * 3600 + 30 * 60, 8 * 3600 + 5 * 60, 9 * 3600 + 5 * 60, 9 * 3600 + 35 * 60
TRANSFER_SECONDS = 120


def clock_seconds(time):
    hours, minutes, seconds = map(int, time.split(':'))
    return hours * 3600 + minutes * 60 + seconds


def dated_connections(dates):
    static = ROOT / 'data/swiss_gtfs'
    regional = ROOT / 'data/zurich_gtfs'
    metadata = load_route_metadata(static)
    with (regional / 'trips.txt').open(encoding='utf-8-sig') as handle:
        trips = {row['trip_id']: row for row in csv.DictReader(handle)
                 if metadata[row['route_id']]['agency_id'] in AGENCIES and row['trip_short_name']}
    services = scheduled_services_by_date(static, dates, {row['service_id'] for row in trips.values()})
    active_trips = {trip for trip, row in trips.items() if any(row['service_id'] in ids for ids in services.values())}
    stops = {}
    with (static / 'stops.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            uic = row['stop_id'].split(':')[0]
            if uic in STATIONS and (uic not in stops or row['stop_id'] == uic):
                stops[uic] = {'name': row['stop_name'], 'latitude': float(row['stop_lat']),
                              'longitude': float(row['stop_lon']), 'transfer_seconds': TRANSFER_SECONDS}
    sequences = defaultdict(list)
    with (regional / 'stop_times.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['trip_id'] in active_trips:
                sequences[row['trip_id']].append((int(row['stop_sequence']), row))
    template = []
    for trip, sequence in sequences.items():
        sequence.sort(key=lambda item: item[0])
        trip_row = trips[trip]
        route = trip_row['route_id']
        agency = metadata[route]['agency_id']
        for (seq_from, start), (seq_to, end) in zip(sequence, sequence[1:]):
            origin, target = (start['stop_id'].split(':')[0], end['stop_id'].split(':')[0])
            if origin not in stops or target not in stops or origin == target:
                continue
            # The regional file omits stops outside Zürich. Requiring successive
            # sequence numbers preserves physical adjacency in this Swiss feed.
            if seq_to != seq_from + 1:
                continue
            dep, arr = clock_seconds(start['departure_time']), clock_seconds(end['arrival_time'])
            if not START <= dep < END or arr < dep or arr > END:
                continue
            template.append({'agency_id': agency, 'operator_id': AGENCIES[agency], 'route_id': route,
                             'route_name': metadata[route]['name'], 'trip_id': trip,
                             'service_id': trip_row['service_id'], 'vehicle_number': trip_row['trip_short_name'],
                             'dep_stop': origin, 'arr_stop': target, 'dep_sequence': seq_from, 'arr_sequence': seq_to,
                             'scheduled_departure_seconds': dep, 'scheduled_arrival_seconds': arr,
                             'pickup_allowed': start['pickup_type'] == '0', 'dropoff_allowed': end['drop_off_type'] == '0'})
    template.sort(key=lambda row: (row['scheduled_departure_seconds'], row['trip_id'], row['dep_sequence']))
    by_date = {day: [dict(row, id=i) for i, row in enumerate(
        row for row in template if row['service_id'] in services[day])] for day in dates}
    return stops, by_date


def endpoint_key(row, kind):
    side = 'dep' if kind == 'departure' else 'arr'
    return (row['operator_id'], row['vehicle_number'], row[f'{side}_stop'],
            kind, row[f'scheduled_{kind}_seconds'] // 60)


def actual_endpoints(day, connections, rows):
    """Join date, operator, train number, BPUIC, event type and planned minute."""
    keys = {endpoint_key(connection, kind) for connection in connections for kind in FIELDS}
    numbers = {key[:2] for key in keys}
    records = defaultdict(set)
    day_text = date.fromisoformat(day).strftime('%d.%m.%Y')
    for row in rows:
        vehicle = row['BETREIBER_ID'], row['LINIEN_ID']
        if vehicle not in numbers or row['BETRIEBSTAG'] != day_text or row['ZUSATZFAHRT_TF'] == 'true':
            continue
        for kind, (planned, actual, status) in FIELDS.items():
            if not row[planned]:
                continue
            key = (*vehicle, row['BPUIC'], kind, seconds_of_service(row[planned], day) // 60)
            if key not in keys:
                continue
            records[key].add(parse_actual_endpoint(row, kind, day))
    return records


def attach_outcomes(day, connections, rows):
    records = actual_endpoints(day, connections, rows)
    gtfs_keys = defaultdict(set)
    for connection in connections:
        for kind in FIELDS:
            side = 'dep' if kind == 'departure' else 'arr'
            gtfs_keys[endpoint_key(connection, kind)].add((connection['trip_id'], connection[f'{side}_sequence']))
    endpoints = Counter()
    legs = Counter()
    for row in connections:
        for kind in FIELDS:
            key = endpoint_key(row, kind)
            options = records[key]
            if len(gtfs_keys[key]) > 1:
                status, time, journey = 'AMBIGUOUS_STATIC', None, None
                reported, canceled, raw_status = None, None, None
            elif len(options) > 1:
                status, time, journey = 'AMBIGUOUS_ACTUAL', None, None
                reported, canceled, raw_status = None, None, None
            elif not options:
                status, time, journey = 'MISSING', None, None
                reported, canceled, raw_status = None, None, None
            else:
                status, time, journey, reported, canceled, raw_status = next(iter(options))
            endpoints[status] += 1
            row[kind] = {'status': status, 'time_seconds': time, 'official_journey': journey,
                         'reported_time_seconds': reported, 'canceled_flag': canceled, 'reported_status': raw_status}
        dep, arr = row['departure'], row['arrival']
        same = dep['official_journey'] is not None and dep['official_journey'] == arr['official_journey']
        if dep['status'] == arr['status'] == 'REAL' and same and dep['time_seconds'] <= arr['time_seconds']:
            row['outcome'] = 'MEASURED'
        elif dep['status'] == arr['status'] == 'CANCELED' and same:
            row['outcome'] = 'CANCELED'
        else:
            row['outcome'] = 'UNRESOLVED'
        legs[row['outcome']] += 1
    return {'n_scheduled_trips': len({row['trip_id'] for row in connections}),
            'n_scheduled_connections': len(connections), 'endpoint_status_counts': dict(endpoints),
            'connection_outcome_counts': dict(legs)}


def prepared_sources(dates, source_dir):
    for day in dates:
        with gzip.open(source_dir / f'{day}.csv.gz', 'rt', encoding='utf-8') as handle:
            yield day, None, list(csv.DictReader(handle, delimiter=';'))


def main(source_dir=None):
    if source_dir is not None:
        source_dir = source_dir.resolve()
    dates = TRAIN_DATES + HOLDOUT_DATES
    stops, scheduled = dated_connections(dates)
    output = {'scope': 'Scheduled SBB adjacency among selected Zurich BPUICs, by service date. '
              'Measured times are REAL and explicit endpoint statuses stay distinct. '
              'Unresolved edges remain in the scheduled action set.',
              'timezone': 'Europe/Zurich', 'archive': ARCHIVE, 'documentation': DOCUMENTATION,
              'static_feed': (ROOT / 'data/swiss_gtfs/feed_info.txt').read_text().strip(),
              'compressed_bytes_requested': 0 if source_dir is not None else 65557 + len(dates) * PREFIX_BYTES,
              'train_dates': TRAIN_DATES, 'holdout_dates': HOLDOUT_DATES,
              'stops': stops, 'od_queries': [{'origin': origin, 'destination': target,
                                             'ready_seconds': DEPARTURE, 'deadline_seconds': DEADLINE}
                                            for origin, target in OD_PAIRS],
              'dates': {}}
    sources = archive_prefixes(dates) if source_dir is None else prepared_sources(dates, source_dir)
    for day, byte_range, rows in sources:
        connections = scheduled[day]
        coverage = attach_outcomes(day, connections, rows)
        output['dates'][day] = {'archive_byte_range': byte_range, 'coverage': coverage, 'connections': connections}
        if source_dir is not None:
            output['dates'][day]['filtered_source'] = str((source_dir / f'{day}.csv.gz').relative_to(ROOT))
        print(day, coverage, flush=True)
    with gzip.open(HERE / 'dated_replay_v2.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(output, handle, ensure_ascii=False, separators=(',', ':'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, help='Directory of small remote-filtered YYYY-MM-DD.csv.gz inputs')
    main(parser.parse_args().source_dir)
