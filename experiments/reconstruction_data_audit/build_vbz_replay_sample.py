"""Link a small Zurich municipal tram/bus measurement sample to GTFS schedules."""
from collections import Counter, defaultdict
import csv
from datetime import date, datetime
import gzip
from io import StringIO
from itertools import groupby
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.gtfs_rt_parser import scheduled_services_by_date

CATALOGUE = 'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd_2023'
SAMPLE = CATALOGUE + '/download/Fahrzeiten_SOLL_IST_20230723_20230729.csv'
STOPS = CATALOGUE + '/download/Haltestelle.csv'
SAMPLE_BYTES = 4 * 1024 ** 2
DATES = ('2023-07-25', '2023-07-26', '2023-07-27')
AGENCIES = {'3849', '849'}


def read_sources():
    if len(sys.argv) == 3:
        return Path(sys.argv[1]).read_bytes(), Path(sys.argv[2]).read_bytes()
    sample_request = Request(SAMPLE, headers={'Range': f'bytes=0-{SAMPLE_BYTES - 1}'})
    with urlopen(sample_request, timeout=25) as response:
        if response.status != 206 or not response.headers['Content-Range'].startswith(f'bytes 0-{SAMPLE_BYTES - 1}/'):
            raise ValueError('The municipal archive did not honor the bounded sample')
        sample = response.read(SAMPLE_BYTES)
    with urlopen(STOPS, timeout=15) as response:
        names = response.read(128 * 1024)
    return sample, names


def csv_rows(content):
    return csv.DictReader(StringIO(content.decode('utf-8-sig')))


def stop_crosswalk(static, raw_stops):
    by_name = defaultdict(set)
    with (static / 'stops.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            name = ' '.join(row['stop_name'].casefold().split())
            by_name[name].add(row['stop_id'].split(':')[0].removeprefix('Parent'))
    result = {}
    for row in csv_rows(raw_stops):
        name = ' '.join(row['halt_lang'].casefold().split())
        matches = by_name[name]
        if len(matches) == 1:
            result[row['halt_id']] = next(iter(matches))
    return result


def measured_rows(raw, stop_ids, normal_only=True):
    outcomes = Counter()
    samples = defaultdict(list)
    date_keys = {date.fromisoformat(day).strftime('%d.%m.%y'): day for day in DATES}
    for row in csv_rows(raw[:raw.rfind(b'\n') + 1]):
        if row['betriebsdatum'] not in date_keys or (normal_only and row['fw_typ'] != '1'):
            continue
        day = date_keys[row['betriebsdatum']]
        start, target = stop_ids.get(row['halt_id_von']), stop_ids.get(row['halt_id_nach'])
        if start is None or target is None:
            outcomes[day, 'stop_identity_unresolved'] += 1
            continue
        values = ('soll_ab_von', 'soll_an_nach', 'ist_ab_von', 'ist_an_nach1')
        if any(not row[key] for key in values):
            outcomes[day, 'time_missing'] += 1
            continue
        dep, arr, actual_dep, actual_arr = map(int, (row[key] for key in values))
        offsets = [(datetime.strptime(row[f'datum_{side}'], '%d.%m.%y').date() - date.fromisoformat(day)).days * 86400
                   for side in ('von', 'nach')]
        dep, arr = dep + offsets[0], arr + offsets[1]
        actual_dep, actual_arr = actual_dep + offsets[0], actual_arr + offsets[1]
        if actual_dep > actual_arr or dep > arr:
            outcomes[day, 'inconsistent_time_order'] += 1
            continue
        key = row['linie'], start, target, dep // 60, arr // 60
        samples[day].append((key, {'provider_trip_id': row['fahrt_id'], 'provider_vehicle': row['fahrzeug'],
                                  'provider_service_kind': row['fw_typ'],
                                  'provider_departure_sequence': int(row['seq_von']),
                                  'provider_arrival_sequence': int(row['seq_nach']),
                                  'provider_scheduled_departure_seconds': dep,
                                  'provider_scheduled_arrival_seconds': arr,
                                  'actual_departure_seconds': actual_dep, 'actual_arrival_seconds': actual_arr}))
    return samples, outcomes


def gtfs_index(static, samples):
    regional = ROOT / 'data/zurich_gtfs'
    wanted = {key for day in samples.values() for key, _ in day}
    lines = {key[0] for key in wanted}
    with (static / 'routes.txt').open(encoding='utf-8-sig') as handle:
        routes = {row['route_id']: row for row in csv.DictReader(handle)
                  if row['agency_id'] in AGENCIES and row['route_short_name'] in lines
                  and row['route_type'].startswith(('7', '9'))}
    with (regional / 'trips.txt').open(encoding='utf-8-sig') as handle:
        trips = {row['trip_id']: row for row in csv.DictReader(handle) if row['route_id'] in routes}
    active = scheduled_services_by_date(static, DATES, {trip['service_id'] for trip in trips.values()})
    union = set().union(*active.values())
    index = defaultdict(list)
    with (regional / 'stop_times.txt').open(encoding='utf-8-sig') as handle:
        stream = csv.DictReader(handle)
        for tid, records in groupby(stream, key=lambda row: row['trip_id']):
            trip = trips.get(tid)
            if trip is None or trip['service_id'] not in union:
                continue
            stops = sorted(records, key=lambda row: int(row['stop_sequence']))
            route = routes[trip['route_id']]
            for a, b in zip(stops, stops[1:]):
                dep, arr = a['departure_time'], b['arrival_time']
                minute_dep = int(dep[:2]) * 60 + int(dep[3:5])
                minute_arr = int(arr[:2]) * 60 + int(arr[3:5])
                origin, target = a['stop_id'].split(':')[0], b['stop_id'].split(':')[0]
                key = route['route_short_name'], origin, target, minute_dep, minute_arr
                seq_a, seq_b = int(a['stop_sequence']), int(b['stop_sequence'])
                if key not in wanted or seq_a + 1 != seq_b:
                    continue
                index[key].append({'trip_id': tid, 'service_id': trip['service_id'], 'route_id': trip['route_id'],
                                   'agency_id': route['agency_id'], 'mode': 'tram' if route['route_type'].startswith('9') else 'bus',
                                   'route_name': key[0], 'dep_stop': origin, 'arr_stop': target,
                                   'dep_sequence': seq_a, 'arr_sequence': seq_b,
                                   'scheduled_departure_seconds': 60 * minute_dep + int(dep[6:8]),
                                   'scheduled_arrival_seconds': 60 * minute_arr + int(arr[6:8])})
    return index, active


def join_day(rows, index, active):
    counts = Counter()
    trips = defaultdict(set)
    matches = []
    for key, measured in rows:
        candidates = [row for row in index[key] if row['service_id'] in active]
        if len(candidates) != 1:
            counts['no_schedule_match' if not candidates else 'ambiguous_schedule_match'] += 1
            continue
        candidate = candidates[0]
        journey = measured['provider_trip_id'], measured['provider_vehicle']
        trips[journey].add(candidate['trip_id'])
        matches.append((journey, dict(candidate, **measured)))
    result = {}
    for journey, row in matches:
        if len(trips[journey]) != 1:
            counts['journey_conflict'] += 1
            continue
        key = row['trip_id'], row['dep_sequence'], row['arr_sequence']
        result.setdefault(key, []).append(row)
    legs = []
    for options in result.values():
        first = options[0]
        if any(row != first for row in options[1:]):
            counts['ambiguous_measurement'] += len(options)
        else:
            counts['duplicate_records'] += len(options) - 1
            legs.append(first)
    counts.update({mode: sum(row['mode'] == mode for row in legs) for mode in ('tram', 'bus')})
    return counts, legs


def observed_runs(legs):
    trips = defaultdict(list)
    for row in legs:
        trips[row['trip_id']].append(row)
    runs = []
    for sequence in trips.values():
        sequence.sort(key=lambda row: row['dep_sequence'])
        current = [sequence[0]]
        for previous, following in zip(sequence, sequence[1:]):
            adjacent = (previous['arr_sequence'] == following['dep_sequence']
                        and previous['provider_trip_id'] == following['provider_trip_id']
                        and previous['provider_vehicle'] == following['provider_vehicle']
                        and previous['actual_arrival_seconds'] <= following['actual_departure_seconds'])
            if adjacent:
                current.append(following)
            else:
                if len(current) >= 2:
                    runs.append(current)
                current = [following]
        if len(current) >= 2:
            runs.append(current)
    return runs


def main():
    raw, names = read_sources()
    static = ROOT / 'data/swiss_gtfs'
    crosswalk = stop_crosswalk(static, names)
    samples, exclusions = measured_rows(raw, crosswalk)
    index, active = gtfs_index(static, samples)
    results = {}
    for day in DATES:
        counts, legs = join_day(samples[day], index, active[day])
        counts.update({key: n for (date_, key), n in exclusions.items() if date_ == day})
        legs.sort(key=lambda row: (row['trip_id'], row['dep_sequence']))
        for i, leg in enumerate(legs):
            leg['id'] = i
        runs = observed_runs(legs)
        results[day] = {'coverage': dict(counts), 'n_contiguous_measured_runs': len(runs),
                        'longest_runs_by_mode': {mode: max((len(run) for run in runs if run[0]['mode'] == mode), default=0)
                                                 for mode in ('tram', 'bus')},
                        'fixed_measured_itineraries': [
                            {'connections': [leg['id'] for leg in run], 'origin': run[0]['dep_stop'],
                             'destination': run[-1]['arr_stop'],
                             'ready_seconds': run[0]['scheduled_departure_seconds'] - 120,
                             'deadline_seconds': run[-1]['scheduled_arrival_seconds'] + 1200}
                            for run in runs], 'legs': legs}
        print(day, dict(counts), len(runs), results[day]['longest_runs_by_mode'], flush=True)
    wanted_stops = {leg[key] for day in results.values() for leg in day['legs'] for key in ('dep_stop', 'arr_stop')}
    stops = {}
    with (static / 'stops.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            uic = row['stop_id'].split(':')[0]
            if uic in wanted_stops and (uic not in stops or row['stop_id'] == uic):
                stops[uic] = {'name': row['stop_name'], 'latitude': float(row['stop_lat']),
                              'longitude': float(row['stop_lon'])}
    output = {'scope': 'The municipal VBZ AVL sample supplies measured arrivals and departures for unique active GTFS '
              'trip and adjacent-stop matches at planned-minute precision. Provider scheduled seconds are retained. '
              'Partial weekly prefix; missing candidates and trip cancellations are unobserved.',
              'catalogue': CATALOGUE, 'sample_url': SAMPLE, 'stops_url': STOPS, 'sample_byte_range': [0, len(raw) - 1],
              'stops_file_bytes': len(names), 'date_scope': 'July VBZ sample separate from October SBB',
              'train_dates': ['2023-07-25'], 'holdout_dates': ['2023-07-26', '2023-07-27'],
              'stops': stops, 'dates': results}
    with gzip.open(HERE / 'vbz_measured_sample.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(output, handle, separators=(',', ':'))


if __name__ == '__main__':
    main()
