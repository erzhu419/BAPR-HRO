"""Prepare all active bus/tram candidates in a declared Zurich geographic window."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.gtfs_rt_parser import scheduled_services_by_date

BBOX = (47.32, 47.43, 8.44, 8.62)
TIME_WINDOW = (7 * 3600 + 30 * 60, 9 * 3600 + 35 * 60)


def seconds(value):
    hour, minute, second = map(int, value.split(':'))
    return 3600 * hour + 60 * minute + second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default='2023-10-25')
    parser.add_argument('--output', type=Path, default=HERE / 'city_schedule.json.gz')
    args = parser.parse_args()
    static = ROOT / 'data/swiss_gtfs'
    stops = {}
    with (static / 'stops.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            uic = row['stop_id'].split(':')[0]
            if not uic.isdecimal():
                continue
            lat, lon = float(row['stop_lat']), float(row['stop_lon'])
            if not BBOX[0] <= lat <= BBOX[1] or not BBOX[2] <= lon <= BBOX[3]:
                continue
            if uic not in stops or row['stop_id'] == uic:
                stops[uic] = {'name': row['stop_name'], 'latitude': lat, 'longitude': lon}
    with (static / 'routes.txt').open(encoding='utf-8-sig') as handle:
        routes = {row['route_id']: row for row in csv.DictReader(handle)
                  if int(row['route_type']) in (0, 3, 11) or 700 <= int(row['route_type']) < 800
                  or 900 <= int(row['route_type']) < 1000}
    with (static / 'trips.txt').open(encoding='utf-8-sig') as handle:
        trips = {row['trip_id']: row for row in csv.DictReader(handle) if row['route_id'] in routes}
    active = scheduled_services_by_date(static, [args.date], {trip['service_id'] for trip in trips.values()})[args.date]
    trips = {tid: trip for tid, trip in trips.items() if trip['service_id'] in active}
    sequence = defaultdict(list)
    # Full static stream avoids importing the historical truncated graph's trip list.
    with (static / 'stop_times.txt').open(encoding='utf-8-sig') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        columns = {name: header.index(name) for name in header}
        for row in reader:
            tid = row[columns['trip_id']]
            if tid not in trips:
                continue
            dep = seconds(row[columns['departure_time']])
            if not TIME_WINDOW[0] <= dep <= TIME_WINDOW[1] + 1200:
                continue
            sequence[tid].append({key: row[index] for key, index in columns.items()})
    connections = []
    boundary = 0
    for tid, stop_times in sequence.items():
        stop_times.sort(key=lambda row: int(row['stop_sequence']))
        trip = trips[tid]
        route = routes[trip['route_id']]
        for a, b in zip(stop_times, stop_times[1:]):
            dep, arr = seconds(a['departure_time']), seconds(b['arrival_time'])
            origin, target = a['stop_id'].split(':')[0], b['stop_id'].split(':')[0]
            if not TIME_WINDOW[0] <= dep < TIME_WINDOW[1]:
                continue
            if origin not in stops or target not in stops:
                boundary += int(origin in stops or target in stops)
                continue
            connections.append({'trip_id': tid, 'service_id': trip['service_id'], 'route_id': trip['route_id'],
                                'route_name': route['route_short_name'], 'agency_id': route['agency_id'],
                                'mode': 'tram' if int(route['route_type']) == 0 or 900 <= int(route['route_type']) < 1000 else 'bus',
                                'vehicle_number': trip['trip_short_name'], 'dep_stop': origin, 'arr_stop': target,
                                'dep_platform': a['stop_id'], 'arr_platform': b['stop_id'],
                                'dep_sequence': int(a['stop_sequence']), 'arr_sequence': int(b['stop_sequence']),
                                'scheduled_departure_seconds': dep, 'scheduled_arrival_seconds': arr,
                                'pickup_allowed': a['pickup_type'] != '1', 'dropoff_allowed': b['drop_off_type'] != '1'})
    connections.sort(key=lambda row: (row['scheduled_departure_seconds'], row['trip_id'], row['dep_sequence']))
    for i, connection in enumerate(connections):
        connection['id'] = i
    output = {'date': args.date, 'scope': 'All active static-GTFS bus/tram adjacency with both endpoints within the declared '
              'Zurich rectangle and departure window. Boundary-crossing services are counted separately.',
              'bounding_box_lat_lon': BBOX, 'time_window_seconds': TIME_WINDOW,
              'static_feed': (static / 'feed_info.txt').read_text().strip(),
              'boundary_connections': boundary, 'stops': stops, 'connections': connections}
    with gzip.open(args.output, 'wt', encoding='utf-8') as handle:
        json.dump(output, handle, ensure_ascii=False, separators=(',', ':'))
    stop_path = args.output.with_name(args.output.name.removesuffix('.json.gz') + '_stops.json')
    stop_path.write_text(json.dumps({'stops': sorted(stops), 'time_window_seconds': TIME_WINDOW}))
    print(args.date, 'stops', len(stops), 'trips', len({c['trip_id'] for c in connections}),
          'connections', len(connections), 'boundary', boundary, 'output', args.output)


if __name__ == '__main__':
    main()
