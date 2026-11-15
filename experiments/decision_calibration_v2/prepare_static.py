"""Filter an official GTFS ZIP into a separate server-side replay directory.

No actual endpoint or RT message is read. The original dated_connections
routine supplies service activation, physical adjacency and the 07:30–09:35
window. Full platform IDs and original sequence numbers remain available.
"""
import argparse
import csv
import gzip
from io import TextIOWrapper
import json
from pathlib import Path, PurePosixPath
from time import perf_counter, process_time
from zipfile import ZipFile

from experiments.reconstruction_data_audit import build_dated_replay as original


def _member(archive, name):
    matches = [item for item in archive.infolist() if PurePosixPath(item.filename).name == name]
    if len(matches) != 1:
        raise ValueError(f'Expected one official GTFS member {name}, found {len(matches)}')
    return matches[0]


def _rows(archive, name):
    with archive.open(_member(archive, name)) as binary:
        with TextIOWrapper(binary, encoding='utf-8-sig', newline='') as handle:
            yield from csv.DictReader(handle)


def _filter(archive, name, path, keep):
    """Stream the source once and retain its actual column schema."""
    total = retained = 0
    with archive.open(_member(archive, name)) as binary:
        with TextIOWrapper(binary, encoding='utf-8-sig', newline='') as handle, path.open('w', newline='') as output:
            reader = csv.DictReader(handle)
            writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                if keep(row):
                    writer.writerow(row)
                    retained += 1
    return dict(source_rows=total, retained_rows=retained, output_bytes=path.stat().st_size)


def extract_static(zip_path, output_dir):
    """Return the isolated runtime paths and a small public-source audit."""
    tick, cpu = perf_counter(), process_time()
    zip_path, output_dir = Path(zip_path), Path(output_dir).resolve()
    if output_dir == original.ROOT.resolve():
        raise ValueError('Static preparation output must be separate from original project data')
    static, regional = output_dir/'data/swiss_gtfs', output_dir/'data/zurich_gtfs'
    static.mkdir(parents=True, exist_ok=True)
    regional.mkdir(parents=True, exist_ok=True)
    counts = {}
    with ZipFile(zip_path) as archive:
        routes = {row['route_id'] for row in _rows(archive,'routes.txt')
                  if row['agency_id'] in original.AGENCIES}
        trips = {row['trip_id']:row for row in _rows(archive,'trips.txt')
                 if row['route_id'] in routes and row['trip_short_name']}
        encountered = set()

        def regional_stop(row):
            keep = row['trip_id'] in trips and row['stop_id'].split(':')[0] in original.STATIONS
            if keep:
                encountered.add(row['trip_id'])
            return keep

        counts['stop_times.txt'] = _filter(archive,'stop_times.txt',regional/'stop_times.txt',regional_stop)
        trips = {trip:row for trip,row in trips.items() if trip in encountered}
        services = {row['service_id'] for row in trips.values()}
        used_routes = {row['route_id'] for row in trips.values()}
        counts['trips.txt'] = _filter(archive,'trips.txt',regional/'trips.txt',lambda row:row['trip_id'] in trips)
        counts['routes.txt'] = _filter(archive,'routes.txt',static/'routes.txt',lambda row:row['route_id'] in used_routes)
        for name in ('calendar.txt','calendar_dates.txt'):
            counts[name] = _filter(archive,name,static/name,lambda row:row['service_id'] in services)
        stops = {row['stop_id']:row for row in _rows(archive,'stops.txt')}
        selected_stops = {key for key in stops if key.split(':')[0] in original.STATIONS}
        selected_stops.update(stops[key]['parent_station'] for key in tuple(selected_stops)
                              if stops[key].get('parent_station'))
        counts['stops.txt'] = _filter(archive,'stops.txt',static/'stops.txt',lambda row:row['stop_id'] in selected_stops)
        counts['transfers.txt'] = _filter(archive,'transfers.txt',static/'transfers.txt',
            lambda row:row['from_stop_id'] in selected_stops and row['to_stop_id'] in selected_stops)
        info = list(_rows(archive,'feed_info.txt'))
        counts['feed_info.txt'] = _filter(archive,'feed_info.txt',static/'feed_info.txt',lambda row:True)
        (regional/'feed_info.txt').write_bytes((static/'feed_info.txt').read_bytes())
        source_members = {name:_member(archive,name).filename for name in counts}
    audit = dict(source_zip=str(zip_path), source_zip_bytes=zip_path.stat().st_size,
        runtime_root=str(output_dir), static_dir=str(static), regional_dir=str(regional),
        feed_info=info, original_stations=sorted(original.STATIONS), agencies=original.AGENCIES,
        row_counts=counts, source_members=source_members,
        original_connection_window=[original.START,original.END],
        filtering='Only SBB 11 named trains and original 16 stations; original sequence IDs and full stop IDs retained. Full-day regional rows retained; original dated_connections applies the time window and physical adjacency.',
        target_actual_or_rt_read=False, original_project_files_changed=False,
        cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick))
    (output_dir/'static_preparation.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


def dated_connections_from_static(dates, runtime_root):
    """Reuse the frozen builder on isolated static files; attach platform IDs."""
    runtime_root = Path(runtime_root).resolve()
    with (runtime_root/'data/swiss_gtfs/feed_info.txt').open(encoding='utf-8-sig') as handle:
        feed = list(csv.DictReader(handle))
    for day in dates:
        compact = day.replace('-','')
        if not any(row['feed_start_date'] <= compact <= row['feed_end_date'] for row in feed):
            raise ValueError(f'Frozen service date {day} is outside this static feed validity')
    old_root = original.ROOT
    try:
        original.ROOT = runtime_root
        stops, by_date = original.dated_connections(dates)
    finally:
        original.ROOT = old_root
    wanted = {(row['trip_id'],row[side+'_sequence']) for rows in by_date.values() for row in rows for side in ('dep','arr')}
    platforms = {}
    with (runtime_root/'data/zurich_gtfs/stop_times.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            key = row['trip_id'],int(row['stop_sequence'])
            if key in wanted:
                platforms[key] = row['stop_id']
    for rows in by_date.values():
        for row in rows:
            row['dep_platform_id'] = platforms[row['trip_id'],row['dep_sequence']]
            row['arr_platform_id'] = platforms[row['trip_id'],row['arr_sequence']]
    return stops, by_date


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dates', nargs='*', default=[])
    args = parser.parse_args()
    result = extract_static(args.zip,args.output)
    if args.dates:
        stops, connections = dated_connections_from_static(args.dates,args.output)
        with gzip.open(args.output/'scheduled.json.gz','wt') as handle:
            json.dump(dict(stops=stops,dates={day:dict(connections=rows) for day,rows in connections.items()},
                scope='Public schedules only; no outcomes attached'),handle)
        result['scheduled_counts'] = {day:len(rows) for day,rows in connections.items()}
    print(json.dumps(result))
