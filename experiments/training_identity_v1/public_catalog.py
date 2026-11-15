"""Full-day active SBB identities and endpoints at the frozen16 stations."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

from experiments.reconstruction_data_audit.build_dated_replay import AGENCIES, STATIONS, clock_seconds
from src.gtfs_rt_parser import load_route_metadata, scheduled_services_by_date


def build_public_catalog(runtime_root, dates):
    """Read static CSVs only; retain active trips even without a graph edge."""
    runtime_root = Path(runtime_root)
    static, regional = runtime_root/'data/swiss_gtfs', runtime_root/'data/zurich_gtfs'
    routes = load_route_metadata(static)
    with (regional/'trips.txt').open(encoding='utf-8-sig') as handle:
        trips = {row['trip_id']:row for row in csv.DictReader(handle)
                 if routes[row['route_id']]['agency_id'] in AGENCIES and row['trip_short_name']}
    active = scheduled_services_by_date(static, dates, {row['service_id'] for row in trips.values()})
    relevant = {trip for trip,row in trips.items() if any(row['service_id'] in active[day] for day in dates)}
    times = defaultdict(list)
    with (regional/'stop_times.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['trip_id'] in relevant and row['stop_id'].split(':')[0] in STATIONS:
                times[row['trip_id']].append(row)
    output = {}
    for day in dates:
        directory, endpoints = [], []
        for trip in sorted(times):
            row = trips[trip]
            if row['service_id'] not in active[day]:
                continue
            identity = dict(trip_id=trip,operator_id=AGENCIES[routes[row['route_id']]['agency_id']],
                            vehicle_number=row['trip_short_name'],route_id=row['route_id'],
                            service_id=row['service_id'])
            directory.append(identity)
            for stop in sorted(times[trip],key=lambda item:int(item['stop_sequence'])):
                for kind in ('arrival','departure'):
                    value = stop[kind+'_time']
                    if not value:
                        continue
                    endpoints.append(dict(**identity,stop_id=stop['stop_id'].split(':')[0],
                        platform_id=stop['stop_id'],stop_sequence=int(stop['stop_sequence']),
                        kind=kind,scheduled_seconds=clock_seconds(value),
                        pickup_allowed=stop['pickup_type']=='0',dropoff_allowed=stop['drop_off_type']=='0',
                        pickup_type=stop['pickup_type'],drop_off_type=stop['drop_off_type']))
        output[day] = dict(trips=directory,endpoints=endpoints)
    return dict(dates=output,source_runtime=str(runtime_root),
                scope='Full-day active SBB named trips touching any of the original16 stations; all regional planned endpoints, without adjacency, pickup/dropoff or07:30–09:35 edge filtering',
                actual_or_rt_or_probabilities_or_losses_read=False)


def write_public_catalog(runtime_root, dates, output):
    value = build_public_catalog(runtime_root, dates)
    output = Path(output)
    with gzip.open(output,'wt') as handle:
        json.dump(value,handle,separators=(',',':'))
    return dict(file=str(output),bytes=output.stat().st_size,
                counts={day:dict(trips=len(row['trips']),endpoints=len(row['endpoints']))
                        for day,row in value['dates'].items()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--dates',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(write_public_catalog(args.runtime_root,args.dates,args.output),indent=2))
