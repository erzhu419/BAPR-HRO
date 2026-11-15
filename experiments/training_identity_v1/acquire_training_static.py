"""Acquire the metadata-selected September release on the server only."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen
from zipfile import ZipFile

from experiments.decision_calibration_v2.prepare_static import extract_static, dated_connections_from_static
from experiments.training_identity_v1.public_catalog import write_public_catalog
from src.gtfs_rt_parser import scheduled_services_by_date

DATES = ('2023-10-01', '2023-10-02')
VERSION = '2023-09-27'
BLOCK_BYTES, WORKERS = 1024**2, 32


def save_gzip(path, value):
    with gzip.open(path, 'wt') as handle:
        json.dump(value, handle, separators=(',', ':'))


def export_old_literals(runtime, query_path, output):
    queries = json.loads(query_path.read_text())
    regional, static = runtime/'data/zurich_gtfs', runtime/'data/swiss_gtfs'
    with (regional/'trips.txt').open(encoding='utf-8-sig') as handle:
        trips = {row['trip_id']:row for row in csv.DictReader(handle)}
    active = scheduled_services_by_date(static, DATES, {row['service_id'] for row in trips.values()})
    wanted = {trip for values in queries.values() for trip in values}
    times = defaultdict(list)
    columns = ('stop_id','stop_sequence','arrival_time','departure_time','pickup_type','drop_off_type')
    with (regional/'stop_times.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['trip_id'] in wanted:
                times[row['trip_id']].append({key:row[key] for key in columns})
    days = {}
    for day in DATES:
        rows = []
        for trip in queries[day]:
            metadata = trips.get(trip)
            rows.append(dict(literal_trip_id=trip,
                status='found_in_regional_static' if metadata else 'absent_from_filtered_static_unknown',
                metadata=metadata,
                active_on_date=metadata['service_id'] in active[day] if metadata else None,
                regional_stop_times=sorted(times.get(trip,[]),key=lambda row:int(row['stop_sequence']))))
        days[day] = rows
    result = dict(static_release=VERSION, feed_scope='SBB named trips touching the original16 stations; full-day regional stop_times, no morning-window filtering',
                  missing_interpretation='Absence from these filtered files remains unknown; it is not evidence of an incorrect identity',
                  actual_or_rt_or_probabilities_or_losses_read=False, days=days)
    save_gzip(output, result)
    return {day:dict(queries=len(rows),found=sum(row['metadata'] is not None for row in rows),
                    active=sum(row['active_on_date'] is True for row in rows),
                    inactive=sum(row['active_on_date'] is False for row in rows),
                    missing_unknown=sum(row['metadata'] is None for row in rows),
                    stop_time_rows=sum(len(row['regional_stop_times']) for row in rows)) for day,rows in days.items()}


def acquire(runtime_root, metadata_path, query_path):
    metadata = json.loads(metadata_path.read_text())
    url, size = metadata['url'], metadata['archive_bytes']
    runtime_root.mkdir(parents=True, exist_ok=True)
    cache = runtime_root/'cache'
    cache.mkdir(exist_ok=True)
    path = cache/url.rsplit('/',1)[1]
    started = datetime.now(timezone.utc).isoformat()
    tick, downloaded = perf_counter(), 0
    unverified_reused_bytes = 0
    integrity = dict(checked=False,method='ZipFile.testzip',bad_member=None)
    reused = path.exists() and path.stat().st_size == size
    if not reused:
        temporary = path.with_suffix('.range-download')
        checkpoint = runtime_root/'workers8_checkpoint.json'
        unverified_offsets = set()
        if temporary.exists() and checkpoint.exists():
            unverified_offsets = set(json.loads(checkpoint.read_text())['unverified_offsets'])
            if temporary.stat().st_size != size:
                raise ValueError('Interrupted archive has a different size')
            unverified_reused_bytes = sum(min(BLOCK_BYTES,size-start) for start in unverified_offsets)
            fd = os.open(temporary,os.O_WRONLY)
        else:
            fd = os.open(temporary, os.O_CREAT|os.O_TRUNC|os.O_WRONLY, 0o644)
            os.ftruncate(fd, size)

        def fetch(start):
            end = min(start+BLOCK_BYTES,size)-1
            with urlopen(Request(url,headers={'Range':f'bytes={start}-{end}'}),timeout=90) as response:
                if response.status != 206 or response.headers['Content-Range'] != f'bytes {start}-{end}/{size}':
                    raise ValueError('Official archive did not honor the requested range')
                raw = response.read()
            if len(raw) != end-start+1:
                raise ValueError('Incomplete static archive range')
            os.pwrite(fd,raw,start)
            return len(raw)

        try:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for future in as_completed([pool.submit(fetch,start) for start in range(0,size,BLOCK_BYTES)
                                            if start not in unverified_offsets]):
                    downloaded += future.result()
                    if downloaded//(8*BLOCK_BYTES) != (downloaded-BLOCK_BYTES)//(8*BLOCK_BYTES):
                        print(json.dumps(dict(stage='download',network_bytes=downloaded,
                            unverified_reused_bytes=unverified_reused_bytes,total=size,seconds=perf_counter()-tick)),flush=True)
        finally:
            os.close(fd)
        validation_tick=perf_counter()
        with ZipFile(temporary) as archive:
            bad_member=archive.testzip()
        integrity=dict(checked=True,method='ZipFile.testzip',bad_member=bad_member,
                       seconds=perf_counter()-validation_tick)
        print(json.dumps(dict(stage='zip_integrity',**integrity)),flush=True)
        if bad_member is not None:
            raise ValueError(f'ZIP validation failed for {bad_member}; no static output produced')
        temporary.rename(path)
    output = runtime_root/'inputs'/f'static_{VERSION}'
    preparation = extract_static(path,output)
    stops, dates = dated_connections_from_static(DATES,output)
    schedule_path = runtime_root/'training_static_schedule.json.gz'
    save_gzip(schedule_path,dict(stops=stops,dates={day:dict(connections=rows) for day,rows in dates.items()},
                               scope='Public schedules only; no outcomes attached'))
    literal_path = runtime_root/'old_literal_trips_by_date.json.gz'
    literals = export_old_literals(output,query_path,literal_path)
    public_catalog = write_public_catalog(output,DATES,runtime_root/'active_public_endpoints.json.gz')
    receipt = dict(started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
        static_version=VERSION,dates=DATES,url=url,source_zip=str(path),archive_bytes=size,
        downloaded_bytes=downloaded,reused_complete_zip=reused,workers=WORKERS,
        unverified_reused_bytes_before_zip_check=unverified_reused_bytes,zip_integrity=integrity,
        seconds=perf_counter()-tick,runtime_root=str(output),static_preparation=preparation,
        scheduled_counts={day:len(rows) for day,rows in dates.items()},
        scheduled_trips={day:len({row['trip_id'] for row in rows}) for day,rows in dates.items()},
        schedule_file=str(schedule_path),schedule_bytes=schedule_path.stat().st_size,
        old_literal_file=str(literal_path),old_literal_bytes=literal_path.stat().st_size,
        old_literal_summary=literals,old_data_or_model_changed=False,target_actual_or_rt_read=False)
    receipt['active_public_catalog'] = public_catalog
    (runtime_root/'static_acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:value for key,value in receipt.items() if key!='static_preparation'},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--old-trip-ids',type=Path,required=True)
    args=parser.parse_args()
    acquire(args.runtime_root,args.metadata,args.old_trip_ids)
