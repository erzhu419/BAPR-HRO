"""Decode source-correct messages only after dates and public menus are frozen."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import tarfile
from time import perf_counter, process_time
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2 as pb

from experiments.decision_calibration_v2.fetch_messages import attach_platform_updates
from experiments.natural_gap_v2.rt_clock import civil_service_seconds
from experiments.reconstruction_v34.messages import SAMPLE_SECONDS, WINDOW, snapshot_frames
from experiments.training_identity_v1.audit import audit_day, canonical_frames, decoder_maps


def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as handle:
        return json.load(handle)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(path,'wt') if str(path).endswith('.gz') else path.open('w')) as handle:
        json.dump(value,handle,allow_nan=False,indent=None if str(path).endswith('.gz') else 2,
                  separators=(',',':') if str(path).endswith('.gz') else None)
        handle.write('\n')


def decode_day(day, connections, evidence, archive):
    tick, cpu = perf_counter(), process_time()
    trips, canonical = decoder_maps(evidence)
    midnight = int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('Europe/Zurich')).timestamp())
    convert = lambda epoch: civil_service_seconds(epoch,day)
    frames, captures, members = [], [], 0
    with tarfile.open(archive,'r|bz2') as source:
        for member in source:
            if not member.name.endswith('.gtfsrt'):
                continue
            members += 1
            captured = convert(datetime.fromisoformat(Path(member.name).stem).timestamp())
            if not WINDOW[0] <= captured < WINDOW[1]+60 or captured % SAMPLE_SECONDS >= 60:
                continue
            captures.append(captured)
            feed = pb.FeedMessage.FromString(source.extractfile(member).read())
            matching = snapshot_frames(feed,day,midnight,captured,member.name,trips,epoch_to_service=convert)
            frames.extend(attach_platform_updates(matching,feed,trips,connections))
    frames.sort(key=lambda f:(f['receipt_seconds'],f['capture_seconds'],f['archive_member'],f['entity_id']))
    canonical_frames(frames,canonical)
    return dict(day=day,frames=frames,source=str(archive),
        source_kind='User-approved third-party traines research capture of official Swiss GTFS-RT feed',
        summary=dict(archive_members=members,sampled_snapshots=len(captures),
            source_parse_complete=True,identity_queries=evidence['summary']['queries'],
            identity_endpoint_statuses=evidence['summary']['statuses'],
            mapped_endpoint_count=evidence['summary']['matched_endpoints'],
            identity_mapping_complete=evidence['summary']['matched_endpoints']==evidence['summary']['queries'],
            matching_bins=sorted({value//600*600 for value in captures}),
            matched_trip_messages=len(frames),distinct_trips=len({f['trip_id'] for f in frames}),
            capture_range=[min(captures),max(captures)] if captures else None,
            endpoint_update_statuses=dict(Counter(u[2] for f in frames for u in f['endpoint_updates'])),
            published_stop_id_kinds=dict(Counter(
                'missing' if not u['published_stop_id'] else
                'platform' if ':' in u['published_stop_id'] else 'station_only'
                for f in frames for u in f['platform_stop_updates'])),
            extraction_wall_seconds=perf_counter()-tick,extraction_cpu_seconds=process_time()-cpu),
        information='Original capture sampling and max(capture,header,TripUpdate) availability in common Swiss civil service seconds. '
            'Raw numeric values unchanged; absolute epoch clocks decoded consistently with GTFS and actual labels. '
            'Platform fields are audit evidence only and are not consumed by the fixed model.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    directory = args.directory
    manifest = read(directory/'date_manifest.json')
    selection = read(directory/'selection.json')
    if not manifest['dates_frozen'] or not selection.get('frozen_at_utc'):
        raise ValueError('Freeze both dates and public menus before target message decoding')
    dates = manifest['selected_dates']
    scheduled = read(directory/'scheduled.json.gz')
    connections = {day:scheduled['dates'][day]['connections'] for day in dates}
    mapping_path, receipt_path = directory/'rt_public_identity_mapping.json.gz',directory/'rt_mapping_receipt.json'
    if receipt_path.exists():
        # The first mapping timestamp is evidence; resuming cannot overwrite it.
        receipt = read(receipt_path)
        if (receipt['dates'] != dates or receipt['date_freeze_utc'] != manifest['frozen_at_utc']
                or receipt['menu_freeze_utc'] != selection['frozen_at_utc']):
            raise ValueError('Source-processing resume does not match the frozen cohort')
        evidence = read(mapping_path)['days']
    else:
        catalog = read(directory/'active_public_endpoints.json.gz')
        evidence = {day:audit_day(day,connections[day],catalog['dates'][day],catalog['dates'][day]) for day in dates}
        write(mapping_path,dict(days=evidence))
        write(receipt_path,dict(
            written_at_utc=datetime.now(timezone.utc).isoformat(),dates=dates,
            date_freeze_utc=manifest['frozen_at_utc'],menu_freeze_utc=selection['frozen_at_utc'],
            all_public_mappings_written_before_rt_decode=True,
            source_version_by_date=catalog['source_version_by_date'],
            statuses={day:evidence[day]['summary'] for day in dates},
            unknown_identity_literal_id_fallback=False,forecast_or_actual_values_used_for_mapping=False))
    print('all source mappings frozen before decode',flush=True)
    decode_started_at_utc = datetime.now(timezone.utc).isoformat()
    messages, errors = {}, []
    for day in dates:
        output, marker = directory/'messages'/f'{day}.json.gz',directory/'messages'/f'{day}.audit.json'
        if output.exists() and marker.exists():
            completed = read(marker)
            if not completed['output_complete'] or not completed['summary']['source_parse_complete']:
                continue
            value = read(output)
            if value['day'] != day or value['summary'] != completed['summary']:
                raise ValueError('Completed per-day message data disagree with their receipt')
            messages[day] = value
    resumed = sorted(messages)
    attempts_path = directory/'rt_decode_attempts.json'
    attempts = read(attempts_path) if attempts_path.exists() else dict(attempts=[])
    attempt = dict(source_processing_started_at_utc=decode_started_at_utc,workers=args.workers,
                   resumed_complete_dates=resumed,pending_dates=[day for day in dates if day not in messages])
    attempts['attempts'].append(attempt)
    write(attempts_path,attempts)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(decode_day,day,connections[day],evidence[day],
            directory/'cache/rt'/f'{day}.tar.bz2'):day for day in dates if day not in messages}
        for future in as_completed(jobs):
            day = jobs[future]
            try:
                result = future.result()
                capture = read(directory/'rt_clocks'/f'{day}.json')
                if result['summary']['matching_bins'] != capture['matching_bins']:
                    raise ValueError('Decoded capture bins differ from the clock-only frozen prescreen')
                messages[day] = result
                write(directory/'messages'/f'{day}.json.gz',result)
                write(directory/'messages'/f'{day}.audit.json',dict(date=day,output_complete=True,summary=result['summary']))
                print(json.dumps(dict(date=day,**result['summary'])),flush=True)
            except Exception as error:
                errors.append(dict(date=day,error=repr(error)))
                print(json.dumps(errors[-1]),flush=True)
    attempt.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),errors=errors)
    write(attempts_path,attempts)
    write(directory/'rt_decode_summary.json',dict(completed_at_utc=attempt['completed_at_utc'],
        source_processing_started_at_utc=attempts['attempts'][0]['source_processing_started_at_utc'],
        final_attempt_started_at_utc=decode_started_at_utc,source_processing_attempts_file='rt_decode_attempts.json',
        resumed_complete_dates=resumed,newly_decoded_dates=[day for day in dates if day not in resumed],
        date_freeze_utc=manifest['frozen_at_utc'],menu_freeze_utc=selection['frozen_at_utc'],
        registered_dates=dates,days={day:messages[day]['summary'] for day in dates if day in messages},
        errors=errors,raw_forecast_values_modified=False,clock_conversion='Europe/Zurich civil service seconds',
        original_model_menu_parameters_modified=False))
    if errors:
        raise RuntimeError('All available dates processed; explicit decode errors retained')
    write(directory/'messages.json.gz',dict(dates={day:messages[day] for day in dates},
        sample_seconds=SAMPLE_SECONDS,capture_window_seconds=list(WINDOW),timezone='Europe/Zurich',
        availability='max(capture,header,individual TripUpdate) in Swiss civil service seconds',
        information_contract='Forecasts and service notices remain public predictions; absent messages are not nonoccurrence.',
        endpoint_update_fields=['source_connection_id','kind','status','predicted_seconds'],
        trip_status_fields=['trip_id','status'],scope='All frozen natural_gap_v2 dates, correct source-native identities'))


if __name__ == '__main__':
    main()
