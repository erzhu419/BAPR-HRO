"""Extract the frozen V34 public-message panel from local GTFS-RT archives.

StopTimeEvent values remain predictions, never actual vehicle events. Each
TripUpdate has its own conservative availability clock; an update cannot be
made visible early by another entity in the same feed snapshot.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import argparse
import gzip
import json
from pathlib import Path
import sys
import tarfile
from time import perf_counter
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2 as pb

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_data_audit.build_dated_replay import dated_connections

DATES = tuple(f'2023-10-{day:02}' for day in (*range(12, 29), 30, 31))
WINDOW = (7 * 3600, 9 * 3600 + 40 * 60)
SAMPLE_SECONDS = 600


def endpoint_map(connections):
    trips = defaultdict(dict)
    for c in connections:
        for kind, side in (('departure', 'dep'), ('arrival', 'arr')):
            trips[c['trip_id']][c[f'{side}_sequence'], kind] = (
                c['id'], c[f'{side}_stop'], c[f'scheduled_{kind}_seconds'])
    return trips


def snapshot_frames(feed, day, midnight, captured, member, trips):
    header = int(feed.header.timestamp) - midnight if feed.header.HasField('timestamp') else None
    output = []
    for entity in feed.entity:
        if not entity.HasField('trip_update'):
            continue
        update = entity.trip_update
        trip = update.trip.trip_id
        if trip not in trips or update.trip.start_date != day.replace('-', ''):
            continue
        measured = int(update.timestamp) - midnight if update.HasField('timestamp') else None
        receipt = max(x for x in (captured, header, measured) if x is not None)
        status = pb.TripDescriptor.ScheduleRelationship.Name(update.trip.schedule_relationship)
        events = []
        if status == 'SCHEDULED':
            for stop in update.stop_time_update:
                if not stop.HasField('stop_sequence'):
                    continue
                stop_status = pb.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(stop.schedule_relationship)
                for kind in ('departure', 'arrival'):
                    key = stop.stop_sequence, kind
                    if key not in trips[trip]:
                        continue
                    cid, uic, scheduled = trips[trip][key]
                    if stop.stop_id and stop.stop_id.split(':')[0] != uic:
                        continue
                    if stop_status in ('NO_DATA', 'SKIPPED'):
                        events.append([cid, kind, stop_status, None])
                    elif stop.HasField(kind):
                        prediction = getattr(stop, kind)
                        if prediction.HasField('time'):
                            clock = int(prediction.time) - midnight
                        elif prediction.HasField('delay'):
                            clock = scheduled + prediction.delay
                        else:
                            continue
                        events.append([cid, kind, 'FORECAST', clock])
        output.append(dict(archive_member=member, entity_id=entity.id, trip_id=trip,
            capture_seconds=captured, header_seconds=header, update_seconds=measured,
            receipt_seconds=receipt, trip_statuses=[[trip, status]], endpoint_updates=events))
    return output


def extract(day, connections):
    started = perf_counter()
    target = HERE / 'message_days' / f'{day}.json.gz'
    if target.exists():
        with gzip.open(target, 'rt') as handle:
            result = json.load(handle)
        result['summary']['trip_status_messages'] = dict(Counter(
            s[1] for f in result['frames'] for s in f['trip_statuses']))
        return day, result, True
    trips = endpoint_map(connections)
    midnight = int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('Europe/Zurich')).timestamp())
    source = ROOT / f'data/swiss_rt/gtfs-rt/{day}.tar.bz2'
    frames, captures = [], []
    members = 0
    with tarfile.open(source, 'r|bz2') as archive:
        for member in archive:
            if not member.name.endswith('.gtfsrt'):
                continue
            members += 1
            captured = int(datetime.fromisoformat(Path(member.name).stem).timestamp()) - midnight
            if not WINDOW[0] <= captured < WINDOW[1] + 60 or captured % SAMPLE_SECONDS >= 60:
                continue
            captures.append(captured)
            feed = pb.FeedMessage.FromString(archive.extractfile(member).read())
            frames.extend(snapshot_frames(feed, day, midnight, captured, member.name, trips))
    frames.sort(key=lambda f: (f['receipt_seconds'], f['capture_seconds'], f['archive_member'], f['entity_id']))
    updates = Counter(u[2] for f in frames for u in f['endpoint_updates'])
    delayed = [f for f in frames if f['receipt_seconds'] > f['capture_seconds']]
    update_delayed = [f for f in frames if f['update_seconds'] is not None and
                      f['update_seconds'] > max(f['capture_seconds'], f['header_seconds'] or f['capture_seconds'])]
    summary = dict(scheduled_trips=len(trips), scheduled_connections=len(connections),
        archive_members=members, sampled_snapshots=len(captures), matched_trip_messages=len(frames),
        distinct_trips=len({f['trip_id'] for f in frames}), endpoint_updates=dict(updates),
        trip_status_messages=dict(Counter(s[1] for f in frames for s in f['trip_statuses'])),
        capture_range=[min(captures), max(captures)] if captures else None,
        receipt_range=[frames[0]['receipt_seconds'], frames[-1]['receipt_seconds']] if frames else None,
        messages_later_than_capture=len(delayed),
        messages_later_than_legacy_receipt=len(update_delayed),
        messages_with_update_timestamp=sum(f['update_seconds'] is not None for f in frames),
        maximum_delay_from_capture=max((f['receipt_seconds']-f['capture_seconds'] for f in frames), default=0),
        archive_bytes=source.stat().st_size, extraction_seconds=perf_counter()-started)
    result = dict(day=day, source=str(source.relative_to(ROOT)), summary=summary, frames=frames)
    target.parent.mkdir(exist_ok=True)
    with gzip.open(target, 'wt') as handle:
        json.dump(result, handle, separators=(',', ':'))
    return day, result, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', choices=DATES, help='Probe or resume one frozen date')
    args = parser.parse_args()
    primary = tuple(f'2023-10-{d}' for d in (25, 26, 27, 28, 30, 31))
    dates = (args.date,) if args.date else primary + tuple(d for d in DATES if d not in primary)
    _, schedules = dated_connections(dates)
    results = {}
    with ProcessPoolExecutor(max_workers=min(2, len(dates))) as pool:
        jobs = [pool.submit(extract, day, schedules[day]) for day in dates]
        for future in as_completed(jobs):
            day, row, reused = future.result()
            results[day] = row
            print(day, 'reused' if reused else 'extracted', json.dumps(row['summary']), flush=True)
    if not args.date:
        output = dict(scope='Frozen 19-day 16-station public prediction panel; no actual outcomes read.',
            dates={day: results[day] for day in DATES}, sample_seconds=SAMPLE_SECONDS,
            capture_window_seconds=list(WINDOW), timezone='Europe/Zurich',
            availability='max(archive capture, feed header, individual TripUpdate timestamp)',
            information_contract='Numeric fields are forecasts. CANCELED, SKIPPED and NO_DATA are message statuses, '
                'not actual service truth. Absence of a message supplies no nonoccurrence evidence. '
                'Every sampled matching entity is retained, including unchanged predictions.',
            endpoint_update_fields=['source_connection_id', 'kind', 'status', 'predicted_seconds'],
            trip_status_fields=['trip_id', 'status'])
        with gzip.open(HERE / 'messages.json.gz', 'wt') as handle:
            json.dump(output, handle, separators=(',', ':'))
        summaries = {day: results[day]['summary'] for day in DATES}
        (HERE / 'messages_summary.json').write_text(json.dumps(dict(
            primary_dates=primary, days=summaries,
            all_dates_total_trip_messages=sum(s['matched_trip_messages'] for s in summaries.values()),
            all_dates_later_than_legacy_receipt=sum(s['messages_later_than_legacy_receipt'] for s in summaries.values()),
            all_dates_with_update_timestamp=sum(s['messages_with_update_timestamp'] for s in summaries.values())), indent=2)+'\n')


if __name__ == '__main__':
    main()
