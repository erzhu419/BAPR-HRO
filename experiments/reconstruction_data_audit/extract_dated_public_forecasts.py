"""Extract date-specific, time-stamped forecast and status receipts for replay."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import gzip
import json
from pathlib import Path
import tarfile
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2 as pb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAMPLE_SECONDS = 10 * 60
WINDOW_START, WINDOW_END = 7 * 3600, 9 * 3600 + 40 * 60


def scan(row):
    day, day_data = row
    day_yyyymmdd = day.replace('-', '')
    midnight = int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('Europe/Zurich')).timestamp())
    trips = defaultdict(dict)
    for connection in day_data['connections']:
        for kind, side in (('departure', 'dep'), ('arrival', 'arr')):
            trips[connection['trip_id']][connection[f'{side}_sequence'], kind] = (
                connection['id'], connection[f'{side}_stop'], connection[f'scheduled_{kind}_seconds'])
    frames = []
    with tarfile.open(ROOT / f'data/swiss_rt/gtfs-rt/{day}.tar.bz2', 'r|bz2') as archive:
        for member in archive:
            if not member.name.endswith('.gtfsrt'):
                continue
            captured = int(datetime.fromisoformat(Path(member.name).stem).timestamp()) - midnight
            if not WINDOW_START <= captured < WINDOW_END + 60 or captured % SAMPLE_SECONDS >= 60:
                continue
            snapshot = pb.FeedMessage.FromString(archive.extractfile(member).read())
            header = snapshot.header.timestamp - midnight if snapshot.header.HasField('timestamp') else None
            receipt = max(captured, header) if header is not None else captured
            trip_statuses = []
            events = []
            for entity in snapshot.entity:
                if not entity.HasField('trip_update'):
                    continue
                update = entity.trip_update
                tid = update.trip.trip_id
                if tid not in trips or update.trip.start_date != day_yyyymmdd:
                    continue
                status = pb.TripDescriptor.ScheduleRelationship.Name(update.trip.schedule_relationship)
                trip_statuses.append((tid, status))
                if status != 'SCHEDULED':
                    continue
                for stop in update.stop_time_update:
                    if not stop.HasField('stop_sequence'):
                        continue
                    status = pb.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(stop.schedule_relationship)
                    for kind in ('departure', 'arrival'):
                        key = stop.stop_sequence, kind
                        if key not in trips[tid]:
                            continue
                        cid, uic, planned = trips[tid][key]
                        if stop.stop_id and stop.stop_id.split(':')[0] != uic:
                            continue
                        if status in ('NO_DATA', 'SKIPPED'):
                            events.append((cid, kind, status, None))
                        elif stop.HasField(kind):
                            estimate = getattr(stop, kind)
                            if estimate.HasField('time'):
                                prediction = estimate.time - midnight
                            elif estimate.HasField('delay'):
                                prediction = planned + estimate.delay
                            else:
                                continue
                            events.append((cid, kind, 'FORECAST', prediction))
            frames.append({'archive_member': member.name, 'capture_seconds': captured, 'header_seconds': header,
                           'receipt_seconds': receipt, 'trip_statuses': trip_statuses, 'endpoint_updates': events})
    frames.sort(key=lambda frame: (frame['receipt_seconds'], frame['archive_member']))
    trip_statuses, events = Counter(), Counter()
    previously_seen, previously_forecast = {}, {}
    for frame in frames:
        changed_trips, changed_events = [], []
        for tid, status in frame['trip_statuses']:
            if previously_seen.get(tid) != status:
                changed_trips.append((tid, status))
                previously_seen[tid] = status
                trip_statuses[status] += 1
        for cid, kind, status, prediction in frame['endpoint_updates']:
            if previously_forecast.get((cid, kind)) != (status, prediction):
                changed_events.append((cid, kind, status, prediction))
                previously_forecast[cid, kind] = status, prediction
                events[status] += 1
        frame['trip_statuses'] = changed_trips
        frame['endpoint_updates'] = changed_events
    return day, {'frames': frames, 'trip_status_revisions': dict(trip_statuses),
                 'endpoint_revisions': dict(events)}


def main():
    with gzip.open(HERE / 'dated_replay.json.gz', 'rt', encoding='utf-8') as handle:
        data = json.load(handle)
    rows = [(day, data['dates'][day]) for day in data['holdout_dates']]
    with ProcessPoolExecutor(max_workers=len(rows)) as pool:
        dates = dict(pool.map(scan, rows))
    output = {'scope': 'Sampled archived GTFS-RT receipts for exact scheduled trips. Numeric values are forecasts. '
              'A receipt becomes available at max(archive capture time, feed header timestamp). '
              'Unchanged revisions are collapsed. Missing trip updates supply no status.',
              'sample_seconds': SAMPLE_SECONDS, 'window_seconds': [WINDOW_START, WINDOW_END],
              'endpoint_update_fields': ['connection_id', 'event_kind', 'status', 'predicted_event_seconds'],
              'trip_status_fields': ['trip_id', 'status'], 'dates': dates}
    with gzip.open(HERE / 'dated_public_forecasts.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(output, handle, separators=(',', ':'))
    print({day: (len(row['frames']), row['trip_status_revisions'], row['endpoint_revisions'])
           for day, row in dates.items()})


if __name__ == '__main__':
    main()
