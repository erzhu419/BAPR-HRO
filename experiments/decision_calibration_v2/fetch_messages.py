"""Download and filter the frozen message panel on the server.

The traines archive is an independent collection of the Swiss official feed.
V34 sampling, endpoint matching and conservative per-entity availability are
reused exactly. Full public stop IDs are retained as platform evidence.
"""
import argparse
from collections import Counter
from datetime import datetime
import gzip
import json
from pathlib import Path
import shutil
import tarfile
from time import perf_counter, process_time
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2 as pb

from experiments.reconstruction_v34.messages import (
    SAMPLE_SECONDS, WINDOW, endpoint_map, snapshot_frames,
)


def download_archive(source, archive_dir):
    """Keep the national archive on the server; reuse a complete local copy."""
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / source['filename']
    if target.exists() and target.stat().st_size == source['size_bytes']:
        return target, 0
    temporary = target.with_suffix(target.suffix + '.part')
    with urlopen(source['url'], timeout=120) as remote, temporary.open('wb') as handle:
        shutil.copyfileobj(remote, handle, length=1024*1024)
    if temporary.stat().st_size != source['size_bytes']:
        raise ValueError(f'Incomplete RT archive for {source["date"]}')
    temporary.replace(target)
    return target, target.stat().st_size


def attach_platform_updates(frames, feed, trips, connections):
    """Add raw platform evidence without changing V34 endpoint observations."""
    planned = {}
    for c in connections:
        for side in ('dep', 'arr'):
            planned[c['trip_id'], c[side+'_sequence']] = c.get(side+'_platform_id')
    by_entity = {(f['entity_id'], f['trip_id']): f for f in frames}
    for entity in feed.entity:
        if not entity.HasField('trip_update'):
            continue
        update = entity.trip_update
        trip = update.trip.trip_id
        frame = by_entity.get((entity.id, trip))
        if frame is None:
            continue
        evidence = []
        for stop in update.stop_time_update:
            if not stop.HasField('stop_sequence'):
                continue
            endpoints = [(kind, trips[trip][stop.stop_sequence, kind])
                         for kind in ('departure', 'arrival')
                         if (stop.stop_sequence, kind) in trips[trip]]
            if not endpoints:
                continue
            expected = sorted({row[1] for _, row in endpoints})
            evidence.append(dict(stop_sequence=stop.stop_sequence,
                published_stop_id=stop.stop_id or None,
                scheduled_platform_id=planned.get((trip, stop.stop_sequence)),
                station_ids=expected,
                station_matches=bool(stop.stop_id) and stop.stop_id.split(':')[0] in expected,
                stop_status=pb.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(
                    stop.schedule_relationship),
                endpoints=[[row[0], kind] for kind, row in endpoints]))
        frame['platform_stop_updates'] = evidence
    return frames


def extract_messages(day, connections, source, output):
    started, cpu = perf_counter(), process_time()
    trips = endpoint_map(connections)
    midnight = int(datetime.fromisoformat(day).replace(
        tzinfo=ZoneInfo('Europe/Zurich')).timestamp())
    frames, captures = [], []
    panel_stations = {c[side+'_stop'] for c in connections for side in ('dep', 'arr')}
    coverage = Counter()
    unmatched_panel_trips = Counter()
    members = 0
    with tarfile.open(source, 'r|bz2') as archive:
        for member in archive:
            if not member.name.endswith('.gtfsrt'):
                continue
            members += 1
            captured = int(datetime.fromisoformat(Path(member.name).stem).timestamp())-midnight
            if not WINDOW[0] <= captured < WINDOW[1]+60 or captured % SAMPLE_SECONDS >= 60:
                continue
            captures.append(captured)
            feed = pb.FeedMessage.FromString(archive.extractfile(member).read())
            for entity in feed.entity:
                if not entity.HasField('trip_update'):
                    continue
                update = entity.trip_update
                if update.trip.start_date != day.replace('-', ''):
                    continue
                coverage['current_day_trip_messages'] += 1
                if update.trip.trip_id in trips:
                    coverage['matched_scheduled_trip_messages'] += 1
                else:
                    coverage['unmatched_current_day_trip_messages'] += 1
                    if any(s.stop_id.split(':')[0] in panel_stations
                           for s in update.stop_time_update):
                        coverage['unmatched_messages_visiting_panel_stations'] += 1
                        unmatched_panel_trips[update.trip.trip_id] += 1
            matching = snapshot_frames(feed, day, midnight, captured, member.name, trips)
            frames.extend(attach_platform_updates(matching, feed, trips, connections))
    frames.sort(key=lambda f:(f['receipt_seconds'], f['capture_seconds'],
                              f['archive_member'], f['entity_id']))
    platform = Counter()
    for frame in frames:
        for row in frame.get('platform_stop_updates', []):
            platform['matched_sequence_updates'] += 1
            if not row['published_stop_id']:
                platform['missing_stop_id'] += 1
            elif not row['station_matches']:
                platform['station_mismatch'] += 1
            elif ':' in row['published_stop_id']:
                platform['published_full_platform_id'] += 1
            else:
                platform['published_station_only'] += 1
            if (row['station_matches'] and row['scheduled_platform_id'] and
                    row['published_stop_id'] != row['scheduled_platform_id']):
                platform['different_from_scheduled_stop_id'] += 1
    summary = dict(scheduled_trips=len(trips), scheduled_connections=len(connections),
        archive_members=members, sampled_snapshots=len(captures),
        matched_trip_messages=len(frames), distinct_trips=len({f['trip_id'] for f in frames}),
        endpoint_updates=dict(Counter(u[2] for f in frames for u in f['endpoint_updates'])),
        trip_status_messages=dict(Counter(s[1] for f in frames for s in f['trip_statuses'])),
        capture_range=[min(captures), max(captures)] if captures else None,
        receipt_range=[frames[0]['receipt_seconds'], frames[-1]['receipt_seconds']] if frames else None,
        messages_later_than_capture=sum(f['receipt_seconds']>f['capture_seconds'] for f in frames),
        messages_with_update_timestamp=sum(f['update_seconds'] is not None for f in frames),
        maximum_delay_from_capture=max((f['receipt_seconds']-f['capture_seconds'] for f in frames), default=0),
        platform_evidence=dict(platform), archive_bytes=Path(source).stat().st_size,
        trip_id_coverage=dict(coverage),
        unmatched_panel_trip_ids=dict(unmatched_panel_trips),
        coverage_scope='The archive is national; unmatched trips also include other operators and '
            'scheduled departures outside the frozen connection window. Counts alone do not prove a static-version mismatch.',
        extraction_seconds=perf_counter()-started, extraction_cpu_seconds=process_time()-cpu)
    result = dict(day=day, source=str(source), summary=summary, frames=frames,
        sample_seconds=SAMPLE_SECONDS, capture_window_seconds=list(WINDOW),
        availability='max(collector filename clock, feed header, individual TripUpdate timestamp)',
        source_kind='Third-party research capture of official Swiss GTFS-RT',
        information_contract='Endpoint times remain forecasts; public stop IDs remain announced '
            'platform evidence, not observed platform truth. Missing messages give no nonoccurrence evidence.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, 'wt') as handle:
        json.dump(result, handle, separators=(',', ':'))
    output.with_suffix('.audit.json').write_text(json.dumps(dict(day=day,
        output_complete=True, filtered_bytes=output.stat().st_size,
        summary=summary), indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata', type=Path,
                        default=Path(__file__).with_name('rt_sources.json'))
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    sources = {row['date']:row for row in metadata['candidate_window']['rt_files']}
    with gzip.open(args.schedule, 'rt') as handle:
        scheduled = json.load(handle)
    connections = scheduled['dates'][args.date]['connections']
    source, downloaded = download_archive(sources[args.date], args.archive_dir)
    result = extract_messages(args.date, connections, source, args.output)
    audit = dict(date=args.date, source_url=sources[args.date]['url'],
        downloaded_bytes=downloaded, filtered_bytes=args.output.stat().st_size,
        schedule=str(args.schedule), summary=result['summary'])
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
