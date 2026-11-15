"""V2 extraction must preserve V34 forecasts while retaining platform evidence."""
from copy import deepcopy
from datetime import datetime
import io
import json
import gzip
from pathlib import Path
import tarfile
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2 as pb

from experiments.decision_calibration_v2.fetch_messages import (
    attach_platform_updates, extract_messages,
)
from experiments.decision_calibration_v2 import cache_rt
from experiments.decision_calibration_v2.correct_rt_identities import (
    canonical_frames, decoding_connections,
)
from experiments.reconstruction_v34.messages import endpoint_map, snapshot_frames


DAY = '2023-03-09'
MIDNIGHT = int(datetime.fromisoformat(DAY).replace(tzinfo=ZoneInfo('Europe/Zurich')).timestamp())
CONNECTIONS = [dict(id=3, trip_id='trip', dep_sequence=1, arr_sequence=2,
    dep_stop='8503000', arr_stop='8503001', scheduled_departure_seconds=27000,
    scheduled_arrival_seconds=27300, dep_platform_id='8503000:0:3',
    arr_platform_id='8503001:0:1')]


def feed_at(clock):
    feed = pb.FeedMessage()
    feed.header.gtfs_realtime_version = '2.0'
    feed.header.timestamp = MIDNIGHT+clock+5
    entity = feed.entity.add(id='e')
    update = entity.trip_update
    update.trip.trip_id = 'trip'
    update.trip.start_date = '20230309'
    update.timestamp = MIDNIGHT+clock+20
    stop = update.stop_time_update.add(stop_sequence=1, stop_id='8503000:0:12')
    stop.departure.delay = 90
    stop = update.stop_time_update.add(stop_sequence=2, stop_id='8503999:0:1')
    stop.arrival.delay = 40
    return feed


def test_platform_audit_keeps_mismatches_without_inventing_forecasts():
    feed = feed_at(25203)
    trips = endpoint_map(CONNECTIONS)
    frames = snapshot_frames(feed, DAY, MIDNIGHT, 25203, 'member', trips)
    original = deepcopy(frames)
    attach_platform_updates(frames, feed, trips, CONNECTIONS)
    assert [{k:v for k,v in f.items() if k != 'platform_stop_updates'} for f in frames] == original
    assert frames[0]['receipt_seconds'] == 25223
    assert frames[0]['endpoint_updates'] == [[3, 'departure', 'FORECAST', 27090]]
    evidence = frames[0]['platform_stop_updates']
    assert evidence[0]['published_stop_id'] == '8503000:0:12'
    assert evidence[0]['scheduled_platform_id'] == '8503000:0:3'
    assert evidence[0]['station_matches']
    assert not evidence[1]['station_matches']


def test_extractor_uses_frozen_sampling_and_preserves_missing_dates(tmp_path):
    archive = tmp_path/'source.tar.bz2'
    with tarfile.open(archive, 'w:bz2') as handle:
        for clock in (25199, 25203, 25263, 25803, 34863):
            name = datetime.fromtimestamp(MIDNIGHT+clock, ZoneInfo('Europe/Zurich')).isoformat()+'.gtfsrt'
            raw = feed_at(clock).SerializeToString()
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            handle.addfile(member, io.BytesIO(raw))
    result = extract_messages(DAY, CONNECTIONS, archive, tmp_path/'messages.json.gz')
    assert result['summary']['archive_members'] == 5
    assert result['summary']['sampled_snapshots'] == 2
    assert [f['capture_seconds'] for f in result['frames']] == [25203, 25803]
    assert result['summary']['platform_evidence']['station_mismatch'] == 2
    ready = json.loads((tmp_path/'messages.json.audit.json').read_text())
    assert ready['output_complete'] and ready['summary'] == result['summary']
    empty = extract_messages(DAY, [], archive, tmp_path/'empty.json.gz')
    assert empty['frames'] == []
    assert empty['summary']['sampled_snapshots'] == 2
    assert empty['summary']['trip_id_coverage']['unmatched_current_day_trip_messages'] == 2


def test_short_archive_range_resumes_after_received_prefix(tmp_path, monkeypatch):
    part = tmp_path/'part'
    part.write_bytes(b'45')

    def fetch(request, timeout):
        assert request.headers['Range'] == 'bytes=6-9'
        response = io.BytesIO(b'6789')
        response.status = 206
        response.headers = {'Content-Range':'bytes 6-9/10'}
        return response

    monkeypatch.setattr(cache_rt, 'urlopen', fetch)
    source = dict(url='https://mirror.traines.eu/test.tar.bz2', size_bytes=10)
    assert cache_rt.download_range(source, 4, 9, part) == 4
    assert part.read_bytes() == b'456789'


def test_real_reused_literal_id_uses_public_bijection_without_fallback():
    path = Path(__file__).resolve().parents[1]/'experiments/decision_calibration_v2/static_identity_mapping.json.gz'
    with gzip.open(path, 'rt') as handle:
        audit = json.load(handle)['days']['2023-03-16']
    literal = audit['reused_literal_id_changes'][0]['trip_id']
    physical_old = next(row['trip_id'] for row in audit['rows'] if row['mapped_trip_id'] == literal)
    assert physical_old != literal
    evidence = dict(rows=[row for row in audit['rows'] if row['trip_id'] in (literal, physical_old)])
    original = [dict(CONNECTIONS[0], trip_id=literal),
        dict(CONNECTIONS[0], id=7, trip_id=physical_old,
             scheduled_departure_seconds=27600, scheduled_arrival_seconds=27900)]
    decoding, reverse = decoding_connections(original, evidence)
    assert [{k:v for k,v in row.items() if k != 'trip_id'} for row in decoding] == [
        {k:v for k,v in row.items() if k != 'trip_id'} for row in original]
    raw = feed_at(25203)
    raw.entity[0].trip_update.trip.trip_id = literal
    decoded = snapshot_frames(raw, DAY, MIDNIGHT, 25203, 'member', endpoint_map(decoding))
    corrected = canonical_frames(decoded, reverse)
    assert corrected[0]['trip_id'] == physical_old
    assert corrected[0]['raw_trip_id'] == literal
    assert corrected[0]['endpoint_updates'] == [[7, 'departure', 'FORECAST', 27690]]
    assert corrected[0]['receipt_seconds'] == 25223
    canonical = pb.FeedMessage()
    canonical.CopyFrom(raw)
    canonical.entity[0].trip_update.trip.trip_id = physical_old
    expected = snapshot_frames(canonical, DAY, MIDNIGHT, 25203, 'member', endpoint_map(original))
    assert [{k:v for k,v in row.items() if k != 'raw_trip_id'} for row in corrected] == expected
