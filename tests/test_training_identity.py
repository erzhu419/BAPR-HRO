"""Training identity repair preserves queries while isolating raw RT meanings."""
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2 as pb
import pytest

from experiments.reconstruction_v34.messages import endpoint_map, snapshot_frames
from experiments.reconstruction_v35.prepare_data import endpoint_rows
from experiments.training_identity_v1.audit import (
    VEHICLE, audit_day, canonical_frames, compare_queries, decoder_maps, query_endpoints,
)

DAY = '2023-10-01'
MIDNIGHT = int(datetime.fromisoformat(DAY).replace(tzinfo=ZoneInfo('Europe/Zurich')).timestamp())


def connection(trip='old', vehicle='11', cid=0):
    return dict(id=cid, trip_id=trip, operator_id='85:11', vehicle_number=vehicle,
        route_id='R', route_name='S1', dep_stop='A', arr_stop='B', dep_sequence=4, arr_sequence=5,
        scheduled_departure_seconds=28800, scheduled_arrival_seconds=28920,
        departure=dict(status='REAL', time_seconds=28810, official_journey='85:11:11:001'),
        arrival=dict(status='REAL', time_seconds=28940, official_journey='85:11:11:001'))


def catalog(connections):
    trips = {row['trip_id']:dict(trip_id=row['trip_id'], **{k:row[k] for k in VEHICLE})
             for row in connections}
    return dict(trips=list(trips.values()), endpoints=query_endpoints(connections))


def message(trip, capture, updates=(), status='SCHEDULED', update_clock=None):
    """Updates are (sequence, station, kind, clock-or-delay, relationship, mode)."""
    feed = pb.FeedMessage()
    feed.header.gtfs_realtime_version = '2.0'
    feed.header.timestamp = MIDNIGHT+capture
    entity = feed.entity.add()
    entity.id = trip
    update = entity.trip_update
    update.trip.trip_id = trip
    update.trip.start_date = DAY.replace('-', '')
    update.trip.schedule_relationship = pb.TripDescriptor.ScheduleRelationship.Value(status)
    if update_clock is not None:
        update.timestamp = MIDNIGHT+update_clock
    for sequence, station, kind, value, relationship, mode in updates:
        stop = update.stop_time_update.add()
        stop.stop_sequence = sequence
        stop.stop_id = station
        stop.schedule_relationship = pb.TripUpdate.StopTimeUpdate.ScheduleRelationship.Value(relationship)
        if value is not None:
            event = getattr(stop, kind)
            if mode == 'time':
                event.time = MIDNIGHT+value
            else:
                event.delay = value
    return feed


def numeric(trip, capture, departure=28820, arrival=28950, mode='time', update_clock=None):
    return message(trip, capture, [
        (4, 'A', 'departure', departure if mode=='time' else departure-28800, 'SCHEDULED', mode),
        (5, 'B', 'arrival', arrival if mode=='time' else arrival-28920, 'SCHEDULED', mode)],
        update_clock=update_clock)


def decode(feeds, trips, canonical=None):
    frames = []
    for capture, feed in feeds:
        frames.extend(snapshot_frames(feed, DAY, MIDNIGHT, capture, f'{capture}.gtfsrt', trips))
    frames.sort(key=lambda frame:(frame['receipt_seconds'], frame['capture_seconds'],
                                  frame['archive_member'], frame['entity_id']))
    return canonical_frames(frames, canonical) if canonical is not None else frames


@pytest.mark.parametrize('mode', ['time', 'delay'])
def test_reused_literal_id_cannot_override_exact_renamed_vehicle_endpoint(mode):
    original = [connection()]
    source = [connection('new', cid=10), connection('old', vehicle='99', cid=20)]
    evidence = audit_day(DAY, original, catalog(original), catalog(source))
    assert {row['status'] for row in evidence['rows']} == {'renamed_id'}
    assert {row['original_literal_identity'] for row in evidence['rows']} == {'different_physical_train'}
    trips, canonical = decoder_maps(evidence)
    feeds = [(28400, numeric('old', 28400, 30000, 30100, mode)),
             (28400, numeric('new', 28400, mode=mode))]
    before = endpoint_rows(DAY, original, decode(feeds, endpoint_map(original)), [])
    frames = decode(feeds, trips, canonical)
    after = endpoint_rows(DAY, original, frames, [])
    assert [row['forecast_seconds'] for row in before] == [30000, 30100]
    assert [row['forecast_seconds'] for row in after] == [28820, 28950]
    assert frames[0]['trip_id'] == 'old' and frames[0]['raw_trip_id'] == 'new'
    compared = compare_queries(before, after, {DAY:evidence})
    assert compared['summary']['queries'] == 2
    assert compared['summary']['eligibility_changes'] == {'retained_changed_numeric_error':2}
    assert all(row['actual_seconds'] == original[0][row['kind']]['time_seconds'] for row in after)


@pytest.mark.parametrize('changed_field', ['scheduled_seconds', 'stop_sequence'])
def test_changed_endpoint_remains_unknown_without_same_id_fallback(changed_field):
    original = [connection()]
    source = catalog([connection('new', cid=10), connection('old', vehicle='99', cid=20)])
    changed = next(row for row in source['endpoints'] if row['trip_id']=='new' and row['kind']=='arrival')
    changed[changed_field] += 60 if changed_field=='scheduled_seconds' else 1
    evidence = audit_day(DAY, original, catalog(original), source)
    assert evidence['rows'][0]['mapped_trip_id'] == 'new'
    assert evidence['rows'][1]['status'] == 'same_service_endpoint_changed_or_absent'
    assert evidence['rows'][1]['mapped_trip_id'] is None
    trips, canonical = decoder_maps(evidence)
    feeds = [(28400, numeric('new', 28400)), (28400, numeric('old', 28400))]
    rows = endpoint_rows(DAY, original, decode(feeds, trips, canonical), [])
    assert len(rows) == 2 and rows[0]['error_eligible']
    assert rows[1]['forecast_seconds'] is None and not rows[1]['error_eligible']
    assert rows[1]['actual_seconds'] == 28940 and rows[1]['prediction_clock'] == 28620


def test_ambiguous_active_identity_has_no_numeric_or_literal_fallback():
    original = [connection()]
    source = [connection('new1', cid=10), connection('new2', cid=11), connection('old', '99', 20)]
    evidence = audit_day(DAY, original, catalog(original), catalog(source))
    assert {row['status'] for row in evidence['rows']} == {'ambiguous_public_identity'}
    trips, canonical = decoder_maps(evidence)
    assert trips == canonical == {}
    feeds = [(28400, numeric(trip, 28400)) for trip in ('old','new1','new2')]
    rows = endpoint_rows(DAY, original, decode(feeds, trips, canonical), [])
    assert len(rows) == 2
    assert all(row['forecast_seconds'] is None and not row['error_eligible'] for row in rows)
    assert all(row['actual_status']=='REAL' for row in rows)


def test_same_train_service_changes_are_distinguished_from_wrong_train_ids():
    original = [connection()]
    source = catalog(original)
    # Another station's regional service can change without invalidating the
    # two original endpoint identities or requiring whole-trip equivalence.
    source['endpoints'].append(dict(source['endpoints'][0], stop_id='C',
                                   stop_sequence=20, scheduled_seconds=31000))
    evidence = audit_day(DAY, original, catalog(original), source)
    assert {row['original_literal_identity'] for row in evidence['rows']} == {'same_public_endpoint'}
    source['endpoints'][1]['scheduled_seconds'] += 60
    evidence = audit_day(DAY, original, catalog(original), source)
    assert evidence['rows'][0]['status'] == 'same_literal_id'
    assert evidence['rows'][1]['original_literal_identity'] == 'same_train_changed_public_endpoint'
    assert evidence['rows'][1]['mapped_trip_id'] is None


@pytest.mark.parametrize('relationship', ['NO_DATA', 'SKIPPED'])
def test_endpoint_notice_clears_old_forecast_and_future_receipt_does_not_restore_it(relationship):
    original = [connection()]
    evidence = audit_day(DAY, original, catalog(original), catalog([connection('new', cid=10)]))
    trips, canonical = decoder_maps(evidence)
    clear = message('new', 28440, [(4, 'A', 'departure', None, relationship, 'time')])
    # Its capture precedes the departure query, but its individual update time
    # is later. Renaming must not change max(capture, header, update) availability.
    future = numeric('new', 28480, departure=28890, update_clock=28501)
    feeds = [(28400, numeric('new', 28400)), (28440, clear), (28480, future)]
    frames = decode(feeds, trips, canonical)
    assert frames[-1]['receipt_seconds'] == 28501
    rows = endpoint_rows(DAY, original, frames, [])
    assert rows[0]['forecast_seconds'] is None and not rows[0]['error_eligible']
    assert rows[0]['message_endpoint_status'] == (relationship, 28440)
    assert rows[1]['error_eligible']


def test_cancellation_and_later_operating_revision_follow_canonical_vehicle():
    original = [connection()]
    evidence = audit_day(DAY, original, catalog(original), catalog([connection('new', cid=10)]))
    trips, canonical = decoder_maps(evidence)
    feeds = [(28400, numeric('new', 28400)),
             (28450, message('new', 28450, status='CANCELED')),
             (28550, numeric('new', 28550, arrival=28970))]
    frames = decode(feeds, trips, canonical)
    assert frames[1]['trip_statuses'] == [['old','CANCELED']]
    rows = endpoint_rows(DAY, original, frames, [])
    assert rows[0]['forecast_seconds'] is None and not rows[0]['error_eligible']
    assert rows[0]['message_trip_status'] == ('CANCELED', 28450)
    assert rows[1]['forecast_seconds'] == 28970 and rows[1]['error_eligible']
    assert rows[1]['message_trip_status'] == ('SCHEDULED', 28550)


@pytest.mark.parametrize('change', ['missing_query', 'actual_seconds', 'prediction_clock'])
def test_query_comparison_rejects_changed_denominator_labels_or_cutoffs(change):
    original = [connection()]
    evidence = audit_day(DAY, original, catalog(original), catalog(original))
    before = endpoint_rows(DAY, original, [], [])
    after = deepcopy(before)
    if change == 'missing_query':
        after.pop()
    else:
        after[0][change] += 1
    with pytest.raises(ValueError, match='denominator changed|Frozen query or label changed'):
        compare_queries(before, after, {DAY:evidence})
