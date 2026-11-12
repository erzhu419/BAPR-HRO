from google.transit import gtfs_realtime_pb2 as pb

from experiments.reconstruction_v34.messages import snapshot_frames


def feed_with_updates():
    feed = pb.FeedMessage()
    feed.header.gtfs_realtime_version = '2.0'
    feed.header.timestamp = 1000
    for trip, measured in [('a', 1030), ('b', 900)]:
        entity = feed.entity.add()
        entity.id = trip
        entity.trip_update.trip.trip_id = trip
        entity.trip_update.trip.start_date = '20231012'
        entity.trip_update.timestamp = measured
        stop = entity.trip_update.stop_time_update.add()
        stop.stop_sequence = 2
        stop.stop_id = '8503000:0:1'
        stop.departure.time = 1200
    return feed


def test_individual_measurement_clock_cannot_be_published_early():
    trips = {t: {(2, 'departure'): (i, '8503000', 1150)} for i, t in enumerate(('a', 'b'))}
    frames = snapshot_frames(feed_with_updates(), '2023-10-12', 0, 1010, 'frame', trips)
    assert [f['receipt_seconds'] for f in frames] == [1030, 1010]
    assert frames[0]['endpoint_updates'] == [[0, 'departure', 'FORECAST', 1200]]


def test_explicit_no_data_retracts_prediction_but_not_event_truth():
    feed = feed_with_updates()
    feed.entity[0].trip_update.stop_time_update[0].schedule_relationship = pb.TripUpdate.StopTimeUpdate.NO_DATA
    trips = {'a': {(2, 'departure'): (0, '8503000', 1150)}}
    frame, = snapshot_frames(feed, '2023-10-12', 0, 1010, 'frame', trips)
    assert frame['endpoint_updates'] == [[0, 'departure', 'NO_DATA', None]]
    assert frame['trip_statuses'] == [['a', 'SCHEDULED']]


def test_trip_date_and_stop_identity_are_required():
    feed = feed_with_updates()
    feed.entity[0].trip_update.trip.start_date = '20231011'
    feed.entity[1].trip_update.stop_time_update[0].stop_id = '8503001'
    trips = {t: {(2, 'departure'): (i, '8503000', 1150)} for i, t in enumerate(('a', 'b'))}
    frames = snapshot_frames(feed, '2023-10-12', 0, 1010, 'frame', trips)
    assert len(frames) == 1 and frames[0]['trip_id'] == 'b'
    assert frames[0]['endpoint_updates'] == []


def test_missing_update_clock_uses_capture_and_header():
    feed = feed_with_updates()
    feed.header.timestamp = 1020
    feed.entity[0].trip_update.ClearField('timestamp')
    trips = {'a': {(2, 'departure'): (0, '8503000', 1150)}}
    frame, = snapshot_frames(feed, '2023-10-12', 0, 1010, 'frame', trips)
    assert frame['receipt_seconds'] == 1020 and frame['update_seconds'] is None


def test_cancellation_notice_keeps_status_without_numeric_event_truth():
    feed = feed_with_updates()
    feed.entity[0].trip_update.trip.schedule_relationship = pb.TripDescriptor.CANCELED
    trips = {'a': {(2, 'departure'): (0, '8503000', 1150)}}
    frame, = snapshot_frames(feed, '2023-10-12', 0, 1010, 'frame', trips)
    assert frame['trip_statuses'] == [['a', 'CANCELED']]
    assert frame['endpoint_updates'] == []
