"""Service statuses and revisions must not create fictitious cancellations/data."""
import unittest
from google.transit import gtfs_realtime_pb2 as pb

from src.gtfs_rt_parser import parse_single_snapshot, summarize_predictions
from experiments.swiss_full.build_route_priors_temporal import build_temporal_priors


def snapshot_bytes(timestamp, route='r', trip='t', status=pb.TripDescriptor.SCHEDULED,
             delay=0, stop_status=pb.TripUpdate.StopTimeUpdate.SCHEDULED):
    feed = pb.FeedMessage()
    feed.header.gtfs_realtime_version = '2.0'
    feed.header.timestamp = timestamp
    update = feed.entity.add(id=trip).trip_update
    update.trip.trip_id = trip
    update.trip.route_id = route
    update.trip.start_date = '20231001'
    update.trip.start_time = '10:00:00'
    update.trip.schedule_relationship = status
    update.timestamp = timestamp - 10
    if status != pb.TripDescriptor.CANCELED:
        stop = update.stop_time_update.add(stop_id='s', stop_sequence=1)
        stop.schedule_relationship = stop_status
        if delay is not None:
            stop.arrival.delay = delay
    return feed.SerializeToString()


def snapshot(*args, **kwargs):
    return parse_single_snapshot(snapshot_bytes(*args, **kwargs))


class EventSemanticsTests(unittest.TestCase):
    def test_streamed_daily_builder_uses_unique_latest_forecasts(self):
        import io
        from pathlib import Path
        import tarfile
        import tempfile
        from src.gtfs_rt_parser import process_day
        from experiments.swiss_full.build_per_day_distributions import _build_one_day, _init_worker
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.tar.bz2'
            with tarfile.open(path, 'w:bz2') as archive:
                for i, data in enumerate((snapshot_bytes(100, delay=0), snapshot_bytes(110, delay=600))):
                    member = tarfile.TarInfo(f'{i}.gtfsrt')
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            first = summarize_predictions(process_day(str(path), sample_interval=1, max_snapshots=1))
            self.assertEqual(first['r']['mean'], 0)
            _init_worker({'r': {'name': 'A', 'agency_id': 'a'}})
            _, full = _build_one_day(('2023-10-01', str(path), 1))
            self.assertEqual(full['r']['mean'], 10)
            self.assertEqual(full['r']['n_obs'], 1)
            self.assertEqual(full['r']['agency_id'], 'a')

    def test_zero_and_absent_delay_are_different(self):
        zero = snapshot(100, delay=0)
        missing = snapshot(100, delay=None)
        self.assertEqual(zero[1]['delay_seconds'], 0)
        self.assertIsNone(missing[1]['delay_seconds'])
        self.assertEqual(zero[1]['published_at'], 100)
        self.assertEqual(zero[1]['measured_at'], 90)
        self.assertIsNone(zero[1]['event_time'])

    def test_severe_delay_does_not_mean_canceled(self):
        records = snapshot(100, trip='late', delay=2400)
        records += snapshot(100, trip='canceled', status=pb.TripDescriptor.CANCELED)
        out = summarize_predictions(records)['r']
        self.assertEqual(out['n_obs'], 1)
        self.assertEqual(out['mean'], 40)
        self.assertEqual(out['n_severe_delay'], 1)
        self.assertEqual(out['cancel_rate'], .5)
        self.assertEqual(out['n_status_trips'], 2)

    def test_repeated_and_revised_predictions_count_once(self):
        first = snapshot(100, delay=0)
        final = snapshot(110, delay=600)
        out = summarize_predictions(first + first + final + first)['r']
        self.assertEqual(out['n_obs'], 1)
        self.assertEqual(out['n_status_trips'], 1)
        self.assertEqual(out['mean'], 10)
        canceled = snapshot(120, status=pb.TripDescriptor.CANCELED)
        out = summarize_predictions(first + final + canceled)['r']
        self.assertEqual(out['n_obs'], 0)
        self.assertEqual(out['cancel_rate'], 1)

    def test_skipped_and_no_data_are_not_trip_cancellations(self):
        records = snapshot(100, trip='skip', delay=1200,
                           stop_status=pb.TripUpdate.StopTimeUpdate.SKIPPED)
        records += snapshot(100, trip='missing', delay=None,
                            stop_status=pb.TripUpdate.StopTimeUpdate.NO_DATA)
        out = summarize_predictions(records)['r']
        self.assertEqual((out['n_skipped'], out['n_no_data']), (1, 1))
        self.assertEqual(out['cancel_rate'], 0)
        self.assertEqual(out['n_obs'], 0)

    def test_equal_names_keep_route_and_agency_identity(self):
        records = snapshot(100, route='r1', delay=0) + snapshot(100, route='r2', delay=600)
        metadata = {'r1': {'name': 'S10', 'agency_id': '78'},
                    'r2': {'name': 'S10', 'agency_id': '11'}}
        out = summarize_predictions(records, metadata)
        self.assertEqual((out['r1']['mean'], out['r2']['mean']), (0, 10))
        self.assertEqual((out['r1']['agency_id'], out['r2']['agency_id']), ('78', '11'))

    def test_sample_scope_filters_before_record_creation(self):
        feed = pb.FeedMessage.FromString(snapshot_bytes(100, route='r1', trip='target'))
        other = feed.entity.add(id='previous-date').trip_update
        other.trip.CopyFrom(feed.entity[0].trip_update.trip)
        other.trip.start_date = '20230930'
        other = feed.entity.add(id='another-route').trip_update
        other.trip.CopyFrom(feed.entity[0].trip_update.trip)
        other.trip.route_id = 'r2'
        records = parse_single_snapshot(feed.SerializeToString(), {'r1'}, '20231001')
        self.assertEqual(len(records), 2)
        self.assertEqual({(row['route_id'], row['service_date']) for row in records}, {('r1', '20231001')})

    def test_all_canceled_day_remains_in_pooled_cancel_denominator(self):
        from experiments.swiss_full.build_route_priors import pool_priors
        metadata = {'r': {'name': 'A', 'agency_id': 'a'}}
        per_day = {'day1': summarize_predictions(snapshot(100, delay=0), metadata),
                   'day2': summarize_predictions(snapshot(110, status=pb.TripDescriptor.CANCELED), metadata)}
        prior = pool_priors(per_day, per_day)['r']
        self.assertEqual(prior['cancel_rate'], .5)
        self.assertEqual(prior['n_obs'], 1)

    def test_exact_trip_lookup_resolves_ambiguous_public_names(self):
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from src.gtfs_rt_parser import recover_graph_route_identity
        graph = SimpleNamespace(connections=[SimpleNamespace(trip_id=trip)
                                             for trip in ('trip-a', 'trip-b', 'older-year')])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'routes.txt').write_text('route_id,agency_id,route_short_name,route_long_name\nr1,a,S10,\nr2,b,S10,\n')
            (path / 'trips.txt').write_text('route_id,service_id,trip_id\nr1,d,trip-a\nr2,d,trip-b\n')
            identity = recover_graph_route_identity(graph, directory)
        self.assertEqual(identity['by_trip']['trip-a']['route_id'], 'r1')
        self.assertEqual(identity['by_trip']['trip-a']['service_id'], 'd')
        self.assertEqual(identity['by_trip']['trip-b']['agency_id'], 'b')
        self.assertEqual(identity['missing_trip_ids'], ['older-year'])
        self.assertEqual(identity['matched_connections'], 2)
        self.assertEqual(identity['ambiguous_trip_ids'], {})

    def test_service_day_applies_calendar_exceptions(self):
        from pathlib import Path
        import tempfile
        from src.gtfs_rt_parser import scheduled_services_by_date
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'calendar.txt').write_text(
                'service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n'
                'weekday,1,1,1,1,1,0,0,20230101,20231231\n')
            (path / 'calendar_dates.txt').write_text(
                'service_id,date,exception_type\nweekday,20231002,2\nreplacement,20231002,1\n')
            active = scheduled_services_by_date(path, ['2023-10-01', '2023-10-02', '2023-10-03'],
                                                {'weekday', 'replacement'})
        self.assertEqual(active, {'2023-10-01': set(), '2023-10-02': {'replacement'}, '2023-10-03': {'weekday'}})

    def test_temporal_prior_excludes_target_and_future(self):
        dates = {}
        for date, delay in [('2023-10-01', 0), ('2023-10-02', 600), ('2023-10-03', 6000)]:
            dates[date] = summarize_predictions(snapshot(100, delay=delay),
                                               {'r': {'name': 'A', 'agency_id': 'a'}})
        temporal = build_temporal_priors(dates)
        self.assertEqual(temporal['2023-10-01']['priors'], {})
        self.assertEqual(temporal['2023-10-02']['train_dates'], ['2023-10-01'])
        self.assertEqual(temporal['2023-10-02']['priors']['r']['mean'], 0)
        self.assertEqual(temporal['2023-10-03']['priors']['r']['mean'], 5)
        self.assertEqual(temporal['2023-10-03']['priors']['r']['std'], 5)


if __name__ == '__main__':
    unittest.main()
