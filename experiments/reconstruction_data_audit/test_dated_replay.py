"""Regression checks for measured endpoint joins and service continuity."""
import unittest

from experiments.reconstruction_data_audit.build_dated_replay import attach_outcomes
from experiments.reconstruction_data_audit.build_vbz_replay_sample import observed_runs


class DatedReplayTests(unittest.TestCase):
    def connection(self):
        return {'id': 0, 'operator_id': '85:11', 'vehicle_number': '17', 'trip_id': 't',
                'dep_stop': 'a', 'arr_stop': 'b', 'dep_sequence': 1, 'arr_sequence': 2,
                'scheduled_departure_seconds': 28800, 'scheduled_arrival_seconds': 28980}

    def actual(self, stop, kind, status='REAL', canceled='false'):
        row = {'BETREIBER_ID': '85:11', 'LINIEN_ID': '17', 'BETRIEBSTAG': '12.10.2023',
               'ZUSATZFAHRT_TF': 'false', 'FAELLT_AUS_TF': canceled, 'DURCHFAHRT_TF': 'false',
               'BPUIC': stop, 'FAHRT_BEZEICHNER': '85:11:17:001',
               'ABFAHRTSZEIT': '', 'AB_PROGNOSE': '', 'AB_PROGNOSE_STATUS': '',
               'ANKUNFTSZEIT': '', 'AN_PROGNOSE': '', 'AN_PROGNOSE_STATUS': ''}
        if kind == 'departure':
            row.update(ABFAHRTSZEIT='12.10.2023 08:00', AB_PROGNOSE='12.10.2023 08:01:20',
                       AB_PROGNOSE_STATUS=status)
        else:
            row.update(ANKUNFTSZEIT='12.10.2023 08:03', AN_PROGNOSE='12.10.2023 08:03:45',
                       AN_PROGNOSE_STATUS=status)
        return row

    def test_only_real_endpoints_form_a_measured_leg(self):
        c = self.connection()
        coverage = attach_outcomes('2023-10-12', [c], [self.actual('a', 'departure'), self.actual('b', 'arrival')])
        self.assertEqual(c['outcome'], 'MEASURED')
        self.assertEqual(c['arrival']['time_seconds'] - c['departure']['time_seconds'], 145)
        self.assertEqual(coverage['connection_outcome_counts'], {'MEASURED': 1})
        c = self.connection()
        attach_outcomes('2023-10-12', [c], [self.actual('a', 'departure'), self.actual('b', 'arrival', 'PROGNOSE')])
        self.assertEqual(c['outcome'], 'UNRESOLVED')
        self.assertIsNone(c['arrival']['time_seconds'])

    def test_cancellation_and_missing_are_separate(self):
        c = self.connection()
        attach_outcomes('2023-10-12', [c], [self.actual('a', 'departure', status='UNBEKANNT', canceled='true'),
                                          self.actual('b', 'arrival', status='UNBEKANNT', canceled='true')])
        self.assertEqual(c['outcome'], 'CANCELED')
        self.assertIsNone(c['departure']['time_seconds'])
        c = self.connection()
        attach_outcomes('2023-10-12', [c], [])
        self.assertEqual(c['outcome'], 'UNRESOLVED')
        self.assertEqual(c['departure']['status'], 'MISSING')

    def test_rail_keeps_real_cancel_conflict_and_raw_fields(self):
        c = self.connection()
        rows = [self.actual('a', 'departure', canceled='true'), self.actual('b', 'arrival', canceled='true')]
        attach_outcomes('2023-10-12', [c], rows)
        self.assertEqual(c['outcome'], 'UNRESOLVED')
        self.assertEqual(c['departure']['status'], 'CONFLICT_REAL_CANCELED')
        self.assertIsNone(c['departure']['time_seconds'])
        self.assertEqual(c['departure']['reported_time_seconds'], 28880)
        self.assertTrue(c['departure']['canceled_flag'])
        self.assertEqual(c['departure']['reported_status'], 'REAL')

    def test_unmatched_and_wrong_operator_candidates_are_retained(self):
        matched = self.connection()
        other = dict(self.connection(), id=1, operator_id='85:82', trip_id='other')
        connections = [matched, other]
        coverage = attach_outcomes('2023-10-12', connections,
                                  [self.actual('a', 'departure'), self.actual('b', 'arrival')])
        self.assertEqual(len(connections), 2)
        self.assertEqual(coverage['n_scheduled_connections'], 2)
        self.assertEqual(other['outcome'], 'UNRESOLVED')
        self.assertEqual(other['departure']['status'], 'MISSING')

    def test_vbz_continuation_requires_same_vehicle_and_time_order(self):
        first = {'trip_id': 't', 'provider_trip_id': 'p', 'provider_vehicle': 'v', 'dep_sequence': 1,
                 'arr_sequence': 2, 'actual_departure_seconds': 100, 'actual_arrival_seconds': 130}
        second = dict(first, dep_sequence=2, arr_sequence=3, actual_departure_seconds=140, actual_arrival_seconds=170)
        self.assertEqual(len(observed_runs([first, second])), 1)
        second['provider_vehicle'] = 'other'
        self.assertEqual(observed_runs([first, second]), [])

    def test_city_actual_keeps_real_cancel_flag_conflict(self):
        import csv
        import gzip
        from pathlib import Path
        import tempfile
        from experiments.reconstruction_data_audit.match_city_actuals import read_observations
        connection = self.connection()
        connection.update(route_name='7', agency_id='3849')
        rows = [self.actual('a', 'departure', canceled='true'), self.actual('b', 'arrival', canceled='true')]
        for row in rows:
            row['BETREIBER_ID'] = '85:3849'
            row['LINIEN_TEXT'] = '7'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'actual.csv.gz'
            with gzip.open(path, 'wt', newline='') as handle:
                writer = csv.DictWriter(handle, rows[0], delimiter=';')
                writer.writeheader()
                writer.writerows(rows)
            observations, _ = read_observations(path, {'date': '2023-10-12', 'connections': [connection]})
        self.assertEqual({row[3] for row in observations}, {'CONFLICT_REAL_CANCELED'})
        self.assertEqual({row[4] for row in observations}, {None})
        self.assertTrue(all(row[5] is not None and row[6] is True for row in observations))


if __name__ == '__main__':
    unittest.main()
