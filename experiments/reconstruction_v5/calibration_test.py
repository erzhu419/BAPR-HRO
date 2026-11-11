"""Receipt causality and probability semantics of the calibration experiment."""
from copy import deepcopy
import unittest

import numpy as np

from src.calibration_model import PublicCalibrationModel, PublicForecasts, SmoothedLaw
from experiments.reconstruction_v5.calibration_probability import fit_isotonic, predict_isotonic


def connection(cid, trip, scheduled=1000):
    return {'id': cid, 'trip_id': trip, 'route_id': 'r', 'scheduled_departure_seconds': scheduled,
            'scheduled_arrival_seconds': scheduled + 120,
            'departure': {'status': 'REAL', 'time_seconds': scheduled + 60},
            'arrival': {'status': 'REAL', 'time_seconds': scheduled + 180}}


class CalibrationTests(unittest.TestCase):
    def test_future_receipts_do_not_enter_public_state(self):
        c = connection(0, 't')
        frames = [{'receipt_seconds': 400, 'trip_statuses': [], 'endpoint_updates': [[0, 'departure', 'FORECAST', 1060]]},
                  {'receipt_seconds': 800, 'trip_statuses': [], 'endpoint_updates': [[0, 'departure', 'FORECAST', 1600]]}]
        public = PublicForecasts([c], frames)
        self.assertEqual(public.at(399).values, {})
        self.assertEqual(public.at(799).values[0, 'departure'], (1060, 400))
        self.assertEqual(public.at(800).values[0, 'departure'], (1600, 800))

    def test_probability_law_normalizes_and_counts_cancellation_once(self):
        law = SmoothedLaw(np.array([1000.]), np.array([1.]), 10., .25)
        self.assertAlmostEqual(law.quantile(.5), 1000., places=5)
        self.assertAlmostEqual(law.board_probability(900, 200), .75, places=8)
        self.assertLess(law.cdf(990), law.cdf(1010))
        self.assertAlmostEqual(law.cdf(10000), 1.)

    def test_prediction_ignores_target_future_actual(self):
        c = connection(0, 't')
        peer = connection(1, 'peer', 1100)
        frames = [{'receipt_seconds': 500, 'trip_statuses': [], 'endpoint_updates': [
            [0, 'departure', 'FORECAST', 1060], [0, 'arrival', 'FORECAST', 1180],
            [1, 'departure', 'FORECAST', 1160], [1, 'arrival', 'FORECAST', 1280]]}]
        public = PublicForecasts([c, peer], frames)
        protocol = {'kernel_sd_seconds': 17.3, 'route_shrinkage_pseudocount': 20,
                    'peer_shrinkage_pseudocount': 20, 'endpoint_cutoff_before_schedule_seconds': 300,
                    'peer_forecast_schedule_window_seconds': [-900, 2700]}
        model = PublicCalibrationModel({'train': [c, peer]}, {'train': public}, protocol)
        original, features = model.predict(c, 'departure', 700, public)
        hidden = deepcopy(c)
        hidden['departure']['time_seconds'] = 99999
        hidden['arrival']['time_seconds'] = 88888
        altered, altered_features = model.predict(hidden, 'departure', 700, public)
        self.assertEqual(original.mean, altered.mean)
        self.assertEqual(original.quantile(.95), altered.quantile(.95))
        self.assertEqual(features, altered_features)
        self.assertEqual(features['peer_trips'], 1)

    def test_notice_withdraws_forecast_without_asserting_actual_service(self):
        c = connection(0, 't')
        frames = [{'receipt_seconds': 400, 'trip_statuses': [], 'endpoint_updates': [[0, 'departure', 'FORECAST', 1060]]},
                  {'receipt_seconds': 600, 'trip_statuses': [['t', 'CANCELED']], 'endpoint_updates': []}]
        public = PublicForecasts([c], frames)
        self.assertIn((0, 'departure'), public.at(500).values)
        self.assertNotIn((0, 'departure'), public.at(600).values)
        self.assertEqual(c['departure']['status'], 'REAL')

    def test_isotonic_equal_scores_and_date_weights(self):
        mapping = fit_isotonic([.1, .1, .3, .7, .9], [0, 1, 0, 1, 1], [1, 3, 1, 1, 1])
        self.assertAlmostEqual(predict_isotonic(mapping, .1), .6)
        self.assertAlmostEqual(predict_isotonic(mapping, .3), .6)
        self.assertEqual(predict_isotonic(mapping, .99), 1.)
        self.assertTrue(np.all(np.diff(mapping['probability']) >= 0))


if __name__ == '__main__':
    unittest.main()
