"""Training-only shrinkage and numerical conditional-law semantics."""
from copy import deepcopy
import unittest

import numpy as np

from experiments.reconstruction_v5.calibration_test import connection
from src.calibration_day_effect import HeavyTailDayModel, NoiseTable
from src.calibration_model import SmoothedLaw
from src.calibration_relative import ShrunkRelativeDelayModel


class TimetableCenter:
    sigma = 17.32

    def prior(self, c, kind):
        return SmoothedLaw(np.asarray([c[f'scheduled_{kind}_seconds']], dtype=float), np.array([1.]), self.sigma, .1)


def pairs():
    result = []
    for index, delay in enumerate((-60, 0, 60, 120)):
        before = connection(index * 2, 'in' + str(index), 1000 + index * 300)
        after = connection(index * 2 + 1, 'out' + str(index), 1300 + index * 300)
        before['arrival']['time_seconds'] = before['scheduled_arrival_seconds'] + delay
        after['departure']['time_seconds'] = after['scheduled_departure_seconds'] + delay / 4
        result.append((before, after))
    return {'train': result}


class DayEffectTests(unittest.TestCase):
    def test_training_regression_recovers_slope_without_target_actual_at_query(self):
        model = ShrunkRelativeDelayModel(TimetableCenter(), pairs())
        self.assertAlmostEqual(model.coefficient, .25)
        incoming, outgoing = pairs()['train'][0]
        original = model.predict(incoming, outgoing, incoming['arrival']['time_seconds'])
        altered = deepcopy(outgoing)
        altered['departure']['time_seconds'] = 999999
        self.assertEqual(original.mean, model.predict(incoming, altered, incoming['arrival']['time_seconds']).mean)

    def test_noise_interpolation_matches_gaussian_empirical_cdf(self):
        values, weights = np.array([-30., 0., 60.]), np.array([.2, .3, .5])
        table = NoiseTable(values, weights, 17.32)
        law = SmoothedLaw(values, weights, 17.32)
        for value in (-100.4, -20.3, .7, 25.2, 80.9):
            self.assertLess(abs(table.cdf(value) - law.cdf(value)), .00011)

    def test_extreme_known_anchor_moves_day_posterior_and_cancellation_is_once(self):
        model = HeavyTailDayModel(TimetableCenter(), pairs(), delta_step=7.5)
        incoming, outgoing = pairs()['train'][0]
        now = incoming['scheduled_arrival_seconds']
        normal = model.predict(incoming, outgoing, now)
        extreme = model.predict(incoming, outgoing, now + 600)
        self.assertGreater(extreme.mean - normal.mean, 500)
        self.assertAlmostEqual(extreme.board_probability(-10000, 30000), .9)
        self.assertAlmostEqual(float(extreme.weights.sum()), 1.)
        self.assertLess(max(d['tail_bound_over_grid_evidence'] for d in model.diagnostics.values()), 1e-12)
        altered = deepcopy(outgoing)
        altered['departure']['time_seconds'] = 999999
        self.assertEqual(extreme.mean, model.predict(incoming, altered, now + 600).mean)


if __name__ == '__main__':
    unittest.main()
