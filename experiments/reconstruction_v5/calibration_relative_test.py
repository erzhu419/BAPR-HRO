"""Checks that relative conditioning uses observed input and keeps censoring."""
from copy import deepcopy
import unittest

import numpy as np

from experiments.reconstruction_v5.calibration_test import connection
from experiments.reconstruction_v5.calibration_relative_run import candidate_pairs, select_menu
from src.calibration_model import PublicCalibrationModel, PublicForecasts
from src.calibration_relative import RelativeDelayModel
from src.evolving_trip_belief import EvolvingTripBelief
from src.transit_graph import Connection, Stop, TransitGraph


class RelativeTests(unittest.TestCase):
    def setUp(self):
        self.before, self.after = connection(0, 'in', 1000), connection(1, 'out', 1300)
        protocol = {'kernel_sd_seconds': 17.3, 'route_shrinkage_pseudocount': 20,
                    'peer_shrinkage_pseudocount': 20, 'endpoint_cutoff_before_schedule_seconds': 300,
                    'peer_forecast_schedule_window_seconds': [-900, 2700]}
        train = [self.before, self.after]
        marginal = PublicCalibrationModel({'train': train}, {'train': PublicForecasts(train, [])}, protocol)
        self.relative = RelativeDelayModel(marginal, {'train': [(self.before, self.after)]})

    def test_predictor_translates_with_observed_arrival_and_ignores_target_truth(self):
        now = self.before['arrival']['time_seconds']
        left = self.relative.predict(self.before, self.after, now)
        changed = deepcopy(self.after)
        changed['departure']['time_seconds'] = 999999
        right = self.relative.predict(self.before, changed, now + 240)
        self.assertAlmostEqual(right.mean - left.mean, 240)
        self.assertAlmostEqual(right.quantile(.95) - left.quantile(.95), 240, places=5)
        self.assertAlmostEqual(right.board_probability(1600), left.board_probability(1360))

    def test_common_training_location_cancels_in_relative_residual(self):
        before, after = deepcopy(self.before), deepcopy(self.after)
        before['arrival']['time_seconds'] += 240
        after['departure']['time_seconds'] += 240
        shifted = RelativeDelayModel(self.relative.marginal, {'train': [(before, after)]})
        self.assertEqual(shifted.samples, self.relative.samples)

    def test_timetable_menu_is_not_reselected_by_unknown_actual_delays(self):
        graph = TransitGraph()
        for sid in range(3):
            graph.add_stop(Stop(sid, str(sid), min_transfer_time=120))
        graph.add_connection(Connection(0, 'r', 'in', 0, 1, 1000, 1120))
        graph.add_connection(Connection(1, 'r', 'out', 1, 2, 1300, 1420))
        self.assertEqual(len(candidate_pairs(graph)), 1)
        actual = {0: {'outcome': 'MEASURED', 'arrival': {'time_seconds': 2120}}, 1: {'outcome': 'UNRESOLVED'}}
        self.assertEqual(candidate_pairs(graph, actual), [])
        self.assertEqual(len(candidate_pairs(graph)), 1)
        model = EvolvingTripBelief(graph, public_occurrences=False)
        parameters = {'r': {'mean': 1., 'std': 2., 'cancel_rate': .01}}
        prior = model.event_distribution(1, 'departure', parameters)
        model.observe_connection_event(0, 'arrival', 240)
        conditioned = model.event_distribution(1, 'departure', parameters)
        np.testing.assert_array_equal(prior[0], conditioned[0])
        np.testing.assert_array_equal(prior[1], conditioned[1])
        self.assertEqual(prior[2], conditioned[2])

    def test_unknown_outcome_can_be_selected_and_is_then_censored(self):
        rows = [{'incoming': 0, 'outgoing': 1, 'scheduled': 100, 'caught': False, 'probabilities': {'m': .8}},
                {'incoming': 0, 'outgoing': 2, 'scheduled': 200, 'caught': None, 'probabilities': {'m': .9}}]
        selected = select_menu(rows, 'm')
        self.assertEqual(selected[0]['outgoing'], 2)
        self.assertIsNone(selected[0]['caught'])


if __name__ == '__main__':
    unittest.main()
