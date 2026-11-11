"""Native secondary baseline shares fixed-world execution without changing its goal."""
import unittest
import numpy as np

from src.native_router import NativeDurnerRouter
from src.pmf import prob_reachable
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import JourneyScenario, RegimeSchedule, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def two_trip_graph():
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=2))
    graph.add_connection(Connection(0, 'A', 'first', 0, 1, 10, 20))
    graph.add_connection(Connection(1, 'B', 'second', 1, 2, 22, 30))
    return graph


PRIORS = {'A': {'mean': 0., 'std': .5, 'cancel_rate': .2},
          'B': {'mean': 0., 'std': .5, 'cancel_rate': .25}}


class NativeRouterTests(unittest.TestCase):
    def test_conversion_counts_operation_once(self):
        graph = two_trip_graph()
        router = NativeDurnerRouter(graph, PRIORS)
        result = router.route(0, 2, 10)
        first, second = graph.connections
        self.assertAlmostEqual(first.dep_distribution.total_prob, 1.)
        self.assertAlmostEqual(second.arr_distribution.total_prob, 1.)
        self.assertAlmostEqual(first.cancel_prob, .2)
        self.assertAlmostEqual(second.cancel_prob, .25)
        label = result.stop_labels[0][-1]
        expected = prob_reachable(first.arr_distribution, second.dep_distribution, 2) * .75
        self.assertAlmostEqual(label.feasibility, expected)
        self.assertAlmostEqual(label.dest_arrival.total_prob, 1.)
        self.assertEqual(router._score(label, 10), label.mean_dest_arrival)
        self.assertGreater(router.scans, 0)
        self.assertGreater(router.label_updates, 0)
        self.assertEqual(len(router.query_ms), 1)

    def test_two_trip_journey_executes_on_one_reusable_scene(self):
        graph = two_trip_graph()
        scenario = JourneyScenario({0: VehicleEvent(0, False, 10, 20),
                                    1: VehicleEvent(0, False, 22, 30)})
        outcomes = []
        for seed in (1, 7):
            router = NativeDurnerRouter(graph, PRIORS)
            result = simulate_bandit_journey(graph, router, 0, 2, 10,
                                             RegimeSchedule([(0, 'normal')]),
                                             np.random.default_rng(seed), max_time=120,
                                             scenario=scenario)
            outcomes.append((result.arrival_time, result.n_transfers,
                             [(event.time, event.route) for event in result.events
                              if event.event_type == 'board']))
        self.assertEqual(outcomes[0], (30, 1, [(10, 'A'), (22, 'B')]))
        self.assertEqual(outcomes[0], outcomes[1])


if __name__ == '__main__':
    unittest.main()
