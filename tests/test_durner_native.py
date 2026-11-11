"""Native strict strategy must preserve its actual published algorithm semantics."""
import unittest
import numpy as np
from src.durner.native import insert_strict_label, native_preprocess, native_topocsa
from src.pmf import PMF
from src.transit_graph import Connection, Stop, StopLabel, TransitGraph


def graph_with(connections):
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=2))
    for route, trip, dep_stop, arr_stop, dep_time, arr_time in connections:
        connection = Connection(-1, route, trip, dep_stop, arr_stop, dep_time, arr_time,
                                PMF.deterministic(dep_time), PMF.deterministic(arr_time))
        graph.add_connection(connection)
    return graph


class NativeDurnerTests(unittest.TestCase):
    def test_strict_dominance_uses_departure_and_preserves_equal_departures(self):
        graph = graph_with([('a', 'a', 0, 3, 10, 30), ('b', 'b', 0, 3, 20, 30),
                            ('c', 'c', 0, 3, 20, 40)])
        labels = []
        for cid in (0, 1, 2):
            c = graph.connections[cid]
            insert_strict_label(labels, StopLabel(cid, c.arr_distribution, c.arr_time, 1), graph)
        self.assertEqual([l.connection_id for l in labels], [2, 1])

    def test_staying_on_trip_ignores_transfer_time(self):
        graph = graph_with([('a', 't', 0, 1, 0, 10), ('a', 't', 1, 3, 10, 20)])
        result = native_topocsa(graph, 0, 3, 0)
        self.assertEqual(result.mean_arrival, 20)
        graph.connections[1].trip_id = 'different'
        self.assertEqual(native_topocsa(graph, 0, 3, 0).mean_arrival, float('inf'))

    def test_explicit_cancellation_changes_success_not_conditional_arrival(self):
        graph = graph_with([('a', 'a', 0, 1, 0, 10), ('b', 'b', 1, 3, 12, 20),
                            ('c', 'c', 1, 3, 22, 40)])
        graph.connections[1].cancel_prob = .5
        result = native_topocsa(graph, 0, 3, 0)
        source = result.stop_labels[0][-1]
        self.assertEqual(source.feasibility, 1)
        self.assertEqual(source.mean_dest_arrival, 30)
        self.assertEqual(source.dest_arrival.total_prob, 1)

    def test_cycle_cut_is_lowest_transfer_gap_not_detected_back_edge(self):
        graph = graph_with([('a', 'a', 0, 1, 0, 5), ('b', 'b', 1, 0, 2, 4)])
        graph.connections[0].dep_distribution = PMF(np.ones(11) / 11, 0)
        graph.connections[1].dep_distribution = PMF(np.ones(11) / 11, 2)
        graph.connections[0].arr_distribution = PMF(np.ones(11) / 11, 7)
        graph.connections[1].arr_distribution = PMF(np.ones(11) / 11, 4)
        order, cuts, _ = native_preprocess(graph, 0, 50)
        self.assertEqual(cuts, {(0, 1)})
        self.assertEqual(order, [0, 1])

    def test_subprobability_representation_is_not_silently_double_counted(self):
        graph = graph_with([('a', 'a', 0, 3, 10, 20)])
        graph.connections[0].dep_distribution = PMF(np.array([.5]), 10)
        with self.assertRaisesRegex(ValueError, 'normalized PMFs'):
            native_topocsa(graph, 0, 3, 0)


if __name__ == '__main__':
    unittest.main()
