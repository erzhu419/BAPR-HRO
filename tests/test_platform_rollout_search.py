from types import SimpleNamespace

import numpy as np

from src.platform_rollout_search import PlatformRolloutSearch
from src.platform_transit import PlatformState
from src.transit_graph import Connection, Stop, TransitGraph


class _TimetableBelief:
    def __init__(self, graph):
        self.graph = graph

    def mean_events(self, parameters):
        return {c.id: SimpleNamespace(departure=c.dep_time, arrival=c.arr_time)
                for c in self.graph.connections}

    def event_distribution(self, cid, kind, parameters):
        return np.array([self.graph.connections[cid].dep_time]), np.ones(1), 0.


def _graph(edges, mct=0):
    graph = TransitGraph()
    for stop in sorted({stop for _, a, b, _, _ in edges for stop in (a, b)}):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=mct))
    for trip, a, b, d, e in edges:
        graph.add_connection(Connection(0, trip, trip, a, b, d, e))
    return graph


def test_future_rollout_ignores_unobserved_arrival_while_choosing_a_public_offer():
    graph = _graph((('now', 0, 1, 5, 28), ('next', 0, 1, 10, 20), ('later', 0, 1, 15, 50)))
    search = PlatformRolloutSearch(graph, _TimetableBelief(graph), {}, PlatformState(0, 0), 1,
                                   60, {cid: (cid,) for cid in range(3)}, None)
    departures = np.array([[5, 10, 15], [5, 10, 15]])
    arrivals = np.array([[28, 20, 16], [28, 20, 50]])
    assert search._rollout_wait(5, departures, arrivals, np.zeros_like(departures, dtype=bool),
                                {0}).tolist() == [20, 20]


def test_nominal_nonoccurrence_retains_departures_before_platform_entry():
    graph = _graph((('now', 0, 1, 30, 80), ('later', 0, 1, 40, 50)))

    class Belief(_TimetableBelief):
        def event_distribution(self, cid, kind, parameters):
            if cid == 1:
                return np.array([5, 10, 40]), np.array([.25, .25, .5]), .2
            return super().event_distribution(cid, kind, parameters)

    root = PlatformState(0, 20, initialized=True, local_since=10)
    search = PlatformRolloutSearch(graph, Belief(graph), {}, root, 1, 120, {0: (0,), 1: (1,)}, None)
    state = PlatformState(0, 30, local_since=10, visible_connections=(0,), initialized=True)

    # Cancellation and the departure at 5 both lead to the deadline. A
    # departure at 10 would have been observed and is ruled out by the log.
    assert np.isclose(search.nominal_wait[10, 1], (.4*120+.4*50)/.8)
    assert search.choose(state, None, 0, None, (0,)) == (0, ())


def test_rollout_matches_public_transfer_failure_and_fallback_timing():
    graph = _graph((('incoming', 0, 1, 5, 10), ('target', 1, 3, 14, 22),
                    ('before_ready', 1, 3, 12, 19), ('first', 1, 2, 13, 16),
                    ('first', 2, 3, 16, 23), ('simultaneous', 1, 3, 13, 25),
                    ('at_expiry', 1, 3, 733, 736)), 3)
    search = PlatformRolloutSearch(graph, _TimetableBelief(graph), {}, PlatformState(0, 0), 3,
                                   800, {0: (0, 1)}, None)
    departures = np.repeat([[c.dep_time for c in graph.connections]], 3, axis=0)
    arrivals = np.repeat([[c.arr_time for c in graph.connections]], 3, axis=0)
    departures[:, 1] = [12, 14, 9]
    arrivals[:, 4] = 30
    arrivals[:, 5] = 18
    assert search._arrival(search.routes[0], departures, arrivals,
                           np.zeros_like(departures, dtype=bool)).tolist() == [30, 22, 736]
