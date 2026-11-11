from dataclasses import replace
from itertools import product

import numpy as np

from src.committed_transit import CommittedState
from src.evolving_trip_belief import EvolvingTripBelief
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_oracle import earliest_arrival
from src.temporal_support import IntervalOperatingSupport, JointOperatingSupport, TemporalReachability
from src.transit_graph import Connection, Stop, TransitGraph


def graph_with(rows):
    graph = TransitGraph()
    for stop in sorted({stop for row in rows for stop in row[1:3]}):
        graph.add_stop(Stop(stop, str(stop)))
    for trip, origin, target, dep, arr, pickup in rows:
        graph.add_connection(Connection(-1, trip, trip, origin, target, dep, arr,
                                        pickup_allowed=pickup))
    return graph


def test_joint_support_matches_full_clock_paths_with_future_evidence(monkeypatch):
    monkeypatch.setattr('src.evolving_trip_belief.DELAYS', np.array([-1, 0, 1]))
    graph = graph_with([('A', 0, 1, 2, 3, True), ('A', 1, 2, 3, 4, True)])
    parameters = {'A': dict(mean=0., std=1., cancel_rate=.3)}
    for canceled in (False, True):
        belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
        belief.observe_connection_event(0, 'cancel' if canceled else 'departure', 0)
        belief.observe_connection_event(1, 'arrival', 0)
        belief.observe_wait(1, 3, 3)
        support = JointOperatingSupport(belief, parameters)
        interval = IntervalOperatingSupport(belief, parameters)
        law = belief._law('A', parameters)
        paths = []
        for ix in product(*(range(len(times)) for times in law.times)):
            value = (law.initial[ix[0]] + sum(e[i] for e, i in zip(law.emissions, ix))
                     + sum(m[i, j] for m, i, j in zip(law.transitions, ix, ix[1:])))
            if np.isfinite(value):
                paths.append(ix)
        assert paths
        for cid in (0, 1):
            dep, arr = 2 * cid, 2 * cid + 1
            for ready in range(6):
                candidates = [law.times[arr][ix[arr]] for ix in paths
                              if not law.canceled[dep][ix[dep]] and law.times[dep][ix[dep]] >= ready]
                expected = min(candidates, default=None)
                assert support(cid, ready) == expected
                if expected is not None:
                    assert interval(cid, ready) <= expected


def test_temporal_label_matches_single_world_at_ties_and_no_pickup():
    graph = graph_with([('A', 0, 1, 1, 6, True), ('A', 1, 2, 6, 10, False),
                        ('B', 1, 2, 5, 7, True), ('C', 1, 2, 6, 8, False),
                        ('D', 1, 2, 7, 9, True)])
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    parameters = {c.route: dict(mean=0., std=1., cancel_rate=0.) for c in graph.connections}
    for c in graph.connections:
        belief.observe_connection_event(c.id, 'departure', 0)
        belief.observe_connection_event(c.id, 'arrival', 0)
    support = JointOperatingSupport(belief, parameters)
    reachability = TemporalReachability(graph)
    start = CommittedState(0, 0, None, 0)
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections},
                            minute=1, publications={})
    for state, expected in ((start, (9, (0, 4))),
                            (replace(start, unavailable_ids=frozenset({4})), (10, (0, 1))),
                            (CommittedState(1, 6, 'A', 6, frozenset({4})), (10, (1,)))):
        label = reachability.solve(state, 2, 12, support)
        assert (label.arrival, label.path) == earliest_arrival(graph, scene, state, 2, 12) == expected


def test_joint_support_keeps_clock_grid_gaps_without_a_runtime_floor():
    graph = graph_with([('A', 0, 1, 100, 110, True)])
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False)
    parameters = {'A': dict(mean=0., std=1., cancel_rate=0.)}
    assert IntervalOperatingSupport(belief, parameters)(0, 112) == 112
    assert JointOperatingSupport(belief, parameters)(0, 112) == 170
