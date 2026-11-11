import numpy as np

from src.committed_transit import CommittedState
from src.evolving_trip_belief import EvolvingTripBelief
from src.temporal_corridor import TemporalCorridor
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks
from src.transit_graph import Connection, Stop, TransitGraph


def test_corridor_keeps_onboard_but_requires_transfer_and_a_suffix():
    graph = TransitGraph()
    for stop in range(5):
        graph.add_stop(Stop(stop, str(stop)))
    edges = [('A', 0, 1, 10, 14, True), ('A', 1, 3, 14, 15, False),
             ('B', 1, 3, 14, 14, True), ('C', 0, 2, 10, 13, True),
             ('D', 4, 3, 11, 13, True), ('E', 0, 3, 10, 20, True)]
    for trip, origin, destination, depart, arrive, pickup in edges:
        graph.add_connection(Connection(-1, trip, trip, origin, destination, depart, arrive,
                                        pickup_allowed=pickup))
    scan = TemporalScan(graph, None)
    corridor = TemporalCorridor(scan)
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    parameters = {trip: dict(mean=0., std=1., cancel_rate=0.) for trip, *_ in edges}
    for c in graph.connections:
        for kind in ('departure', 'arrival'):
            belief.observe_connection_event(c.id, kind, 0)
    root = CommittedState(0, 10, None, 10)
    assert corridor.connections(belief, parameters, root, 3, 20, transfer=1) == {0, 1}
    assert corridor.connections(belief, parameters, root, 3, 20, transfer=0) == {0, 1, 2}
    assert corridor.connections(belief, parameters, root, 3, 15, transfer=1) == set()
    assert corridor.connections_and_floor(belief, parameters, root, 3, 20, transfer=1)[1] == 5
    assert corridor.connections_and_floor(belief, parameters, root, 3, 20, transfer=0)[1] == 4
    # A route arriving at the cap is equivalent to the capped fallback.
    assert corridor.connections_and_floor(belief, parameters, root, 3, 15, transfer=1) == (set(), 5.)


def test_corridor_preserves_packed_arrivals_after_observed_clock_shift(tmp_path):
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop)))
    for trip, origin, destination, depart, arrive in [
        ('A', 0, 1, 8, 12), ('A', 1, 2, 12, 15), ('B', 1, 2, 13, 17), ('X', 0, 3, 9, 15),
    ]:
        graph.add_connection(Connection(-1, trip, trip, origin, destination, depart, arrive))
    scan = TemporalScan(graph, build_temporal_kernel(tmp_path / 'temporal.so'))
    corridor = TemporalCorridor(scan)
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    parameters = {c.route: dict(mean=1., std=2., cancel_rate=.2) for c in graph.connections}
    # The second history moves A beyond the original +64-tick grid edge.
    assert corridor.trips(belief, parameters, CommittedState(1, 85, 'A', 85), 2, 94) == set()
    belief.observe_connection_event(0, 'arrival', 73)
    root = CommittedState(1, 85, 'A', 85, frozenset({0}))
    transfers = [0, 4, 0, 0]
    kept = corridor.trips(belief, parameters, root, 2, 94, transfer=transfers)
    assert kept == {'A'}
    all_worlds = sample_clocks(belief, parameters, np.random.default_rng(19), 32)
    kept_worlds = sample_clocks(belief, parameters, np.random.default_rng(19), 32, trips=kept)
    complete = scan.arrivals(*all_worlds, root, 2, 94, transfers)
    pruned = scan.arrivals(*kept_worlds, root, 2, 94, transfers)
    np.testing.assert_array_equal(complete, pruned)
    assert np.any(complete < 94)
    _, floor = corridor.connections_and_floor(belief, parameters, root, 2, 94, transfer=transfers)
    assert np.all(complete - root.time >= floor)
