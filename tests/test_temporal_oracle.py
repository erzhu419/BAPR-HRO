from src.committed_transit import CommittedState
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_oracle import earliest_arrival
from src.transit_graph import Connection, Stop, TransitGraph


def test_oracle_enforces_realized_time_order_and_no_pickup_on_transfer():
    graph = TransitGraph()
    for i in range(3):
        graph.add_stop(Stop(i, str(i)))
    for trip, origin, destination, departure, arrival, pickup in [
        ('A', 0, 1, 1, 6, True), ('A', 1, 2, 6, 10, False),
        ('B', 1, 2, 5, 7, True), ('C', 1, 2, 6, 8, False), ('D', 1, 2, 7, 9, True),
    ]:
        graph.add_connection(Connection(-1, trip, trip, origin, destination,
                                        departure, arrival, pickup_allowed=pickup))
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections},
                            minute=1, publications={})
    start = CommittedState(0, 0, None, 0)
    assert earliest_arrival(graph, scene, start, 2, 12) == (9, (0, 4))
    assert earliest_arrival(graph, scene, start, 2, 12, transfer_seconds=2) == (10, (0, 1))
