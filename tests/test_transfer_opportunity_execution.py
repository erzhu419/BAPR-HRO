from experiments.reconstruction_v24.run import public_case, route_replay
from src.simulator import VehicleEvent
from src.transfer_opportunity import RouteOption
from src.transit_graph import Connection, Stop, TransitGraph


def graph_and_events():
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=5))
    for cid, trip, dep, arr, clock, end in ((0, 'root', 0, 1, 10, 20),
                                          (1, 'fast', 1, 2, 900, 920),
                                          (2, 'slow', 1, 2, 900, 950)):
        graph.add_connection(Connection(cid, 'R', trip, dep, arr, clock, end))
    events = {0: VehicleEvent(0, False, 10, 20), 1: VehicleEvent(0, False, 900, 920),
              2: VehicleEvent(0, False, 900, 950)}
    return graph, events


def test_executor_waits_for_late_first_tail_and_uses_same_tie_and_mct_rules():
    graph, events = graph_and_events()
    option = RouteOption((0,), {1: (1,), 2: (2,)})
    assert route_replay(graph, events, option, 10, 1000) == (920, 'arrived', 1)
    # The nominally faster tail leaves during physical transfer and is lost.
    events[1] = VehicleEvent(0, False, 24, 44)
    assert route_replay(graph, events, option, 10, 1000) == (950, 'arrived', 2)


def test_rejecting_an_offer_cannot_recover_a_vehicle_that_left_before_entry():
    graph = TransitGraph()
    for sid in range(2):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=0))
    graph.add_connection(Connection(0, 'R', 'a', 0, 1, 10, 30))
    graph.add_connection(Connection(1, 'R', 'b', 0, 1, 0, 20))
    events = {0: VehicleEvent(0, False, 10, 30), 1: VehicleEvent(0, False, 0, 20)}
    # Root candidate 1 has left before the passenger entered. Public history
    # still admits it, but choosing to wait for it must incur deadline loss.
    case = public_case(events, (0, 1), 5, 5, 1000, {0: (20, 0), 1: (10, 1)})
    assert case == dict(status='offer', clock=10, visible=(0,), other=1)
    assert route_replay(graph, events, RouteOption((1,), {}), case['clock']+1, 1000) == (
        1000, 'deadline', None)
