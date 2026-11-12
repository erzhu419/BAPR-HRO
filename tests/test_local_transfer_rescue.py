from dataclasses import replace

from src.background_policy import execute_background
from src.interruptible_recourse import InterruptibleRecourse
from src.local_transfer_rescue import compile_local_transfer_rescue
from src.platform_transit import PlatformState, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.tail_rescue import TailRescuePlan
from src.transit_graph import Connection, Stop, TransitGraph


def replay(target_departure, *, fallback=True):
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=3))
    for trip, start, end, departure, arrival, pickup in (
        ('incoming', 0, 1, 5, 10, True),
        ('target', 1, 3, 14, 22, True),
        ('before_ready', 1, 3, 12, 19, True),
        ('first', 1, 2, 13, 16, True),
        ('first', 2, 3, 16, 23, False),
        ('simultaneous', 1, 3, 13, 25, True),
        ('at_expiry', 1, 3, 733, 736, True),
    ):
        graph.add_connection(Connection(0, trip, trip, start, end, departure, arrival,
                                        pickup_allowed=pickup))
    root = PlatformState(0, 0)
    plan = TailRescuePlan((0, 1))
    controller = compile_local_transfer_rescue(graph, plan, root, 3, 800)
    assert controller.fallback_paths == {2: (2,), 3: (3, 4), 5: (5,), 6: (6,)}
    binding = (controller if fallback else InterruptibleRecourse(graph, plan, root, 800)).bind()
    events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections}
    events[1] = replace(events[1], departure=target_departure)
    events[4] = replace(events[4], arrival=30)
    events[5] = replace(events[5], arrival=18)
    model = build_platform_model(graph, [(1., JourneyScenario(events))], 0, 3, 0, 800,
                                 initial_state=root, poll_interval=1)
    result = execute_background(model, model.scenarios[0], binding.policy, binding.policy, 0)
    rides = [row['ridden_connection'] for row in result['trace']
             if row['ridden_connection'] is not None]
    events = [row for row in binding.trace if row['branch'] == 'local_transfer_rescue']
    return result['cost'], rides, events


def test_public_miss_uses_legal_first_visible_through_path_and_stays_on_vehicle():
    loss, rides, fallback = replay(12)
    assert (loss, rides) == (30, [0, 3, 4])
    assert fallback[0]['time'] == 13
    assert fallback[0]['failure_clock'] == 12
    assert fallback[0]['reason'] == 'local_departure_unboardable'
    assert replay(14) == replay(14, fallback=False) == (22, [0, 1], [])


def test_unobserved_departure_opens_fallback_at_original_expiry():
    loss, rides, fallback = replay(9)
    assert (loss, rides) == (736, [0, 6])
    assert fallback[0]['time'] == fallback[0]['failure_clock'] == 733
    assert fallback[0]['reason'] == 'attempt_expired'
