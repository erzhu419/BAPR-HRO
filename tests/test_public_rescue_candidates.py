from dataclasses import replace

from src.platform_transit import PlatformState, build_platform_model
from src.background_policy import execute_background
from src.interruptible_recourse import InterruptibleRecourse
from src.public_rescue_candidates import PublicRescueCandidates, screen_public_itineraries
from src.simulator import JourneyScenario, VehicleEvent
from src.tail_rescue import TailRescuePlan
from src.transit_graph import Connection, Stop, TransitGraph


def fixture(rescue_arrival):
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=2))
    for route, origin, target, dep, arr, pickup in (
        ('A', 0, 2, 12, 18, True),
        ('B', 0, 2, 11, 20, True),
        ('R', 0, 1, 10, 13, True),
        ('R', 1, 2, 13, rescue_arrival, False),
        ('excluded', 0, 2, 10, 11, False),
        ('incoming', 3, 0, 5, 8, True),
    ):
        graph.add_connection(Connection(0, route, route, origin, target, dep, arr, pickup_allowed=pickup))
    root = PlatformState(0, 8, None, 8, 5, frozenset({5}), local_since=8)
    plan = TailRescuePlan((0,), 0, (1,), True)
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                             for c in graph.connections})
    return graph, root, plan, scene


def replay(graph, root, plan, scene, *, margin):
    options = screen_public_itineraries(graph, root, 2, 40, (0, 1), 12)
    assert set(options) == {2}
    assert options[2].path == (2, 3)
    candidate = PublicRescueCandidates(graph, plan, root, 40, options, margin=margin).bind('expanded')
    original = InterruptibleRecourse(graph, plan, root, 40, mode='first_visible').bind()
    model = build_platform_model(graph, [(1., scene)], 0, 2, 8, 40, initial_state=root, poll_interval=1)
    result = execute_background(model, model.scenarios[0], original.policy, candidate.policy, 9)
    rides = [step['ridden_connection'] for step in result['trace'] if step['ridden_connection'] is not None]
    return result['cost'], rides


def test_guarded_first_rescue_can_continue_without_a_second_pickup():
    graph, root, plan, scene = fixture(16)
    assert replay(graph, root, plan, scene, margin=0) == (8, [2, 3])


def test_guard_rejects_slower_visible_rescue_and_obeys_actual_boardability():
    graph, root, plan, scene = fixture(30)
    assert replay(graph, root, plan, scene, margin=0) == (12, [1])
    assert replay(graph, root, plan, scene, margin=None) == (22, [2, 3])
    events = dict(scene.events)
    events[2] = replace(events[2], departure=9)
    assert replay(graph, root, plan, replace(scene, events=events), margin=None) == (12, [1])
