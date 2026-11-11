"""Physical timetables with analytically known causal full/H optima."""
import pytest

from src.finite_transit import OBSERVE, WAIT, build_transit_model
from src.information_planner import ExactPlanner
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def graph_with(connections):
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=2))
    for connection in connections:
        graph.add_connection(connection)
    return graph


def scheduled_scene(graph):
    return JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                            for c in graph.connections})


@pytest.mark.parametrize("same_trip,expected", [(True, 2), (False, 5)])
def test_same_vehicle_and_different_vehicle_transfer(same_trip, expected):
    graph = graph_with([Connection(0, "A", "vehicle", 0, 1, 0, 1),
                        Connection(1, "A", "vehicle" if same_trip else "other", 1, 2, 1, 2),
                        Connection(2, "B", "direct", 0, 2, 0, 5)])
    model = build_transit_model(graph, [(1, scheduled_scene(graph))], 0, 2, 0, 10)
    planner = ExactPlanner(model)
    assert model.actions(planner.root.state) == (OBSERVE,)
    assert planner.solve().value == expected
    assert planner.evaluate(planner.optimal_action) == expected


def test_initial_public_observation_and_candidate_loss():
    graph = graph_with([Connection(0, "A", "fast", 0, 2, 0, 2),
                        Connection(1, "B", "backup", 0, 2, 0, 5)])
    available = scheduled_scene(graph)
    canceled = JourneyScenario({0: VehicleEvent(999, True, 0, 2),
                                1: available.events[1]})
    model = build_transit_model(graph, [(0.5, available), (0.5, canceled)], 0, 2, 0, 10)
    full = ExactPlanner(model)
    restricted = ExactPlanner(model, lambda node: tuple(
        action for action in model.actions(node.state) if action != ("board", 0)))
    assert full.solve().value == 3.5
    certificate = restricted.certificate(restricted.optimal_action)
    assert certificate.restricted_optimum == 5
    assert certificate.support_loss == 1.5
    assert certificate.within_candidate_loss == 0


def test_future_arrivals_do_not_leak_before_boarding():
    # The two hub services depart together. Their different future arrivals
    # reveal which is fast only after the boarding decision is irrevocable.
    graph = graph_with([Connection(0, "feeder", "feeder", 0, 1, 0, 1),
                        Connection(1, "A", "A", 1, 2, 3, 4),
                        Connection(2, "B", "B", 1, 2, 3, 4),
                        Connection(3, "direct", "direct", 0, 2, 0, 5)])
    scenes = []
    for first_fast in (True, False):
        events = dict(scheduled_scene(graph).events)
        events[1] = VehicleEvent(0, False, 3, 4 if first_fast else 10)
        events[2] = VehicleEvent(0, False, 3, 10 if first_fast else 4)
        scenes.append(JourneyScenario(events))
    model = build_transit_model(graph, [(0.5, scene) for scene in scenes], 0, 2, 0, 12)
    planner = ExactPlanner(model)
    assert planner.solve().value == 5
    assert planner.clairvoyant_lower_bound() == 4
    # Public information after the required initial observation is identical.
    initial = [model.transition(world, planner.root.state, OBSERVE)
               for world in model.scenarios]
    assert initial[0] == initial[1]
    assert all(event[0] <= 0 for event in initial[0].state.public_log)


def test_zero_minute_continuation_and_late_arrival_use_common_cap():
    graph = graph_with([Connection(0, "A", "same", 0, 1, 0, 0),
                        Connection(1, "A", "same", 1, 2, 0, 1)])
    model = build_transit_model(graph, [(1, scheduled_scene(graph))], 0, 2, 0, 1)
    assert ExactPlanner(model).solve().value == 1
    late_graph = graph_with([Connection(0, "A", "late", 0, 2, 0, 20)])
    late = build_transit_model(late_graph, [(1, scheduled_scene(late_graph))], 0, 2, 0, 5)
    planner = ExactPlanner(late)
    policy = lambda node: OBSERVE if not node.state.initialized else ("board", 0)
    assert planner.evaluate(policy) == planner.solve().value == 5
    assert build_transit_model(late_graph, [(1, scheduled_scene(late_graph))],
                               0, 0, 0, 5).terminal_cost(None, planner.root.state) == 0


def test_waiting_past_departure_cannot_reboard_a_departed_vehicle():
    graph = graph_with([Connection(0, "A", "A", 0, 2, 0, 2)])
    model = build_transit_model(graph, [(1, scheduled_scene(graph))], 0, 2, 0, 5)
    world = model.scenarios[0]
    initial = model.transition(world, ExactPlanner(model).root.state, OBSERVE).state
    assert ("board", 0) in model.actions(initial)
    waited = model.transition(world, initial, WAIT).state
    assert model.actions(waited) == (WAIT,)
    assert ExactPlanner(model).evaluate(lambda node: OBSERVE if not node.state.initialized
                                        else WAIT) == 5
