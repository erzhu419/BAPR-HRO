import pytest

from src.committed_transit import build_committed_model
from src.information_planner import ExactPlanner
from src.platform_transit import build_platform_model, WAIT
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import TransitGraph, Stop, Connection


def pair():
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for trip in ('A', 'B'):
        graph.add_connection(Connection(-1, 'R', trip, 0, 1, 20, 30))
    worlds = []
    for times in ((20, 40), (40, 20)):
        events = {i: VehicleEvent(t - 20, False, t, t + 10) for i, t in enumerate(times)}
        worlds.append((.5, JourneyScenario(events, minute=60, publications={})))
    return graph, worlds


def test_local_event_response_is_thirty_versus_committed_forty():
    graph, worlds = pair()
    committed = build_committed_model(graph, worlds, 0, 1, 0, 120)
    platform = build_platform_model(graph, worlds, 0, 1, 0, 120)
    assert ExactPlanner(committed).solve().value == pytest.approx(40)
    assert ExactPlanner(platform).solve().value == pytest.approx(30)


def test_remote_actual_does_not_end_a_local_wait_or_reveal_world():
    graph, worlds = pair()
    graph.add_connection(Connection(-1, 'R', 'remote', 2, 1, 5, 6))
    worlds = [(p, JourneyScenario(dict(scene.events) | {2: VehicleEvent(0, False, 5 + i, 6 + i)},
                   minute=60, publications={})) for i, (p, scene) in enumerate(worlds)]
    model = build_platform_model(graph, worlds, 0, 1, 0, 120)
    states = [model.transition(w, model.initial_state, ('observe',)).state for w in model.scenarios]
    assert states[0] == states[1]
    for world, state in zip(model.scenarios, states):
        result = model.transition(world, state, WAIT)
        assert result.state.time == 20
        assert not any(cid == 2 for _, cid, _ in result.state.public_log)


def test_explicit_future_publication_only_arrives_at_its_receipt_time():
    graph, worlds = pair()
    worlds = [(p, JourneyScenario(dict(scene.events) | {1: VehicleEvent(999, True, 20, 30)},
               minute=60, publications={'1:cancel': 10})) for p, scene in worlds]
    model = build_platform_model(graph, worlds, 0, 1, 0, 120)
    w = model.scenarios[0]
    state = model.transition(w, model.initial_state, ('observe',)).state
    assert not state.public_log
    result = model.transition(w, state, WAIT)
    assert result.state.time == 10
    assert result.state.public_log == ((10, 1, 'published_cancel'),)


def test_positive_walking_shortcut_is_a_real_action():
    graph, worlds = pair()
    model = build_platform_model(graph, worlds, 0, 1, 0, 120, footpaths=((0, 1, 5),))
    assert ExactPlanner(model).solve().value == pytest.approx(5)


def test_walk_past_deadline_does_not_report_a_phantom_arrival():
    graph, worlds = pair()
    model = build_platform_model(graph, worlds, 0, 1, 0, 3, footpaths=((0, 1, 5),))
    world = model.scenarios[0]
    state = model.transition(world, model.initial_state, ('observe',)).state
    result = model.transition(world, state, ('walk', 0))
    assert result.terminal and result.state.time == 3 and result.state.stop == 0


def test_unidentified_platform_pair_does_not_get_station_max_implicitly():
    graph, worlds = pair()
    graph.add_connection(Connection(-1, 'R', 'tail', 1, 2, 60, 70))
    worlds = [(p, JourneyScenario(dict(scene.events) | {2: VehicleEvent(0, False, 60, 70)},
                   minute=60, publications={})) for p, scene in worlds]
    model = build_platform_model(graph, worlds, 0, 2, 0, 120,
                                 transfer_time=lambda state, cid: None)
    assert ExactPlanner(model).solve().value == pytest.approx(120)


def test_waiting_onboard_does_not_pay_transfer_walking_time_simultaneously():
    graph = TransitGraph()
    for i in range(4):
        graph.add_stop(Stop(i, str(i), min_transfer_time=30))
    for row in [('A', 0, 1, 0, 10), ('A', 1, 3, 100, 110), ('B', 1, 2, 50, 60)]:
        trip, a, b, dep, arr = row
        graph.add_connection(Connection(-1, 'R', trip, a, b, dep, arr))
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
        for c in graph.connections}, minute=60, publications={})
    model = build_platform_model(graph, [(1., scene)], 0, 2, 0, 150)
    w = model.scenarios[0]
    state = model.transition(w, model.initial_state, ('observe',)).state
    state = model.transition(w, state, ('board', 0)).state
    state = model.transition(w, state, WAIT).state
    assert state.time == 50 and state.current_trip == 'A'
    assert ('board', 2) not in model.actions(state)
    state = model.transition(w, state, ('alight',)).state
    assert ('board', 2) not in model.actions(state)
    assert ExactPlanner(model).solve().value == pytest.approx(60)


def test_busy_passenger_stays_on_the_vehicle_without_a_new_board_command():
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=420))
    for a, b, dep, arr in [(0, 1, 0, 10), (1, 2, 10, 20)]:
        graph.add_connection(Connection(-1, 'R', 'A', a, b, dep, arr))
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
        for c in graph.connections}, minute=60, publications={})
    model = build_platform_model(graph, [(1., scene)], 0, 2, 0, 100)
    world = model.scenarios[0]
    state = model.transition(world, model.initial_state, ('observe',)).state
    state = model.transition(world, state, ('board', 0)).state
    result = model.transition(world, state, WAIT)
    assert result.terminal and result.state.stop == 2 and result.state.time == 20
