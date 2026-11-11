"""Differential physical/observation checks against the online executor."""
import numpy as np
import pytest

from src.committed_transit import OBSERVE, WAIT, build_committed_model
from src.durner.topocsa import HyperpathResult
from src.information_planner import ExactPlanner
from src.pmf import PMF
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import JourneyScenario, RegimeSchedule, VehicleEvent
from src.transit_graph import Connection, Stop, StopLabel, TransitGraph


def make_graph(connections):
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=2))
    for connection in connections:
        graph.add_connection(connection)
    return graph


class FirstIDRouter:
    def __init__(self, graph):
        self.graph, self.unavailable = graph, set()
        labels = {stop: [] for stop in graph.stops}
        for c in graph.connections:
            labels[c.dep_stop].append(StopLabel(c.id, PMF.deterministic(c.arr_time), c.arr_time, 1))
        self.cached_result = HyperpathResult(labels, None, 0, set(range(len(graph.connections))), 0, set())
        self.learn_calls, self.decision_times = [], []

    def route(self, *args):
        return self.cached_result

    def set_execution_context(self, current_trip, last_arrival, unavailable):
        self.unavailable = set(unavailable)

    def select_connection(self, stop, time, rng, top_k):
        self.decision_times.append(time)
        for label in sorted(self.cached_result.stop_labels.get(stop, []), key=lambda label: label.connection_id):
            c = self.graph.connections[label.connection_id]
            if c.id in self.unavailable or c.dep_time > time + 25:
                continue
            if c.dep_distribution is not None and c.dep_distribution.support_max < time:
                continue
            return label, 0
        return None

    def observe_delay(self, route, delay):
        self.learn_calls.append((False, route, delay))

    def observe_cancel(self, route, kind):
        self.learn_calls.append((True, route, 0))


def follow_first_id(model):
    world, state = model.scenarios[0], model.initial_state
    loss, times = 0, []
    for _ in range(model.horizon):
        actions = model.actions(state)
        if not actions:
            break
        if OBSERVE in actions:
            action = OBSERVE
        else:
            times.append(state.time)
            action = next((a for a in actions if a[0] == "select"), WAIT)
        outcome = model.transition(world, state, action)
        loss += outcome.cost
        state = outcome.state
        if outcome.terminal:
            break
    return loss + model.terminal_cost(world, state), state, times


def compare_executor(graph, scene, cap):
    model = build_committed_model(graph, [(1, scene)], 0, 2, 0, cap)
    loss, state, times = follow_first_id(model)
    router = FirstIDRouter(graph)
    runtime = simulate_bandit_journey(graph, router, 0, 2, 0,
                                      RegimeSchedule([(0, "fixed")]),
                                      np.random.default_rng(7), max_time=cap, scenario=scene)
    assert loss == runtime.arrival_time
    assert times == router.decision_times
    assert [vars(message) for message in state.messages] == [vars(message) for message in runtime.observations]
    assert [(canceled, route, 0 if canceled else delay)
            for _, route, canceled, delay in state.learning_log] == router.learn_calls
    return loss, state, times


@pytest.mark.parametrize("same_trip,expected", [(True, 2), (False, 6)])
def test_commitment_transfer_and_same_trip_match_executor(same_trip, expected):
    graph = make_graph([Connection(0, "A", "first", 0, 1, 0, 1),
                        Connection(1, "A", "first" if same_trip else "second", 1, 2, 1, 2),
                        Connection(2, "B", "rescue", 1, 2, 4, 6)])
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections})
    loss, _, _ = compare_executor(graph, scene, 10)
    assert loss == expected


@pytest.mark.parametrize("event", [VehicleEvent(0, False, 20, 30),
                                    VehicleEvent(20, False, 40, 50),
                                    VehicleEvent(999, True, 20, 30)])
def test_unpublished_future_waits_twelve_minutes_in_all_worlds(event):
    graph = make_graph([Connection(0, "A", "future", 0, 2, 20, 30)])
    loss, state, times = compare_executor(graph, JourneyScenario({0: event}), 13)
    assert loss == 13
    assert times == [0, 12]
    assert state.messages == state.learning_log == ()
    assert [feedback[2] for feedback in state.feedback] == [12, 13]


def test_local_five_message_limit_and_once_per_trip_learning_match_executor():
    graph = make_graph([Connection(i, "A", f"canceled-{i}", 0, 1, 0, 2) for i in range(7)])
    scene = JourneyScenario({c.id: VehicleEvent(999, True, 0, 2) for c in graph.connections})
    model = build_committed_model(graph, [(1, scene)], 0, 2, 0, 2)
    observed = model.transition(model.scenarios[0], model.initial_state, OBSERVE).state
    assert len(observed.messages) == 5
    assert observed.unavailable_ids == frozenset(range(5))
    _, state, _ = compare_executor(graph, scene, 2)
    assert len(state.messages) == len(state.learned_trips) == 7


def test_local_window_excludes_remote_and_old_events():
    graph = make_graph([Connection(0, "old", "old", 0, 1, -11, -10),
                        Connection(1, "remote", "remote", 1, 2, 0, 2),
                        Connection(2, "future", "future", 0, 2, 3, 5)])
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections})
    model = build_committed_model(graph, [(1, scene)], 0, 2, 0, 10)
    state = model.transition(model.scenarios[0], model.initial_state, OBSERVE).state
    assert state.messages == ()
    assert set(model.actions(state)) == {WAIT, ("select", 0), ("select", 2)}


def test_zero_minute_segment_and_fixed_world_do_not_reappear():
    graph = make_graph([Connection(0, "A", "same", 0, 0, 0, 0),
                        Connection(1, "A", "same", 0, 2, 0, 10)])
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections})
    loss, state, times = compare_executor(graph, scene, 20)
    assert loss == 10
    assert times == [0, 0]
    assert len(state.learning_log) == 1
    assert [feedback[1] for feedback in state.feedback] == [0, 1]


def test_causal_full_and_fixed_candidate_optima_share_information():
    graph = make_graph([Connection(0, "A", "primary", 0, 2, 2, 4),
                        Connection(1, "B", "backup", 0, 2, 5, 7)])
    normal = JourneyScenario({0: VehicleEvent(0, False, 2, 4), 1: VehicleEvent(0, False, 5, 7)})
    canceled = JourneyScenario({0: VehicleEvent(999, True, 2, 4), 1: normal.events[1]})
    model = build_committed_model(graph, [(0.5, normal), (0.5, canceled)], 0, 2, 0, 10)
    planner = ExactPlanner(model)
    assert planner.solve().value == 5.5
    restricted = ExactPlanner(model, lambda node: tuple(a for a in model.actions(node.state)
                                                        if a != ("select", 1)))
    certificate = restricted.certificate(restricted.optimal_action)
    assert certificate.restricted_optimum == 7
    assert certificate.support_loss == 1.5
    assert certificate.within_candidate_loss == 0


def test_continuing_an_existing_public_state_returns_remaining_loss():
    graph = make_graph([Connection(0, "A", "same", 0, 1, 0, 1),
                        Connection(1, "A", "same", 1, 2, 1, 2)])
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                             for c in graph.connections})
    bank = [(1, scene)]
    model = build_committed_model(graph, bank, 0, 2, 0, 10)
    observed = model.transition(model.scenarios[0], model.initial_state, OBSERVE).state
    context = model.transition(model.scenarios[0], observed, ("select", 0)).state
    continued = build_committed_model(graph, bank, 0, 2, 0, 10, initial_state=context)
    assert continued.initial_state == context
    assert OBSERVE not in continued.actions(context)
    assert ExactPlanner(continued).solve().value == 1
    assert context.time + ExactPlanner(continued).solve().value == 2


@pytest.mark.parametrize("incoming_shift,target_shift,expected", [(0, 6, 24), (6, 0, 36)])
def test_same_shift_attribution_fixture_matches_runtime_before_information(
    incoming_shift, target_shift, expected,
):
    graph = make_graph([Connection(0, "incoming", "incoming", 0, 1, 1, 7),
                        Connection(1, "target", "target", 1, 2, 10, 18),
                        Connection(2, "backup", "backup", 1, 2, 16, 36),
                        Connection(3, "direct", "direct", 0, 2, 0, 28)])
    scene = JourneyScenario({
        0: VehicleEvent(incoming_shift, False, 1 + incoming_shift, 7 + incoming_shift),
        1: VehicleEvent(target_shift, False, 10 + target_shift, 18 + target_shift),
        2: VehicleEvent(0, False, 16, 36),
        3: VehicleEvent(0, False, 0, 28),
    })
    model = build_committed_model(graph, [(1, scene)], 0, 2, 0, 60)
    initial = model.transition(model.scenarios[0], model.initial_state, OBSERVE).state
    assert [message.event_id for message in initial.messages] == ["3:departure"]
    assert all(event.departure <= event.arrival for event in scene.events.values())
    loss, _, _ = compare_executor(graph, scene, 60)
    assert loss == expected
