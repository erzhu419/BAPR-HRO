from dataclasses import replace
from functools import lru_cache

import pytest

from src.causal_prefix import causal_prefix_lower
from src.event_platform_solver import EventPlatform
from src.information_planner import ExactPlanner
from src.platform_transit import OBSERVE, WAIT, PlatformState
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def timetable(rows, transfer=0):
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=transfer))
    for trip, dep_stop, arr_stop, dep, arr in rows:
        graph.add_connection(Connection(-1, 'route', trip, dep_stop, arr_stop, dep, arr))
    return graph


def world(graph, *, overrides=None, publications=None):
    events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections}
    events.update(overrides or {})
    return JourneyScenario(events, minute=1, publications=publications or {})


def oracle(platform):
    @lru_cache(None)
    def remaining(index, state):
        model = replace(platform.model, initial_state=state,
                        scenarios=(replace(platform.model.scenarios[index], probability=1.),))
        return ExactPlanner(model).solve_compressed().value

    return remaining


def test_idle_wait_reveals_possible_local_events_and_explicit_publications_only():
    graph = timetable([('A', 0, 1, 12, 16), ('remote', 2, 3, 2, 3)])
    bank = [
        (.5, world(graph, publications={'1:departure': 5})),
        (.5, world(graph, overrides={0: VehicleEvent(8, False, 20, 24)})),
    ]
    platform = EventPlatform(graph, bank, PlatformState(0, 0), 1, 40)
    model = platform.model
    state = model.transition(model.scenarios[0], model.initial_state, OBSERVE).state
    assert state == model.transition(model.scenarios[1], model.initial_state, OBSERVE).state

    joined = dict(platform.outcomes(state, (0, 1), WAIT, cutoff=40))
    assert [outcome.state.time for outcome in joined.values()] == [5, 5]
    assert joined[0].state.public_log == ((5, 1, 'published_departure'),)
    assert joined[1].state.public_log == ()
    for index, outcome in joined.items():
        unit = state
        while unit.time < 5:
            unit = model.transition(model.scenarios[index], unit, WAIT).state
        assert outcome.state == unit and outcome.cost == 5

    _, masked = next(platform.outcomes(state, (1,), WAIT, cutoff=40))
    assert masked.state.time == 20 and masked.state.visible_connections == (0,)
    _, cut = next(platform.outcomes(state, (1,), WAIT, cutoff=13))
    assert cut.state.time == 13 and cut.state.visible_connections == ()


@pytest.mark.parametrize('weights, expected', [((.5, .5), 30), ((1., 0.), 24)])
def test_event_solver_preserves_alighting_and_same_trip_inertia(weights, expected):
    graph = timetable([('A', 3, 0, -2, 0), ('A', 0, 1, 20, 24),
                       ('B', 0, 1, 12, 30)], transfer=10)
    initial = PlatformState(0, 0, 'A', 0, last_connection=0, used_connections=frozenset({0}))
    bank = [(.5, world(graph)), (.5, world(graph, overrides={1: VehicleEvent(0, False, 20, 65)}))]
    platform = EventPlatform(graph, bank, initial, 1, 80)
    remaining = oracle(platform)
    unit_model = replace(platform.model, scenarios=tuple(
        replace(scenario, probability=weight) for scenario, weight in zip(platform.model.scenarios, weights)))
    full = causal_prefix_lower(unit_model, remaining, elapsed_horizon=80, stop_when=lambda _: False)
    result = platform.solve(remaining, stop_after_first_ride=False, weights=weights)
    assert result.value == full.value == expected
    assert result.information_nodes < full.information_nodes
    first_ride = platform.solve(remaining, elapsed_horizon=14, weights=weights)
    unit_prefix = causal_prefix_lower(unit_model, remaining, elapsed_horizon=14)
    assert first_ride.value == unit_prefix.value


def test_exact_singleton_shortcut_keeps_original_world_indices_and_falls_through():
    graph = timetable([('A', 0, 1, 12, 16)])
    bank = [(.5, world(graph)), (.5, world(graph, overrides={0: VehicleEvent(4, False, 16, 20)}))]
    platform = EventPlatform(graph, bank, PlatformState(0, 0), 1, 30)
    remaining = oracle(platform)
    seen = []

    def certified(index, state):
        seen.append((index, state.initialized))
        return remaining(index, state) if state.initialized else None

    trace = []
    result = platform.solve(remaining, stop_after_first_ride=False, weights=(0., 1.),
                            singleton_exact=certified, trace=lambda *node: trace.append(node))
    assert result.value == 20
    assert seen == [(1, False), (1, True)]
    assert result.boundary_kind_mass == {'exact_continuation': 1.}
    assert trace[-1][1:] == ((1,), (), 'exact_continuation')
    with pytest.raises(ValueError, match='full deadline'):
        platform.solve(remaining, singleton_exact=certified)


def test_oracle_pruning_preserves_value_and_keeps_full_action_rankings_when_tracing():
    graph = timetable([('A', 0, 1, 12, 18), ('B', 0, 1, 12, 25), ('C', 0, 1, 16, 30)])
    bank = [(.5, world(graph)), (.5, world(graph, overrides={0: VehicleEvent(0, False, 12, 20)}))]
    platform = EventPlatform(graph, bank, PlatformState(0, 0), 1, 40)
    remaining = oracle(platform)
    full = platform.solve(remaining, stop_after_first_ride=False)
    pruned = platform.solve(remaining, stop_after_first_ride=False, prune_with_oracle=True)
    assert pruned.value == full.value == 19
    assert pruned.action_branches_pruned >= 2
    assert pruned.information_nodes < full.information_nodes
    trace = []
    traced = platform.solve(remaining, stop_after_first_ride=False, prune_with_oracle=True,
                            trace=lambda *node: trace.append(node))
    assert traced.action_branches_pruned == 0
    values = next(dict(actions) for state, _, actions, _ in trace if state.time == 12)
    assert values == {WAIT: 18, ('board', 0): 7, ('board', 1): 13}
