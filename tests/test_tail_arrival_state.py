"""Arrival evidence remains distinguishable after two public paths converge."""
from dataclasses import replace

import numpy as np

from src.causal_planning_router import BudgetedPolicySearch, CausalPlanningRouter
from src.committed_transit import CommittedState, WAIT, build_committed_model
from src.continuation_router import nominal_action
from src.evolving_trip_belief import EvolvingTripBelief
from src.simulator import JourneyScenario, VehicleEvent, generate_delay_observations
from src.transit_graph import Connection, Stop, TransitGraph


def test_tail_cache_and_tree_keep_the_earlier_boarded_arrival():
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=120 if stop == 2 else 60))
    rows = [('A', 0, 1, 600, 660), ('B', 1, 2, 840, 900),
            ('A', 2, 3, 1020, 1080), ('C', 2, 3, 1080, 1140)]
    for cid, (trip, start, end, dep, arr) in enumerate(rows):
        graph.add_connection(Connection(cid, trip, trip, start, end, dep, arr))
    parameters = {route: dict(mean=0, std=1, cancel_rate=0) for route in 'ABC'}
    original = EvolvingTripBelief(graph, public_occurrences=False,
                                  travel_sd=.3, dwell_sd=.3, gap_sd=.2)
    router = CausalPlanningRouter(graph, parameters, event_belief=original, max_time=1200)
    scenes = []
    for arrival_delay in (0, 120):
        events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections}
        events[0] = VehicleEvent(0, False, 600, 660 + arrival_delay)
        events[2] = VehicleEvent(arrival_delay, False, 1020 + arrival_delay, 1080 + arrival_delay)
        scenes.append(JourneyScenario(events, minute=60, publications={}))
    initial = CommittedState(0, 600, None, 600, initialized=True)
    model = build_committed_model(graph, [(0.5, s) for s in scenes], 0, 3, 600, 1800,
                                  initial_state=initial)
    states, beliefs = [], []
    for world in model.scenarios:
        state, belief = initial, original.copy()
        for cid in (0, 1):
            outcome = model.transition(world, state, ('select', cid))
            router._advance_belief(belief, state, outcome.state, outcome.terminal)
            state = outcome.state
        states.append(state)
        beliefs.append(belief)
    assert replace(states[0], arrivals=()) == replace(states[1], arrivals=())
    assert states[0].arrivals == ((0, 660), (1, 900))
    assert states[1].arrivals == ((0, 780), (1, 900))

    def tail(state, belief, actions):
        return nominal_action(graph, state, 3, belief.mean_events(parameters), set(range(4)),
                              1800, 1500, minute=60)

    search = BudgetedPolicySearch(model, original, tail, router._advance_belief,
                                  np.random.default_rng(2), 64, 1200)
    actions = (WAIT, ('select', 2), ('select', 3))
    assert [search._tail(state, b, actions) for state, b in zip(states, beliefs)] == [
        ('select', 2), ('select', 3)]
    search.tree = {state: {} for state in states}
    assert len(search.tree) == 2


def test_delayed_actual_receipt_with_no_absence_inference():
    graph = TransitGraph()
    graph.add_stop(Stop(0, '0'))
    graph.add_stop(Stop(1, '1'))
    graph.add_connection(Connection(0, 'A', 'T', 0, 1, 600, 840))
    belief = EvolvingTripBelief(graph, public_occurrences=False)
    scenario = JourneyScenario({0: VehicleEvent(0, False, 600, 840)}, minute=60,
                                publications={'0:departure': 660})
    seen = set()
    for time in (620, 660):
        before = set(seen)
        messages = generate_delay_observations(graph, 0, time, '', None,
                                               scenario=scenario, seen_events=seen)
        belief.observe_poll(0, time, messages, before)
    times, probabilities, cancel = belief.event_distribution(
        0, 'departure', {'A': dict(mean=0, std=1, cancel_rate=.2)})
    assert times[probabilities.argmax()] == 600
    assert probabilities.max() == 1
    assert cancel == 0


def test_synthetic_empty_poll_is_retained_after_boarding_paths_converge():
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=120))
    rows = [('B', 0, 1, 900, 960), ('A', 0, 2, 620, 650),
            ('A', 1, 3, 1080, 1140), ('C', 1, 3, 1140, 1200)]
    for cid, (trip, start, end, dep, arr) in enumerate(rows):
        graph.add_connection(Connection(cid, trip, trip, start, end, dep, arr))
    parameters = {route: dict(mean=0, std=1, cancel_rate=0) for route in 'ABC'}
    belief = EvolvingTripBelief(graph, public_occurrences=True,
                                travel_sd=.3, dwell_sd=.3, gap_sd=.2)
    router = CausalPlanningRouter(graph, parameters, event_belief=belief, max_time=1800)
    events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections}
    events[1] = VehicleEvent(120, False, 740, 770)
    events[2] = VehicleEvent(120, False, 1200, 1260)
    initial = CommittedState(0, 600, None, 600, initialized=True)
    model = build_committed_model(graph, [(1., JourneyScenario(events, minute=60))],
                                  0, 3, 600, 2400, initial_state=initial)
    states, beliefs = [], []
    for actions in [(('select', 0),), (WAIT, ('select', 0))]:
        state, branch = initial, belief.copy()
        branch.observe_poll(0, 600, [], set())
        for action in actions:
            outcome = model.transition(model.scenarios[0], state, action)
            router._advance_belief(branch, state, outcome.state, outcome.terminal)
            state = outcome.state
        states.append(state)
        beliefs.append(branch)
    assert replace(states[0], poll_history=()) == replace(states[1], poll_history=())
    assert states[0] != states[1]

    def tail(state, branch, actions):
        return nominal_action(graph, state, 3, branch.mean_events(parameters), set(range(4)),
                              2400, 1500, minute=60)

    search = BudgetedPolicySearch(model, belief, tail, router._advance_belief,
                                  np.random.default_rng(2), 512, 1800)
    assert [search._tail(s, b, model.actions(s)) for s, b in zip(states, beliefs)] == [
        ('select', 2), ('select', 3)]
    search.tree = {state: {} for state in states}
    assert len(search.tree) == 2
