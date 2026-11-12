import pytest

from src.conditional_rescue_policy import ConditionalItineraries
from src.evolving_trip_belief import EvolvingTripBelief
from src.local_transfer_rescue import compile_local_transfer_rescue
from src.local_transfer_value import add_transfer_rescue_scores
from src.platform_transit import PlatformState, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.tail_rescue import TailRescuePlan
from src.transit_graph import Connection, Stop, TransitGraph


@pytest.mark.parametrize('target_departure,expected', [(29, 56), (30, 47), (31, 47), (33, 40)])
def test_transfer_score_uses_the_same_public_miss_time_and_fallback_as_execution(target_departure, expected):
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=120))
    for i, (origin, target, departure, arrival) in enumerate((
        (0, 1, 20, 30), (1, 2, 31, 40), (1, 2, 36, 49), (1, 2, 36, 47), (1, 2, 47, 56),
    )):
        graph.add_connection(Connection(-1, 'R', str(i), origin, target, 60*departure, 60*arrival))
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False)
    events = {}
    for c in graph.connections:
        departure = 60*target_departure if c.id == 1 else c.dep_time
        delay = departure-c.dep_time
        belief.observe_connection_event(c.id, 'departure', delay)
        belief.observe_connection_event(c.id, 'arrival', 0)
        events[c.id] = VehicleEvent(delay, False, departure, c.arr_time)
    root = PlatformState(0, 600, last_arrival=600, local_since=600)
    goal, deadline = 2, 3900
    plan = TailRescuePlan((0, 1))
    pars = {'R': {'mean': 0., 'std': 1., 'cancel_rate': 0.}}
    scores = ConditionalItineraries(belief, pars, graph, root, goal, deadline, {0: plan})
    recourse = compile_local_transfer_rescue(graph, plan, root, goal, deadline)
    original = scores.rows[0][1200]
    info = add_transfer_rescue_scores(scores, belief, pars, goal, {0: recourse})
    assert original.expected_arrival_or_deadline == (2400 if target_departure == 33 else deadline)
    assert scores.rows[0][1200].expected_arrival_or_deadline == pytest.approx(60*expected)
    assert scores.rows[0][1200].success_probability == pytest.approx(1.)
    assert info[0][1200]['failure_probability'] == pytest.approx(float(target_departure != 33))

    model = build_platform_model(graph, [(1., JourneyScenario(events, minute=60, publications={}))],
                                 0, goal, root.time, deadline, initial_state=root, poll_interval=60)
    state = model.initial_state
    binding = recourse.bind('test')
    for _ in range(model.horizon):
        action = binding.action(state)
        outcome = model.transition(model.scenarios[0], state, action)
        state = outcome.state
        if outcome.terminal:
            break
    assert state.stop == goal and state.time == 60*expected
