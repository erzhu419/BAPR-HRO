from dataclasses import replace

import pytest

from src.committed_transit import CommittedState, build_committed_model
from src.tail_rescue import TailRescueBounds, compile_tail_rescue, freeze_tail_plan
from tests.test_risk_bounds import SmallGrid, bank, graph


def network():
    timetable = graph([('R', 'A', 0, 1, 60, 120), ('S', 'B', 1, 2, 180, 240),
                       ('R', 'RESCUE', 1, 2, 960, 1020)], transfer=61)
    belief = SmallGrid(timetable, minute=60, public_occurrences=False,
                       travel_sd=.5, dwell_sd=.5)
    parameters = {'R': dict(mean=0., std=.7, cancel_rate=0.),
                  'S': dict(mean=0., std=.7, cancel_rate=.2)}
    return timetable, belief, parameters


def run(plan, model, world, timetable):
    state, attempted = model.initial_state, []
    for _ in range(model.horizon):
        if not model.actions(state) or not plan.primary:
            break
        action = plan.action(state, timetable)
        outcome = model.transition(world, state, action)
        if action[0] == 'select':
            attempted.append(action[1])
        caught = any(kind == 'board' for kind, _, _ in outcome.state.feedback[len(state.feedback):])
        plan = plan.advance(action, caught)
        state = outcome.state
        if outcome.terminal:
            break
    return state.time - model.initial_state.time + model.terminal_cost(world, state), attempted


def test_one_downstream_rescue_matches_the_exact_committed_bank():
    timetable, belief, parameters = network()
    parameters['R']['cancel_rate'] = .1
    state = CommittedState(0, 0, None, 0, initialized=True)
    worlds = bank(belief, parameters)
    model = build_committed_model(timetable, worlds, 0, 2, 0, 1200, initial_state=state)
    bounds = TailRescueBounds(belief, parameters, timetable, 2, 1200)
    plan, value = compile_tail_rescue(bounds, state, (0, 1))
    assert (plan.failure_index, plan.rescue) == (1, (2,))
    results = [run(plan, model, scenario, timetable) for scenario in model.scenarios]
    expected = sum(p * result[0] for (p, _), result in zip(worlds, results))
    assert value.upper == pytest.approx(expected)
    assert value.upper == pytest.approx(bounds.evaluate(state, plan).upper)
    assert value.upper < value.primary.upper
    assert 0 < value.rescue_completion_probability <= value.failed_probability < 1
    assert [0] in [attempted for _, attempted in results]
    assert [0, 1, 2] in [attempted for _, attempted in results]
    assert [0, 1] in [attempted for _, attempted in results]
    with pytest.raises(ValueError, match='exclude'):
        bounds.evaluate(state, replace(plan, rescue=(0,)))


def test_remainder_preserves_frozen_thresholds_and_the_designated_failure():
    timetable, belief, parameters = network()
    root = TailRescueBounds(belief, parameters, timetable, 2, 1200)
    plan = freeze_tail_plan(root, (0, 1), failure_index=1, rescue=(2,))
    remainder = plan.advance(('select', 0), True)
    assert (remainder.primary, remainder.failure_index, remainder.rescue) == ((1,), 0, (2,))
    assert not remainder.first_action_now and remainder.departures == plan.departures

    state = CommittedState(1, 120, 'A', 120, frozenset({0}), initialized=True)
    belief.observe_connection_event(0, 'arrival', 0)
    belief.observe_forecast(1, 'departure', 360, 25)
    worlds = bank(belief, parameters)
    model = build_committed_model(timetable, worlds, 1, 2, 120, 1200, initial_state=state)
    bound = TailRescueBounds(belief, parameters, timetable, 2, 1200).evaluate(state, remainder)
    expected = sum(p * run(remainder, model, scenario, timetable)[0]
                   for (p, _), scenario in zip(worlds, model.scenarios))
    assert bound.upper == pytest.approx(expected)
    fallback = remainder.advance(('select', 1), False)
    assert fallback.primary == (2,) and fallback.failure_index is None
    assert not fallback.rescue and not fallback.first_action_now
