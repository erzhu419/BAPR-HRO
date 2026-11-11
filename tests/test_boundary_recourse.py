from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.boundary_recourse import BoundaryRecourseBounds, choose_boundary_tail
from src.committed_transit import CommittedState, build_committed_model
from src.tail_rescue import TailRescueBounds, freeze_tail_plan
from tests.test_risk_bounds import SmallGrid, bank, graph


def prefix_exit(plan, model, world, timetable):
    state, active = model.initial_state, plan.prefix
    for _ in range(model.horizon):
        if not active.primary:
            return state, True
        action = active.action(state, timetable, minute=1)
        outcome = model.transition(world, state, action)
        caught = any(kind == 'board' for kind, _, _ in outcome.state.feedback[len(state.feedback):])
        state = outcome.state
        if action[0] == 'select' and not caught:
            return state, False
        active = active.advance(action, caught)
        if outcome.terminal:
            return state, not active.primary
    raise AssertionError('Prefix did not finish in the model horizon')


def tail_cost(plan, model, world, timetable):
    state = model.initial_state
    for _ in range(model.horizon):
        if not model.actions(state) or not plan.primary:
            break
        action = plan.action(state, timetable, minute=1)
        outcome = model.transition(world, state, action)
        caught = any(kind == 'board' for kind, _, _ in outcome.state.feedback[len(state.feedback):])
        plan = plan.advance(action, caught)
        state = outcome.state
        if outcome.terminal:
            break
    return state.time - model.initial_state.time + model.terminal_cost(world, state)


@pytest.mark.parametrize('same_trip_tail', [True, False])
def test_boundary_value_and_every_conditional_tail_match_complete_joint_bank(same_trip_tail):
    if same_trip_tail:
        rows = [('R', 'A', 0, 1, 2, 3), ('R', 'A', 1, 2, 4, 5),
                ('S', 'B', 1, 2, 5, 6)]
        prefix, paths, destination, q = (0,), ((1,), (2,)), 2, .2
    else:
        rows = [('R', 'A', 0, 1, 2, 3), ('R', 'A', 1, 2, 3, 4),
                ('S', 'B', 2, 3, 5, 6), ('S', 'C', 2, 3, 7, 8)]
        prefix, paths, destination, q = (0, 1), ((2,), (3,)), 3, 0.
    timetable = graph(rows, transfer=1)
    belief = SmallGrid(timetable, minute=1, public_occurrences=False, travel_sd=.5, dwell_sd=.5)
    parameters = {'R': dict(mean=0., std=.7, cancel_rate=q),
                  'S': dict(mean=0., std=.7, cancel_rate=0.)}
    state = CommittedState(0, 0, None, 0, initialized=True)
    bounds = BoundaryRecourseBounds(belief, parameters, timetable, destination, 20)
    tails = tuple(freeze_tail_plan(bounds, path, first_action_now=False) for path in paths)
    plan, value = bounds.evaluate(state, prefix, tails)
    incoming = bounds.prefix_arrivals(state, prefix)
    assert incoming.success_probability == pytest.approx(value.prefix_success_probability)
    worlds = bank(belief, parameters)
    model = build_committed_model(timetable, worlds, 0, destination, 0, 20, initial_state=state)
    expected, phase_mass, phase_loss = 0., {}, {}
    for world in model.scenarios:
        child, succeeded = prefix_exit(plan, model, world, timetable)
        if not succeeded:
            expected += world.probability * 20
            continue
        remaining = SimpleNamespace(initial_state=child, horizon=model.horizon, actions=model.actions,
                                    transition=model.transition, terminal_cost=model.terminal_cost)
        losses = np.array([tail_cost(tail, remaining, world, timetable) for tail in tails])
        phase_mass[child.time] = phase_mass.get(child.time, 0.) + world.probability
        phase_loss[child.time] = phase_loss.get(child.time, np.zeros(len(tails))) + world.probability * losses
        chosen = dict(plan.choices)[child.time]
        expected += world.probability * (child.time + losses[chosen])
    assert value.upper == pytest.approx(expected, abs=1e-10)
    for phase in value.phases:
        assert phase.probability == pytest.approx(phase_mass[phase.arrival])
        np.testing.assert_allclose(phase.conditional_uppers,
                                   phase_loss[phase.arrival] / phase_mass[phase.arrival], atol=1e-10)
    assert value.upper <= min(value.fixed_tail_uppers) + 1e-10
    if same_trip_tail:
        assert value.prefix_success_probability < 1
    else:
        assert len({phase.selected for phase in value.phases}) == 2
        assert value.upper < min(value.fixed_tail_uppers) - .01
    assert plan.tail_for(-999) == tails[0]
    assert belief._known_times == belief._known_status == {}
    with pytest.raises(ValueError, match='past incoming-trip'):
        bounds.evaluate(state, prefix, (replace(tails[0], primary=(prefix[0],)),))


def test_conditional_family_retains_frozen_rescue_and_phase_aligned_waits():
    timetable = graph([('R', 'A', 0, 1, 2, 3), ('S', 'B', 1, 2, 4, 5),
                       ('R', 'C', 1, 2, 19, 20)], transfer=1)
    belief = SmallGrid(timetable, minute=1, public_occurrences=False, travel_sd=.5)
    parameters = {'R': dict(mean=0., std=.7, cancel_rate=0.),
                  'S': dict(mean=0., std=.7, cancel_rate=.2)}
    state = CommittedState(0, 0, None, 0, initialized=True)
    bounds = BoundaryRecourseBounds(belief, parameters, timetable, 2, 25)
    tails = (freeze_tail_plan(bounds, (1,), first_action_now=False, failure_index=0, rescue=(2,)),
             freeze_tail_plan(bounds, (2,), first_action_now=False))
    plan, value = bounds.evaluate(state, (0,), tails)
    worlds = bank(belief, parameters)
    model = build_committed_model(timetable, worlds, 0, 2, 0, 25, initial_state=state)
    expected = 0.
    for world in model.scenarios:
        child, succeeded = prefix_exit(plan, model, world, timetable)
        assert succeeded
        remaining = SimpleNamespace(initial_state=child, horizon=model.horizon, actions=model.actions,
                                    transition=model.transition, terminal_cost=model.terminal_cost)
        expected += world.probability * (child.time + tail_cost(plan.tail_for(child.time), remaining,
                                                               world, timetable))
    assert value.upper == pytest.approx(expected, abs=1e-10)
    assert value.upper < value.fixed_tail_uppers[1]


def test_offgrid_observed_arrival_reselects_from_public_posterior_and_frozen_family():
    timetable = graph([('R', 'A', 0, 1, 2, 3), ('R', 'A', 1, 2, 3, 4),
                       ('S', 'B', 2, 3, 5, 6), ('S', 'C', 2, 3, 7, 8)], transfer=1)
    belief = SmallGrid(timetable, minute=1, public_occurrences=False, travel_sd=.5, dwell_sd=.5)
    parameters = {route: dict(mean=0., std=.7, cancel_rate=0.) for route in ('R', 'S')}
    root = CommittedState(0, 0, None, 0, initialized=True)
    bounds = BoundaryRecourseBounds(belief, parameters, timetable, 3, 20)
    tails = tuple(freeze_tail_plan(bounds, (cid,), first_action_now=False) for cid in (2, 3))
    plan, original = bounds.evaluate(root, (0, 1), tails)
    assert 6 not in {phase.arrival for phase in original.phases}
    assert plan.tail_for(6) == tails[0]

    # Only receipts from the two actually traversed prefix connections arrive.
    public = belief.copy()
    for cid, kind, delay in [(0, 'departure', 0), (0, 'arrival', 0),
                             (1, 'departure', 0), (1, 'arrival', 2)]:
        public.observe_connection_event(cid, kind, delay)
    observed = replace(root, stop=2, time=6, last_arrival=6, current_trip='A',
                       unavailable_ids=frozenset({0, 1}))
    updated = TailRescueBounds(public, parameters, timetable, 3, 20)
    choice = choose_boundary_tail(updated, observed, tails)
    assert choice.selected == 1
    assert choice.upper < choice.conditional_uppers[0]
    assert tails == plan.tails  # the callback did not regenerate paths/thresholds
    worlds = bank(public, parameters)
    model = build_committed_model(timetable, worlds, 2, 3, 6, 20, initial_state=observed)
    expected = tuple(sum(world.probability * tail_cost(tail, model, world, timetable)
                         for world in model.scenarios) for tail in tails)
    np.testing.assert_allclose(choice.conditional_uppers, expected, atol=1e-10)
