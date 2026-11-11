"""Measured background publication for the no-feed commitment contract."""

from dataclasses import dataclass, replace
from math import ceil
from time import perf_counter, process_time

from .committed_transit import WAIT


@dataclass(frozen=True)
class ContinuationUpdate:
    plan: object = None
    retain: bool = False
    defer: bool = False
    work: dict | None = None


def replay_continuation(model, world, initial_plan, plan_update, observe, applicable, known,
                        *, graph, deadline, minute, charged=True):
    """The incumbent may commit while the worker is busy.

    A commitment in progress cannot be interrupted. Wait timers keep their
    original phase through worker completion. A proposed replacement must
    survive a public-state check; an obsolete update still consumes work.
    """
    state, active = model.initial_state, initial_plan
    due = state.time
    decisions, actions = [], []
    missing = None
    terminal = not model.actions(state)

    def step(clock, background):
        nonlocal state, active, due, terminal, missing
        state = replace(state, time=clock)
        if clock >= deadline:
            terminal = True
            return False
        action = active.action(state, graph, minute=minute) if active is not None else WAIT
        if action == WAIT:
            due = min(deadline, clock + minute)
            return False
        if action not in model.actions(state):
            active = None
            due = min(deadline, clock + minute)
            return False
        cid = action[1]
        if not known(state, cid):
            missing = cid
            return True
        before = state
        outcome = model.transition(world, state, action)
        state, terminal = outcome.state, outcome.terminal
        feedback = state.feedback[len(before.feedback):]
        caught = bool(feedback and feedback[-1][0] == 'board')
        observe(before, state, terminal)
        active = active.advance(action, caught)
        if active is not None and not active.primary:
            active = None
        due = state.time
        actions.append(dict(clock=clock, next_clock=state.time, stop=before.stop,
                            next_stop=state.stop, connection=cid, caught=caught,
                            feedback=feedback, while_computing=background))
        return True

    while not terminal and state.time < deadline and missing is None:
        ordinal = len(decisions)
        requested_at = state.time
        active_first = active.primary[0] if active and active.primary else None
        started, cpu = perf_counter(), process_time()
        proposal = plan_update(state, active, ordinal)
        wall, cpu_seconds = perf_counter() - started, process_time() - cpu
        release = requested_at + (ceil(wall) if charged else 0)
        acted = False
        if active is None:
            due = max(due, min(release, deadline))
        while due < release and not terminal and missing is None:
            acted = step(due, True) or acted
        receipt = dict(clock=requested_at, ready_at=release, wall_seconds=wall,
                       cpu_seconds=cpu_seconds, active_first=active_first,
                       **(proposal.work or {}))
        if terminal or missing is not None:
            receipt['publication'] = 'closed_during_compute'
        elif acted:
            receipt['publication'] = 'incumbent_advanced'
        else:
            state = replace(state, time=min(deadline, max(state.time, release)))
            if state.time >= deadline:
                terminal = True
                receipt['publication'] = 'deadline'
            elif proposal.retain:
                receipt['publication'] = 'retained'
            elif proposal.defer:
                if proposal.plan is None or proposal.plan is active or applicable(state, proposal.plan):
                    active = proposal.plan
                due = min(deadline, state.time + minute)
                receipt['publication'] = 'wait'
            elif proposal.plan is not None and applicable(state, proposal.plan):
                active = proposal.plan
                due = state.time
                receipt['publication'] = 'installed'
            else:
                receipt['publication'] = 'inapplicable'
            if not terminal and due <= state.time:
                step(state.time, False)
        decisions.append(receipt)
        if not terminal and missing is None and due > state.time:
            state = replace(state, time=due)

    return state, missing, decisions, actions
