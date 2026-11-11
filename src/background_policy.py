"""One common publication schedule while an issued causal policy keeps acting."""
from dataclasses import dataclass
from time import perf_counter, process_time


@dataclass(frozen=True)
class IssuedPolicy:
    action: object
    applicable: object = lambda state: True
    name: str = 'issued'


def measure_plan(planner, public_snapshot):
    """The worker receives only its immutable public query snapshot."""
    wall, cpu = perf_counter(), process_time()
    policy = planner(public_snapshot)
    return policy, {'wall_seconds': perf_counter() - wall, 'cpu_seconds': process_time() - cpu}


def execute_background(model, world, incumbent, candidate, publish_at, *, blocking=False, unresolved=None):
    """Replay worker latency through the physical model before publishing.

    The model must offer public clock ticks no longer than one execution tick
    (platform reference: poll_interval=1). Candidate applicability sees only the
    actual public state. A ride can cross completion: publication is deferred
    to its next decision boundary, where its assumptions are checked again.
    blocking is the measured legacy comparator, not an extra policy privilege.
    """
    state, cost, active = model.initial_state, 0., incumbent
    published = rejected = False
    trace = []
    missing = ()
    for _ in range(model.horizon):
        actions = model.actions(state)
        if not actions:
            break
        if actions == (('observe',),):
            action = ('observe',)
        else:
            if not published and not rejected and state.time >= publish_at:
                if candidate.applicable(state):
                    active, published = candidate, True
                else:
                    rejected = True
            executing = None if blocking and state.time < publish_at else active
            action = ('wait',) if executing is None else executing.action(state)
            missing = unresolved(state, executing, action) if unresolved is not None else ()
            if missing:
                break
            if action not in actions:
                raise ValueError('Issued policy is not executable at this public state')
        before = state
        outcome = model.transition(world, state, action)
        state, cost = outcome.state, cost + outcome.cost
        trace.append(dict(time=before.time, stop=before.stop, action=action,
            next_time=state.time, next_stop=state.stop,
            ridden_connection=state.last_connection if state.used_connections != before.used_connections else None,
            worker_running=before.time < publish_at, policy=active.name))
        if outcome.terminal:
            break
    return dict(cost=None if missing else cost + model.terminal_cost(world, state), state=state,
        published=published, rejected=rejected, trace=trace, missing_connections=missing,
        incumbent_actions_during_compute=sum(t['worker_running'] and
            (t['ridden_connection'] is not None or t['action'][0] == 'walk') for t in trace))
