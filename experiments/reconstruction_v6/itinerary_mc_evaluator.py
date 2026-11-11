"""Execute a fixed public itinerary/rescue through the real committed model."""
from dataclasses import dataclass

from src.committed_transit import WAIT


@dataclass(frozen=True)
class FixedWitness:
    primary: tuple
    rescue: tuple = ()
    first_action_now: bool = True


def execute_fixed_witness(model, world, graph, policy, departures, deadline, *, minute=60, lookahead=None):
    """A future action depends only on public state and root-frozen thresholds.

    The first root miss switches to the fixed rescue; any later miss stops
    until deadline. No current-world clock or updated posterior chooses an
    action. Every WAIT/select is a counted committed-model transition.
    """
    lookahead = 25 * minute if lookahead is None else lookahead
    state, cost, work = model.initial_state, 0., 0
    trace = []
    path, position, root_pending = policy.primary, 0, True
    first_now = policy.first_action_now
    while position < len(path) and state.stop != graph.connections[path[-1]].arr_stop and state.time < deadline:
        cid = path[position]
        c = graph.connections[cid]
        transfer = (0 if state.current_trip is None or state.current_trip == c.trip_id
                    else graph.stops[state.stop].min_transfer_time)
        if not first_now:
            while state.time < deadline:
                ready = max(state.time, state.last_arrival + transfer)
                opened = c.dep_time <= state.time + lookahead
                nominal_ready = departures[cid] <= ready + 12 * minute
                if opened and nominal_ready:
                    break
                before = state
                outcome = model.transition(world, state, WAIT)
                state, cost, work = outcome.state, cost + outcome.cost, work + 1
                trace.append((before.time, before.stop, WAIT, state.time, state.stop))
                if outcome.terminal:
                    break
        if state.time >= deadline:
            break
        action = ('select', cid)
        if action not in model.actions(state):
            raise ValueError('The fixed witness selected an illegal public action')
        before = state
        outcome = model.transition(world, state, action)
        state, cost, work = outcome.state, cost + outcome.cost, work + 1
        trace.append((before.time, before.stop, action, state.time, state.stop))
        feedback = state.feedback[len(before.feedback):]
        caught = bool(feedback and feedback[-1][0] == 'board')
        if outcome.terminal:
            break
        if caught:
            position += 1
            root_pending = False
        elif root_pending and policy.rescue:
            path, position, root_pending = policy.rescue, 0, False
        else:
            break
        first_now = False
    cost += model.terminal_cost(world, state)
    return {'cost': float(cost), 'transitions': work,
            'arrived': state.stop == graph.connections[policy.primary[-1]].arr_stop if policy.primary else False,
            'trace': trace, 'final_time': state.time, 'final_stop': state.stop}
