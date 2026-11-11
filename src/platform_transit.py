"""Finite causal reference for interruptible, locally observed platform waiting.

Local departures are observable at occurrence. This idealized reference permits
a previously issued boarding rule to react then; it does not infer door-open
or human reaction times from an actual departure timestamp. No remote actual
events are published implicitly. Explicit publications retain their timestamps.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace

from .information_planner import FiniteScenarioModel, Outcome, Scenario

OBSERVE = ('observe',)
WAIT = ('wait',)
ALIGHT = ('alight',)


@dataclass(frozen=True)
class PlatformState:
    stop: int
    time: int
    current_trip: str | None = None
    last_arrival: int = 0
    last_connection: int | None = None
    used_connections: frozenset = frozenset()
    visible_connections: tuple = ()
    public_log: tuple = ()
    initialized: bool = False
    local_since: int = 0


def build_platform_model(graph, bank, origin, destination, departure, deadline, *,
                         transfer_time=None, footpaths=(), poll_interval=60,
                         initial_state=None):
    """WAIT ends at a local event or timer, never at an unobserved remote event.

    transfer_time(state,cid) returns an operational requirement in clock ticks;
    None means that this pair's feasibility is unidentified and is not offered.
    Footpaths are explicit (from_stop,to_stop,positive_duration) records. Their
    absence is a declared network scope, not a claim that no real walk exists.
    Full optimality refers only to this exact finite-world execution model.
    """
    bank = tuple(bank)
    connections = {c.id: c for c in graph.connections}
    paths = tuple(footpaths)
    for _, _, duration in paths:
        if duration <= 0:
            raise ValueError('The physical footpath reference uses positive durations')

    def ready(state, c):
        if state.current_trip is not None and c.trip_id != state.current_trip:
            return None  # Walking to another vehicle starts only after alighting.
        if state.current_trip is None and state.last_connection is None:
            return state.time
        if state.current_trip is not None and c.trip_id == state.current_trip:
            return state.time
        duration = (transfer_time(state, c.id) if transfer_time is not None
                    else graph.stops[state.stop].min_transfer_time)
        return None if duration is None else max(state.time, state.last_arrival + duration)

    def timelines(scene):
        local, announced = defaultdict(list), []
        for cid, event in scene.events.items():
            if cid in connections and not event.canceled:
                c = connections[cid]
                local[c.dep_stop].append((event.departure, cid, 'departure'))
                local[c.arr_stop].append((event.arrival, cid, 'arrival'))
        for key, receipt in (scene.publications or {}).items():
            raw_cid, kind = key.split(':', 1)
            cid = int(raw_cid)
            if cid in connections and cid in scene.events and kind in ('cancel', 'departure', 'arrival'):
                announced.append((receipt, cid, 'published_' + kind))
        return {stop: sorted(events) for stop, events in local.items()}, sorted(announced)

    schedules = [timelines(scene) for _, scene in bank]

    def between(events, start, end):
        return events[bisect_left(events, (start,)):bisect_right(events, (end, float('inf')))]

    def publish(timeline, state):
        local, announced = timeline
        local_events = local.get(state.stop, ())
        visible = []
        for _, cid, kind in between(local_events, state.time, state.time):
            if kind != 'departure':
                continue
            c = connections[cid]
            r = ready(state, c)
            if (c.id not in state.used_connections and r is not None and r <= state.time
                    and (c.pickup_allowed or state.current_trip == c.trip_id)):
                visible.append(c.id)
        last_logged = state.public_log[-1][0] if state.public_log else departure
        local_start = max(state.local_since, last_logged)
        new_events = between(local_events, local_start, state.time)
        new_announced = between(announced, max(last_logged, departure), state.time)
        log = tuple(sorted(set(state.public_log).union(new_events, new_announced)))
        return replace(state, visible_connections=tuple(sorted(visible)), public_log=log, initialized=True)

    def actions(state):
        if state.stop == destination or state.time >= deadline:
            return ()
        if not state.initialized:
            return (OBSERVE,)
        walking = tuple(('walk', i) for i, (a, _, _) in enumerate(paths)
                        if a == state.stop and state.current_trip is None)
        alight = (ALIGHT,) if state.current_trip is not None else ()
        return (WAIT,) + alight + tuple(('board', cid) for cid in state.visible_connections) + walking

    def transition(world, state, action):
        index = world.factors['bank_index']
        scene, timeline = bank[index][1], schedules[index]
        if action not in actions(state):
            raise ValueError('Action is outside the currently observed physical menu')
        before = state.time
        if action == OBSERVE:
            updated = publish(timeline, state)
        elif action == WAIT:
            staying = [cid for cid in state.visible_connections
                       if state.current_trip is not None and connections[cid].trip_id == state.current_trip]
            if staying:
                # Physical inertia applies even while a planner is busy.
                # Only an explicit ALIGHT leaves a vehicle already boarded.
                return transition(world, state, ('board', min(staying)))
            until = min(deadline, state.time + poll_interval)
            local, announced = timeline
            upcoming = between(local.get(state.stop, ()), before + 1, until)
            next_messages = between(announced, before + 1, until)
            time = min(until, upcoming[0][0] if upcoming else until,
                       next_messages[0][0] if next_messages else until)
            updated = publish(timeline, replace(state, time=time))
        elif action == ALIGHT:
            updated = publish(timeline, replace(state, current_trip=None, last_arrival=state.time))
        elif action[0] == 'walk':
            _, stop, duration = paths[action[1]]
            time = min(deadline, before + duration)
            reached = stop if before + duration <= deadline else state.stop
            updated = publish(timeline, replace(state, stop=reached, time=time, current_trip=None,
                last_arrival=time, last_connection=None, local_since=time))
        else:
            cid = action[1]
            c, event = connections[cid], scene.events[cid]
            time = min(deadline, event.arrival)
            updated = publish(timeline, replace(state, stop=c.arr_stop if event.arrival <= deadline else state.stop,
                time=time, current_trip=c.trip_id or f'connection:{cid}',
                last_arrival=time, last_connection=cid, used_connections=state.used_connections | {cid},
                local_since=time))
        return Outcome(updated, updated.public_log, updated.time - before,
                       updated.stop == destination or updated.time >= deadline)

    def terminal_cost(world, state):
        return 0 if state.stop == destination else deadline - state.time

    initial = initial_state or PlatformState(origin, departure, last_arrival=departure, local_since=departure)
    worlds = tuple(Scenario(str(i), p, {'bank_index': i}) for i, (p, _) in enumerate(bank))
    return FiniteScenarioModel(worlds, initial, actions, transition,
        deadline - initial.time + 2 * len(connections) + 3, terminal_cost)
