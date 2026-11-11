"""Minute-resolution transit adapter for the finite causal reference solver.

The observation convention is explicit: departures and arrivals are globally
published when they occur, cancellations at scheduled departure. This is a
small-network reference model, not the local-feed executor in simulator.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from .information_planner import FiniteScenarioModel, Outcome, Scenario
from .simulator import JourneyScenario
from .transit_graph import TransitGraph


OBSERVE = ("observe",)
WAIT = ("wait",)


@dataclass(frozen=True)
class TransitState:
    stop: int
    time: int
    onboard_trip: tuple | None
    transfer_ready: int
    used_connections: frozenset[int] = frozenset()
    visible_connections: tuple[int, ...] = ()
    public_log: tuple = ()
    initialized: bool = False


def build_transit_model(
    graph: TransitGraph,
    bank: list[tuple[float, JourneyScenario]],
    origin: int,
    destination: int,
    departure: int,
    deadline: int,
) -> FiniteScenarioModel:
    """Build full/H-compatible causal planning with elapsed-time capped loss.

    Each bank entry is (probability, pre-generated JourneyScenario). A unique
    initial OBSERVE action conditions on the actual public log before boarding.
    Later actions are WAIT for one minute or ('board', connection_id) for a
    currently departing visible vehicle. The same-trip continuation buffer is
    zero; after changing vehicles the stop's minimum transfer time applies.
    Every unsuccessful journey costs deadline-departure, including horizon
    exhaustion; successful journeys cost their actual elapsed duration.
    """
    if deadline < departure:
        raise ValueError("Deadline must not precede departure")
    bank = tuple(bank)
    connections = {connection.id: connection for connection in graph.connections}
    for _, scene in bank:
        for cid in connections:
            event = scene.events[cid]
            if not event.canceled and event.arrival < event.departure:
                raise ValueError("Actual connection arrival must not precede departure")

    def trip_key(connection):
        return (("trip", connection.trip_id) if connection.trip_id
                else ("connection", connection.id))

    def public_state(scene, stop, time, onboard, ready, used):
        log = []
        visible = []
        for cid, connection in connections.items():
            event = scene.events[cid]
            if event.canceled:
                if connection.dep_time <= time:
                    log.append((connection.dep_time, cid, "cancel"))
                continue
            if event.departure <= time:
                log.append((event.departure, cid, "departure"))
            if event.arrival <= time:
                log.append((event.arrival, cid, "arrival"))
            if (connection.dep_stop == stop and event.departure == time
                    and cid not in used
                    and (connection.pickup_allowed or onboard == trip_key(connection))
                    and (onboard == trip_key(connection) or time >= ready)):
                visible.append(cid)
        return TransitState(stop, time, onboard, ready, used,
                            tuple(sorted(visible)), tuple(sorted(log)), True)

    def actions(state):
        if state.stop == destination or state.time >= deadline:
            return ()
        if not state.initialized:
            return (OBSERVE,)
        return (WAIT,) + tuple(("board", cid) for cid in state.visible_connections)

    def transition(world, state, action):
        scene = bank[world.factors["bank_index"]][1]
        if action == OBSERVE:
            updated = public_state(scene, state.stop, state.time, state.onboard_trip,
                                   state.transfer_ready, state.used_connections)
            return Outcome(updated, updated.public_log, 0)
        if action == WAIT:
            # Ignoring this vehicle's departure means the passenger alights;
            # its later visit cannot receive a false same-vehicle exemption.
            leaves = any(trip_key(connections[cid]) == state.onboard_trip
                         for cid in state.visible_connections)
            onboard = None if leaves else state.onboard_trip
            time = min(deadline, state.time + 1)
            updated = public_state(scene, state.stop, time, onboard,
                                   state.transfer_ready, state.used_connections)
            return Outcome(updated, updated.public_log, time - state.time,
                           time == deadline)
        cid = action[1]
        if action[0] != "board" or cid not in state.visible_connections:
            raise ValueError("Boarding requires a currently departing observed vehicle")
        connection, event = connections[cid], scene.events[cid]
        used = state.used_connections | {cid}
        if event.arrival > deadline:
            updated = public_state(scene, state.stop, deadline, None,
                                   state.transfer_ready, used)
            return Outcome(updated, updated.public_log, deadline - state.time, True)
        updated = public_state(scene, connection.arr_stop, event.arrival,
                               trip_key(connection),
                               event.arrival + graph.stops[connection.arr_stop].min_transfer_time,
                               used)
        return Outcome(updated, updated.public_log, event.arrival - state.time,
                       connection.arr_stop == destination or event.arrival == deadline)

    def terminal_cost(world, state):
        return 0 if state.stop == destination else max(0, deadline - state.time)

    scenarios = tuple(Scenario(str(i), probability, {"bank_index": i})
                      for i, (probability, _) in enumerate(bank))
    # WAIT increases time; every ride consumes a distinct connection, including
    # zero-minute rides. One extra stage covers the forced initial observation.
    horizon = deadline - departure + len(connections) + 1
    return FiniteScenarioModel(scenarios,
                               TransitState(origin, departure, None, departure),
                               actions, transition, horizon, terminal_cost)
