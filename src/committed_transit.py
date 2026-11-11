"""Finite-world reference for simulate_bandit_journey's commitment semantics.

The executor's local message function is reused directly (last ten minutes,
maximum five unseen events per decision). Choosing a connection commits to its
transfer time and at most twelve further minutes before another decision.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .information_planner import FiniteScenarioModel, Outcome, Scenario
from .simulator import JourneyScenario, generate_delay_observations
from .transit_graph import TransitGraph


OBSERVE = ("observe",)
WAIT = ("wait",)


@dataclass(frozen=True)
class LocalMessage:
    route: str
    stop_id: int
    scheduled_time: int
    predicted_time: int
    actual_time: int
    timestamp: int
    event_id: str
    is_canceled: bool

    @property
    def actual_delay(self):
        return self.actual_time - self.scheduled_time


@dataclass(frozen=True)
class CommittedState:
    stop: int
    time: int
    current_trip: str | None
    last_arrival: int
    unavailable_ids: frozenset[int] = frozenset()
    seen_events: frozenset[str] = frozenset()
    learned_trips: frozenset[str] = frozenset()
    messages: tuple[LocalMessage, ...] = ()
    learning_log: tuple = ()
    feedback: tuple = ()
    initialized: bool = False


def build_committed_model(
    graph: TransitGraph,
    bank: list[tuple[float, JourneyScenario]],
    origin: int,
    destination: int,
    departure: int,
    deadline: int,
    candidate_ids=None,
    lookahead: int = 25,
    initial_state: CommittedState | None = None,
) -> FiniteScenarioModel:
    """Exact causal full/H planning with the current online executor actions.

    The physical reference menu comprises the supplied fixed connection IDs at
    the current stop, excluding known unavailable IDs, scheduled departures
    beyond time+lookahead and (if supplied) PMF support ending before now. It
    does not inspect future actual departures. Scoring-dependent top-k pruning
    belongs in a separate candidate_rule H, not this common physical menu.

    OBSERVE is forced once initially; subsequent transitions include the local
    publication before the next decision. WAIT represents selector returning
    None. A state retains public observations and once-per-trip estimator
    updates separately, as in the executor. Exact planning uses the known bank;
    it does not imitate an approximate router's estimator or dynamic menus.

    An initial_state continues an existing public history. Its bank must already
    be conditioned on that history. Set initialized=True if this decision's
    local publication already happened; False requests one initial publication.
    Values from such a context are remaining loss; add the elapsed prefix when
    reporting complete journey loss. model.actions and model.transition are pure
    functions usable for rollout without invoking ExactPlanner.
    """
    if deadline < departure:
        raise ValueError("Deadline must not precede departure")
    bank = tuple(bank)
    connections = {c.id: c for c in graph.connections}
    candidate_ids = frozenset(connections if candidate_ids is None else candidate_ids)
    for _, scene in bank:
        for cid in connections:
            event = scene.events[cid]
            if not event.canceled and event.arrival < event.departure:
                raise ValueError("Actual arrival must not precede departure")

    def learn(state, connection, canceled, delay):
        key = connection.trip_id or f"connection:{connection.id}"
        if key in state.learned_trips:
            return state
        return replace(state, learned_trips=state.learned_trips | {key},
                       learning_log=state.learning_log + ((key, connection.route, canceled, delay),))

    def publish(scene, state):
        seen = set(state.seen_events)
        batch = generate_delay_observations(
            graph, state.stop, state.time, "finite_known_bank", None,
            n_obs=5, scenario=scene, seen_events=seen)
        unavailable = set(state.unavailable_ids)
        for observation in batch:
            cid = int(observation.event_id.split(":", 1)[0])
            state = learn(state, connections[cid], observation.is_canceled,
                          observation.actual_delay)
            if observation.is_canceled or observation.event_id.endswith(":departure"):
                if observation.is_canceled or observation.actual_time < state.time:
                    unavailable.add(cid)
        messages = tuple(LocalMessage(**vars(observation)) for observation in batch)
        return replace(state, unavailable_ids=frozenset(unavailable),
                       seen_events=frozenset(seen), messages=state.messages + messages,
                       initialized=True)

    def actions(state):
        if state.stop == destination or state.time >= deadline:
            return ()
        if not state.initialized:
            return (OBSERVE,)
        if not any(c.id in candidate_ids for c in graph.get_connections_from(state.stop)):
            return ()  # fixed hyperpath has no labels here: executor is stuck
        available = []
        for c in graph.get_connections_from(state.stop):
            if (c.id not in candidate_ids or c.id in state.unavailable_ids
                    or c.dep_time > state.time + lookahead):
                continue
            if c.dep_distribution is not None and c.dep_distribution.support_max < state.time:
                continue
            available.append(("select", c.id))
        return (WAIT,) + tuple(available)

    def transition(world, state, action):
        scene = bank[world.factors["bank_index"]][1]
        start_time = state.time
        if action == OBSERVE:
            updated = publish(scene, state)
            return Outcome(updated, updated.messages, 0)
        if action == WAIT:
            state = replace(state, time=min(deadline, state.time + 1))
        else:
            if action not in actions(state) or action[0] != "select":
                raise ValueError("Selected connection is outside the public fixed menu")
            connection = connections[action[1]]
            same_trip = state.current_trip is not None and connection.trip_id == state.current_trip
            ready = state.last_arrival + (
                0 if state.current_trip is None or same_trip
                else graph.stops[state.stop].min_transfer_time)
            state = replace(state, time=min(deadline, max(state.time, ready)))
            if state.time < deadline:
                event = scene.events[connection.id]
                if event.canceled:
                    wait_end = min(deadline, state.time + 12)
                    if state.time < connection.dep_time and connection.dep_time > wait_end:
                        state = replace(state, time=wait_end,
                                        feedback=state.feedback + (("wait_late", connection.id, wait_end),))
                    else:
                        state = replace(state, time=max(state.time, connection.dep_time),
                                        unavailable_ids=state.unavailable_ids | {connection.id})
                        state = learn(state, connection, True, 0)
                        state = replace(state, feedback=state.feedback +
                                        (("cancel", connection.id, state.time),))
                elif event.departure < state.time:
                    state = replace(state, unavailable_ids=state.unavailable_ids | {connection.id},
                                    feedback=state.feedback + (("missed", connection.id, state.time),))
                elif event.departure > min(deadline, state.time + 12):
                    wait_end = min(deadline, state.time + 12)
                    state = replace(state, time=wait_end,
                                    feedback=state.feedback + (("wait_late", connection.id, wait_end),))
                else:
                    state = learn(state, connection, False, event.delay)
                    state = replace(state, unavailable_ids=state.unavailable_ids | {connection.id},
                                    current_trip=connection.trip_id or f"connection:{connection.id}",
                                    feedback=state.feedback +
                                    (("board", connection.id, event.departure),))
                    if event.arrival > deadline:
                        state = replace(state, time=deadline)
                    else:
                        state = replace(state, stop=connection.arr_stop, time=event.arrival,
                                        last_arrival=event.arrival)
        terminal = state.stop == destination or state.time >= deadline
        if not terminal:
            state = publish(scene, state)
        return Outcome(state, (state.messages, state.learning_log, state.feedback),
                       state.time - start_time, terminal)

    def terminal_cost(world, state):
        return 0 if state.stop == destination else max(0, deadline - state.time)

    scenarios = tuple(Scenario(str(i), probability, {"bank_index": i})
                      for i, (probability, _) in enumerate(bank))
    # Every zero-time action removes a connection; all others advance at least
    # one minute. Add the one forced observation to cover every valid journey.
    initial_state = initial_state or CommittedState(origin, departure, None, departure)
    horizon = max(0, deadline - initial_state.time) + len(connections) + 1
    return FiniteScenarioModel(scenarios,
                               initial_state,
                               actions, transition, horizon, terminal_cost)
