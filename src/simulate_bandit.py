"""Execute static and online routers in a fixed, causally observed vehicle world."""

from __future__ import annotations

import numpy as np
import math
from time import perf_counter
from typing import Optional

from .transit_graph import TransitGraph
from .adaptive_bandit_router import AdaptiveBetaBanditRouter
from .matched_router import MatchedRouter
from .simulator import (
    JourneyEvent, JourneyResult, RegimeSchedule, JourneyScenario,
    generate_delay_observations, set_regime_dist_fn, _regime_dist_cache,
)


def simulate_bandit_journey(
    graph: TransitGraph,
    router,
    s_source: int,
    s_dest: int,
    t_depart: int,
    regime_schedule: RegimeSchedule,
    rng: np.random.Generator,
    max_time: int = 180,
    scenario: Optional[JourneyScenario] = None,
    charge_computation: bool = False,
) -> JourneyResult:
    """Run one policy without redrawing vehicles or exposing future realizations.

    ``scenario`` is reusable across methods; its RNG is independent of policy
    random choices. Local actual events are published only after occurrence.
    Each trip contributes at most one outcome to a route's posterior per
    journey. A waiting timeout is recorded separately from cancellation.
    Timetable and journey times use the scenario's clock ticks.
    With charge_computation, decision processing advances that same clock,
    rounded up to its tick resolution. A selected vehicle can then be missed.
    """
    if scenario is None:
        scenario = JourneyScenario.sample(
            graph, regime_schedule,
            np.random.default_rng(int(rng.integers(0, 2**63))),
        )
    minute = scenario.minute
    is_online = all(hasattr(router, name) for name in (
        'observe_delay', 'observe_cancel', 'select_connection'))
    if isinstance(router, AdaptiveBetaBanditRouter):
        router.begin_journey()
    decision_started = perf_counter()
    initial = router.route(s_source, s_dest, t_depart)
    hyperpath = initial if is_online else initial.hyperpath
    current_stop, current_time = s_source, t_depart
    last_arrival, current_trip = t_depart, None
    deadline = t_depart + max_time
    unavailable_ids: set[int] = set()
    seen_events: set[str] = set()
    learned_trips: set[str] = set()
    observation_log = []
    n_transfers = 0
    events = [JourneyEvent(t_depart, s_source, 'depart', details='Initial route computed')]
    censored = False

    def learn(connection, canceled: bool, delay: int, kind='departure'):
        if hasattr(router, 'observe_connection_event'):
            router.observe_connection_event(connection.id, 'cancel' if canceled else kind, delay)
        elif hasattr(router, 'observe_vehicle_event'):
            router.observe_vehicle_event(connection.id, canceled, delay)
        key = connection.trip_id or f'connection:{connection.id}'
        if not is_online or key in learned_trips:
            return
        learned_trips.add(key)
        if canceled:
            router.observe_cancel(connection.route, kind='true')
        else:
            router.observe_delay(connection.route, delay / minute if minute != 1 else delay)

    while current_stop != s_dest and current_time < deadline:
        decision_clock = current_time
        seen_before = set(seen_events)
        observations = generate_delay_observations(
            graph, current_stop, current_time,
            regime_schedule.get_regime(current_time), rng,
            scenario=scenario, seen_events=seen_events,
        )
        if hasattr(router, 'observe_public_messages'):
            router.observe_public_messages(current_stop, current_time, observations, seen_before)
        observation_log.extend(observations)
        for obs in observations:
            cid = int(obs.event_id.split(':', 1)[0])
            c_obs = graph.connections[cid]
            learn(c_obs, obs.is_canceled, obs.actual_delay, obs.event_id.split(':', 1)[1])
            if obs.is_canceled or obs.event_id.endswith(':departure'):
                # A departure at this instant remains boardable.
                if obs.is_canceled or obs.actual_time < current_time:
                    unavailable_ids.add(cid)

        if hasattr(router, 'set_execution_context'):
            router.set_execution_context(current_trip, last_arrival, unavailable_ids)
        if hasattr(router, 'set_information_context'):
            router.set_information_context(current_stop, current_time, current_trip, last_arrival,
                                           unavailable_ids, seen_events, learned_trips)
        precommitted = (router.precommitted_connection(current_stop, current_time, current_trip)
                        if hasattr(router, 'precommitted_connection') else None)
        hyperpath = getattr(router, 'cached_result', None) or hyperpath
        labels = hyperpath.stop_labels.get(current_stop, [])
        if (not labels and isinstance(router, MatchedRouter)
                and router._valued_version != router._version):
            # A public observation can restore a connection to the menu.
            # Normally the selector refreshes this cache before scoring.
            router._value()
            hyperpath = router.cached_result
            labels = hyperpath.stop_labels.get(current_stop, [])
        if not labels and not (hasattr(router, 'has_unlabeled_continuation')
                               and router.has_unlabeled_continuation(current_stop)):
            events.append(JourneyEvent(current_time, current_stop, 'stuck'))
            break

        if is_online:
            # Existing selectors share this cache; exclude known impossible
            # connections before scoring so they cannot be retried forever.
            original_labels = hyperpath.stop_labels.get(current_stop, [])
            hyperpath.stop_labels[current_stop] = [
                label for label in original_labels
                if label.connection_id not in unavailable_ids and (
                    graph.connections[label.connection_id].pickup_allowed
                    or graph.connections[label.connection_id].trip_id == current_trip)]
            try:
                selected = router.select_connection(current_stop, current_time, rng, top_k=5)
            finally:
                hyperpath.stop_labels[current_stop] = original_labels
            c = None if selected is None else graph.connections[selected[0].connection_id]
        else:
            c = next((graph.connections[label.connection_id]
                      for label in reversed(labels)
                      if label.connection_id not in unavailable_ids
                      and (graph.connections[label.connection_id].pickup_allowed
                           or graph.connections[label.connection_id].trip_id == current_trip)
                      and current_time - minute <= graph.connections[label.connection_id].dep_time
                      <= current_time + 25 * minute), None)
        elapsed = perf_counter() - decision_started
        issued_stay = c is not None and c.id == precommitted
        if charge_computation:
            if not issued_stay:
                current_time = min(deadline, current_time + math.ceil(elapsed * minute / 60))
            events.append(JourneyEvent(current_time, current_stop, 'compute',
                                       details=f'decision_clock={decision_clock}; seconds={elapsed:.6f}; preissued_stay={issued_stay}'))
        if hasattr(router, 'observe_computation_elapsed'):
            router.observe_computation_elapsed(decision_clock, current_time, elapsed)
        if c is None:
            current_time = min(deadline, current_time + minute)
            decision_started = perf_counter()
            continue

        if c.id in unavailable_ids:
            # A selector that recomputed its cache will see the exclusion on
            # its next invocation, without spending fictitious waiting time.
            current_time = min(deadline, current_time + minute)
            decision_started = perf_counter()
            continue
        if not c.pickup_allowed and c.trip_id != current_trip:
            current_time = min(deadline, current_time + minute)
            decision_started = perf_counter()
            continue
        same_trip = current_trip is not None and c.trip_id == current_trip
        action_started = current_time

        def report_outcome(kind):
            nonlocal decision_started
            if hasattr(router, 'observe_attempt_outcome'):
                router.observe_attempt_outcome(c.id, kind, action_started, current_time)
            decision_started = perf_counter()

        ready = last_arrival + (0 if current_trip is None or same_trip
                               else graph.stops[current_stop].min_transfer_time)
        # Choosing a transfer commits the passenger to the known transfer
        # time before another decision can be made at the boarding platform.
        current_time = min(deadline, max(current_time, ready))
        if current_time >= deadline:
            report_outcome('deadline')
            break
        event = scenario.events.get(c.id)
        if event is None:
            events.append(JourneyEvent(current_time, current_stop, 'unresolved', route=c.route,
                                       details=f'connection={c.id}; measured outcome unavailable'))
            report_outcome('unresolved')
            censored = True
            break
        if event.canceled:
            notice = (c.dep_time if scenario.publications is None
                      else scenario.publications.get(f'{c.id}:cancel'))
            wait_end = min(deadline, current_time + 12 * minute)
            if notice is None or notice > wait_end:
                current_time = wait_end
                events.append(JourneyEvent(current_time, current_stop, 'wait_late', route=c.route,
                                           details=f'connection={c.id}; no vehicle during patience window'))
                report_outcome('wait_late')
                continue
            current_time = max(current_time, notice)
            unavailable_ids.add(c.id)
            learn(c, True, 0)
            events.append(JourneyEvent(current_time, current_stop, 'cancel', route=c.route,
                                       details=f'connection={c.id}; explicit cancellation'))
            report_outcome('cancel')
            continue

        if event.departure < current_time:
            receipt = (event.departure if scenario.publications is None
                       else scenario.publications.get(f'{c.id}:departure'))
            if receipt is None or receipt > current_time:
                current_time = min(deadline, current_time + 12 * minute)
                events.append(JourneyEvent(current_time, current_stop, 'wait_late', route=c.route,
                                           details=f'connection={c.id}; no vehicle during patience window'))
                report_outcome('wait_late')
                continue
            unavailable_ids.add(c.id)
            report_outcome('missed')
            continue
        wait_end = min(deadline, current_time + 12 * minute)
        if event.departure > wait_end:
            # Censoring is not a cancellation or a realized delay sample.
            current_time = wait_end
            events.append(JourneyEvent(current_time, current_stop, 'wait_late', route=c.route,
                                       details=f'connection={c.id}; patience window'))
            report_outcome('wait_late')
            continue
        if event.departure > deadline:
            current_time = deadline
            break
        learn(c, False, event.departure - c.dep_time)
        events.append(JourneyEvent(event.departure, current_stop, 'board', route=c.route,
                                   details=f'connection={c.id}; to={c.arr_stop}; trip={c.trip_id}'))
        # A connection is one physical segment and cannot be boarded twice,
        # including the zero-minute self-loops present in the Swiss graph.
        unavailable_ids.add(c.id)
        n_transfers += int(current_trip is not None and not same_trip)
        current_trip = c.trip_id or f'connection:{c.id}'
        if event.arrival > deadline:
            current_time = deadline
            report_outcome('board')
            break
        current_time = last_arrival = event.arrival
        current_stop = c.arr_stop
        if hasattr(router, 'observe_connection_event'):
            router.observe_connection_event(c.id, 'arrival', event.arrival - c.arr_time)
        report_outcome('board')
        if current_stop != s_dest:
            events.append(JourneyEvent(current_time, current_stop, 'arrive', route=c.route,
                                       details=f'connection={c.id}; delay={(event.arrival - c.arr_time) / minute:g}min'))

    arrived = current_stop == s_dest
    arrival = current_time if arrived or censored else deadline
    events.append(JourneyEvent(arrival, current_stop, 'finish',
                               details='arrived' if arrived else 'censored' if censored else 'timeout'))
    if isinstance(router, AdaptiveBetaBanditRouter):
        router.end_journey(arrival - t_depart)
    return JourneyResult(
        origin=s_source, destination=s_dest, departure_time=t_depart,
        arrival_time=arrival, events=events, n_transfers=n_transfers,
        n_replans=getattr(router, 'n_replans', 0),
        observations=observation_log,
    )
