"""Journey simulator: simulate a passenger traveling through the network.

Generates realistic delay observations and tracks the passenger as they
board buses, transfer, and eventually arrive at the destination.
Tests static vs adaptive routing under controlled regime shifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from types import MappingProxyType
from collections.abc import Mapping

from .transit_graph import TransitGraph, Connection
from .bocd.regime_detector import DelayObservation
from .synthetic_network import create_bus_story_network, create_regime_distributions
from .router import StaticRouter, AdaptiveRouter, PeriodicRouter, RoutingDecision


@dataclass
class JourneyEvent:
    """A single event during a journey."""
    time: int           # minutes from midnight
    stop_id: int
    event_type: str     # "depart", "arrive", "transfer", "replan", "board"
    route: str = ""
    details: str = ""


@dataclass
class JourneyResult:
    """Result of a simulated journey."""
    origin: int
    destination: int
    departure_time: int
    arrival_time: int       # actual arrival at destination
    events: list[JourneyEvent] = field(default_factory=list)
    n_transfers: int = 0
    n_replans: int = 0
    total_computation_ms: float = 0.0
    regime_changes_encountered: int = 0
    observations: list[DelayObservation] = field(default_factory=list)


@dataclass
class RegimeSchedule:
    """Schedule of regime shifts during simulation.

    Example: [(0, "normal"), (500, "disrupted_402"), (560, "normal")]
    means normal until t=500, then 402 disrupted, then normal again at t=560.
    """
    shifts: list[tuple[int, str]]  # (time, regime_name)

    def get_regime(self, t: int) -> str:
        """Get active regime at time t."""
        regime = self.shifts[0][1]
        for shift_time, shift_regime in self.shifts:
            if t >= shift_time:
                regime = shift_regime
        return regime


_regime_dist_cache: dict[str, dict] = {}
_regime_dist_fn = create_regime_distributions


def set_regime_dist_fn(fn):
    """Set the regime distribution factory (for large network etc.)."""
    global _regime_dist_fn
    _regime_dist_fn = fn
    _regime_dist_cache.clear()


def _get_regime_dists(regime: str) -> dict:
    if regime not in _regime_dist_cache:
        _regime_dist_cache[regime] = _regime_dist_fn(regime)
    return _regime_dist_cache[regime]


def sample_actual_delay(connection: Connection, regime: str, rng: np.random.Generator) -> int:
    """Sample an actual delay for a connection under a given regime.

    Returns delay in minutes (can be negative for early arrivals).
    999 = sentinel for canceled connection.
    """
    dists = _get_regime_dists(regime)
    route = connection.route

    if route in dists:
        info = dists[route]
        # Check cancellation
        probs = info["delay_probs"]
        # Either an explicit probability or a sub-PMF may encode cancellation.
        cancel_prob = info.get("cancel_prob", max(0.0, 1.0 - float(probs.sum())))
        if cancel_prob > 0 and rng.random() < cancel_prob:
            return 999  # canceled

        if probs.sum() <= 0:
            return 999
        offset = info.get("delay_offset", 0)
        probs_norm = probs / probs.sum()
        delay_idx = rng.choice(len(probs_norm), p=probs_norm)
        return offset + delay_idx
    else:
        return int(rng.normal(1, 2))


@dataclass(frozen=True)
class VehicleEvent:
    """One immutable realization of a scheduled connection."""
    delay: int
    canceled: bool
    departure: int
    arrival: int


@dataclass(frozen=True)
class JourneyScenario:
    """Exogenous trip realizations, sampled before any policy decision.

    ``sample`` uses a shared trip delay and cancellation. Measured scenarios
    can supply distinct segment clocks and partial cancellations. An explicit
    publication map controls availability of exact facts, including direct
    cancellation receipts. An empty map supplies only first-hand ride facts.
    """
    events: Mapping[int, VehicleEvent]
    minute: int = 1  # ticks per minute; measured replay uses integer seconds
    publications: Mapping[str, int] | None = None

    def __post_init__(self):
        if self.minute < 1:
            raise ValueError("A minute must contain at least one clock tick")
        object.__setattr__(self, "events", MappingProxyType(dict(self.events)))
        if self.publications is not None:
            object.__setattr__(self, "publications", MappingProxyType(dict(self.publications)))

    @classmethod
    def sample(cls, graph: TransitGraph, regime_schedule: RegimeSchedule,
               rng: np.random.Generator) -> JourneyScenario:
        trips = {}
        events = {}
        for c in sorted(graph.connections, key=lambda c: (c.dep_time, c.id)):
            key = ("trip", c.trip_id) if c.trip_id else ("connection", c.id)
            if key not in trips:
                regime = regime_schedule.get_regime(c.dep_time)
                trips[key] = sample_actual_delay(c, regime, rng)
            delay = trips[key]
            events[c.id] = VehicleEvent(
                delay=delay, canceled=delay == 999,
                departure=c.dep_time + delay if delay != 999 else c.dep_time,
                arrival=c.arr_time + delay if delay != 999 else c.arr_time,
            )
        return cls(events)

    def event(self, connection: Connection) -> VehicleEvent:
        return self.events[connection.id]


@dataclass(frozen=True)
class LazyTripScenario(JourneyScenario):
    """Internal ordered-trip sampler with immutable, deferred event values.

    Its mapping owns fixed draws for every trip before policy execution. The
    trip generator guarantees ordered clocks when those draws are evaluated.
    """

    def __post_init__(self):
        if self.minute < 1:
            raise ValueError("A minute must contain at least one clock tick")
        object.__setattr__(self, 'events', MappingProxyType(self.events))
        if self.publications is not None:
            object.__setattr__(self, 'publications', MappingProxyType(dict(self.publications)))


def generate_delay_observations(
    graph: TransitGraph,
    current_stop: int,
    current_time: int,
    regime: str,
    rng: np.random.Generator,
    n_obs: int = 5,
    scenario: Optional[JourneyScenario] = None,
    seen_events: Optional[set[str]] = None,
) -> list[DelayObservation]:
    """Publish completed local vehicle events, with no future actual times.

    Cancellations are explicitly announced at scheduled departure. A caller
    simulating more than one step supplies one scenario and a persistent set
    of consumed event IDs. No forecasts are modeled by this interface.
    """
    if scenario is None:
        scenario = JourneyScenario.sample(graph, RegimeSchedule([(0, regime)]), rng)
    if scenario.publications is not None and not scenario.publications:
        return []
    observations = []
    pending = []
    for c in graph.get_connections_from(current_stop):
        event = scenario.events.get(c.id)
        if event is None:
            continue
        pending.append((event.departure, c, "cancel" if event.canceled else "departure", c.dep_time))
    for c in graph.get_connections_to(current_stop):
        event = scenario.events.get(c.id)
        if event is not None and not event.canceled:
            pending.append((event.arrival, c, "arrival", c.arr_time))
    released = []
    for actual, c, kind, scheduled in pending:
        publication = (actual if scenario.publications is None
                       else scenario.publications.get(f'{c.id}:{kind}'))
        if publication is not None:
            released.append((publication, actual, c, kind, scheduled))
    for publication, actual, c, kind, scheduled in sorted(released, key=lambda x: (x[0], x[2].id, x[3])):
        event_id = f"{c.id}:{kind}"
        if (not current_time - 10 * scenario.minute <= publication <= current_time
                or (kind != 'cancel' and actual > publication)):
            continue
        if seen_events is not None and event_id in seen_events:
            continue
        observations.append(DelayObservation(
            route=c.route, stop_id=current_stop, scheduled_time=scheduled,
            predicted_time=scheduled, actual_time=publication if kind == 'cancel' else actual,
            timestamp=publication,
            event_id=event_id, is_canceled=kind == "cancel",
        ))
        if seen_events is not None:
            seen_events.add(event_id)
        if len(observations) >= n_obs:
            break
    return observations


def simulate_journey(
    graph: TransitGraph,
    router,  # StaticRouter | AdaptiveRouter | PeriodicRouter
    s_source: int,
    s_dest: int,
    t_depart: int,
    regime_schedule: RegimeSchedule,
    rng: np.random.Generator,
    max_time: int = 300,  # max journey duration in minutes
    scenario: Optional[JourneyScenario] = None,
) -> JourneyResult:
    """Simulate a passenger journey from source to destination.

    The passenger follows the hyperpath: at each stop, they board the
    first available connection from the hyperpath's recommendations.
    At transfer stops, the adaptive router may re-plan.
    """
    events = []
    observation_log = []
    current_stop = s_source
    current_time = t_depart
    n_transfers = 0
    n_replans = 0
    total_comp_ms = 0.0
    regime_changes = 0
    prev_regime = regime_schedule.get_regime(t_depart)
    if scenario is None:
        scenario = JourneyScenario.sample(graph, regime_schedule,
            np.random.default_rng(int(rng.integers(0, 2**63))))
    seen_events: set[str] = set()
    current_trip = None
    last_arrival = t_depart

    # Initial route computation
    decision = router.route(s_source, s_dest, t_depart)
    total_comp_ms += decision.computation_ms
    events.append(JourneyEvent(current_time, current_stop, "depart",
                               details=f"Initial route computed"))

    canceled_routes_at_stop: set[int] = set()  # individual connections, not entire lines
    boarded_connection_ids: set[int] = set()
    prev_stop = -1

    while current_stop != s_dest and current_time < t_depart + max_time:
        # Reset canceled routes when we arrive at a new stop
        if current_stop != prev_stop:
            canceled_routes_at_stop = set()
            prev_stop = current_stop

        # Get current regime
        current_regime = regime_schedule.get_regime(current_time)
        if current_regime != prev_regime:
            regime_changes += 1
            prev_regime = current_regime

        # Generate delay observations at current stop
        observations = generate_delay_observations(
            graph, current_stop, current_time, current_regime, rng,
            scenario=scenario, seen_events=seen_events)
        observation_log.extend(observations)

        # Re-plan only at transfer stops (stops served by multiple routes)
        is_transfer = len(graph.get_routes_at_stop(current_stop)) > 1
        if hasattr(router, 'replan') and is_transfer:
            decision = router.replan(current_stop, s_dest, current_time, observations)
            total_comp_ms += decision.computation_ms
            if decision.recomputed:
                n_replans += 1
                events.append(JourneyEvent(
                    current_time, current_stop, "replan",
                    details=f"regime={decision.regime_name} conf={decision.confidence:.2f}"))

        # Find the best connection to board from current stop
        hyperpath = decision.hyperpath
        labels = hyperpath.stop_labels.get(current_stop, [])
        if not labels:
            # No route found from here, journey failed
            events.append(JourneyEvent(current_time, current_stop, "stuck",
                                       details="No connections in hyperpath"))
            break

        # Boarding strategy differs between static and adaptive passengers:
        # - Static: take the FIRST bus that shows up (greedy, no ranking trust)
        # - Adaptive: trust the hyperpath ranking, prefer BEST-RANKED even if
        #   it means waiting a few extra minutes. The ranking reflects current
        #   regime awareness (after reweight).
        is_adaptive = hasattr(router, 'detector')
        patience = 12
        boarded = False
        action_completed = False
        tried_routes = set()

        # Real-world constraint: passenger only considers TOP-K alternatives
        # (like a navigation app showing 3 routes). This is where adaptive
        # routing gains advantage: static top-K was computed under normal
        # distributions, but adaptive top-K reflects current regime.
        TOP_K_ROUTES = 3
        viable = []
        seen_routes = set()
        for label in reversed(labels):  # best mean_dest_arrival first
            c = graph.connections[label.connection_id]
            if not c.pickup_allowed and c.trip_id != current_trip:
                continue
            if c.dep_time < current_time - 1:
                continue
            if c.dep_time > current_time + 25:
                continue
            if label.feasibility < 0.2:
                continue
            if c.id in canceled_routes_at_stop or c.id in boarded_connection_ids:
                continue  # known canceled, don't try again
            if c.route in seen_routes:
                continue
            seen_routes.add(c.route)
            viable.append((label, c))
            if len(seen_routes) >= TOP_K_ROUTES:
                break  # only consider top K distinct routes

        # Try candidates in order
        tried_routes = set()
        for label, c in viable:
            if c.route in tried_routes:
                continue
            tried_routes.add(c.route)

            same_trip = current_trip is not None and c.trip_id == current_trip
            ready = last_arrival + (0 if current_trip is None or same_trip
                                   else graph.stops[current_stop].min_transfer_time)
            current_time = min(t_depart + max_time, max(current_time, ready))
            if current_time >= t_depart + max_time:
                action_completed = True
                break
            event = scenario.event(c)
            delay = event.delay
            if event.canceled:
                # Commit to the same waiting window before any outcome is
                # visible; a scheduled cancellation announcement may end it.
                if current_time < c.dep_time:
                    wait_end = min(t_depart + max_time, current_time + patience)
                    if c.dep_time > wait_end:
                        current_time = wait_end
                        action_completed = True
                        events.append(JourneyEvent(current_time, current_stop, "wait_late",
                                                   route=c.route, details=f"connection={c.id}; patience window"))
                        break
                    current_time = c.dep_time
                canceled_routes_at_stop.add(c.id)
                action_completed = True
                events.append(JourneyEvent(
                    current_time, current_stop, "wait_cancel",
                    route=c.route,
                    details=f"connection={c.id}; explicit cancellation"))
                break  # back to main loop → replan → try with updated info

            actual_dep = event.departure
            wait_end = min(t_depart + max_time, current_time + patience)
            if actual_dep > wait_end:
                current_time = wait_end
                action_completed = True
                events.append(JourneyEvent(
                    current_time, current_stop, "wait_late",
                    route=c.route,
                    details=f"connection={c.id}; patience window"))
                break

            if actual_dep < current_time:
                continue  # already left

            actual_arr = event.arrival
            if actual_dep > t_depart + max_time:
                current_time = t_depart + max_time
                break

            events.append(JourneyEvent(
                actual_dep, current_stop, "board",
                route=c.route,
                details=f"→stop {c.arr_stop} (sched {c.dep_time}, actual {actual_dep})"))
            boarded_connection_ids.add(c.id)

            if actual_arr > t_depart + max_time:
                current_time = t_depart + max_time
                break
            current_time = actual_arr
            current_stop = c.arr_stop
            transferred = current_trip is not None and not same_trip
            n_transfers += int(transferred)
            current_trip = c.trip_id or f"connection:{c.id}"
            last_arrival = actual_arr

            if current_stop != s_dest:
                events.append(JourneyEvent(
                    current_time, current_stop, "arrive",
                    route=c.route,
                    details=f"arrived (delay={actual_dep - c.dep_time} min)"))

            boarded = True
            break  # exit the label loop after boarding

        if not boarded and not action_completed:
            # Nothing worked in this iteration, advance time slightly
            current_time += 1

    arrival_time = current_time if current_stop == s_dest else t_depart + max_time

    events.append(JourneyEvent(arrival_time, current_stop, "finish",
                               details="arrived" if current_stop == s_dest else "timeout"))

    return JourneyResult(
        origin=s_source,
        destination=s_dest,
        departure_time=t_depart,
        arrival_time=arrival_time,
        events=events,
        n_transfers=n_transfers,
        n_replans=n_replans,
        total_computation_ms=total_comp_ms,
        regime_changes_encountered=regime_changes,
        observations=observation_log,
    )
