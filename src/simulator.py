"""Journey simulator: simulate a passenger traveling through the network.

Generates realistic delay observations and tracks the passenger as they
board buses, transfer, and eventually arrive at the destination.
Tests static vs adaptive routing under controlled regime shifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Optional

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
        cancel_prob = info.get("cancel_prob", 0)
        if cancel_prob > 0 and rng.random() < cancel_prob:
            return 999  # canceled

        probs = info["delay_probs"]
        offset = info.get("delay_offset", 0)
        probs_norm = probs / probs.sum()
        delay_idx = rng.choice(len(probs_norm), p=probs_norm)
        return offset + delay_idx
    else:
        return int(rng.normal(1, 2))


def generate_delay_observations(
    graph: TransitGraph,
    current_stop: int,
    current_time: int,
    regime: str,
    rng: np.random.Generator,
    n_obs: int = 5,
) -> list[DelayObservation]:
    """Generate simulated GTFS-RT observations at a stop.

    Looks at connections departing from current_stop in the near future
    and samples actual delays according to the current regime.
    """
    observations = []
    nearby_conns = [c for c in graph.get_connections_from(current_stop)
                    if current_time - 5 <= c.dep_time <= current_time + 15]
    nearby_conns += [c for c in graph.get_connections_to(current_stop)
                     if current_time - 10 <= c.arr_time <= current_time + 5]

    # Note: removed global network sampling — it pollutes per-route beliefs
    # with unrelated disruption signals. Real GTFS-RT observations are
    # per-vehicle, so only nearby connections are observed.

    for c in nearby_conns[:n_obs]:
        delay = sample_actual_delay(c, regime, rng)
        if delay == 999:
            # Canceled: generate a "no-show" observation with very high delay.
            # The bus was expected but never came — this is a strong surprise signal.
            obs = DelayObservation(
                route=c.route,
                stop_id=current_stop,
                scheduled_time=c.dep_time,
                predicted_time=c.dep_time + 2,  # system predicted near on-time
                actual_time=c.dep_time + 30,     # treat as 30 min delay (no-show)
                timestamp=current_time,
            )
        else:
            obs = DelayObservation(
                route=c.route,
                stop_id=current_stop,
                scheduled_time=c.dep_time,
                predicted_time=c.dep_time + max(0, delay - rng.integers(0, 3)),
                actual_time=c.dep_time + delay,
                timestamp=current_time,
            )
        observations.append(obs)

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
) -> JourneyResult:
    """Simulate a passenger journey from source to destination.

    The passenger follows the hyperpath: at each stop, they board the
    first available connection from the hyperpath's recommendations.
    At transfer stops, the adaptive router may re-plan.
    """
    events = []
    current_stop = s_source
    current_time = t_depart
    n_transfers = 0
    n_replans = 0
    total_comp_ms = 0.0
    regime_changes = 0
    prev_regime = regime_schedule.get_regime(t_depart)

    # Initial route computation
    decision = router.route(s_source, s_dest, t_depart)
    total_comp_ms += decision.computation_ms
    events.append(JourneyEvent(current_time, current_stop, "depart",
                               details=f"Initial route computed"))

    canceled_routes_at_stop: set[str] = set()  # routes known canceled at current stop
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
            graph, current_stop, current_time, current_regime, rng)

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
        patience = 10
        boarded = False
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
            if c.dep_time < current_time - 1:
                continue
            if c.dep_time > current_time + 25:
                continue
            if label.feasibility < 0.2:
                continue
            if c.route in canceled_routes_at_stop:
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

            # Sample actual delay
            delay = sample_actual_delay(c, current_regime, rng)
            if delay == 999:
                # Canceled. Blacklist this route at this stop and try others.
                canceled_routes_at_stop.add(c.route)
                current_time += patience
                events.append(JourneyEvent(
                    current_time, current_stop, "wait_cancel",
                    route=c.route,
                    details=f"waited {patience}min, {c.route} canceled"))
                break  # back to main loop → replan → try with updated info

            actual_dep = c.dep_time + delay
            if actual_dep > current_time + patience:
                current_time += patience
                events.append(JourneyEvent(
                    current_time, current_stop, "wait_late",
                    route=c.route,
                    details=f"dep={c.dep_time} actual={actual_dep}, too late"))
                break

            if actual_dep < current_time:
                continue  # already left

            actual_arr = c.arr_time + delay

            events.append(JourneyEvent(
                actual_dep, current_stop, "board",
                route=c.route,
                details=f"→stop {c.arr_stop} (sched {c.dep_time}, actual {actual_dep})"))

            current_time = actual_arr
            current_stop = c.arr_stop

            if current_stop != s_dest:
                n_transfers += 1
                events.append(JourneyEvent(
                    current_time, current_stop, "arrive",
                    route=c.route,
                    details=f"arrived (delay={actual_dep - c.dep_time} min)"))

            boarded = True
            break  # exit the label loop after boarding

        if not boarded:
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
    )
