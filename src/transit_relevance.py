"""Clock-support relevance for independent trip factors.

The index is an optimistic transport graph, not a routing policy. Its output
closes over *whole trips*: an off-path segment can reveal a useful trip's state.
Omitted factors remain available on demand if physical actions are retained.
Deleting actions additionally needs the wait-simulation condition in the note;
the public extension alone is not a coarse-WAIT policy equivalence proof.
The model assumptions and the public-message extension are in the V5 theory
note. No trip probabilities or HMM marginals are evaluated here.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable

from .transit_graph import TransitGraph


ClockSupport = Callable[[int, str], tuple[int, int] | None]


def trip_key(connection):
    return connection.trip_id or f"connection:{connection.id}"


@dataclass(frozen=True)
class EdgeEnvelope:
    departure_low: int
    departure_high: int
    arrival_low: int
    arrival_high: int
    minimum_travel: int = 0

    def earliest(self, ready):
        departure = max(ready, self.departure_low)
        arrival = max(self.arrival_low, departure + self.minimum_travel)
        return (arrival if departure <= self.departure_high
                and arrival <= self.arrival_high else math.inf)

    def latest(self, due):
        arrival = min(due, self.arrival_high)
        departure = min(self.departure_high, arrival - self.minimum_travel)
        return (departure if arrival >= self.arrival_low
                and departure >= self.departure_low else -math.inf)


@dataclass(frozen=True)
class RelevanceResult:
    transport_edge_ids: frozenset[int]
    transport_trip_ids: frozenset[str]
    information_trip_ids: frozenset[str]
    required_trip_ids: frozenset[str]
    retained_connection_ids: frozenset[int]
    earliest: dict[int, float]
    latest: dict[int, float]
    work: dict[str, int]
    public_occurrences: bool

    def affected_inference(self, changed_trips):
        """Trip-factor invalidations when hard support and information stay fixed."""
        return self.required_trip_ids.intersection(changed_trips)


class RelevanceIndex:
    """Reuse support metadata; recompute cheap reachability after support edits.

    ``refresh`` must name every trip whose support changed. A new support or
    information rule requires a new index. Graph topology is fixed for an index.
    It deliberately does not reuse policy values or observation posteriors.
    """

    def __init__(self, graph: TransitGraph, hard_support: ClockSupport,
                 minimum_travel: Callable[[int], int] | None = None):
        self.graph = graph
        self.hard_support = hard_support
        self.minimum_travel = minimum_travel or (lambda cid: 0)
        self.by_trip = {}
        self.envelopes = {}
        self.supports = {}
        self.support_queries = 0
        for connection in graph.connections:
            self.by_trip.setdefault(trip_key(connection), []).append(connection.id)
        self.refresh(self.by_trip)

    def refresh(self, changed_trips):
        """Refresh only declared trip supports; subsequent solve scans globally."""
        before = self.support_queries
        for trip in changed_trips:
            for cid in self.by_trip.get(trip, ()):
                departure = self.hard_support(cid, "departure")
                arrival = self.hard_support(cid, "arrival")
                self.support_queries += 2
                self.supports[cid] = departure, arrival
                tau = self.minimum_travel(cid)
                if tau < 0:
                    raise ValueError("A hard minimum travel time must be nonnegative")
                self.envelopes[cid] = (EdgeEnvelope(*departure, *arrival, tau)
                                       if departure is not None and arrival is not None else None)
        return self.support_queries - before

    def solve(self, origin, destination, start, deadline, *,
              public_occurrences=False, publication_lookback=10):
        """Return a superset of every goal-reaching ride and useful trip factor.

        ``publication_lookback`` uses graph clock units (600 for second clocks).
        True denotes the executor's immediate occurred-event synthetic feed,
        including all possible first-five masking events at reachable stops.
        False denotes no future external feed, not a delayed unknown feed.
        """
        graph = self.graph
        earliest = {stop: math.inf for stop in graph.stops}
        latest = {stop: -math.inf for stop in graph.stops}
        earliest[origin] = start
        queue = [(start, origin)]
        scans_forward = scans_reverse = 0
        while queue:
            clock, stop = heapq.heappop(queue)
            if clock != earliest[stop]:
                continue
            for cid in graph._dep_index.get(stop, ()):
                scans_forward += 1
                envelope = self.envelopes[cid]
                arrival = envelope.earliest(clock) if envelope is not None else math.inf
                target = graph.connections[cid].arr_stop
                if arrival <= deadline and arrival < earliest[target]:
                    earliest[target] = arrival
                    heapq.heappush(queue, (arrival, target))
        latest[destination] = deadline
        queue = [(-deadline, destination)]
        while queue:
            negative_clock, stop = heapq.heappop(queue)
            clock = -negative_clock
            if clock != latest[stop]:
                continue
            for cid in graph._arr_index.get(stop, ()):
                scans_reverse += 1
                envelope = self.envelopes[cid]
                departure = envelope.latest(clock) if envelope is not None else -math.inf
                source = graph.connections[cid].dep_stop
                if departure >= start and departure > latest[source]:
                    latest[source] = departure
                    heapq.heappush(queue, (-departure, source))
        edges, transport = set(), set()
        for connection in graph.connections:
            envelope = self.envelopes[connection.id]
            if (envelope is not None and math.isfinite(earliest[connection.dep_stop])
                    and envelope.earliest(earliest[connection.dep_stop])
                    <= min(deadline, latest[connection.arr_stop])):
                edges.add(connection.id)
                transport.add(trip_key(connection))

        information = set()
        if public_occurrences:
            for connection in graph.connections:
                departure, arrival = self.supports[connection.id]
                events = ((connection.dep_stop, departure), (connection.arr_stop, arrival),
                          # Canceled notices occur at the scheduled departure.
                          (connection.dep_stop, (connection.dep_time, connection.dep_time)))
                if any(bounds is not None and earliest[stop] <= deadline
                       and bounds[0] <= deadline
                       and bounds[1] >= earliest[stop] - publication_lookback
                       for stop, bounds in events):
                    information.add(trip_key(connection))
        required = transport | information
        retained = {cid for trip in required for cid in self.by_trip[trip]}
        return RelevanceResult(frozenset(edges), frozenset(transport), frozenset(information),
                               frozenset(required), frozenset(retained), earliest, latest,
                               {"support_queries_total": self.support_queries,
                                "forward_edge_scans": scans_forward,
                                "reverse_edge_scans": scans_reverse,
                                "closure_edge_scans": len(graph.connections),
                                "publication_edge_scans": len(graph.connections) if public_occurrences else 0},
                               public_occurrences)


def compute_relevance(graph, origin, destination, start, deadline, hard_support, *,
                      public_occurrences=False, minimum_travel=None,
                      publication_lookback=10):
    return RelevanceIndex(graph, hard_support, minimum_travel).solve(
        origin, destination, start, deadline, public_occurrences=public_occurrences,
        publication_lookback=publication_lookback)


def product_coupling_bound(cap, trip_total_variations, proxy_optimization_gap=0):
    """Standard product coupling bound for a supplied independent-factor error.

    This does not estimate TV from samples or claim posterior calibration. With
    identical observation/execution maps and bounded loss [0,cap], optimizing a
    proxy product law has regret at most 2*cap*delta plus its optimization gap.
    """
    errors = tuple(trip_total_variations)
    if cap < 0 or proxy_optimization_gap < 0 or any(not 0 <= e <= 1 for e in errors):
        raise ValueError("Use nonnegative losses/gaps and TV errors in [0,1]")
    delta = 1 - math.prod(1 - error for error in errors)
    return {"world_coupling_error": delta,
            "policy_value_error": cap * delta,
            "policy_regret": min(cap, 2 * cap * delta + proxy_optimization_gap)}
