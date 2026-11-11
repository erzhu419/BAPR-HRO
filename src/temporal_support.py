"""Optimistic full-network temporal reachability on frozen edge supports."""
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf

import numpy as np

from .evolving_trip_belief import EvolvingTripBelief


@dataclass(frozen=True)
class TemporalLabel:
    arrival: int
    path: tuple[int, ...]
    edge_queries: int
    settled_labels: int


class TemporalReachability:
    """Find earliest arrival for FIFO, nonnegative-time connection maps.

    The map returns the earliest possible arrival for a given ready time,
    or None if no operating departure can be caught. It may use optimistic
    edge times from different worlds. The search allows zero transfer time,
    including alighting and reboarding the same trip. It keeps a separate
    onboard label only when needed for a no-pickup connection.
    """

    def __init__(self, graph):
        self.graph = graph
        pickup, onboard = defaultdict(list), defaultdict(list)
        for c in graph.connections:
            if c.pickup_allowed:
                pickup[c.dep_stop].append(c.id)
            else:
                onboard[c.dep_stop, c.trip_id or f'connection:{c.id}'].append(c.id)
        self.pickup = dict(pickup)
        self.onboard = dict(onboard)

    def solve(self, state, destination, deadline, arrival_at):
        serial = count()
        queue, best = [], {}
        edge_queries = settled = 0

        def offer(stop, trip, time, path):
            key = stop, trip
            if time <= deadline and time < best.get(key, inf):
                best[key] = time
                heappush(queue, (time, next(serial), stop, trip, path))

        offer(state.stop, None, state.time, ())
        if (state.stop, state.current_trip) in self.onboard:
            offer(state.stop, state.current_trip, state.time, ())
        while queue:
            ready, _, stop, trip, path = heappop(queue)
            if ready != best[stop, trip]:
                continue
            settled += 1
            if stop == destination:
                return TemporalLabel(ready, path, edge_queries, settled)
            options = self.pickup.get(stop, ()) if trip is None else self.onboard[stop, trip]
            for cid in options:
                if cid in state.unavailable_ids:
                    continue
                edge_queries += 1
                arrival = arrival_at(cid, ready)
                if arrival is None or arrival > deadline:
                    continue
                connection = self.graph.connections[cid]
                next_path = path + (cid,)
                offer(connection.arr_stop, None, arrival, next_path)
                next_trip = connection.trip_id or f'connection:{cid}'
                if (connection.arr_stop, next_trip) in self.onboard:
                    offer(connection.arr_stop, next_trip, arrival, next_path)
        return TemporalLabel(deadline, (), edge_queries, settled)


class IntervalOperatingSupport:
    """Relax endpoint supports to closed intervals subject to departure <= arrival."""

    def __init__(self, belief, parameters):
        self.belief = belief.copy()
        self.parameters = {route: dict(p) for route, p in parameters.items()}
        self._intervals = {}

    def __call__(self, cid, ready):
        if cid not in self._intervals:
            self._intervals[cid] = tuple(self.belief.hard_support(cid, kind, self.parameters)
                                         for kind in ('departure', 'arrival'))
        departure, arrival = self._intervals[cid]
        if departure is None or arrival is None:
            return None
        earliest_departure = max(ready, departure[0])
        earliest_arrival = max(earliest_departure, arrival[0])
        return earliest_arrival if earliest_departure <= departure[1] and earliest_arrival <= arrival[1] else None


def _ordered_operating_clocks(law, service_steps):
    """Exact prefix/suffix operating support on an ordered Gaussian chain.

    Each service-state transition permits all nondecreasing clock pairs.
    The earliest reachable clock for each preceding service state determines
    whether a slot can be reached. The suffix uses the latest extendable clock
    by following service state. Within a ride both endpoints are operating.
    """
    allowed = [np.isfinite(emission) for emission in law.emissions]
    reachable = [-inf, -inf]
    departures = []
    for i, (clocks, states, emission) in enumerate(zip(law.times, law.canceled, allowed)):
        new_reachable = [inf, inf]
        for status in (False, True):
            if i == 0:
                possible = emission & np.isfinite(law.initial) & (states == status)
            else:
                threshold = min((reachable[prev] for prev in (0, 1)
                                 if service_steps[i - 1][prev, int(status)]), default=inf)
                possible = emission & (states == status) & (clocks >= threshold)
            feasible = clocks[possible]
            if not status and i % 2 == 0:
                departures.append(feasible)
            if feasible.size:
                new_reachable[int(status)] = int(feasible[0])
        reachable = new_reachable

    extendable = [inf, inf]
    arrivals = []
    for i in range(len(law.times) - 1, -1, -1):
        clocks, states, emission = law.times[i], law.canceled[i], allowed[i]
        new_extendable = [-inf, -inf]
        for status in (False, True):
            threshold = (inf if i == len(law.times) - 1 else
                         max((extendable[nxt] for nxt in (0, 1)
                              if service_steps[i][int(status), nxt]), default=-inf))
            feasible = clocks[emission & (states == status) & (clocks <= threshold)]
            if not status and i % 2 == 1:
                arrivals.append(feasible)
            if feasible.size:
                new_extendable[int(status)] = int(feasible[-1])
        extendable = new_extendable
    return tuple(zip(departures, reversed(arrivals)))


class JointOperatingSupport:
    """Exact operating support of each edge, derived lazily per vehicle trip.

    Prefix and suffix supports can be joined through the ride's ordered
    departure-to-arrival arc. No tiny but positive state is discarded. Across
    route edges the search relaxes the requirement of a shared realization.
    """

    def __init__(self, belief, parameters):
        if type(belief) is not EvolvingTripBelief:
            raise TypeError('This support projection requires the ordered Gaussian trip model')
        self.belief = belief.copy()
        self.parameters = {route: dict(p) for route, p in parameters.items()}
        self._clocks = {}
        self.trips_built = 0
        self.grid_atoms = 0

    def clocks(self, cid):
        if cid not in self._clocks:
            trip, _ = self.belief._positions[cid, 'departure']
            slots = self.belief._trips[trip]
            law = self.belief._law(trip, self.parameters)
            route = self.belief.graph.connections[slots[0].cid].route
            q = self.parameters.get(route, {'cancel_rate': .01})['cancel_rate']
            persistence = self.belief.cancel_persistence
            same = np.eye(2, dtype=bool)
            next_cancel = persistence * np.array([0., 1.]) + (1. - persistence) * q
            between = np.column_stack((next_cancel < 1., next_cancel > 0.))
            steps = tuple(same if slot.kind == 'arrival' else between for slot in slots[1:])
            for dep, pair in zip(slots[::2], _ordered_operating_clocks(law, steps)):
                self._clocks[dep.cid] = pair
            self.trips_built += 1
            self.grid_atoms += sum(len(times) for times in law.times)
        return self._clocks[cid]

    def __call__(self, cid, ready):
        departures, arrivals = self.clocks(cid)
        departure = int(departures.searchsorted(ready))
        if departure == len(departures):
            return None
        arrival = int(arrivals.searchsorted(departures[departure]))
        return int(arrivals[arrival]) if arrival < len(arrivals) else None
