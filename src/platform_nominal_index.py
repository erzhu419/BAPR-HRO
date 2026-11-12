"""Indexed nominal paths for the positive-transfer platform contract."""

from bisect import bisect_right
from collections import defaultdict
from types import SimpleNamespace

from .committed_transit import CommittedState
from .recourse_itinerary_router import build_trip_blocks, expand_trip_blocks, nominal_itinerary, trip_key


class PlatformNominalIndex:
    """One reverse timetable sweep plus forced, conditional first-trip queries.

    Boarding another trip pays the destination stop's transfer time. Staying
    on the same trip follows its consecutive contiguous segments and can use
    no-pickup edges. The index uses scheduled clocks after the first trip and
    only chooses a nominal path. Public platform execution still determines
    realized legality, timing and missed transfers.
    """

    def __init__(self, graph, goal, earliest, deadline, *, unavailable=(), trip_blocks=None):
        if any(stop.min_transfer_time <= 0 for stop in graph.stops.values()):
            raise ValueError('The reverse timetable index requires positive cross-trip transfer times')
        self.graph, self.goal, self.earliest, self.deadline = graph, goal, earliest, deadline
        self.unavailable = frozenset(unavailable)
        self.trip_blocks = build_trip_blocks(graph) if trip_blocks is None else trip_blocks
        blocks, positions = self.trip_blocks
        self.following = {cid: next_cid for block in blocks.values()
                          for cid, next_cid in zip(block, block[1:])}
        self.departures, self.options = defaultdict(list), defaultdict(list)
        self.arrivals, self.successors = {}, {}
        connections = graph.connections
        active = (c for c in connections if c.id not in self.unavailable
                  and earliest <= c.dep_time <= c.arr_time <= deadline and c.dep_time < deadline)
        for c in sorted(active, key=lambda c: (c.dep_time, positions[c.id], c.id), reverse=True):
            if c.arr_stop == goal:
                arrival, successor = c.arr_time, None
            else:
                trip = trip_key(c)
                ready = c.arr_time + graph.stops[c.arr_stop].min_transfer_time
                transfer = self._first(c.arr_stop, ready, trip)
                next_cid = self.following.get(c.id)
                through = None
                if next_cid in self.arrivals:
                    following = connections[next_cid]
                    if following.dep_stop == c.arr_stop and following.dep_time >= c.arr_time:
                        through = next_cid
                successor = min((cid for cid in (through, transfer) if cid is not None),
                                key=lambda cid: (self.arrivals[cid], cid != through, cid), default=None)
                if successor is None:
                    continue
                arrival = self.arrivals[successor]
            self.arrivals[c.id] = arrival
            self.successors[c.id] = successor
            if c.pickup_allowed:
                self._add(c)

    def _add(self, connection):
        """Keep the two best suffixes with distinct first trips at this cutoff."""
        stop = connection.dep_stop
        current = self.options[stop][-1] if self.options[stop] else ()
        ranked = sorted((*current, connection.id), key=lambda cid: (self.arrivals[cid], cid))
        first = ranked[0]
        trip = trip_key(self.graph.connections[first])
        second = next((cid for cid in ranked[1:] if trip_key(self.graph.connections[cid]) != trip), None)
        self.departures[stop].append(-connection.dep_time)
        self.options[stop].append((first,) if second is None else (first, second))

    def _first(self, stop, ready, excluded_trip):
        offset = bisect_right(self.departures.get(stop, ()), -ready)
        if offset == 0:
            return None
        return next((cid for cid in self.options[stop][offset-1]
                     if trip_key(self.graph.connections[cid]) != excluded_trip), None)

    def _unroll(self, cid, excluded_trip=None):
        path, previous = [], None
        seen = set() if excluded_trip is None else {excluded_trip}
        while cid is not None:
            trip = trip_key(self.graph.connections[cid])
            if trip != previous and trip in seen:
                return None
            path.append(cid)
            seen.add(trip)
            previous = trip
            cid = self.successors[cid]
        return tuple(path)

    def path(self, stop, ready, *, excluded_trip=None):
        """Return a nominal route from an alighted passenger's ready clock."""
        if stop == self.goal:
            return ()
        first = self._first(stop, ready, excluded_trip)
        if first is None:
            return None
        path = self._unroll(first, excluded_trip)
        if path is not None:
            return path
        scheduled = self._scheduled()
        state = CommittedState(stop, ready, None, ready, self.unavailable)
        return nominal_itinerary(self.graph, state, self.goal, scheduled, scheduled, self.deadline, 1500,
                                 excluded_trips=(() if excluded_trip is None else (excluded_trip,)),
                                 trip_blocks=self.trip_blocks)

    def _scheduled(self):
        return {c.id: SimpleNamespace(departure=c.dep_time, arrival=c.arr_time)
                for c in self.graph.connections}

    def conditional_path(self, cid, nominal, tick):
        """Force one legal root boarding and value transfers off its conditional trip."""
        connections = self.graph.connections
        first, event = connections[cid], nominal[cid]
        if cid in self.unavailable or not first.pickup_allowed or event.arrival is None:
            return None
        arrival = max(tick, round(event.arrival))
        trip = trip_key(first)
        prefix, best = [], None
        while arrival <= self.deadline:
            connection = connections[cid]
            prefix.append(cid)
            if connection.arr_stop == self.goal:
                direct = (arrival, tuple(prefix))
                return direct[1] if best is None or direct < best else best[1]
            ready = arrival + self.graph.stops[connection.arr_stop].min_transfer_time
            transfer = self._first(connection.arr_stop, ready, trip)
            if transfer is not None and (best is None or self.arrivals[transfer] <= best[0]):
                suffix = self._unroll(transfer, trip)
                if suffix is None:
                    suffix = self.path(connection.arr_stop, ready, excluded_trip=trip)
                if suffix:
                    candidate = (connections[suffix[-1]].arr_time, (*prefix, *suffix))
                    if best is None or candidate < best:
                        best = candidate
            next_cid = self.following.get(cid)
            if next_cid is None or next_cid in self.unavailable:
                break
            following = connections[next_cid]
            if following.dep_stop != connection.arr_stop:
                break
            event = nominal[next_cid]
            if event.departure is None or event.arrival is None:
                break
            departure = round(event.departure)
            attempt = max(arrival, following.dep_time-1500, departure-720)
            if (departure < attempt or departure >= self.deadline
                    or (following.dep_distribution is not None and following.dep_distribution.support_max < attempt)):
                break
            arrival = departure + round(event.arrival-event.departure)
            cid = next_cid
        return None if best is None else best[1]
