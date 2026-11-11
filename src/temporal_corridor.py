"""Conservative source-to-destination corridor for the station-MCT oracle."""

from heapq import heappop, heappush
from math import inf

import numpy as np


class TemporalCorridor:
    """Keep every trip that could improve the capped clairvoyant arrival.

    Work on the scan's station-ready and same-vehicle labels. For a single
    operating ride, relax departure and arrival to hard support intervals
    subject to departure <= arrival. Across rides permit different worlds.
    Both relaxations preserve every route from every world in the posterior.
    """

    def __init__(self, scan):
        self.scan = scan
        self.arcs = tuple(map(tuple, scan.arcs.tolist()))
        size = len(scan.stops) + len(scan.onboard)
        self.outgoing = [[] for _ in range(size)]
        self.incoming = [[] for _ in range(size)]
        for cid, (pickup, onboard, stop, seat) in enumerate(self.arcs):
            for label in (pickup, onboard):
                if label != -1:
                    self.outgoing[label].append(cid)
            for label in (stop, seat):
                if label != -1:
                    self.incoming[label].append(cid)
        self._memos = {}
        self._windows = [None] * len(self.arcs)

    def _refresh_windows(self, belief, parameters):
        for trip, slots in belief._trips.items():
            memo = belief._cache[trip]
            route = belief.graph.connections[slots[0].cid].route
            canceled = parameters.get(route, {}).get('cancel_rate') == 1
            if trip in self._memos:
                cached_memo, cached_canceled = self._memos[trip]
                if cached_memo is memo and cached_canceled == canceled:
                    continue
            for slot in slots[::2]:
                cid = slot.cid
                departure = belief.hard_support(cid, 'departure', parameters)
                arrival = belief.hard_support(cid, 'arrival', parameters)
                self._windows[cid] = (*departure, *arrival) if departure is not None and arrival is not None else None
            self._memos[trip] = memo, canceled

    def connections(self, belief, parameters, state, goal, deadline, *, transfer=0):
        """Return connections on some feasible path in the interval relaxation.

        Any path arriving strictly before the cap survives. Paths arriving
        exactly at the cap have the same loss as the deadline fallback.
        """
        return self.connections_and_floor(belief, parameters, state, goal, deadline,
                                          transfer=transfer)[0]

    def connections_and_floor(self, belief, parameters, state, goal, deadline, *, transfer=0):
        """Return the corridor and its deterministic elapsed-arrival floor.

        The forward labels already minimize arrival in the hard-support
        relaxation. Recording the earliest goal arrival adds no graph pass.
        Every posterior world has at least this capped loss; this is not a
        minimum over sampled worlds.
        """
        if state.time >= deadline or state.stop == goal:
            return set(), 0.
        transfers = np.broadcast_to(transfer, (len(self.scan.stops),)).tolist()
        origin, onboard, ready = self.scan.root_labels(state, transfers)
        destination = self.scan.stops[goal]
        horizon = deadline - 1
        self._refresh_windows(belief, parameters)
        windows, unavailable = self._windows, state.unavailable_ids
        latest = [-inf] * len(self.incoming)
        # A goal arrival need not wait through its station transfer duration.
        latest[destination] = horizon + transfers[destination]
        reverse = [(-latest[destination], destination)]
        while reverse:
            neg_time, label = heappop(reverse)
            time = -neg_time
            if latest[label] != time:
                continue
            arrival_cap = time - transfers[label] if label < len(transfers) else time
            for cid in self.incoming[label]:
                interval = windows[cid]
                if interval is None or cid in unavailable:
                    continue
                dlo, dhi, alo, ahi = interval
                arrival = min(ahi, arrival_cap)
                departure = min(dhi, arrival)
                if arrival < alo or departure < max(dlo, state.time):
                    continue
                for source in self.arcs[cid][:2]:
                    if source != -1 and departure > latest[source]:
                        latest[source] = departure
                        heappush(reverse, (-departure, source))

        earliest = [inf] * len(latest)
        forward = []

        def enqueue(label, time):
            if label != -1 and time <= latest[label] and time < earliest[label]:
                earliest[label] = time
                heappush(forward, (time, label))

        enqueue(origin, ready)
        enqueue(onboard, state.time)
        retained = set()
        earliest_goal = deadline
        while forward:
            time, label = heappop(forward)
            if earliest[label] != time:
                continue
            for cid in self.outgoing[label]:
                interval = windows[cid]
                if interval is None or cid in unavailable:
                    continue
                dlo, dhi, alo, ahi = interval
                departure = max(dlo, time)
                arrival = max(alo, departure)
                if departure > dhi or arrival > ahi:
                    continue
                stop, seat = self.arcs[cid][2:]
                station_ready = arrival + transfers[stop]
                if station_ready <= latest[stop] or (seat != -1 and arrival <= latest[seat]):
                    retained.add(cid)
                    if stop == destination:
                        earliest_goal = min(earliest_goal, arrival)
                    else:
                        enqueue(stop, station_ready)
                        enqueue(seat, arrival)
        return retained, float(earliest_goal - state.time)

    def trips(self, belief, parameters, state, goal, deadline, *, transfer=0):
        """Retain whole ordered chains for the surviving connection inventory."""
        connections = self.connections(belief, parameters, state, goal, deadline, transfer=transfer)
        return {belief._trip_by_cid[cid] for cid in connections}
