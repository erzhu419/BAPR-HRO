"""Arrival bounds from the probability of crossing time labels on stops."""
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np

from .opportunity_bounds import ExitWindow, OrderedTripOpportunities


@dataclass(frozen=True)
class FrontierProbability:
    threshold: int
    survival: float
    union_survival: float
    crossings: int
    trips: int
    repeated_trips: int


def time_labels(graph, start, destination, deadline, windows, *, unavailable=(), walks=()):
    """Forward and backward cuts from the same optimistic timetable.

    Each virtual edge accepts a passenger up to its latest departure and
    releases at the larger of the entry time and its earliest arrival.
    Windows generate labels only. The probability calculation separately
    checks the entire posterior and does not assume these windows are hard.
    """
    unavailable = frozenset(unavailable)
    outgoing_walks, incoming_walks = defaultdict(list), defaultdict(list)
    for u, v, duration in walks:
        outgoing_walks[u].append((v, duration))
        incoming_walks[v].append((u, duration))
    arrival = {start.stop: start.time}
    queue = [(start.time, start.stop)]
    while queue:
        time, stop = heappop(queue)
        if time != arrival[stop]:
            continue
        if stop == destination:
            break
        rides = []
        for c in graph.get_connections_from(stop):
            window = windows.get(c.id)
            if c.id not in unavailable and window is not None and time <= window[0]:
                rides.append((c.arr_stop, max(time, window[1])))
        rides.extend((target, time + duration) for target, duration in outgoing_walks[stop])
        for target, reached in rides:
            if reached < arrival.get(target, deadline):
                arrival[target] = reached
                heappush(queue, (reached, target))
    threshold = arrival.get(destination, deadline)
    forward = {stop: min(threshold, arrival.get(stop, threshold)) for stop in graph.stops}

    latest = {destination: threshold-1}
    queue = [(1-threshold, destination)]
    while queue:
        neg_time, stop = heappop(queue)
        time = -neg_time
        if time != latest[stop]:
            continue
        rides = []
        for c in graph.get_connections_to(stop):
            window = windows.get(c.id)
            if c.id not in unavailable and window is not None and window[1] <= time:
                rides.append((c.dep_stop, min(time, window[0])))
        rides.extend((source, time - duration) for source, duration in incoming_walks[stop])
        for source, ready in rides:
            if ready > latest.get(source, start.time-1):
                latest[source] = ready
                heappush(queue, (-ready, source))
    backward = {stop: max(start.time, latest.get(stop, start.time-1)+1) for stop in graph.stops}
    return forward, backward


def integrate_survival(start, points):
    """Integrate a stepwise lower envelope of P(arrival >= threshold)."""
    envelope = {}
    for threshold, probability in points:
        if threshold > start:
            envelope[threshold] = max(envelope.get(threshold, 0.), probability)
    expectation = probability = 0.
    for threshold in sorted(envelope, reverse=True):
        expectation += (threshold-start) * max(0., envelope[threshold]-probability)
        probability = max(probability, envelope[threshold])
    return expectation


class TemporalFrontier:
    """Integrate the first possible time-label violation by independent trip."""

    def __init__(self, bounds, start, *, walks=()):
        self.bounds = bounds
        self.start = start
        self.walks = tuple(walks)
        self.graph = bounds.graph
        self.edges = []
        for c in self.graph.connections:
            if c.id in start.unavailable_ids:
                continue
            dep = bounds.belief.hard_support(c.id, 'departure', bounds.parameters)
            arr = bounds.belief.hard_support(c.id, 'arrival', bounds.parameters)
            if dep is not None and arr is not None and dep[1] >= start.time and arr[0] < bounds.deadline:
                self.edges.append((c, dep[1], arr[0]))

    def evaluate(self, labels):
        """Evaluate proposed label vectors without a model search per vector.

        A path arriving too early must cross at least one of these windows.
        The window on edge (u,v) is dep >= label[u], arr < label[v]. Every
        traversable trip edge is considered, including no-pickup stay legs.
        """
        labels = tuple(labels)
        grouped = defaultdict(lambda: defaultdict(list))
        thresholds = [row[self.bounds.destination] for row in labels]
        stats = [dict(crossings=0, trips=0, repeated_trips=0) for _ in labels]
        probabilities = np.ones(len(labels))
        union_failures = np.zeros(len(labels))
        for i, row in enumerate(labels):
            threshold = thresholds[i]
            if (row[self.start.stop] > self.start.time or threshold > self.bounds.deadline
                    or any(row[v] > row[u]+duration for u, v, duration in self.walks)):
                raise ValueError('Time labels cut the initial state, deadline or a deterministic walk')
            for c, dep_high, arr_low in self.edges:
                ready, arrival = row[c.dep_stop], row[c.arr_stop]
                if arrival <= ready or dep_high < ready or arr_low >= arrival:
                    continue
                trip = self.bounds.belief._trip_by_cid[c.id]
                grouped[trip][i].append(ExitWindow(c.id, ready, threshold-arrival))
                stats[i]['crossings'] += 1

        integrator = OrderedTripOpportunities(self.bounds)
        for trip, cuts in grouped.items():
            for i, windows in cuts.items():
                threshold = np.asarray([thresholds[i]-1.])
                none = float(integrator.no_event(trip, windows, threshold)[0])
                probabilities[i] *= none
                union_failures[i] += 1.-none
                stats[i]['trips'] += 1
                stats[i]['repeated_trips'] += int(len(windows)>1)
            # These matrices are reused across all cuts of this trip. No
            # downstream trip needs this trip's dense probability tables.
            integrator.edge_tail.clear()
            self.bounds._posterior_edges.clear()

        return [FrontierProbability(threshold, float(probabilities[i]),
                    float(max(0., 1.-union_failures[i])), **stats[i])
                for i, threshold in enumerate(thresholds)], integrator.stats
