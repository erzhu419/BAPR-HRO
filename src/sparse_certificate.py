"""Finite-bank action lower bounds with demand-driven sparse transit searches.

The first action has the committed executor's physical clock semantics. Its
continuation gets world knowledge, free transfers, unlimited waiting precision,
and unrestricted boarding. This is a lower bound, not a causal policy value.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from heapq import heappop, heappush
from time import perf_counter

from .committed_transit import OBSERVE, WAIT


class SparseCertificateIndex:
    """Cache exact relaxed EA intervals instead of stop-by-clock label arrays.

    The complete bank is already conditioned on the supplied public root.
    Lazy scenarios are accepted without materializing their unqueried factors.
    Every graph connection remains available to the relaxation; candidate H
    never defines the alleged full lower bound.
    """

    def __init__(self, graph, bank, destination, start, deadline):
        self.graph, self.bank = graph, tuple(bank)
        self.destination, self.start, self.deadline = destination, start, deadline
        if not self.bank or deadline < start:
            raise ValueError('A nonempty bank and nonnegative horizon are required')
        self.weights = tuple(Fraction(p) for p, _ in self.bank)
        if min(self.weights) < 0 or sum(self.weights) <= 0:
            raise ValueError('A bank needs nonnegative weights with positive total')
        self.minute = self.bank[0][1].minute
        if any(scene.minute != self.minute for _, scene in self.bank):
            raise ValueError('All scenarios must use the same clock unit')
        self._intervals = defaultdict(list)
        self.stats = defaultdict(float)

    def _event(self, world, cid):
        self.stats['event_reads'] += 1
        event = self.bank[world][1].events[cid]
        if not event.canceled and event.arrival < event.departure:
            raise ValueError('A modeled ride cannot arrive before departure')
        return event

    def arrival(self, world, stop, time):
        """Worldwise earliest arrival with free transfers, capped at deadline.

        One search settles at most |V| stop labels and scans their outgoing
        connections. Multiple distinct query intervals can rescan the graph;
        the overall worst case is O(Q*(E+V)*log V), not one global scan.
        """
        if stop == self.destination:
            return min(time, self.deadline)
        if time >= self.deadline:
            return self.deadline
        intervals = self._intervals[world, stop]
        for low, high, value in reversed(intervals):
            if low <= time <= high:
                self.stats['ea_cache_hits'] += 1
                return value
        started = perf_counter()
        self.stats['ea_queries'] += 1
        queue = [(time, stop, None)]
        best = {stop: time}
        arrival, first_departure = self.deadline, self.deadline
        while queue:
            clock, current, first = heappop(queue)
            if clock != best[current]:
                continue
            self.stats['settled_stop_labels'] += 1
            if current == self.destination:
                arrival = clock
                first_departure = first
                break
            for cid in self.graph._dep_index.get(current, ()):
                self.stats['connection_scans'] += 1
                event = self._event(world, cid)
                if event.canceled or event.departure < clock or event.arrival > self.deadline:
                    continue
                c = self.graph.connections[cid]
                if event.arrival < best.get(c.arr_stop, self.deadline + 1):
                    best[c.arr_stop] = event.arrival
                    heappush(queue, (event.arrival, c.arr_stop,
                                     event.departure if first is None else first))
        # For any later query through the chosen path's first departure, the
        # same path is still feasible. Monotonicity supplies the reverse bound.
        # If the capped value is deadline, it stays deadline for all later t.
        high = self.deadline if arrival == self.deadline else first_departure
        intervals.append((time, high, arrival))
        self.stats['cached_ea_intervals'] += 1
        self.stats['maximum_search_labels'] = max(self.stats['maximum_search_labels'], len(best))
        self.stats['ea_seconds'] += perf_counter() - started
        return arrival

    def first_projection(self, world, state, action):
        """Exact physical (stop,time) after one legal committed action.

        No observation or policy is evaluated; the continuation is already
        clairvoyant. Missing/late real publications keep the actual12-minute
        wait. Current-trip exemptions and platform MCT affect the ready clock.
        """
        if action == OBSERVE:
            return state.stop, state.time
        if action == WAIT:
            return state.stop, min(self.deadline, state.time + self.minute)
        if action[0] != 'select':
            raise ValueError('Use committed observe/wait/select actions')
        c = self.graph.connections[action[1]]
        same_trip = state.current_trip is not None and c.trip_id == state.current_trip
        transfer = (0 if state.current_trip is None or same_trip
                    else self.graph.stops[state.stop].min_transfer_time)
        ready = min(self.deadline, max(state.time, state.last_arrival + transfer))
        if ready >= self.deadline:
            return state.stop, self.deadline
        self.stats['root_event_reads'] += 1
        event = self._event(world, c.id)
        scene = self.bank[world][1]
        wait_end = min(self.deadline, ready + 12 * self.minute)
        if event.canceled:
            receipt = c.dep_time if scene.publications is None else scene.publications.get(f'{c.id}:cancel')
            return state.stop, (wait_end if receipt is None or receipt > wait_end else max(ready, receipt))
        if event.departure < ready:
            receipt = (event.departure if scene.publications is None
                       else scene.publications.get(f'{c.id}:departure'))
            return state.stop, wait_end if receipt is None or receipt > ready else ready
        if event.departure > wait_end:
            return state.stop, wait_end
        return ((state.stop, self.deadline) if event.arrival > self.deadline
                else (c.arr_stop, event.arrival))

    def action_lower(self, state, actions, support=None):
        """Fraction lower bounds in remaining-loss units for every given action.

        ``support`` must be exactly the bank indices compatible with this
        history. Merely selecting favorable worlds would invalidate the bound.
        """
        started = perf_counter()
        actions = tuple(actions)
        support = tuple(range(len(self.bank))) if support is None else tuple(support)
        weight = sum(self.weights[i] for i in support)
        if weight <= 0:
            raise ValueError('Public history has zero scenario mass')
        result = dict.fromkeys(actions, Fraction(0))
        for world in support:
            if not self.weights[world]:
                continue
            for action in actions:
                stop, time = self.first_projection(world, state, action)
                earliest = self.arrival(world, stop, time)
                result[action] += self.weights[world] * Fraction(earliest - state.time)
        self.stats['action_lower_seconds'] += perf_counter() - started
        self.stats['root_actions_evaluated'] += len(actions)
        return {action: value / weight for action, value in result.items()}
