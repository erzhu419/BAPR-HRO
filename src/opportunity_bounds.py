"""First-exit arrival probability envelopes for frozen independent trip chains."""

from collections import defaultdict
from dataclasses import dataclass
from math import inf, isclose

import numpy as np

from .intermediate_boundary import FirstExitEnvelope
from .risk_bounds import CommitmentBounds


@dataclass(frozen=True)
class ExitWindow:
    connection: int
    ready: int
    remaining: float


@dataclass(frozen=True)
class OpportunityValue:
    lower: float
    trip_union_lower: float
    edge_union_lower: float
    static_lower: float
    probability_lower: float
    thresholds: tuple
    survival_lower: tuple
    local_trajectories: int
    unique_exit_summaries: int


class OrderedTripOpportunities:
    """Compute no qualifying ride across a trip's coupled event slots.

    A ride qualifies when it operates, departs at/after ready, and arrives
    by the threshold minus its remaining optimistic travel time.  Endpoint
    constraints and intra-trip dependence enter the same posterior chain.
    """

    def __init__(self, bounds: CommitmentBounds):
        self.bounds = bounds
        self.trip_by_cid = bounds.belief._trip_by_cid
        self.positions = bounds.belief._positions
        self.edge_tail = {}
        self.stats = {'single_connection_cells': 0, 'chain_edge_cells': 0,
                      'chain_queries': 0, 'single_connection_queries': 0}

    def connection_probability(self, window, thresholds):
        trip, slot = self.positions[window.connection, 'departure']
        law = self.bounds._law(trip)
        operating_departure = ~law.canceled[slot]
        operating_arrival = ~law.canceled[slot + 1]
        if window.connection not in self.edge_tail:
            edge = self.bounds._posterior_edge(trip, slot)
            joint = law.marginal(slot)[operating_departure, None] * edge[np.ix_(
                operating_departure, operating_arrival)]
            self.edge_tail[window.connection] = joint[::-1].cumsum(axis=0)[::-1].cumsum(axis=1)
            self.stats['single_connection_cells'] += joint.size
        self.stats['single_connection_queries'] += 1
        table = self.edge_tail[window.connection]
        departure = np.searchsorted(law.times[slot][operating_departure], window.ready)
        arrivals = np.searchsorted(law.times[slot + 1][operating_arrival],
                                   thresholds - window.remaining, side='right') - 1
        if departure == len(table) or not table.shape[1]:
            return np.zeros(len(thresholds))
        return np.where(arrivals >= 0, table[departure, np.maximum(arrivals, 0)], 0.)

    def no_event(self, trip, windows, thresholds):
        if len(windows) == 1:
            return np.clip(1. - self.connection_probability(windows[0], thresholds), 0., 1.)
        self.stats['chain_queries'] += 1
        slots = {self.positions[w.connection, 'departure'][1]: w for w in windows}
        first, last = min(slots), max(slots)
        law = self.bounds._law(trip)
        mass = np.broadcast_to(law.marginal(first), (len(thresholds), len(law.times[first])))
        for slot in range(first, last + 1):
            edge = self.bounds._posterior_edge(trip, slot)
            if slot not in slots:
                mass = mass @ edge
            else:
                w = slots[slot]
                eligible = (law.times[slot] >= w.ready) & ~law.canceled[slot]
                timely = law.times[slot + 1][None, :] <= thresholds[:, None] - w.remaining
                mass = (mass[:, ~eligible] @ edge[~eligible]
                        + (mass[:, eligible] @ edge[eligible]) * ~timely)
            self.stats['chain_edge_cells'] += len(thresholds) * edge.size
        return np.clip(mass.sum(axis=1), 0., 1.)


class FirstExitOpportunities:
    """An optimistic distribution of final loss from a complete exit inventory.

    The local marginal must cover the chosen fragment's clocks and service,
    including canceled segments. Other local-trip clocks may be omitted: such
    exits then use support-wide deterministic floors.  Every external trip is
    independent of the local trip and of other external trips at this frozen
    query. Correlated future messages may affect the policy but do not defeat
    a necessary event for a physical route.
    """

    def __init__(self, opportunities, graph, destination, *, runtime_floor=None,
                 departure_floor=None, footpaths=(), transfer_floor=None):
        self.opportunities = opportunities
        self.graph, self.destination = graph, destination
        runtime = {c.id: 0 for c in graph.connections} if runtime_floor is None else runtime_floor
        departure = {c.id: -inf for c in graph.connections} if departure_floor is None else departure_floor
        self.static = FirstExitEnvelope(graph, destination, runtime, departure,
                                       footpaths=footpaths, transfer_floor=transfer_floor)

    def _inventory(self, path, used):
        visited = set(used)
        stop = self.graph.connections[path[0]].dep_stop
        trip = self.opportunities.trip_by_cid[path[0]]
        stages = []
        for cid in (*path, None):
            if stop == self.destination:
                break
            if cid is not None:
                c = self.graph.connections[cid]
                if c.dep_stop != stop or self.opportunities.trip_by_cid[cid] != trip or cid in visited:
                    raise ValueError('The fragment must be an unused, contiguous same-vehicle path')
            stages.append((stop, tuple(c for c in self.graph.get_connections_from(stop)
                          if c.id != cid and c.id not in visited
                          and (c.pickup_allowed or self.opportunities.trip_by_cid[c.id] == trip))))
            if cid is not None:
                visited.add(cid)
                stop = self.graph.connections[cid].arr_stop
        return trip, stages

    def _summary(self, trajectory, path, trip, inventory, start, deadline, incoming_cid):
        deterministic = deadline
        windows = {}
        time, previous = start, incoming_cid
        for prefix, (stop, exits) in enumerate(inventory):
            for _, target, duration in self.static._walk_index[stop]:
                deterministic = min(deterministic, time + duration + self.static.distance.get(target, inf))
            for c in exits:
                remaining = self.static.distance.get(c.arr_stop, inf)
                if not np.isfinite(remaining):
                    continue
                other_trip = self.opportunities.trip_by_cid[c.id]
                if other_trip == trip:
                    if c.id in trajectory:
                        event = trajectory[c.id]
                        if not event.canceled and event.departure >= time:
                            deterministic = min(deterministic, event.arrival + remaining)
                    else:
                        departure = max(time, self.static.departure_floor[c.id])
                        deterministic = min(deterministic, departure + self.static.runtime_floor[c.id] + remaining)
                else:
                    ready = time + self.static.transfer_floor.get((previous, c.id), 0)
                    if ready + remaining < min(deadline, deterministic):
                        windows[c.id] = min(windows.get(c.id, inf), ready)
            if prefix == len(path):
                break
            cid = path[prefix]
            event = trajectory[cid]
            if event.canceled or event.departure < time or event.arrival >= deadline:
                break
            time, previous = event.arrival, cid
            if self.graph.connections[cid].arr_stop == self.destination:
                deterministic = min(deterministic, time)
                break
        return deterministic, tuple(sorted(windows.items()))

    def evaluate(self, local_law, path, start, deadline, thresholds, *, incoming_cid=None, used=()):
        """Group the integer-clock survival sum at the end of each time bin."""
        path, local_law, used = tuple(path), tuple(local_law), frozenset(used)
        thresholds = np.asarray(thresholds, dtype=float)
        if (not len(thresholds) or thresholds[-1] != deadline
                or not np.all(np.diff(np.r_[start, thresholds]) > 0)
                or np.any(thresholds != np.floor(thresholds)) or start != int(start)):
            raise ValueError('Integer thresholds must increase from start and end at deadline')
        if not isclose(sum(weight for weight, _ in local_law), 1., abs_tol=1e-12):
            raise ValueError('Local trajectory weights must sum to one')
        trip, inventory = self._inventory(path, used)
        signatures = defaultdict(float)
        static_lower = 0.
        for weight, trajectory in local_law:
            if weight == 0:
                continue
            static_lower += weight * self.static.local_lower(trajectory, path, start,
                deadline=deadline, incoming_cid=incoming_cid, used=used)[0]
            signatures[self._summary(trajectory, path, trip, inventory,
                                     start, deadline, incoming_cid)] += weight
        # For integer Y, E[Y-t] sums P(Y>j), j=t,...,D-1. Each bin uses
        # its last j, giving a lower sum even on a coarser threshold grid.
        cutoffs = thresholds - 1
        survival = np.zeros((3, len(cutoffs)))
        for (deterministic, windows), weight in signatures.items():
            threshold_mask = cutoffs < deterministic
            if not threshold_mask.any():
                continue
            clocks = cutoffs[threshold_mask]
            by_trip = defaultdict(list)
            for cid, ready in windows:
                remaining = self.static.distance[self.graph.connections[cid].arr_stop]
                if ready + remaining <= clocks[-1]:
                    by_trip[self.opportunities.trip_by_cid[cid]].append(ExitWindow(cid, ready, remaining))
            product = np.ones(len(clocks))
            trip_union = np.zeros(len(clocks))
            edge_union = np.zeros(len(clocks))
            for external_trip, exits in by_trip.items():
                none = self.opportunities.no_event(external_trip, exits, clocks)
                product *= none
                trip_union += 1. - none
                if len(exits) == 1:
                    edge_union += 1. - none
                else:
                    for window in exits:
                        edge_union += self.opportunities.connection_probability(window, clocks)
            survival[:, threshold_mask] += weight * np.stack(
                (product, np.maximum(0., 1. - trip_union), np.maximum(0., 1. - edge_union)))
        widths = np.diff(np.r_[start, thresholds])
        probability, trip_union, edge_union = (float(value) for value in survival @ widths)
        return OpportunityValue(max(static_lower, probability), max(static_lower, trip_union),
            max(static_lower, edge_union), static_lower, probability, tuple(cutoffs.tolist()),
            tuple(survival[0].tolist()), len(local_law), len(signatures))
