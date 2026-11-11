"""Reuse feasible departure ranges across simulations of one UCT query."""

from bisect import bisect_left
from collections import defaultdict

from .causal_planning_router import CausalPlanningRouter
from .committed_transit import WAIT
from .continuation_router import nominal_action
from .evolving_trip_belief import EvolvingTripBelief


class NominalDepartureIndex:
    """Lazy index of unchanged trip means at the query's public history.

    It prunes only edges the original nominal search rejects unconditionally:
    no operating mean, rounded arrival past deadline, or departure before the
    current label. An observation can change a whole trip, so the branch reads
    those edges separately. Arrival ordering and tie handling stay in the
    original search, with the same connection iteration order.
    """

    def __init__(self, root, parameters, deadline):
        self.root = root
        self.parameters = parameters
        self.deadline = deadline
        self.root_events = root.mean_events(parameters)
        self.stops = {}
        self.stats = dict(indexed_stops=0, indexed_edges=0, labels=0,
                          candidate_edges=0, branch_edges=0, tail_calls=0)

    def edges_at(self, stop):
        if stop not in self.stops:
            entries = []
            for c in self.root.graph.get_connections_from(stop):
                event = self.root_events[c.id]
                if self._eligible(event):
                    entries.append((round(event.departure), c.id, c))
                self.stats['indexed_edges'] += 1
            self.stops[stop] = sorted(entries)
            self.stats['indexed_stops'] += 1
        return self.stops[stop]

    def _eligible(self, event):
        return (event.departure is not None and
                round(event.departure) + round(event.arrival - event.departure) <= self.deadline)

    def branch(self, belief):
        return _NominalBranch(self, belief)


class _NominalBranch:
    def __init__(self, index, belief):
        self.index = index
        self.events = None
        self.belief = belief
        self.changed = {trip for trip, revision in belief._revision.items()
                        if revision != index.root._revision.get(trip, 0)}
        self.by_stop = defaultdict(list)
        for trip in self.changed:
            for slot in belief._trips[trip][::2]:
                c = belief.graph.connections[slot.cid]
                self.by_stop[c.dep_stop].append(c)

    def __getitem__(self, cid):
        if self.belief._trip_by_cid[cid] not in self.changed:
            return self.index.root_events[cid]
        if self.events is None:
            self.events = self.belief.mean_events(self.index.parameters)
        return self.events[cid]

    def candidates(self, stop, clock):
        entries = self.index.edges_at(stop)
        first = bisect_left(entries, (clock, -1))
        trip_by_cid = self.belief._trip_by_cid
        result = [c for _, cid, c in entries[first:] if trip_by_cid[cid] not in self.changed]
        for c in self.by_stop.get(stop, ()):
            event = self[c.id]
            if self.index._eligible(event) and round(event.departure) >= clock:
                result.append(c)
        self.index.stats['branch_edges'] += len(self.by_stop.get(stop, ()))
        self.index.stats['candidate_edges'] += len(result)
        self.index.stats['labels'] += 1
        return sorted(result, key=lambda c: c.id)


class IndexedNominalTailRouter(CausalPlanningRouter):
    def _nominal_tail(self, parameters):
        if not isinstance(self.trip_belief, EvolvingTripBelief):
            return super()._nominal_tail(parameters)
        index = NominalDepartureIndex(self.trip_belief, parameters, self.journey_deadline)
        self.nominal_index_stats = index.stats

        def tail(state, belief, actions):
            boardings = {action[1] for action in actions if action[0] == 'select'}
            if not boardings:
                return WAIT
            index.stats['tail_calls'] += 1
            branch = index.branch(belief)
            return nominal_action(self.graph, state, self.destination, branch, self.menu,
                self.journey_deadline, self.lookahead, root_boardings=boardings, minute=self.minute,
                connection_candidates=branch.candidates)

        return tail
