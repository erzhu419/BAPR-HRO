"""Shared-trip physical draws and frozen, causal tail-menu controllers.

The decision compiler never receives a sampled or archived future. Physical
outcomes are integrated separately, preserving correlations between options
that refer to the same vehicle. A tail variant is fixed when its first vehicle
is boarded; its final platform takes the first legal offered vehicle.
"""

from collections import defaultdict
from functools import cached_property

import numpy as np

from src.transfer_opportunity import RouteOption, TransferOpportunityScores


class TailMenu:
    def __init__(self, graph, belief, parameters, hub, deadline):
        self.graph, self.belief, self.deadline = graph, belief, deadline
        self.hub, self.variants, self.compilers = hub, defaultdict(list), {}
        for index, row in enumerate(hub['options']):
            cid = row['first_cid']
            option = RouteOption(tuple(row['first_path']),
                                 {p[0]: tuple(p) for p in row['tail_paths']})
            self.variants[cid].append(index)
            self.compilers[index] = TransferOpportunityScores(
                graph, belief, parameters, deadline, {cid: option})
        self.cids = tuple(sorted(self.variants))
        self.ranks = {cid: (min(hub['options'][i]['scheduled_arrival']
                                for i in self.variants[cid]), cid) for cid in self.cids}
        self._scores = {}

    def departure_distribution(self, cid):
        return self.compilers[self.variants[cid][0]].departure_distribution(cid)

    def best(self, cid, clock):
        key = cid, float(clock)
        if key not in self._scores:
            values = []
            for index in self.variants[cid]:
                score = self.compilers[index].board_score(cid, clock)
                if score is not None:
                    values.append((score.exact, index))
            self._scores[key] = min(values) if values else (float(self.deadline), None)
        return self._scores[key]

    def clock_nominal_score(self, cid, clock):
        # FullMenuOpportunity's interface name is historical. Here all waiting
        # controls use the SAME full conditional fixed-variant mark.
        return self.best(cid, clock)[0]

    @cached_property
    def paths(self):
        return [tuple(p) for row in self.hub['options']
                for p in [row['first_path'], *row['tail_paths']]]


class TripWorlds:
    """Draw each physical vehicle once, independent of where it is requested."""
    def __init__(self, belief, parameters, paths, count, seed):
        self.belief, self.count = belief, count
        self.events = {}
        trips = sorted({belief._positions[cid, 'departure'][0]
                        for path in paths for cid in path})
        rng = np.random.default_rng(seed)
        for trip in trips:
            law = belief._law(trip, parameters)
            draws = law.draw_uniforms_batch(rng.random((count, len(law.times))))
            for slot, event in enumerate(belief._trips[trip]):
                selected = draws[:, slot]
                self.events[event.cid, event.kind] = (
                    law.times[slot][selected], law.canceled[slot][selected])
        self._paths = {}

    def path(self, path, deadline):
        path = tuple(path)
        if path not in self._paths:
            value = np.full(self.count, deadline, dtype=float)
            live = np.ones(self.count, dtype=bool)
            previous = np.full(self.count, -np.inf)
            for cid in path:
                departure, dc = self.events[cid, 'departure']
                arrival, ac = self.events[cid, 'arrival']
                live &= ~dc & ~ac & (departure >= previous) & (arrival >= departure)
                live &= (departure < deadline) & (arrival <= deadline)
                previous = arrival
            value[live] = previous[live]
            self._paths[path] = value
        return self._paths[path]

    def option(self, graph, row, deadline):
        first = self.path(row['first_path'], deadline)
        if not row['tail_paths']:
            return first
        ready = first + graph.stops[row['transfer_stop']].min_transfer_time
        selected = np.full(self.count, deadline, dtype=float)
        first_time = np.full(self.count, np.inf)
        # Tie order is fixed scheduled duration then cid, as in V25.
        tails = sorted(row['tail_paths'], key=lambda p: (
            graph.connections[p[-1]].arr_time-graph.connections[p[0]].dep_time, p[0]))
        for path in tails:
            departure, canceled = self.events[path[0], 'departure']
            use = ~canceled & (departure >= ready) & (departure < deadline) & (departure < first_time)
            selected[use] = self.path(path, deadline)[use]
            first_time[use] = departure[use]
        return selected


class TailOutcomes:
    """Evaluate frozen public-prefix decisions in shared physical worlds."""
    def __init__(self, menu, worlds):
        self.menu, self.worlds = menu, worlds
        self.departures = np.column_stack([
            np.where(worlds.events[cid, 'departure'][1], np.inf,
                     worlds.events[cid, 'departure'][0]) for cid in menu.cids])
        self.order = np.argsort(self.departures, axis=1, kind='stable')
        self.boarded = np.full(self.departures.shape, menu.deadline, dtype=float)
        self.variant_choices = np.full(self.departures.shape, -1, dtype=int)
        for column, cid in enumerate(menu.cids):
            for clock in np.unique(self.departures[:, column]):
                if not np.isfinite(clock):
                    continue
                _, index = menu.best(cid, clock)
                rows = self.departures[:, column] == clock
                if index is not None:
                    value = worlds.option(menu.graph, menu.hub['options'][index], menu.deadline)
                    self.boarded[rows, column] = value[rows]
                    self.variant_choices[rows, column] = index

    def evaluate(self, tables, entry, ready, methods):
        n, k = self.departures.shape
        bits = 1 << np.arange(k)
        full = (1 << k)-1
        result = {m: np.full(n, self.menu.deadline, dtype=float) for m in methods}
        live = {m: np.ones(n, dtype=bool) for m in methods}
        rows = np.arange(n)
        previous = np.full(n, -np.inf)
        for position in range(k):
            clock = self.departures[rows, self.order[:, position]]
            valid = (clock >= ready) & (clock < self.menu.deadline) & (clock != previous)
            previous = clock
            if not valid.any():
                continue
            ids = rows[valid]
            now = clock[valid]
            observed = (self.departures[ids] >= entry) & (self.departures[ids] <= now[:, None])
            seen = observed @ bits
            present = (self.departures[ids] == now[:, None]) @ bits
            time_index = np.searchsorted(tables['times'], now)
            if not np.array_equal(tables['times'][time_index], now):
                raise AssertionError('Sampled departure is outside the frozen public event grid')
            for method in methods:
                choice = tables['choices'][method][time_index, full ^ seen, present]
                take = (choice >= 0) & live[method][ids]
                chosen_rows = ids[take]
                result[method][chosen_rows] = self.boarded[chosen_rows, choice[take]]
                live[method][chosen_rows] = False
        return result


def paired_statistics(left, right):
    """Positive gain means the right-hand executable policy is better."""
    difference = np.asarray(left, float)-np.asarray(right, float)
    mean = float(difference.mean())
    se = float(difference.std(ddof=1)/np.sqrt(len(difference))) if len(difference) > 1 else 0.
    return dict(n=len(difference), mean_gain=mean, se=se, ci95=[mean-1.96*se, mean+1.96*se],
                wins=int((difference > 1e-9).sum()), harms=int((difference < -1e-9).sum()),
                ties=int((np.abs(difference) <= 1e-9).sum()))
