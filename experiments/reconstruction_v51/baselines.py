"""Ordinary policies on one common action menu and public information state.

The first five policies reuse existing conditional planners/waiting controls.
The deterministic snapshot baseline scans menu-lifted connections: it cannot
create a transfer at an exit that the common menu does not declare. None of
these controls receives an archived world or unrevealed publication frames.
"""
from itertools import groupby
from math import inf

import numpy as np

from experiments.reconstruction_v33.execute import Executor as WaitingControls
from experiments.reconstruction_v33.nodes import Ride
from experiments.reconstruction_v39.state import raw_history_key
from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine


CHEAP_LABELS = ('cheap', 'mean', 'single', 'pair', 'one_step', 'snapshot_csa')
CONTROLS = {
    'cheap': ('rolling_fixed', 'combined'),
    'mean': ('nominal', 'nominal'),
    'single': ('rolling_fixed', 'single'),
    'pair': ('rolling_fixed', 'pair'),
    'one_step': ('one_step_rollout', 'pair'),
}


class MenuLayout:
    """Lift occurrences so shared stops cannot add undeclared transfers."""

    def __init__(self, node):
        self.rides, self.platforms, self.records = {}, {}, []
        if isinstance(node, Ride):
            self.add_ride(node, None)
        else:
            self.add_platform(node)

    def add_platform(self, platform):
        pid = id(platform)
        if pid in self.platforms:
            return pid
        self.platforms[pid] = platform
        for ride in platform.rides.values():
            self.add_ride(ride, pid)
        return pid

    def add_ride(self, ride, parent):
        # A Ride can be reused at distinct platform occurrences. Its separate
        # entry edges lead to the same physical suffix, which is safe to share.
        rid = id(ride)
        if rid in self.rides:
            if parent is not None:
                self.rides[rid]['parents'].add(parent)
            return rid
        self.rides[rid] = dict(ride=ride, parents=set() if parent is None else {parent})
        for position, cid in enumerate(ride.path):
            destination = cid in ride.exits and ride.exits[cid] is None
            tail = ride.exits.get(cid)
            tail_id = None if tail is None else self.add_platform(tail)
            self.records.append(dict(cid=cid, rid=rid, position=position,
                destination=destination, tail=tail_id))
            # Physical execution ends at the first declared destination.
            if destination:
                break
        return rid


class MeanSnapshotCSA:
    """Connection scan on the current W conditional-mean timetable.

This is deterministic earliest arrival within the supplied menu, not CSA-MEAT
or a stochastic optimum. Canceled-only endpoints and temporally inconsistent
mean connections are unavailable in this snapshot. The physical action menu
is unchanged and every later visible offer triggers a new public snapshot.
"""

    def __init__(self, graph, planner, deadline):
        self.graph, self.planner, self.deadline = graph, planner, float(deadline)
        self._layouts, self._snapshots = {}, {}
        self.stats = dict(csa_snapshot_builds=0, csa_snapshot_hits=0,
            csa_endpoint_means=0, csa_unavailable_connections=0,
            csa_inconsistent_mean_connections=0, csa_scans=0,
            csa_connection_visits=0, csa_equal_time_passes=0)

    def _snapshot(self, node, history, clock):
        key = node, float(clock), raw_history_key(history, node.trips(self.planner.model))[0]
        if key in self._snapshots:
            self.stats['csa_snapshot_hits'] += 1
            return self._snapshots[key]
        if node not in self._layouts:
            self._layouts[node] = MenuLayout(node)
        layout = self._layouts[node]
        means = {}
        for record in layout.records:
            cid = record['cid']
            for kind in ('departure', 'arrival'):
                if (cid, kind) in means:
                    continue
                trip, slot = self.planner.model._positions[cid, kind]
                law = history.law(trip)
                mass = law.marginal(slot)
                active = (~law.canceled[slot]) & (mass > 0)
                times, probability = law.times[slot][active], mass[active]
                value = (None if not len(times) else float(times[0]) if len(times) == 1
                         else float(times @ probability / probability.sum()))
                means[cid, kind] = value
                self.stats['csa_endpoint_means'] += 1
        records = []
        inconsistent = unavailable = 0
        for sequence, record in enumerate(layout.records):
            departure, arrival = (means[record['cid'], kind] for kind in ('departure', 'arrival'))
            if departure is None or arrival is None:
                unavailable += 1
                continue
            if arrival < departure:
                inconsistent += 1
                continue
            records.append(dict(record, departure=departure, arrival=arrival, sequence=sequence))
        records.sort(key=lambda row: (row['departure'], row['sequence']))
        result = dict(layout=layout, records=records, means=means,
                      inconsistent=inconsistent, unavailable=unavailable)
        self._snapshots[key] = result
        self.stats['csa_snapshot_builds'] += 1
        self.stats['csa_unavailable_connections'] += unavailable
        self.stats['csa_inconsistent_mean_connections'] += inconsistent
        return result

    def _scan(self, snapshot, clock, *, platform=None, ready=None, ride=None, position=None):
        """Standard forward connection scanning on the lifted action graph."""
        self.stats['csa_scans'] += 1
        layout = snapshot['layout']
        platforms, riding = {}, {}
        if platform is not None:
            platforms[id(platform)] = float(ready)
        else:
            riding[id(ride), position] = float(clock)
        best = self.deadline

        def lower(table, key, value):
            if value < table.get(key, inf):
                table[key] = value
                return True
            return False

        for departure, group in groupby(snapshot['records'], key=lambda row: row['departure']):
            if departure < clock:
                continue
            if departure >= best:
                break
            block = tuple(group)
            # Equal-clock zero-duration edges must not depend on menu order.
            # Each pass strictly adds/decreases a finite lifted-state label.
            changed = True
            while changed:
                changed = False
                self.stats['csa_equal_time_passes'] += 1
                for row in block:
                    self.stats['csa_connection_visits'] += 1
                    rid, index = row['rid'], row['position']
                    accessible = riding.get((rid, index), inf) <= departure
                    if index == 0:
                        accessible |= any(platforms.get(parent, inf) <= departure
                                          for parent in layout.rides[rid]['parents'])
                    if not accessible:
                        continue
                    arrival = row['arrival']
                    if row['destination']:
                        best = min(best, arrival)
                        continue
                    updated = lower(riding, (rid, index+1), arrival)
                    changed |= updated and arrival <= departure
                    if row['tail'] is not None:
                        tail = layout.platforms[row['tail']]
                        transfer_ready = arrival+self.graph.stops[tail.stop].min_transfer_time
                        updated = lower(platforms, row['tail'], transfer_ready)
                        changed |= updated and transfer_ready <= departure
        return float(best)

    def onboard(self, ride, cid, clock, history):
        snapshot = self._snapshot(ride, history, clock)
        tail = ride.exits[cid]
        alight = (float(clock) if tail is None else self._scan(snapshot, clock,
            platform=tail, ready=clock+self.graph.stops[tail.stop].min_transfer_time))
        next_position = ride.path.index(cid)+1
        continuation = (self.deadline if next_position == len(ride.path) else
            self._scan(snapshot, clock, ride=ride, position=next_position))
        choice = alight < self.deadline and alight <= continuation+1e-9
        return choice, dict(method='public_mean_snapshot_csa', alight=alight,
            continue_cost=continuation, mean_timetable_inconsistent=snapshot['inconsistent'],
            scope='Deterministic current-W operating means; original menu exits only')

    def platform(self, platform, entry, ready, history, visible, observed, clock):
        snapshot = self._snapshot(platform, history, clock)
        visible, removed = set(visible), set(observed)-set(visible)
        ranks = self.planner.ranks(platform)
        values = {}
        for cid, ride in platform.rides.items():
            if cid in removed:
                continue
            departure = snapshot['means'].get((cid, 'departure'))
            if departure is None or departure < ready or departure < clock:
                continue
            # Current visible offers have exact departure evidence. Unseen
            # offers at this clock are not legal instantaneous boarding acts.
            if cid not in visible and departure <= clock:
                continue
            values[cid] = self._scan(snapshot, departure, ride=ride, position=0)
        target = min(values, key=lambda cid: (values[cid], ranks[cid]), default=None)
        if target is not None and values[target] >= self.deadline:
            target = None
        return (target if target in visible else None), dict(method='public_mean_snapshot_csa',
            target=target, values=values, entry=float(entry), ready=float(ready),
            mean_timetable_inconsistent=snapshot['inconsistent'],
            scope='Deterministic current-W operating means; original menu exits only')


class BaselinePolicy:
    """Hooks for the common V50 observation executor."""

    def __init__(self, graph, planner, deadline, label):
        if label not in CHEAP_LABELS:
            raise ValueError(label)
        self.label, self.graph, self.planner, self.deadline = label, graph, planner, deadline
        self.engine = PhysicalDecisionEngine(planner, variant='frontier')
        self.csa = MeanSnapshotCSA(graph, planner, deadline) if label == 'snapshot_csa' else None
        self.method, self.waiting = CONTROLS.get(label, ('rolling_fixed', 'combined'))
        self.waiting_control = WaitingControls(graph, planner, deadline,
            method=self.method, waiting=self.waiting, trace=False)

    def onboard(self, ride, cid, clock, history):
        if self.csa is not None:
            return self.csa.onboard(ride, cid, clock, history)
        return self.engine.onboard(ride, cid, clock, history, method=self.method)

    def platform(self, platform, entry, ready, history, visible, observed, clock):
        if self.csa is not None:
            return self.csa.platform(platform, entry, ready, history, visible, observed, clock)
        return self.engine.platform(platform, entry, ready, history, visible, observed, clock,
            choose=self.waiting_control.choose, method=self.method)

    @property
    def stats(self):
        result = dict(self.engine.stats)
        if self.csa is not None:
            result.update(self.csa.stats)
        return result
