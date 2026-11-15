"""Separate station observations from the action menu without changing motion.

Native execution delegates to the original executor. Under a common contract,
all declared local departures are observed while present at that station;
only currently visible action-menu vehicles can be boarded. Publications are
consumed at their public clock, and frozen rollouts never advance the stream.
"""
from collections import defaultdict
from time import perf_counter

from experiments.reconstruction_v34.execute import Executor as PublicExecutor
from experiments.reconstruction_v39.executor import SnapshotExecutor as OriginalSnapshotExecutor


class ObservationContract:
    """A root-frozen station-to-departure mapping, independent of action menus."""

    def __init__(self, station_to_cids):
        self.station_to_cids = {int(stop):tuple(sorted(set(cids)))
                                for stop, cids in station_to_cids.items()}

    @classmethod
    def from_menu(cls, node):
        by_stop = defaultdict(set)

        def visit(current):
            if current is None:
                return
            if hasattr(current, 'path'):
                for child in current.exits.values():
                    visit(child)
            else:
                by_stop[current.stop].update(current.rides)
                for ride in current.rides.values():
                    visit(ride)

        visit(node)
        return cls(by_stop)

    def cids_at(self, stop):
        return self.station_to_cids.get(stop, ())

    def trips(self, model):
        return {model._positions[cid, 'departure'][0]
                for cids in self.station_to_cids.values() for cid in cids}

    def paths(self):
        """Include these slots when constructing a shared physical world bank."""
        return [(cid,) for cid in sorted({cid for cids in self.station_to_cids.values() for cid in cids})]

    def describe(self):
        return dict(scope='Root-frozen station-local departures; independent of action menu',
            station_to_cids=[dict(stop=stop, cids=list(cids))
                             for stop, cids in sorted(self.station_to_cids.items())],
            observation_window='Original platform entry through departure from station; entry and ready remain distinct',
            current_offer='All simultaneous observed departures; only visible action-menu vehicles are boardable')


class ObservationMixin:
    """Use with a public executor whose choose_* hooks implement the policy."""

    def __init__(self, *args, observation_contract=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_contract = observation_contract
        if observation_contract is not None:
            self.counts.update(local_observation_epochs=0, observation_only_epochs=0)

    def observation_evidence(self, history, clock):
        return self.decision_history(history, clock)

    def platform(self, platform, entry, history, world, *, after=float('-inf')):
        if self.observation_contract is None:
            if after == float('-inf'):
                return super().platform(platform, entry, history, world)
            return super().platform(platform, entry, history, world, after=after)
        cids = self.observation_contract.cids_at(platform.stop)
        if not set(platform.rides).issubset(cids):
            raise ValueError('Every action-menu departure must be visible in the observation contract')
        ready = entry+self.graph.stops[platform.stop].min_transfer_time
        departures = {cid:world.event(cid, 'departure') for cid in cids}
        clocks = sorted({clock for clock, canceled, _ in departures.values()
                         if not canceled and entry <= clock < self.deadline and clock > after})
        for clock in clocks:
            observed = {cid:time for cid, (time, canceled, _) in departures.items()
                        if not canceled and entry <= time <= clock}
            observed_visible = tuple(cid for cid, (time, canceled, _) in departures.items()
                                     if not canceled and time == clock)
            visible = tuple(cid for cid in platform.rides if cid in observed_visible)
            tick = perf_counter()
            local_receipts = history.observe_platform(cids, entry, clock, observed)
            evidence = self.observation_evidence(history, clock)
            self.counts['local_observation_epochs'] += 1
            public = dict(stop=platform.stop, entry=entry, ready=ready, clock=clock,
                observation_cids=cids, observed=observed, observed_visible=observed_visible,
                visible=visible, local_receipts=local_receipts, **evidence)
            if clock < ready or not visible:
                self.counts['observation_only_epochs'] += 1
                self.compute_wall += perf_counter()-tick
                self.record(dict(kind='observation', **public))
                continue
            # Observation-only vehicles affect the shared history, not the
            # action IDs and seen IDs expected by a menu-local controller.
            action_observed = {cid:time for cid, time in observed.items() if cid in platform.rides}
            choice, receipt = self.choose_platform(platform, entry, ready, history,
                visible, action_observed, clock)
            if choice is not None and choice not in visible:
                raise ValueError('A controller selected a vehicle that is not a current legal offer')
            self.compute_wall += perf_counter()-tick
            self.counts['platform'] += 1
            self.record(dict(kind='platform', choice=choice, receipt=receipt,
                action_observed=action_observed, **public))
            if choice is not None:
                # Returning here ends every local nonoccurrence window. A
                # later onboard clock never updates this station's absence.
                return self.ride(platform.rides[choice], clock, history, world)
        return self.deadline, 'no_legal_offer'


class Executor(ObservationMixin, PublicExecutor):
    """Actual archive execution with current publications and common sight."""


class SnapshotExecutor(ObservationMixin, OriginalSnapshotExecutor):
    """The same local observation contract with publications held frozen."""

    def observation_evidence(self, history, clock):
        if not self.keep_trace:
            return {}
        return dict(public_history=history.public_state(),
                    message_state=dict(frame_index=history.frame_index, clock=history.message_clock))

    def choose_platform(self, platform, entry, ready, history, visible, observed, clock):
        return self.engine.platform(platform, entry, ready, history, visible, observed, clock,
            choose=self.choose, method=self.method)
