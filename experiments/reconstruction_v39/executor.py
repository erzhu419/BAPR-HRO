"""The original physical snapshot executor with decision-engine hooks only."""
from time import perf_counter

import numpy as np

from experiments.reconstruction_v34.conditional_reference import (
    ProjectionIncompatible, SnapshotExecutor as OriginalSnapshotExecutor,
)


class SnapshotExecutor(OriginalSnapshotExecutor):
    def __init__(self, graph, planner, deadline, *, engine, method='rolling_fixed', waiting='combined', trace=True):
        super().__init__(graph,planner,deadline,method=method,waiting=waiting,trace=trace)
        self.engine = engine

    def platform(self, platform, entry, history, world, *, after=-np.inf):
        ready = entry+self.graph.stops[platform.stop].min_transfer_time
        departures = {cid: world.event(cid, 'departure') for cid in platform.rides}
        clocks = sorted({t for t, canceled, _ in departures.values()
                         if not canceled and ready <= t < self.deadline and t > after})
        for clock in clocks:
            observed = {cid: t for cid, (t, canceled, _) in departures.items()
                        if not canceled and entry <= t <= clock}
            visible = tuple(cid for cid, (t, canceled, _) in departures.items()
                            if not canceled and t == clock)
            tick = perf_counter()
            history.observe_platform(tuple(platform.rides), entry, clock, observed)
            choice, receipt = self.engine.platform(platform,entry,ready,history,visible,observed,clock,
                choose=self.choose,method=self.method)
            self.compute_wall += perf_counter()-tick
            self.counts['platform'] += 1
            self.record(dict(kind='platform', stop=platform.stop, clock=clock, choice=choice))
            if choice is not None:
                return self.ride(platform.rides[choice], clock, history, world)
        return self.deadline, 'no_legal_offer'

    def ride(self, ride, ready, history, world):
        previous, vehicle = ready, None
        for cid in ride.path:
            departure, canceled, identity = world.event(cid, 'departure')
            if canceled or departure >= self.deadline:
                return self.deadline, 'service_or_deadline_failure'
            if departure < previous:
                raise ProjectionIncompatible(f'{cid}:departure={departure}<known_previous={previous}')
            if vehicle is not None and identity != vehicle:
                raise ValueError('A sampled ride changed its physical vehicle')
            vehicle = identity
            arrival, canceled, arrival_identity = world.event(cid, 'arrival')
            if canceled:
                return self.deadline, 'service_failure'
            if arrival_identity != vehicle:
                raise ValueError('Sampled arrival changed its physical vehicle')
            if arrival < departure:
                raise ProjectionIncompatible(f'{cid}:arrival={arrival}<departure={departure}')
            if arrival >= self.deadline:
                return self.deadline, 'deadline'
            history.observe(cid, 'departure', departure)
            history.observe(cid, 'arrival', arrival)
            previous = arrival
            if cid not in ride.exits:
                continue
            platform = ride.exits[cid]
            if platform is None:
                return arrival, 'arrived'
            tick = perf_counter()
            alight, receipt = self.engine.onboard(ride,cid,arrival,history,method=self.method)
            self.compute_wall += perf_counter()-tick
            self.counts['onboard'] += 1
            self.record(dict(kind='onboard', cid=cid, clock=arrival,
                             choice='alight' if alight else 'continue'))
            if alight:
                return self.platform(platform, arrival, history, world)
        return self.deadline, 'exhausted_ride'
