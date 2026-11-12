"""End-to-end local observation, shared history, and repeated alighting.

Only this executor reads realized events. The planner receives a posterior
history and a public clock, never a world index or unrealized endpoint.
"""

from time import perf_counter

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from src.plan_waiting_controls import SingleTargetOpportunity, CombinedWaitingOpportunity, TwoTargetOpportunity


class MissingEndpoint(ValueError):
    pass


class ArchivedWorld:
    def __init__(self, metadata):
        self.metadata = metadata

    def event(self, cid, kind):
        row = self.metadata[cid]
        endpoint = row[kind]
        if endpoint['status'] == 'CANCELED':
            return row['scheduled_'+kind+'_seconds'], True, endpoint.get('official_journey')
        if endpoint['status'] != 'REAL':
            raise MissingEndpoint(f'{cid}:{kind}:{endpoint["status"]}')
        return endpoint['time_seconds'], False, endpoint.get('official_journey')


class DrawnWorld:
    def __init__(self, bank, index):
        self.bank, self.index = bank, index

    def event(self, cid, kind):
        times, canceled = self.bank.events[cid, kind]
        trip = self.bank.belief._positions[cid, kind][0]
        return int(times[self.index]), bool(canceled[self.index]), trip


class Executor:
    def __init__(self, graph, planner, deadline, *, method='one_step_rollout', waiting='pair', trace=True):
        self.graph, self.planner, self.deadline = graph, planner, deadline
        self.method, self.waiting, self.keep_trace = method, waiting, trace
        self.trace, self.counts, self.compute_wall = [], dict(platform=0, onboard=0), 0.

    def record(self, row):
        if self.keep_trace:
            self.trace.append(row)

    def choose(self, reference, visible, seen, clock):
        if self.waiting == 'first':
            return reference.choose(visible, seen, clock, 'first_offer')
        single = SingleTargetOpportunity(reference)
        if self.waiting == 'single':
            return single.choose(visible, seen, clock)
        combined = CombinedWaitingOpportunity(single)
        if self.waiting == 'combined':
            return combined.choose(visible, seen, clock)
        if self.waiting == 'pair':
            return TwoTargetOpportunity(combined).choose(visible, seen, clock)
        if self.waiting == 'nominal':
            # Deterministic replanning under current public conditional means.
            # A cancellation-only or already observed alternative is unavailable.
            values = {cid: reference.score(cid, clock) for cid in visible}
            for cid in reference.cids:
                if cid in seen or cid in visible:
                    continue
                times, mass, _ = reference.distributions[cid]
                legal = (times > clock) & (times < self.deadline)
                probability = mass[legal].sum()
                if probability > 0:
                    expected = float(times[legal] @ mass[legal]/probability)
                    values[cid] = reference.score(cid, expected)
            target = min(values, key=lambda cid: (values[cid], reference.ranks[cid]), default=None)
            return (target if target in visible else None), dict(method='rolling_mean', target=target, values=values)
        raise ValueError(self.waiting)

    def platform(self, platform, entry, history, world):
        ready = entry+self.graph.stops[platform.stop].min_transfer_time
        departures = {cid: world.event(cid, 'departure') for cid in platform.rides}
        clocks = sorted({clock for clock, canceled, _ in departures.values()
                         if not canceled and ready <= clock < self.deadline})
        for clock in clocks:
            observed = {cid: time for cid, (time, canceled, _) in departures.items()
                        if not canceled and entry <= time <= clock}
            visible = tuple(cid for cid, (time, canceled, _) in departures.items()
                            if not canceled and time == clock)
            tick = perf_counter()
            history.observe_platform(tuple(platform.rides), entry, clock, observed)
            reference = self.planner.platform_reference(platform, history, entry, ready, self.method)
            choice, receipt = self.choose(reference, visible, tuple(observed), clock)
            self.compute_wall += perf_counter()-tick
            self.counts['platform'] += 1
            self.record(dict(kind='platform', stop=platform.stop, entry=entry, ready=ready,
                             clock=clock, observed=observed, visible=visible, choice=choice, receipt=receipt))
            if choice is not None:
                return self.ride(platform.rides[choice], clock, history, world)
        return self.deadline, 'no_legal_offer'

    def ride(self, ride, ready, history, world):
        previous, vehicle = ready, None
        for cid in ride.path:
            departure, canceled, identity = world.event(cid, 'departure')
            if canceled or departure < previous or departure >= self.deadline:
                return self.deadline, 'service_or_clock_failure'
            if identity is None or vehicle is not None and identity != vehicle:
                raise MissingEndpoint(f'{cid}:inconsistent_vehicle_identity')
            vehicle = identity
            arrival, canceled_arrival, arrival_identity = world.event(cid, 'arrival')
            if canceled_arrival:
                return self.deadline, 'service_failure'
            if arrival_identity != vehicle or arrival < departure:
                raise MissingEndpoint(f'{cid}:inconsistent_arrival_identity_or_clock')
            if arrival >= self.deadline:
                return self.deadline, 'deadline'
            history.observe(cid, 'departure', departure)
            history.observe(cid, 'arrival', arrival)
            previous = arrival
            if cid not in ride.exits:
                continue
            platform = ride.exits[cid]
            if platform is None:
                self.record(dict(kind='destination', cid=cid, clock=arrival))
                return arrival, 'arrived'
            tick = perf_counter()
            compiled = self.planner.compile_ride(ride, history, self.method)
            alight, receipt = compiled.decision(cid, arrival)
            self.compute_wall += perf_counter()-tick
            self.counts['onboard'] += 1
            self.record(dict(kind='onboard', cid=cid, clock=arrival, choice='alight' if alight else 'continue',
                             receipt=receipt))
            if alight:
                return self.platform(platform, arrival, history, world)
        return self.deadline, 'exhausted_ride'

    def run(self, ride, start, history, world):
        began = perf_counter()
        unsupported_evidence = None
        try:
            arrival, status = self.ride(ride, start, history, world)
        except MissingEndpoint as error:
            arrival, status = None, 'unresolved:'+str(error)
        except UnsupportedHistory as error:
            arrival, status = None, 'unsupported_public_history:'+str(error)
            unsupported_evidence = history.public_state()
        return dict(loss=None if arrival is None else arrival-start, status=status,
            trace=self.trace, decisions=self.counts, compute_wall=self.compute_wall,
            unsupported_evidence=unsupported_evidence,
            total_wall=perf_counter()-began)
