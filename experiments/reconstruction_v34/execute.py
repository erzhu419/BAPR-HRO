"""Physical execution with messages revealed at the current decision clock."""

from time import perf_counter

from experiments.reconstruction_v33.execute import (
    ArchivedWorld, DrawnWorld, Executor as PhysicalExecutor, MissingEndpoint,
)


class Executor(PhysicalExecutor):
    """Use the existing controls and execution contract on a message history.

    Posterior planning keeps the received history frozen. Only these actual
    platform and onboard decisions advance the publication stream.
    """

    def decision_history(self, history, clock):
        first_frame = history.frame_index
        receipt = history.advance(clock)
        receipt['first_frame_index'] = first_frame
        receipt['next_frame_index'] = history.frame_index
        for index, item in enumerate(receipt['receipts'], first_frame):
            frame = history.frames[index]
            item.update(frame_index=index, consumed_at_clock=float(clock),
                publication={key: frame[key] for key in ('archive_member', 'entity_id', 'trip_id',
                    'capture_seconds', 'header_seconds', 'update_seconds', 'receipt_seconds') if key in frame})
        return dict(message_receipt=receipt, public_history=history.public_state(),
                    message_state=dict(frame_index=history.frame_index,
                                       clock=history.message_clock))

    def choose_platform(self, platform, entry, ready, history, visible, observed, clock):
        reference = self.planner.platform_reference(platform, history, entry, ready, self.method)
        return self.choose(reference, visible, tuple(observed), clock)

    def choose_onboard(self, ride, cid, clock, history):
        compiled = self.planner.compile_ride(ride, history, self.method)
        return compiled.decision(cid, clock)

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
            evidence = self.decision_history(history, clock)
            choice, receipt = self.choose_platform(platform, entry, ready, history, visible, observed, clock)
            self.compute_wall += perf_counter()-tick
            self.counts['platform'] += 1
            self.record(dict(kind='platform', stop=platform.stop, entry=entry, ready=ready,
                clock=clock, observed=observed, visible=visible, choice=choice,
                receipt=receipt, **evidence))
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
            evidence = self.decision_history(history, arrival)
            alight, receipt = self.choose_onboard(ride, cid, arrival, history)
            self.compute_wall += perf_counter()-tick
            self.counts['onboard'] += 1
            self.record(dict(kind='onboard', cid=cid, clock=arrival,
                choice='alight' if alight else 'continue', receipt=receipt, **evidence))
            if alight:
                return self.platform(platform, arrival, history, world)
        return self.deadline, 'exhausted_ride'
