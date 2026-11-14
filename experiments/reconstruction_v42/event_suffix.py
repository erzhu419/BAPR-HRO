"""Same-world suffix reuse at public events with future-live trip contexts.

This is contextual policy evaluation, not a new policy or a new posterior.
The full/live variants have identical checkpoints and menu views; only their
history and basis scope differs. Tables never survive a paired sample world.
"""

from time import perf_counter, process_time

from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v33.execute import DrawnWorld
from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v34.conditional_reference import ProjectionIncompatible
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v39.executor import SnapshotExecutor
from experiments.reconstruction_v39.state import DecisionEngine, raw_history_key
from experiments.reconstruction_v41.continuation import (
    empirical_pair_summary, evaluate as original_evaluate,
)


def new_stats():
    return dict(checks=0, hits=0, avoided_decisions=0, check_cpu=0.,
                key_trips=0, key_record_units=0)


class EventSuffixExecutor(SnapshotExecutor):
    """Reuse complete future execution after equal public event contexts."""

    def __init__(self, *args, suffixes, meeting_stats, mode='event_live', **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in ('event_full', 'event_live'):
            raise ValueError(mode)
        self.suffixes, self.meeting_stats, self.mode = suffixes, meeting_stats, mode
        self._ride_views, self._platform_views = {}, {}

    def ride_view(self, ride, cid):
        key = self.planner.structure(ride), cid
        if key not in self._ride_views:
            position = ride.path.index(cid)
            path = ride.path[position:]
            self._ride_views[key] = Ride(cid, path,
                {c: ride.exits[c] for c in path if c in ride.exits})
        return self._ride_views[key]

    def platform_view(self, platform, visible, observed):
        removed = set(observed)-set(visible)
        retained = tuple(cid for cid in platform.rides if cid not in removed)
        key = self.planner.structure(platform), retained
        if key not in self._platform_views:
            self._platform_views[key] = Platform(platform.stop,
                {cid: platform.rides[cid] for cid in retained})
        return self._platform_views[key], tuple(cid for cid in observed if cid in retained)

    def event_key(self, phase, view, history, clock, context):
        """A future-closed projection; never just current selected forecasts."""
        trips = self.planner.relevant_trips(view) if self.mode == 'event_live' else None
        public, count, units = raw_history_key(history, trips)
        # evaluate() conditions every relevant trip before cloning worlds.
        # Simulated receipts remain on that shared physical basis.
        bases = getattr(history, '_physical_bases', {})
        basis = tuple((trip, id(base)) for trip, base in sorted(bases.items())
                      if trips is None or trip in trips)
        self.meeting_stats['key_trips'] += count
        self.meeting_stats['key_record_units'] += units
        return (phase, self.planner.structure(view), float(clock), context,
                public, basis, history.message_clock, history.frame_index)

    def suffix(self, phase, view, history, clock, context, compute):
        began = process_time()
        key = self.event_key(phase, view, history, clock, context)
        saved = self.suffixes.get(key)
        self.meeting_stats['checks'] += 1
        self.meeting_stats['check_cpu'] += process_time()-began
        if saved is not None:
            value, trace, counts = saved
            self.trace.extend(trace)
            for kind, count in counts.items():
                self.counts[kind] += count
            self.meeting_stats['hits'] += 1
            self.meeting_stats['avoided_decisions'] += sum(counts.values())
            return value
        first, before = len(self.trace), dict(self.counts)
        value = compute()
        self.suffixes[key] = (value, tuple(self.trace[first:]),
            {kind: count-before[kind] for kind, count in self.counts.items()})
        return value

    def platform(self, platform, entry, history, world, *, after=float('-inf')):
        ready = entry+self.graph.stops[platform.stop].min_transfer_time
        departures = {cid: world.event(cid, 'departure') for cid in platform.rides}
        clocks = sorted({t for t, canceled, _ in departures.values()
                         if not canceled and ready <= t < self.deadline and t > after})
        return self._platform_from(platform, entry, ready, history, world, departures, clocks, 0)

    def _platform_from(self, platform, entry, ready, history, world, departures, clocks, index):
        if index == len(clocks):
            return self.deadline, 'no_legal_offer'
        clock = clocks[index]
        observed = {cid: t for cid, (t, canceled, _) in departures.items()
                    if not canceled and entry <= t <= clock}
        visible = tuple(cid for cid, (t, canceled, _) in departures.items()
                        if not canceled and t == clock)
        tick = perf_counter()
        history.observe_platform(tuple(platform.rides), entry, clock, observed)
        self.compute_wall += perf_counter()-tick
        view, seen = self.platform_view(platform, visible, observed)

        def continuation():
            tick = perf_counter()
            choice, _ = self.engine.platform(platform, entry, ready, history, visible, observed,
                clock, choose=self.choose, method=self.method)
            self.compute_wall += perf_counter()-tick
            self.counts['platform'] += 1
            self.record(dict(kind='platform', stop=platform.stop, clock=clock, choice=choice))
            if choice is not None:
                return self.ride(platform.rides[choice], clock, history, world)
            return self._platform_from(platform, entry, ready, history, world,
                                       departures, clocks, index+1)

        context = (float(entry), float(ready), visible, seen)
        return self.suffix('platform_event', view, history, clock, context, continuation)

    def ride(self, ride, ready, history, world):
        return self._ride_from(ride, 0, ready, None, history, world)

    def _ride_from(self, ride, index, previous, vehicle, history, world):
        for position in range(index, len(ride.path)):
            cid = ride.path[position]
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
            view = self.ride_view(ride, cid)

            def continuation():
                tick = perf_counter()
                alight, _ = self.engine.onboard(ride, cid, arrival, history, method=self.method)
                self.compute_wall += perf_counter()-tick
                self.counts['onboard'] += 1
                self.record(dict(kind='onboard', cid=cid, clock=arrival,
                                 choice='alight' if alight else 'continue'))
                if alight:
                    return self.platform(platform, arrival, history, world)
                return self._ride_from(ride, position+1, arrival, vehicle, history, world)

            return self.suffix('onboard_event', view, history, arrival, (cid, vehicle), continuation)
        return self.deadline, 'exhausted_ride'


def evaluate(graph, model, parameters, deadline, node, row, history, *,
             planner_type, count, seed, mode='event_live', world_type=DrawnWorld):
    """V41 samples, policy, accounting scope and scientific output unchanged."""
    if mode in ('full', 'entry'):
        result = original_evaluate(graph, model, parameters, deadline, node, row, history,
            planner_type=planner_type, count=count, seed=seed,
            meeting=mode == 'entry', world_type=world_type)
        result['mode'] = mode
        return result
    if mode not in ('event_full', 'event_live'):
        raise ValueError(mode)
    if count not in (16, 128):
        raise ValueError('V41 freezes 16 or 128 paired worlds')
    began, cpu = perf_counter(), process_time()
    snapshot = freeze_snapshot(history, row['clock'])
    for trip in sorted(node.trips(model)):
        snapshot.law(trip)
    conditioning = dict(wall=perf_counter()-began, process_cpu=process_time()-cpu)
    tick, tick_cpu = perf_counter(), process_time()
    bank = TripWorlds(snapshot.as_belief(), parameters, node.paths(), 128, seed)
    generation = dict(wall=perf_counter()-tick, process_cpu=process_time()-tick_cpu)
    tick, tick_cpu = perf_counter(), process_time()
    planner = planner_type(graph, model, parameters, deadline)
    if hasattr(snapshot, 'basis_context'):
        from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine
        engine = PhysicalDecisionEngine(planner, variant='frontier')
    else:
        engine = DecisionEngine(planner, variant='frontier')
    setup = dict(wall=perf_counter()-tick, process_cpu=process_time()-tick_cpu)
    rows = dict(alight=[], continue_=[])
    costs = {arm: dict(process_cpu=0., wall=0.) for arm in rows}
    stats = new_stats()
    for index in range(count):
        suffixes = {}
        for arm, action in (('alight', 'alight'), ('continue_', 'continue')):
            tick, tick_cpu = perf_counter(), process_time()
            executor = EventSuffixExecutor(graph, planner, deadline, engine=engine,
                method='rolling_fixed', waiting='combined', suffixes=suffixes,
                meeting_stats=stats, mode=mode)
            outcome = executor.forced(node, row, action, snapshot.clone(), world_type(bank, index))
            outcome['world_index'] = index
            rows[arm].append(outcome)
            costs[arm]['process_cpu'] += process_time()-tick_cpu
            costs[arm]['wall'] += perf_counter()-tick
    rows['continue'] = rows.pop('continue_')
    costs['continue'] = costs.pop('continue_')
    paired = empirical_pair_summary([r['loss'] for r in rows['alight']],
        [r['loss'] for r in rows['continue']], deadline-row['clock'])
    paired['orientation'] = 'alight loss minus continue loss'
    return dict(seed=seed, count=count, generated_count=128,
        public_history=history.public_state(), future_messages=0,
        rows=rows, paired=paired, meeting=True, mode=mode, meeting_stats=stats,
        cost=dict(conditioning=conditioning, generation=generation, planner_setup=setup,
            continuations=costs, total_process_cpu=process_time()-cpu,
            total_wall=perf_counter()-began, planner_stats=dict(planner.stats),
            decision_stats=dict(engine.stats)))
