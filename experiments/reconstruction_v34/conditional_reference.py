"""Paired current-snapshot action checks; no future publications are generated.

The physical trip law and its grid are unchanged. Public off-grid receipts
are anchored at their observed times during execution; an incompatible future
draw is reported as unresolved rather than shifted, resampled or failed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v32.journey import TripWorlds, paired_statistics
from experiments.reconstruction_v33.execute import DrawnWorld, Executor
from experiments.reconstruction_v33.nodes import Ride, root_ride
from experiments.reconstruction_v33.run import jsonable
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.forecast import ForecastPlanner, MessageHistory
from experiments.reconstruction_v34.run import setup


HERE = Path(__file__).resolve().parent


def restore_snapshot(model, parameters, public_state, clock):
    """Restore the entire public state, without attaching an archive stream."""
    history = MessageHistory(model, parameters)
    history.message_clock = float(clock)
    for row in public_state:
        mode, trip = row['evidence'], row['trip']
        if mode in ('observed', 'unseen'):
            value = (float(row['clock']) if mode == 'observed'
                     else tuple(tuple(map(float, w)) for w in row['windows']))
            record = (int(row['cid']), row['kind'], mode, value)
            history._records[trip] = (*history._records.get(trip, ()), record)
        elif mode == 'forecast':
            history._forecasts[int(row['cid']), row['kind']] = (
                float(row['predicted_time']), float(row['sd_seconds']), float(row['receipt_seconds']))
        elif mode == 'trip_status':
            history._trip_statuses[trip] = row['status'], float(row['receipt_seconds'])
        elif mode in ('endpoint_status', 'endpoint_service_notice'):
            target = history._endpoint_statuses if mode == 'endpoint_status' else history._service_statuses
            target[int(row['cid']), row['kind']] = row['status'], float(row['receipt_seconds'])
        else:
            raise ValueError(mode)
    return history


def locate_node(ride, trace, index):
    """Follow public executed choices, not a global CID lookup or future path."""
    platform = None
    for i, row in enumerate(trace[:index+1]):
        if row['kind'] == 'onboard':
            if row['cid'] not in ride.exits:
                raise ValueError('Trace arrival is outside the current retained ride')
            if i == index:
                return ride
            if row['choice'] == 'alight':
                platform = ride.exits[row['cid']]
        elif row['kind'] == 'platform':
            if platform is None or platform.stop != row['stop']:
                raise ValueError('Trace platform does not follow the previous alighting action')
            if i == index:
                return platform
            if row['choice'] is not None:
                ride, platform = platform.rides[row['choice']], None
        else:
            raise ValueError('A reference cannot start after arrival at the destination')
    raise IndexError(index)


class ProjectionIncompatible(ValueError):
    pass


class AnchoredWorld(DrawnWorld):
    """Known receipts stay known; only unrealized events come from the draw."""
    def __init__(self, bank, index, public_state):
        super().__init__(bank, index)
        self.known = {(int(r['cid']), r['kind']): float(r['clock'])
                      for r in public_state if r['evidence'] == 'observed'}

    def event(self, cid, kind):
        key = cid, kind
        if key in self.known:
            return self.known[key], False, self.bank.belief._positions[key][0]
        return super().event(cid, kind)


class SnapshotExecutor(Executor):
    """Common continuation with a fixed received snapshot and fresh local facts."""
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
            reference = self.planner.platform_reference(platform, history, entry, ready, self.method)
            choice, receipt = self.choose(reference, visible, tuple(observed), clock)
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
            compiled = self.planner.compile_ride(ride, history, self.method)
            alight, receipt = compiled.decision(cid, arrival)
            self.compute_wall += perf_counter()-tick
            self.counts['onboard'] += 1
            self.record(dict(kind='onboard', cid=cid, clock=arrival,
                             choice='alight' if alight else 'continue'))
            if alight:
                return self.platform(platform, arrival, history, world)
        return self.deadline, 'exhausted_ride'

    def forced(self, node, row, action, history, world):
        clock = row['clock']
        began, cpu = perf_counter(), process_time()
        try:
            if row['kind'] == 'platform':
                if action is None:
                    arrival, status = self.platform(node, row['entry'], history, world, after=clock)
                else:
                    if action not in row['visible']:
                        raise ValueError('Forced boarding is not a currently public legal offer')
                    arrival, status = self.ride(node.rides[action], clock, history, world)
            elif action == 'alight':
                arrival, status = self.platform(node.exits[row['cid']], clock, history, world)
            elif action == 'continue':
                path = node.path[node.path.index(row['cid'])+1:]
                if path:
                    suffix = Ride(path[0], path, {cid: child for cid, child in node.exits.items() if cid in path})
                    arrival, status = self.ride(suffix, clock, history, world)
                else:
                    arrival, status = self.deadline, 'exhausted_ride'
            else:
                raise ValueError(action)
        except ProjectionIncompatible as error:
            arrival, status = None, 'projection_incompatible:'+str(error)
        except UnsupportedHistory as error:
            arrival, status = None, 'unsupported_public_history:'+str(error)
        return dict(loss=None if arrival is None else arrival-clock, status=status,
                    trace=self.trace, decisions=self.counts, compute_wall=self.compute_wall,
                    total_wall=perf_counter()-began, total_process_cpu=process_time()-cpu)


def summarize_pairs(subject, baseline, remaining):
    """The unresolved full-bank interval retains every sampled world's mass."""
    good = [i for i, (a, b) in enumerate(zip(subject, baseline)) if a is not None and b is not None]
    low, high = [], []
    for a, b in zip(subject, baseline):
        low.append((0. if a is None else a)-(remaining if b is None else b))
        high.append((remaining if a is None else a)-(0. if b is None else b))
    return dict(orientation='subject action loss minus baseline action loss; positive favors baseline',
        worlds=len(subject), complete_pairs=len(good), unresolved_pairs=len(subject)-len(good),
        complete_pair_mean_subject_loss=float(np.mean([subject[i] for i in good])) if good else None,
        complete_pair_mean_baseline_loss=float(np.mean([baseline[i] for i in good])) if good else None,
        complete_pair_statistics=paired_statistics([subject[i] for i in good], [baseline[i] for i in good]) if good else None,
        full_bank_mean_gain_interval=[float(np.mean(low)), float(np.mean(high))],
        interval_scope='Empirical finite-bank missing-outcome bounds; not a population confidence interval.')


def run_case(case_index):
    began, cpu = perf_counter(), process_time()
    protocol, data, fit, selection, catalog = (read(HERE/name) for name in (
        'protocol.json', 'data.json.gz', 'model_fit.json', 'selection.json', 'divergence_catalog.json'))
    selected = catalog['reference_states'][case_index]
    root = selection['selected'][selected['root_index']]
    archive = read(HERE/selected['subject_source']['file'])
    trace = archive['rows']['public_messages'][selected['method']]['outcome']['trace']
    row = trace[selected['first']['trace_index']]
    graph, _, _, _, _, _ = context(data, root['day'])
    model, parameters = setup(graph, fit)
    ride = root_ride(graph, root)
    node = locate_node(ride, trace, selected['first']['trace_index'])
    history = restore_snapshot(model, parameters, row['public_history'], row['clock'])
    if json.dumps(history.public_state(), sort_keys=True) != json.dumps(row['public_history'], sort_keys=True):
        raise AssertionError('Snapshot reconstruction changed a public fact')
    seed, count = protocol['reference_seed']+case_index, protocol['reference_worlds']
    bank = TripWorlds(history.as_belief(), parameters, ride.paths(), count, seed)
    preparation = dict(wall=perf_counter()-began, process_cpu=process_time()-cpu)
    actions = {name: selected['first'][name+'_action'][-1] for name in ('subject', 'baseline')}
    results, planner_costs = {}, {}
    for label, action in actions.items():
        tick, tick_cpu = perf_counter(), process_time()
        planner = ForecastPlanner(graph, model, parameters, root['deadline'])
        rows = []
        for index in range(count):
            executor = SnapshotExecutor(graph, planner, root['deadline'], method='rolling_fixed', waiting='combined')
            result = executor.forced(node, row, action, history.clone(), AnchoredWorld(bank, index, row['public_history']))
            result['world_index'] = index
            rows.append(result)
        results[label] = rows
        planner_costs[label] = dict(wall=perf_counter()-tick, process_cpu=process_time()-tick_cpu,
                                   planner_stats=dict(planner.stats))
        print('reference', case_index, root['root_id'], label, action,
              {status: sum(r['status'].split(':')[0] == status for r in rows)
               for status in sorted({r['status'].split(':')[0] for r in rows})}, flush=True)
    result = dict(case_index=case_index, selected=selected, root_id=root['root_id'],
        clock=row['clock'], deadline=root['deadline'], current_actions=actions,
        public_history=row['public_history'], continuation='combined_rolling; frozen received messages; new local observations',
        seed=seed, count=count, sampling='Independent original conditional physical trip chains; each trip once per world',
        projection='Off-grid public receipts are anchored. Incompatible future grid draws remain unresolved with their original mass.',
        future_messages=0, rows=results,
        paired=summarize_pairs([r['loss'] for r in results['subject']],
            [r['loss'] for r in results['baseline']], root['deadline']-row['clock']),
        cost=dict(preparation=preparation, action_continuations=planner_costs,
                  total_wall=perf_counter()-began, total_process_cpu=process_time()-cpu))
    target = HERE/'conditional_reference'/f'{case_index:02}.json.gz'
    target.parent.mkdir(exist_ok=True)
    with gzip.open(target, 'wt') as handle:
        json.dump(result, handle, default=jsonable)
    return case_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    catalog = read(HERE/'divergence_catalog.json')
    pending = [i for i in range(len(catalog['reference_states']))
               if (args.case_index is None or i == args.case_index)
               and not (HERE/'conditional_reference'/f'{i:02}.json.gz').exists()]
    if args.workers == 1:
        for index in pending:
            run_case(index)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_case, pending))
    completed = [read(HERE/'conditional_reference'/f'{i:02}.json.gz')
                 for i in range(len(catalog['reference_states']))
                 if (HERE/'conditional_reference'/f'{i:02}.json.gz').exists()]
    summary = dict(selected=len(catalog['reference_states']), completed=len(completed),
        scope='Current-snapshot Monte Carlo action checks with common combined_rolling continuation; not an exact future-message optimum.',
        cases=[{key: row[key] for key in ('case_index','root_id','clock','current_actions','seed','count','paired','cost')}
               for row in completed])
    (HERE/'conditional_reference.json').write_text(json.dumps(summary, indent=2, default=jsonable)+'\n')


if __name__ == '__main__':
    main()
