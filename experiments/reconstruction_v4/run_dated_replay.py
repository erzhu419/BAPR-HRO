"""Held-out dated railway replay with prior-only and issued-forecast policies."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import copy
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from datetime import datetime, timezone

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.causal_planning_router import CausalPlanningRouter
from src.evolving_trip_belief import EvolvingTripBelief
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import JourneyScenario, RegimeSchedule, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph
from src.trip_belief import NominalEvent

METHODS = ('schedule_nominal', 'trained_nominal', 'evolving_rollout', 'forecast_rollout')
IMPLEMENTATION = 'dated_active_seconds_poll_history_v6'
DATA_PATH = ROOT / 'experiments/reconstruction_data_audit/dated_replay_v2.json.gz'
FORECAST_PATH = ROOT / 'experiments/reconstruction_data_audit/dated_public_forecasts.json.gz'


def freeze_replay_inputs(output, *, data, forecasts, training, settings, sources):
    code_paths = {str(ROOT / path) for path in (
        'src/evolving_trip_belief.py', 'src/committed_transit.py', 'src/causal_planning_router.py',
        'src/continuation_router.py', 'src/simulate_bandit.py', 'src/transit_graph.py', 'src/simulator.py',
        'experiments/reconstruction_v4/run_dated_replay.py')}
    code_paths.update(str(Path(source) if Path(source).is_absolute() else ROOT / source)
                      for source in sources if str(source).endswith('.py'))
    protocol = {'implementation': IMPLEMENTATION, 'data': data, 'forecasts': forecasts,
                'training': training, 'settings': settings, 'sources': sources,
                'source_text': {path: Path(path).read_text() for path in sorted(code_paths)},
                'libraries': {'numpy': np.__version__, 'scipy': scipy.__version__}}
    protocol_path = output.with_name(output.stem + '_protocol.json.gz')
    plain_path = output.with_name(output.stem + '_protocol.json')
    receipts = output.with_name(output.stem + '_cases')
    if protocol_path.exists():
        with gzip.open(protocol_path, 'rt') as handle:
            previous = json.load(handle)
        if previous != protocol:
            raise ValueError(f'Inputs or protocol changed. Use a new output: {protocol_path}')
        return protocol_path
    if plain_path.exists():
        previous = json.loads(plain_path.read_text())
        if previous != protocol:
            raise ValueError(f'Inputs or protocol changed. Use a new output: {plain_path}')
    elif output.exists() or any(receipts.glob('*.json')):
        raise ValueError(f'Existing replay receipts lack a frozen protocol. Use a new output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(protocol_path, 'wt') as handle:
        json.dump(protocol, handle, ensure_ascii=False)
    if plain_path.exists():
        with gzip.open(protocol_path, 'rt') as handle:
            if json.load(handle) != previous:
                raise ValueError('Compressed protocol changed its input values')
        plain_path.unlink()
    return protocol_path


def read_inputs(data_path=DATA_PATH, forecast_path=FORECAST_PATH):
    with gzip.open(data_path, 'rt') as handle:
        data = json.load(handle)
    with gzip.open(forecast_path, 'rt') as handle:
        forecasts = json.load(handle)
    with (DATA_PATH.parent / 'station_transfer_bounds.json').open() as handle:
        data['station_transfer_bounds'] = json.load(handle)['stations']
    return data, forecasts


def fit_training(data, retain_no_pickup=False):
    departures, statuses = defaultdict(list), defaultdict(Counter)
    travel, dwell, gap = [], [], []
    for day in data['train_dates']:
        trips = defaultdict(list)
        for c in data['dates'][day]['connections']:
            if not c['dropoff_allowed'] or (not c['pickup_allowed'] and not retain_no_pickup):
                continue
            trips[c['trip_id']].append(c)
            dep = c['departure']
            if dep['status'] in ('REAL', 'CANCELED'):
                statuses[c['route_id']][dep['status']] += 1
                statuses['pooled'][dep['status']] += 1
            if dep['status'] == 'REAL':
                shift = (dep['time_seconds'] - c['scheduled_departure_seconds']) / 60
                departures[c['route_id']].append(shift)
                departures['pooled'].append(shift)
            if c['outcome'] == 'MEASURED':
                travel.append(((c['arrival']['time_seconds'] - dep['time_seconds']) -
                               (c['scheduled_arrival_seconds'] - c['scheduled_departure_seconds'])) / 60)
        for rows in trips.values():
            rows.sort(key=lambda c: c['dep_sequence'])
            for before, after in zip(rows, rows[1:]):
                if before['arrival']['status'] == after['departure']['status'] == 'REAL':
                    residual = ((after['departure']['time_seconds'] - before['arrival']['time_seconds']) -
                                (after['scheduled_departure_seconds'] - before['scheduled_arrival_seconds'])) / 60
                    if before['arr_stop'] != after['dep_stop']:
                        gap.append(residual)
                    elif before['arr_sequence'] == after['dep_sequence']:
                        dwell.append(residual)

    priors = {}
    for route, shifts in departures.items():
        counts = statuses[route]
        priors[route] = {'mean': float(np.mean(shifts)), 'std': max(.5, float(np.std(shifts))),
                         'cancel_rate': counts['CANCELED'] / counts.total(),
                         'n_departures': len(shifts), 'status_counts': dict(counts)}
    increments = {f'{name}_{stat}': float(value) for name, values in [('travel', travel), ('dwell', dwell), ('gap', gap)]
                  for stat, value in [('drift', np.mean(values)), ('sd', max(1 / 60, np.std(values)))]}
    return {'train_dates': data['train_dates'], 'route_priors': priors, 'increments': increments,
            'travel_count': len(travel), 'dwell_count': len(dwell), 'gap_count': len(gap),
            'units': 'Route delay and running/dwell residual increments in minutes.',
            'floors_minutes': {'route_std': .5, 'increment_sd': 1 / 60},
            'forecast_sd_rule': 'Route training departure spread times 60 seconds, with 30-second floor. This bandwidth is not fitted to held-out forecast errors.'}


def dated_graph(data, day, transfer_rule='assumed120', retain_no_pickup=False):
    graph = TransitGraph()
    stop_ids = {key: i for i, key in enumerate(sorted(data['stops']))}
    for key, row in data['stops'].items():
        official = data.get('station_transfer_bounds', {}).get(key, {}).get('conservative_max_stated_seconds')
        seconds = official if transfer_rule == 'gtfs_station_max' and official is not None else 120
        graph.add_stop(Stop(stop_ids[key], row['name'], row['latitude'], row['longitude'], seconds))
    metadata, source_to_local, omitted, events = {}, {}, [], {}
    for row in data['dates'][day]['connections']:
        if not row['dropoff_allowed'] or (not row['pickup_allowed'] and not retain_no_pickup):
            omitted.append(row['id'])
            continue
        cid = len(graph.connections)
        source_to_local[row['id']] = cid
        metadata[cid] = row
        graph.add_connection(Connection(cid, row['route_id'], row['trip_id'], stop_ids[row['dep_stop']],
                                        stop_ids[row['arr_stop']], row['scheduled_departure_seconds'],
                                        row['scheduled_arrival_seconds'], pickup_allowed=row['pickup_allowed']))
        c = graph.connections[cid]
        if row['outcome'] == 'MEASURED':
            dep, arr = row['departure']['time_seconds'], row['arrival']['time_seconds']
            events[cid] = VehicleEvent(dep - c.dep_time, False, dep, arr)
        elif row['outcome'] == 'CANCELED':
            events[cid] = VehicleEvent(999, True, c.dep_time, c.arr_time)
    world = JourneyScenario(events, minute=60, publications={})
    return graph, world, stop_ids, source_to_local, metadata, omitted


class ScheduledClock:
    """Schedule forecasts revised only by public endpoint/attempt evidence."""
    minute = 60

    def __init__(self, graph):
        self.graph, self.known, self.canceled, self.excluded = graph, {}, set(), {}

    def copy(self):
        result = copy(self)
        result.known, result.canceled, result.excluded = self.known.copy(), self.canceled.copy(), self.excluded.copy()
        return result

    def observe_poll(self, stop, time, messages, seen_before):
        for msg in messages:
            cid, kind = msg.event_id.split(':', 1)
            self.observe_connection_event(int(cid), kind, msg.actual_delay)

    def observe_connection_event(self, cid, kind, delay):
        if kind == 'cancel':
            self.canceled.add(cid)
        else:
            c = self.graph.connections[cid]
            self.known[cid, kind] = (c.dep_time if kind == 'departure' else c.arr_time) + delay

    def observe_attempt(self, cid, kind, time):
        if kind == 'missed':
            self.excluded[cid] = (-np.inf, time)

    def observe_wait(self, cid, start, end):
        self.excluded[cid] = (start, end)

    def mean_events(self, parameters):
        result = {}
        for c in self.graph.connections:
            dep = self.known.get((c.id, 'departure'), c.dep_time)
            arr = self.known.get((c.id, 'arrival'), c.arr_time)
            lo, hi = self.excluded.get(c.id, (np.inf, -np.inf))
            result[c.id] = (NominalEvent(None, None, None, 1.0) if c.id in self.canceled or lo <= dep <= hi
                            else NominalEvent(dep - c.dep_time, dep, max(dep, arr), 0.0))
        return result


class ReplayRouter(CausalPlanningRouter):
    def __init__(self, *args, frames=(), source_to_local=None, forecast_sd=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames, self.frame_index = frames, 0
        self.source_to_local = source_to_local or {}
        self.forecast_sd = forecast_sd or {}
        self.replay_decisions = []
        self._trip_cids = defaultdict(list)
        for c in self.graph.connections:
            self._trip_cids[c.trip_id].append(c.id)

    def consume_forecasts(self, time):
        receipts, counts = [], Counter()
        while self.frame_index < len(self.frames) and self.frames[self.frame_index]['receipt_seconds'] <= time:
            frame = self.frames[self.frame_index]
            self.frame_index += 1
            receipts.append(frame['receipt_seconds'])
            for trip, status in frame['trip_statuses']:
                counts[f'trip_{status}'] += 1
                if status == 'CANCELED':
                    for cid in self._trip_cids.get(trip, ()):
                        for kind in ('departure', 'arrival'):
                            self.trip_belief.observe_forecast(cid, kind, None, 1.)
            for old_cid, kind, status, predicted in frame['endpoint_updates']:
                if old_cid not in self.source_to_local:
                    continue
                cid = self.source_to_local[old_cid]
                counts[status] += 1
                if status == 'FORECAST':
                    self.trip_belief.observe_forecast(cid, kind, predicted, self.forecast_sd[cid])
                elif status == 'SKIPPED':
                    self.trip_belief.observe_forecast(cid, kind, None, 1.)
        return {'receipts': receipts, 'updates': dict(counts)}

    def select_connection(self, stop, time, rng, top_k=5):
        full_started = perf_counter()
        public = self.consume_forecasts(time)
        forecast_ms = (perf_counter() - full_started) * 1000
        started = perf_counter()
        decision = super().select_connection(stop, time, rng, top_k)
        cid = None if decision is None else decision[0].connection_id
        self.replay_decisions.append({'clock_seconds': time, 'stop': stop,
                                      'current_trip': self.information_state.current_trip,
                                      'action': 'wait' if cid is None else cid,
                                      'forecast': public,
                                      'forecast_ms': forecast_ms,
                                      'planning_ms': (perf_counter() - started) * 1000,
                                      'decision_total_ms': (perf_counter() - full_started) * 1000,
                                      'model_transitions': self.predictions[-1]['work'] if self.algorithm != 'nominal' else 0,
                                      'complete_common_rounds': self.predictions[-1]['complete_common_rounds'] if self.algorithm != 'nominal' else None,
                                      'root_actions': self.predictions[-1]['root_actions'] if self.algorithm != 'nominal' else None,
                                      'outcome': 'wait' if cid is None else 'pending',
                                      'end_seconds': min(time + 60, self.journey_deadline) if cid is None else None})
        return decision

    def observe_attempt_outcome(self, cid, kind, started, time):
        super().observe_attempt_outcome(cid, kind, started, time)
        if self.replay_decisions and self.replay_decisions[-1]['action'] == cid:
            self.replay_decisions[-1].update(outcome=kind, end_seconds=time)


def run_one(spec, data, forecasts, fit, samples, budget, transfer_rule='assumed120',
            retain_no_pickup=False, algorithm_override=None):
    day, qi, method = spec
    end_to_end_started = perf_counter()
    graph, world, stop_ids, source_map, rows, omitted = dated_graph(data, day, transfer_rule, retain_no_pickup)
    graph_ms = (perf_counter() - end_to_end_started) * 1000
    setup_started = perf_counter()
    query = data['od_queries'][qi]
    cap = query['deadline_seconds'] - query['ready_seconds']
    pooled = fit['route_priors']['pooled']
    priors = {c.route: fit['route_priors'].get(c.route, pooled) for c in graph.connections}
    belief = (ScheduledClock(graph) if method == 'schedule_nominal' else
              EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments']))
    router = ReplayRouter(graph, priors, event_belief=belief, menu='full',
                          algorithm=algorithm_override or ('nominal' if method.endswith('nominal') else 'rollout'),
                          samples=samples, work_budget=budget, max_time=cap,
                          seed=int(day.replace('-', '')) + qi * 1000,
                          frames=forecasts['dates'][day]['frames'] if method == 'forecast_rollout' else (),
                          source_to_local=source_map,
                          forecast_sd={c.id: priors[c.route]['std'] * 60 for c in graph.connections})
    router.online_mean = router.online_cancel = False
    router._initial_cancels = {route: p['cancel_rate'] for route, p in priors.items()}
    setup_ms = (perf_counter() - setup_started) * 1000
    started = perf_counter()
    result = simulate_bandit_journey(graph, router, stop_ids[query['origin']], stop_ids[query['destination']],
                                      query['ready_seconds'], RegimeSchedule([(0, 'measured')]),
                                      np.random.default_rng(8000 + qi), max_time=cap, scenario=world)
    end = result.events[-1].details
    for decision in router.replay_decisions:
        if isinstance(decision['action'], int):
            decision['source_connection'] = rows[decision['action']]['id']
    return {'implementation': IMPLEMENTATION, 'samples': samples, 'transition_budget_per_decision': budget,
            'transfer_rule': transfer_rule, 'graph_build_ms': graph_ms, 'policy_setup_ms': setup_ms,
            'end_to_end_ms': (perf_counter() - end_to_end_started) * 1000,
            'worker_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'date': day, 'query': qi, 'method': method, 'od': query, 'status': end,
            'loss_seconds': None if end == 'censored' else result.arrival_time - query['ready_seconds'],
            'elapsed_observed_seconds': result.arrival_time - query['ready_seconds'],
            'compute_ms': (perf_counter() - started) * 1000, 'planning_transitions': router.rollout_steps,
            'decisions': router.replay_decisions, 'events': [asdict(e) for e in result.events],
            'public_actual_receipts': router.polls, 'forecast_frames_consumed': router.frame_index,
            'graph_connections': len(graph.connections), 'known_world_events': len(world.events),
            'excluded_pickup_dropoff_source_ids': omitted,
            'route_prior_pool_fallbacks': sorted({c.route for c in graph.connections if c.route not in fit['route_priors']})}


def summarize(rows, methods=METHODS):
    result = {}
    for method in methods:
        group = [r for r in rows if r.get('method') == method]
        known = [r['loss_seconds'] for r in group if r.get('loss_seconds') is not None]
        completed = [r for r in group if r['status'] != 'error']
        times = [r['compute_ms'] for r in completed]
        queries = [d['decision_total_ms'] for r in completed for d in r['decisions']]
        result[method] = {'n': len(group), 'statuses': dict(Counter(r.get('status', 'error') for r in group)),
                          'uncensored_mean_loss_seconds': float(np.mean(known)) if known else None,
                          'mean_compute_ms': float(np.mean(times)) if times else None,
                          'mean_end_to_end_ms': float(np.mean([r['end_to_end_ms'] for r in completed])) if completed else None,
                          'mean_graph_build_ms': float(np.mean([r['graph_build_ms'] for r in completed])) if completed else None,
                          'mean_policy_setup_ms': float(np.mean([r['policy_setup_ms'] for r in completed])) if completed else None,
                          'decision_p50_p95_p99_ms': np.quantile(queries, [.5, .95, .99]).tolist() if queries else None,
                          'largest_worker_peak_kib': max([r['worker_peak_kib'] for r in completed], default=None),
                          'mean_loss_lower_seconds': float(np.mean([r['elapsed_observed_seconds'] for r in completed])) if completed else None,
                          'mean_loss_upper_seconds': float(np.mean([
                              r['loss_seconds'] if r['loss_seconds'] is not None else
                              r['od']['deadline_seconds'] - r['od']['ready_seconds'] for r in completed])) if completed else None}
    return result


def paired_summary(rows, methods=METHODS):
    by_query = defaultdict(dict)
    for row in rows:
        if row['status'] != 'error':
            by_query[row['date'], row['query']][row['method']] = row
    result = {}
    for method in methods[1:]:
        references = ('schedule_nominal', 'trained_nominal', 'evolving_rollout') if method == 'forecast_rollout' else ('schedule_nominal', 'trained_nominal')
        for reference in references:
            if method == reference:
                continue
            common = [group for group in by_query.values() if method in group and reference in group]
            known = [group[method]['loss_seconds'] - group[reference]['loss_seconds'] for group in common
                     if group[method]['loss_seconds'] is not None and group[reference]['loss_seconds'] is not None]
            lower, upper = [], []
            for group in common:
                action, baseline = group[method], group[reference]
                def high(row):
                    return (row['loss_seconds'] if row['loss_seconds'] is not None else
                            row['od']['deadline_seconds'] - row['od']['ready_seconds'])
                lower.append(action['elapsed_observed_seconds'] - high(baseline))
                upper.append(high(action) - baseline['elapsed_observed_seconds'])
            result[f'{method}_minus_{reference}'] = {'n_common': len(common), 'n_jointly_uncensored': len(known),
                                                     'jointly_uncensored_mean_seconds': float(np.mean(known)) if known else None,
                                                     'all_common_mean_lower_seconds': float(np.mean(lower)) if lower else None,
                                                     'all_common_mean_upper_seconds': float(np.mean(upper)) if upper else None}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--data', type=Path, default=DATA_PATH)
    parser.add_argument('--forecasts', type=Path, default=FORECAST_PATH)
    parser.add_argument('--transfer-rule', choices=('assumed120', 'gtfs_station_max'), default='gtfs_station_max')
    parser.add_argument('--output', type=Path, default=HERE / 'dated_replay_final_results.json')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.data, args.forecasts = args.data.resolve(), args.forecasts.resolve()
    started_at = datetime.now(timezone.utc).isoformat()
    input_started = perf_counter()
    data, forecasts = read_inputs(args.data, args.forecasts)
    input_ms = (perf_counter() - input_started) * 1000
    fit_started = perf_counter()
    fit = fit_training(data)
    fit_ms = (perf_counter() - fit_started) * 1000
    protocol_path = freeze_replay_inputs(
        args.output, data=data, forecasts=forecasts, training=fit,
        settings={'methods': args.methods, 'samples': args.samples, 'budget': args.budget,
                  'transfer_rule': args.transfer_rule, 'workers': args.workers},
        sources=[str(args.data), str(args.forecasts),
                 'experiments/reconstruction_data_audit/station_transfer_bounds.json',
                 str(Path(__file__).relative_to(ROOT)), 'src/evolving_trip_belief.py',
                 'src/committed_transit.py', 'src/causal_planning_router.py', 'src/simulate_bandit.py'])
    folder = args.output.with_name(args.output.stem + '_cases')
    folder.mkdir(parents=True, exist_ok=True)
    specs = [(day, qi, method) for day in data['holdout_dates'] for qi in range(len(data['od_queries']))
             for method in args.methods]
    rows, todo = [], []
    for spec in specs:
        path = folder / f'{spec[0]}_q{spec[1]:02}_{spec[2]}.json'
        if path.exists():
            row = json.loads(path.read_text())
            if row.get('status') != 'error':
                if (row.get('implementation') != IMPLEMENTATION or row.get('samples') != args.samples
                        or row.get('transition_budget_per_decision') != args.budget
                        or row.get('transfer_rule') != args.transfer_rule):
                    raise ValueError(f'Use a separate output for this changed protocol: {path}')
                rows.append(row)
                continue
        todo.append((spec, path))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_one, spec, data, forecasts, fit, args.samples, args.budget, args.transfer_rule): (spec, path)
                for spec, path in todo}
        for job in as_completed(jobs):
            spec, path = jobs[job]
            try:
                row = job.result()
            except Exception as error:
                row = {'date': spec[0], 'query': spec[1], 'method': spec[2], 'status': 'error',
                       'error': f'{type(error).__name__}: {error}'}
            path.write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
            print(*spec, row['status'], row.get('loss_seconds'), flush=True)
    modeled_coverage = {}
    transfer_values = {}
    for key in data['stops']:
        official = data['station_transfer_bounds'].get(key, {}).get('conservative_max_stated_seconds')
        transfer_values[key] = official if args.transfer_rule == 'gtfs_station_max' and official is not None else 120
    for day in data['dates']:
        kept = [c for c in data['dates'][day]['connections'] if c['pickup_allowed'] and c['dropoff_allowed']]
        modeled_coverage[day] = {'connections': len(kept),
                                 'outcomes': dict(Counter(c['outcome'] for c in kept)),
                                 'excluded_pickup_dropoff': [c['id'] for c in data['dates'][day]['connections']
                                                            if not c['pickup_allowed'] or not c['dropoff_allowed']]}
    checkpoint_times = [path.stat().st_mtime for path in folder.glob('*.json')]
    output = {'implementation': IMPLEMENTATION, 'summary_generated_at_utc': started_at,
              'source_data': str(args.data), 'source_forecasts': str(args.forecasts),
              'frozen_protocol': str(protocol_path.relative_to(ROOT)),
              'checkpoint_window_utc': [datetime.fromtimestamp(t, timezone.utc).isoformat()
                                        for t in (min(checkpoint_times), max(checkpoint_times))],
              'scope': 'Three held-out dates with eight OD queries each on dated active SBB adjacency among 16 selected stations. Unknown measured outcomes remain candidates and censor attempted use.',
              'clock_unit': 'seconds', 'train_dates': data['train_dates'], 'holdout_dates': data['holdout_dates'],
              'samples': args.samples, 'transition_budget_per_decision': args.budget,
              'transfer_rule': args.transfer_rule, 'input_loading_ms': input_ms, 'offline_fit_ms': fit_ms,
              'transfer_seconds_by_stop': transfer_values,
              'stations_without_stated_bound_using_assumed120': [key for key in data['stops']
                  if data['station_transfer_bounds'].get(key, {}).get('conservative_max_stated_seconds') is None],
              'timing_scope': 'end_to_end_ms includes dated graph/world construction, belief/router setup and complete execution/planning. Offline fit and input load are reported once. Worker peak is process-lifetime ru_maxrss, not a per-journey memory peak.',
              'method_definitions': {
                  'schedule_nominal': 'Deterministic scheduled unknown clocks, revised by actual attempted/boarded endpoint evidence.',
                  'trained_nominal': 'Mean clocks of the trained evolving event model under observed endpoint and waiting evidence.',
                  'evolving_rollout': f'Same trained evolving model and nominal tail, with {args.samples} draws and at most {args.budget} model transitions per decision.',
                  'forecast_rollout': 'Matched evolving rollout with causally received numeric forecasts as additional soft evidence.'},
              'route_fit_policy': 'Route mean, spread and cancellation priors remain fixed from training dates. Event clocks and segment availability condition on the evolving public history.',
              'public_actual_information': 'No synthetic occurrence publications. Only attempt and actually boarded endpoint feedback.',
              'forecast_information': 'Sampled GTFS-RT frames consumed only at receipt<=decision clock. Forecasts are noisy emissions. SKIPPED/cancellation notices clear forecasts and never set actual cancellation truth.',
              'forecast_tail': 'Each planning query freezes its received forecast set. Later issued forecast messages are consumed at the next actual decision, without simulated future forecast issuance.',
              'training': fit, 'coverage_by_date': {d: data['dates'][d]['coverage'] for d in data['dates']},
              'modeled_coverage_by_date': modeled_coverage,
              'summary': summarize(rows), 'paired': paired_summary(rows),
              'rows': sorted(rows, key=lambda r: (r['date'], r['query'], r['method']))}
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
