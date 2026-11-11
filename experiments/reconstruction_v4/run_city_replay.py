"""Complete declared city-candidate replay with explicit prior provenance."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.reconstruction_v4.run_dated_replay import (
    ROOT, HERE, IMPLEMENTATION, fit_training, run_one, summarize, paired_summary, freeze_replay_inputs)

METHODS = ('schedule_nominal', 'trained_nominal', 'evolving_rollout', 'evolving_uct')
OD_NAMES = (('Zürich, Paradeplatz', 'Zürich, Goldbrunnenplatz'),
            ('Zürich, Heuried', 'Zürich, Sternen Oerlikon'))


def load_city(path):
    with gzip.open(path, 'rt') as handle:
        original = json.load(handle)
    if 'dates' in original:
        dates = original['dates']
        train, holdout = original['train_dates'], original['holdout_dates']
    else:
        dates = {original['date']: original}
        train, holdout = [], [original['date']]
    stops = {key: value for day in dates.values() for key, value in day['stops'].items()}
    by_name = {value['name']: key for key, value in sorted(stops.items())}
    queries = [{'origin': by_name[a], 'destination': by_name[b], 'origin_name': a, 'destination_name': b,
                'ready_seconds': 8 * 3600 + 5 * 60, 'deadline_seconds': 9 * 3600 + 5 * 60}
               for a, b in OD_NAMES]
    with (ROOT / 'experiments/reconstruction_data_audit/station_transfer_bounds.json').open() as handle:
        transfer = json.load(handle)['stations']
    return {'stops': stops, 'dates': dates, 'train_dates': train, 'holdout_dates': holdout,
            'od_queries': queries, 'station_transfer_bounds': transfer}, original


def past_forecast_fit(day):
    with (ROOT / 'data/route_priors_temporal_corrected.pkl').open('rb') as handle:
        prior = pickle.load(handle)[day]
    rows = list(prior['priors'].values())
    n = sum(r['n_obs'] for r in rows)
    mean = sum(r['mean'] * r['n_obs'] for r in rows) / n
    variance = sum((r['std'] ** 2 + r['mean'] ** 2) * r['n_obs'] for r in rows) / n - mean ** 2
    statuses = sum(r['n_status_trips'] for r in rows)
    pooled = {'mean': mean, 'std': max(.25, variance) ** .5,
              'cancel_rate': sum(r['n_canceled'] for r in rows) / statuses}
    return {'train_dates': prior['train_dates'], 'route_priors': dict(prior['priors'], pooled=pooled),
            'increments': {'travel_sd': 2., 'dwell_sd': 1., 'gap_sd': 2.,
                           'travel_drift': 0., 'dwell_drift': 0., 'gap_drift': 0.},
            'route_fit_source': 'Strictly earlier archived GTFS-RT forecasts and explicit statuses; missing routes use pooled earlier-date fit.',
            'increment_source': 'Declared fallback increments, in minutes. Not fitted on target-day actuals.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
    parser.add_argument('--output', type=Path, default=HERE / 'city_replay_results.json')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--queries', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--dates', nargs='+')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--budget', type=int, default=512)
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    data, original = load_city(args.input)
    if args.dates:
        if not set(args.dates) <= set(data['holdout_dates']):
            parser.error('Selected dates must be declared holdouts')
        data['holdout_dates'] = args.dates
    before = perf_counter()
    fits = ({day: fit_training(data, retain_no_pickup=True) for day in data['holdout_dates']}
            if data['train_dates'] else {day: past_forecast_fit(day) for day in data['holdout_dates']})
    fit_ms = (perf_counter() - before) * 1000
    forecasts = {'dates': {day: {'frames': []} for day in data['holdout_dates']}}
    protocol_path = freeze_replay_inputs(
        args.output, data=data, forecasts=forecasts, training=fits,
        settings={'methods': args.methods, 'queries': args.queries, 'samples': args.samples,
                  'budget': args.budget, 'workers': args.workers, 'transfer_rule': 'gtfs_station_max',
                  'retain_no_pickup': True},
        sources=[str(args.input.relative_to(ROOT)), 'data/route_priors_temporal_corrected.pkl',
                 'experiments/reconstruction_data_audit/station_transfer_bounds.json',
                 str(Path(__file__).relative_to(ROOT)), 'experiments/reconstruction_v4/run_dated_replay.py',
                 'src/evolving_trip_belief.py', 'src/causal_planning_router.py', 'src/committed_transit.py'])
    folder = args.output.with_name(args.output.stem + '_cases')
    folder.mkdir(parents=True, exist_ok=True)
    specs = [(day, qi, method) for day in data['holdout_dates'] for qi in args.queries for method in args.methods]
    completed, pending = [], []
    for spec in specs:
        path = folder / f'{spec[0]}_q{spec[1]:02}_{spec[2]}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['status'] != 'error':
                if (saved.get('implementation') != IMPLEMENTATION or saved.get('samples') != args.samples
                        or saved.get('transition_budget_per_decision') != args.budget
                        or saved.get('transfer_rule') != 'gtfs_station_max'):
                    raise ValueError(f'Use a new output for changed protocol: {path}')
                completed.append(saved)
                continue
        pending.append((spec, path))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_one, spec, data, forecasts, fits[spec[0]], args.samples, args.budget,
                            'gtfs_station_max', True, 'uct' if spec[2] == 'evolving_uct' else None): (spec, path)
                for spec, path in pending}
        for future in as_completed(jobs):
            spec, path = jobs[future]
            try:
                row = future.result()
            except Exception as error:
                row = {'date': spec[0], 'query': spec[1], 'method': spec[2], 'status': 'error',
                       'error': f'{type(error).__name__}: {error}'}
            path.write_text(json.dumps(row, indent=2) + '\n')
            completed.append(row)
            print(*spec, row['status'], row.get('loss_seconds'), flush=True)
    coverage = {day: {'source_connections': len(record['connections']), 'coverage': record['coverage'],
                       'no_pickup_connections_retained': sum(not c['pickup_allowed'] for c in record['connections']),
                       'no_dropoff_connections_omitted': sum(not c['dropoff_allowed'] for c in record['connections'])}
                for day, record in data['dates'].items()}
    output = {'implementation': IMPLEMENTATION, 'summary_generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'frozen_protocol': str(protocol_path.relative_to(ROOT)),
              'input': str(args.input.relative_to(ROOT)), 'scope': original.get('scope', 'Date-specific complete declared city candidates.'),
              'source_metadata': {key: value for key, value in original.items() if key != 'dates'},
              'query_selection': 'Fixed stop-name pairs selected from static schedule identities, with common08:05 departure and60-minute cap.',
              'queries': data['od_queries'], 'train_dates': data['train_dates'], 'holdout_dates': data['holdout_dates'],
              'fits': fits, 'fit_ms': fit_ms, 'coverage_by_date': coverage,
              'transfer_rule': 'Maximum stated station-interior GTFS time; missing bounds use assumed120seconds. No-pickup segments remain available to same-trip passengers.',
              'no_public_actual_feed': True, 'samples': args.samples, 'budget': args.budget,
              'timing_scope': 'Per-row end-to-end cost includes graph/world+belief/router setup+execution. Worker memory is lifetime peak. Fit is reported separately.',
              'summary': summarize(completed, METHODS), 'paired': paired_summary(completed, METHODS),
              'rows': sorted(completed, key=lambda row: (row['date'], row['query'], row['method']))}
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
