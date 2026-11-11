"""One training-selected duration fit, matched to archived city budget64."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import freeze_replay_inputs, run_one, summarize

VERSION = 'v5_training_only_normalized_duration_grid_fit'
METHODS = ('trained_nominal', 'evolving_uct')


def execute(spec, data, forecasts, fit, clock, model):
    from src.evolving_trip_belief import EvolvingTripBelief
    if model == 'cdf20':
        from src.ordered_duration_belief import OrderedDurationBelief
        belief_type = OrderedDurationBelief
    else:
        belief_type = EvolvingTripBelief
    row = run_one(spec, data, forecasts, fit, 4, 64, 'gtfs_station_max', True,
                  'uct' if spec[2] == 'evolving_uct' else None, charge_computation=clock == 'charged',
                  belief_type=belief_type)
    row.update(implementation=VERSION, clock=clock, duration_model=model)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clock', choices=('logical', 'charged'), required=True)
    parser.add_argument('--model', choices=('grid60', 'cdf20'), required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    output = HERE / f'city_ordered_{args.model}_64_{args.clock}_results.json'
    baseline_path = HERE / f'city_budget64_{args.clock}_results.json'
    baseline = json.loads(baseline_path.read_text())
    nominal_path = HERE / f'city_frontier_{args.clock}_results.json'
    baseline_rows = ([r for r in baseline['rows'] if r['method'] == 'evolving_uct']
                     + [r for r in json.loads(nominal_path.read_text())['rows'] if r['method'] == 'trained_nominal'])
    fit_path = HERE / 'ordered_duration_training_fit.json'
    fit = json.loads(fit_path.read_text())['candidate_training' if args.model == 'grid60' else 'continuous_training']
    data, _ = load_city(ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
    data['holdout_dates'] = ['2023-07-26', '2023-07-27']
    forecasts = {'dates': {day: {'frames': []} for day in data['holdout_dates']}}
    settings = {'samples': 4, 'budget': 64, 'clock': args.clock, 'model': args.model, 'workers': args.workers,
                'dates': data['holdout_dates'], 'queries': [0, 1], 'methods': list(METHODS),
                'fit_selected_by': 'Training-only declared coarse or continuous likelihood; fixed grid recipe, no test tuning.',
                'baseline': [str(p.relative_to(ROOT)) for p in (baseline_path, nominal_path)]}
    model_sources = ['src/ordered_duration_belief.py'] if args.model == 'cdf20' else []
    protocol = freeze_replay_inputs(output, data=data, forecasts=forecasts, training=fit, settings=settings,
                                    sources=[str(Path(__file__)), str(HERE / 'runtime_fit_ordered.py'),
                                             'src/ordered_transition_fit.py', str(fit_path),
                                             str(baseline_path), str(nominal_path), *model_sources], implementation=VERSION)
    folder = output.with_name(output.stem + '_cases')
    folder.mkdir(exist_ok=True)
    rows, pending = [], []
    for day in data['holdout_dates']:
        for qi in (0, 1):
            for method in METHODS:
                spec = day, qi, method
                path = folder / f'{day}_{qi}_{method}.json'
                if path.exists():
                    row = json.loads(path.read_text())
                    if row['status'] != 'error':
                        rows.append(row)
                        continue
                pending.append((spec, path))
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(execute, spec, data, forecasts, fit, args.clock, args.model): (spec, path) for spec, path in pending}
        for job in as_completed(jobs):
            spec, path = jobs[job]
            try:
                row = job.result()
            except Exception as error:
                row = {'date': spec[0], 'query': spec[1], 'method': spec[2],
                       'status': 'error', 'error': repr(error)}
            path.write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
            print(*spec, row['status'], row.get('loss_seconds'), row.get('end_to_end_ms'), flush=True)
    lookup = {(r['date'], r['query'], r['method']): r for r in baseline_rows}
    comparison = []
    for row in rows:
        previous = lookup[row['date'], row['query'], row['method']]
        comparison.append({'date': row['date'], 'query': row['query'], 'method': row['method'],
                           'baseline_status': previous['status'], 'status': row['status'],
                           'baseline_loss': previous['loss_seconds'], 'loss': row.get('loss_seconds'),
                           'baseline_end_to_end_ms': previous['end_to_end_ms'],
                           'end_to_end_ms': row.get('end_to_end_ms')})
    good = [r for r in rows if r['status'] != 'error']
    output.write_text(json.dumps({'implementation': VERSION, 'settings': settings,
        'frozen_protocol': str(protocol.relative_to(ROOT)), 'training_fit': str(fit_path.relative_to(ROOT)),
        'run_wall_seconds': perf_counter() - started, 'summary': summarize(good, METHODS),
        'matching_note': 'Same dates, static OD, seeds, graph, 4 worlds and 64 transition budget; '
                         'archived baseline worker regime was also 2. Charged clocks reflect each run actual wall time.',
        'baseline_summary': summarize(baseline_rows, METHODS),
        'comparisons': sorted(comparison, key=lambda r: (r['date'], r['query'], r['method'])),
        'rows': sorted(rows, key=lambda r: (r['date'], r['query'], r['method']))}, indent=2) + '\n')


if __name__ == '__main__':
    main()
