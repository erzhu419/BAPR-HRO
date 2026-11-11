"""Matched small railway mechanism panel for one shared-day joint belief."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from calibration_run import data_inputs, load
from experiments.reconstruction_v4.run_dated_replay import freeze_replay_inputs, run_one, summarize
from src.evolving_trip_belief import EvolvingTripBelief
from src.joint_day_belief import JointDayBelief, TRAINING_DAY_SCALE_SECONDS

METHODS = ('independent_nominal', 'joint_nominal', 'independent_rollout', 'joint_rollout')
DATES = ('2023-10-12', '2023-10-25', '2023-10-27')


def inputs():
    data = data_inputs()
    fit = json.loads((HERE / 'calibration_frozen_fit.json').read_text())['evolving_fit']
    data['train_dates'] = fit['train_dates']
    data['holdout_dates'] = list(DATES)
    frames = load(HERE / 'calibration_public_frames.json.gz')
    return data, frames, fit


def run_spec(spec, data, frames, fit, charge_computation=False):
    day, qi, method = spec
    joint = method.startswith('joint_')
    algorithm = 'nominal' if method.endswith('nominal') else 'rollout'
    row = run_one((day, qi, 'trained_nominal' if algorithm == 'nominal' else 'evolving_rollout'),
                  data, frames, fit, 4, 512, 'gtfs_station_max',
                  algorithm_override=algorithm, belief_type=JointDayBelief if joint else EvolvingTripBelief,
                  charge_computation=charge_computation)
    row.update(method=method, cohort='previously_observed_development', shared_day=joint,
               delta_grid_seconds=list(range(-360, 361, 60)) if joint else None,
               day_scale_seconds=TRAINING_DAY_SCALE_SECONDS if joint else None,
               joint_day_implementation='v2_raw_operating_mass')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, default=HERE / 'joint_day_v2_results.json')
    parser.add_argument('--charge-computation', action='store_true')
    args = parser.parse_args()
    data, frames, fit = inputs()
    output = args.output.resolve()
    methods = ('independent_rollout', 'joint_rollout') if args.charge_computation else METHODS
    freeze_replay_inputs(output, data=data, forecasts=frames, training=fit,
        settings={'methods': methods, 'dates': DATES, 'queries': [0, 1], 'samples': 4, 'budget': 512,
                  'workers': args.workers, 'transfer_rule': 'gtfs_station_max',
                  'scope': 'Shared joint-model mechanism on already examined dates, not independent confirmation.',
                  'delta_grid_seconds': list(range(-360, 361, 60)), 'day_scale_seconds': TRAINING_DAY_SCALE_SECONDS,
                  'logic_time': not args.charge_computation, 'charge_computation': args.charge_computation,
                  'public_numeric_forecasts': False},
        sources=['src/joint_day_belief.py', str(Path(__file__).relative_to(ROOT))],
        implementation='joint_day_finite_mixture_v2_raw_operating_mass')
    folder = output.with_name(output.stem + '_cases')
    folder.mkdir(exist_ok=True)
    rows, todo = [], []
    for day in DATES:
        for qi in (0, 1):
            for method in methods:
                spec = day, qi, method
                path = folder / f'{day}_q{qi:02}_{method}.json'
                if path.exists():
                    row = json.loads(path.read_text())
                    if row['status'] != 'error':
                        rows.append(row)
                        continue
                todo.append((spec, path))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_spec, spec, data, frames, fit, args.charge_computation): (spec, path) for spec, path in todo}
        for job in as_completed(jobs):
            spec, path = jobs[job]
            try:
                row = job.result()
            except Exception as error:
                row = {'date': spec[0], 'query': spec[1], 'method': spec[2], 'status': 'error',
                       'error': f'{type(error).__name__}: {error}'}
            path.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
            rows.append(row)
            output.write_text(json.dumps({'scope': 'Previously observed development mechanism panel.',
                'training_dates': fit['train_dates'], 'charge_computation': args.charge_computation,
                'rows': rows, 'summary': summarize(rows, methods)}, indent=2) + '\n')
            print(*spec, row['status'], row.get('loss_seconds'), round(row.get('end_to_end_ms', 0) / 1000, 2), flush=True)


if __name__ == '__main__':
    main()
