"""Charge measured planning time for shared, matching-marginal and cheap priors."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from joint_day_run import DATES, inputs
from experiments.reconstruction_v4.run_dated_replay import freeze_replay_inputs, run_one, summarize
from src.evolving_trip_belief import EvolvingTripBelief
from src.independent_day_mixture_belief import IndependentDayMixtureBelief
from src.joint_day_belief import JointDayBelief, TRAINING_DAY_SCALE_SECONDS

TYPES = {'independent_rollout': EvolvingTripBelief, 'trip_mixture_rollout': IndependentDayMixtureBelief,
         'joint_rollout': JointDayBelief}


def run_charged(spec, data, frames, fit):
    day, qi, method = spec
    row = run_one((day, qi, 'evolving_rollout'), data, frames, fit, 4, 512, 'gtfs_station_max',
                  algorithm_override='rollout', belief_type=TYPES[method], charge_computation=True)
    row.update(method=method, cohort='previously_observed_development',
               joint_day_implementation='v2_raw_operating_mass', charge_computation=True)
    return row


def main():
    data, frames, fit = inputs()
    output = HERE / 'joint_day_coupling_charged_results.json'
    freeze_replay_inputs(output, data=data, forecasts=frames, training=fit,
        settings={'methods': list(TYPES), 'dates': DATES, 'queries': [0, 1], 'samples': 4, 'budget': 512,
                  'workers': 2, 'transfer_rule': 'gtfs_station_max', 'charge_computation': True,
                  'delta_grid_seconds': list(range(-360, 361, 60)), 'day_scale_seconds': TRAINING_DAY_SCALE_SECONDS,
                  'scope': 'Paired measured-computation validation of the frozen development coupling contrast.'},
        sources=['src/joint_day_belief.py', 'src/independent_day_mixture_belief.py',
                 str(Path(__file__).relative_to(ROOT)), 'experiments/reconstruction_v5/joint_day_run.py'],
        implementation='joint_day_v2_charged_coupling_comparison')
    folder = HERE / 'joint_day_coupling_charged_cases'
    folder.mkdir(exist_ok=True)
    rows, todo = [], []
    for day in DATES:
        for qi in (0, 1):
            for method in TYPES:
                spec = day, qi, method
                path = folder / f'{day}_q{qi:02}_{method}.json'
                if path.exists():
                    row = json.loads(path.read_text())
                    if row['status'] != 'error':
                        rows.append(row)
                        continue
                todo.append((spec, path))
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(run_charged, spec, data, frames, fit): (spec, path) for spec, path in todo}
        for job in as_completed(jobs):
            spec, path = jobs[job]
            try:
                row = job.result()
            except Exception as error:
                row = {'date': spec[0], 'query': spec[1], 'method': spec[2], 'status': 'error',
                       'error': f'{type(error).__name__}: {error}'}
            path.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
            rows.append(row)
            output.write_text(json.dumps({'charge_computation': True, 'rows': rows,
                                         'summary': summarize(rows, tuple(TYPES))}, indent=2) + '\n')
            print(*spec, row['status'], row.get('loss_seconds'), round(row.get('end_to_end_ms', 0) / 1000, 2), flush=True)


if __name__ == '__main__':
    main()
