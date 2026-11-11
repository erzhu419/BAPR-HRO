"""Match compiled nominal and analytic recourse policies to city baselines."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import sys
from time import process_time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import (
    ReplayRouter, fit_training, run_one, freeze_replay_inputs, summarize)
from src.recourse_itinerary_router import RecourseItineraryRouter
from src.latency_recourse_router import LatencyRecourseRouter

METHODS = ('trained_nominal', 'evolving_uct', 'compiled_nominal', 'analytic_recourse', 'latency_recourse')
VERSION = 'v6_exact_itinerary_first_failure_rescue'


class ItineraryReplayRouter(ReplayRouter, RecourseItineraryRouter):
    def select_connection(self, *args, **kwargs):
        result = super().select_connection(*args, **kwargs)
        prediction = self.predictions[-1]
        for key in ('compiled', 'value', 'primary', 'rescue', 'full_model_lower',
                    'full_model_gap', 'globally_removed', 'family_screened',
                    'itinerary_evaluations', 'analytic_work'):
            if key in prediction:
                self.replay_decisions[-1][key] = prediction[key]
        return result


class LatencyReplayRouter(ReplayRouter, LatencyRecourseRouter):
    def select_connection(self, *args, **kwargs):
        result = super().select_connection(*args, **kwargs)
        prediction = self.predictions[-1]
        for key in ('compiled', 'value', 'primary', 'rescue', 'full_model_lower', 'full_model_gap',
                    'globally_removed', 'family_screened', 'itinerary_evaluations', 'analytic_work',
                    'estimated_execution_clock', 'bounds_scope', 'dispatch_policy_evaluations'):
            if key in prediction:
                self.replay_decisions[-1][key] = prediction[key]
        return result


def execute(spec, data, forecasts, fit, settings):
    is_compiled = spec[2] in ('compiled_nominal', 'analytic_recourse', 'latency_recourse')
    if is_compiled:
        base = LatencyReplayRouter if spec[2] == 'latency_recourse' else ItineraryReplayRouter
        class ConfiguredRouter(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, optimize=spec[2] != 'compiled_nominal',
                                 max_itineraries=settings['max_itineraries'], **kwargs)
        router_type = ConfiguredRouter
    else:
        router_type = ReplayRouter
    started = process_time()
    row = run_one(spec, data, forecasts, fit, 4, 64, 'gtfs_station_max', True,
                  'uct' if spec[2] == 'evolving_uct' else None,
                  router_type=router_type, charge_computation=settings['clock'] == 'charged')
    row.update(implementation=VERSION, clock=settings['clock'], cpu_seconds=process_time() - started)
    if is_compiled:
        row['transition_budget_per_decision'] = None
        row['candidate_itinerary_limit'] = settings['max_itineraries'] if spec[2] != 'compiled_nominal' else 0
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
    parser.add_argument('--clock', choices=('logical', 'charged'), required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS[:4]))
    parser.add_argument('--max-itineraries', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.clock != 'charged' and 'latency_recourse' in args.methods:
        parser.error('LatencyRecourseRouter values the charged execution clock')
    data, _ = load_city(args.input)
    fit = fit_training(data, retain_no_pickup=True)
    forecasts = {'dates': {day: {'frames': []} for day in data['holdout_dates']}}
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key != 'output'}
    output = args.output.resolve()
    protocol = freeze_replay_inputs(output, data=data, forecasts=forecasts, training=fit,
        settings=settings, implementation=VERSION, sources=[str(args.input), str(Path(__file__)),
            'src/risk_bounds.py', 'src/recourse_itinerary_router.py', 'src/first_branch_belief.py',
            'src/latency_recourse_router.py'])
    folder = output.with_name(output.stem + '_cases')
    folder.mkdir(exist_ok=True)
    rows, pending = [], []
    for day in data['holdout_dates']:
        for qi in range(len(data['od_queries'])):
            for method in args.methods:
                spec = day, qi, method
                path = folder / f'{day}_{qi}_{method}.json'
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                else:
                    pending.append((spec, path))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(execute, spec, data, forecasts, fit, settings): (spec, path)
                for spec, path in pending}
        for job in as_completed(jobs):
            spec, path = jobs[job]
            row = job.result()
            path.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
            rows.append(row)
            print(*spec, row['status'], row['loss_seconds'], round(row['end_to_end_ms'] / 1000, 3), flush=True)
    output.write_text(json.dumps(dict(implementation=VERSION, settings=settings,
        frozen_protocol=str(protocol.relative_to(ROOT)), summary=summarize(rows, METHODS), rows=rows),
        indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
