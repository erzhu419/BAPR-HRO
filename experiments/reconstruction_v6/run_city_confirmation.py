"""Evaluate the four predeclared methods using the literal locked controller."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import importlib.util
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

name = 'src._confirmation_locked_recourse'
specification = importlib.util.spec_from_file_location(name, HERE / 'confirmation_locked_recourse.py')
module = importlib.util.module_from_spec(specification)
sys.modules[name] = module
specification.loader.exec_module(module)


class LockedReplayRouter(ReplayRouter, module.RecourseItineraryRouter):
    def select_connection(self, *args, **kwargs):
        result = super().select_connection(*args, **kwargs)
        prediction = self.predictions[-1]
        for key in ('compiled', 'value', 'primary', 'rescue', 'full_model_lower',
                    'full_model_gap', 'globally_removed', 'family_screened',
                    'itinerary_evaluations', 'analytic_work'):
            if key in prediction:
                self.replay_decisions[-1][key] = prediction[key]
        return result


def execute(spec, data, forecasts, fit, clock):
    compiled = spec[2] in ('compiled_nominal', 'analytic_recourse')
    if compiled:
        class Configured(LockedReplayRouter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, optimize=spec[2] == 'analytic_recourse', max_itineraries=8, **kwargs)
        router_type = Configured
    else:
        router_type = ReplayRouter
    started = process_time()
    row = run_one(spec, data, forecasts, fit, 4, 64, 'gtfs_station_max', True,
        'uct' if spec[2] == 'evolving_uct' else None, router_type=router_type,
        charge_computation=clock == 'charged')
    row.update(implementation='v6_literal_locked_city_confirmation', clock=clock,
               cpu_seconds=process_time() - started)
    if compiled:
        row['transition_budget_per_decision'] = None
        row['candidate_itinerary_limit'] = 8 if spec[2] == 'analytic_recourse' else 0
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clock', choices=('logical', 'charged'), required=True)
    args = parser.parse_args()
    lock = json.loads((HERE / 'city_confirmation_protocol.json').read_text())
    data, _ = load_city(HERE / 'city_confirmation_actuals.json.gz')
    fit = fit_training(data, retain_no_pickup=True)
    forecasts = {'dates': {day: {'frames': []} for day in data['holdout_dates']}}
    methods = tuple(lock['settings']['methods'])
    output = HERE / f'city_confirmation_{args.clock}.json'
    protocol = freeze_replay_inputs(output, data=data, forecasts=forecasts, training=fit,
        settings=dict(original_lock=lock, clock=args.clock, workers=2),
        sources=[str(Path(__file__)), str(HERE / 'confirmation_locked_recourse.py'), 'src/risk_bounds.py'],
        implementation='v6_literal_locked_city_confirmation')
    folder = output.with_name(output.stem + '_cases')
    folder.mkdir(exist_ok=True)
    rows, todo = [], []
    for day in data['holdout_dates']:
        for qi in range(len(data['od_queries'])):
            for method in methods:
                spec = day, qi, method
                path = folder / f'{day}_{qi}_{method}.json'
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                else:
                    todo.append((spec, path))
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(execute, spec, data, forecasts, fit, args.clock): (spec, path) for spec, path in todo}
        for job in as_completed(jobs):
            spec, path = jobs[job]
            row = job.result()
            path.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
            rows.append(row)
            print(*spec, row['status'], row['loss_seconds'], round(row['end_to_end_ms'] / 1000, 2), flush=True)
    output.write_text(json.dumps(dict(original_lock='city_confirmation_protocol.json',
        frozen_protocol=str(protocol.relative_to(ROOT)), summary=summarize(rows, methods), rows=rows),
        indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
