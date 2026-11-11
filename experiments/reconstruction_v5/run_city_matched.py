"""Match full-city policy quality and account for online computation time."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city, METHODS as BASE_METHODS, past_forecast_fit
from experiments.reconstruction_v4.run_dated_replay import (
    ReplayRouter, fit_training, run_one, summarize, paired_summary, freeze_replay_inputs)
from src.certified_router import CertifiedPlanningRouter
from src.evolving_trip_belief import EvolvingTripBelief

VERSION = 'v5_frontier_witness_first_cutoff_certificates'
METHODS = BASE_METHODS + ('certified_rollout',)


class CertifiedReplayRouter(ReplayRouter, CertifiedPlanningRouter):
    def select_connection(self, *args, **kwargs):
        result = super().select_connection(*args, **kwargs)
        prediction = self.predictions[-1]
        self.replay_decisions[-1].update({key: prediction[key] for key in (
            'finite_bank_lower', 'finite_bank_upper', 'finite_bank_gap',
            'certified_removed', 'bound_ms', 'bound_work', 'action_bounds')})
        return result


def execute(spec, data, forecasts, fit, settings):
    belief_type = EvolvingTripBelief
    if settings['engine'] == 'reference':
        name = 'src._runtime_reference_evolving'
        module_spec = importlib.util.spec_from_file_location(name, HERE / 'runtime_reference_evolving.py')
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        belief_type = module.EvolvingTripBelief
    row = run_one(spec, data, forecasts, fit, settings['samples'], settings['budget'],
                  'gtfs_station_max', True, 'uct' if spec[2] == 'evolving_uct' else None,
                  belief_type=belief_type,
                  router_type=CertifiedReplayRouter if spec[2] == 'certified_rollout' else ReplayRouter,
                  charge_computation=settings['clock'] == 'charged')
    row.update(implementation=VERSION, engine=settings['engine'], clock=settings['clock'])
    fields = {}
    with open('/proc/self/status') as handle:
        for line in handle:
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, value, _ = line.split()
                fields[key.rstrip(':')] = int(value)
    row['process_image_memory_kib'] = fields
    return row


def compare_archived(rows):
    source = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    original = json.loads(source.read_text())
    lookup = {(r['date'], r['query'], r['method']): r for r in original['rows']}
    fields = ('clock_seconds', 'stop', 'current_trip', 'action', 'outcome', 'end_seconds',
              'model_transitions', 'complete_common_rounds', 'root_actions')
    changed = []
    for row in rows:
        if (row['date'], row['query'], row['method']) not in lookup:
            continue
        old = lookup[row['date'], row['query'], row['method']]
        a = [{k: d[k] for k in fields} for d in row['decisions']]
        b = [{k: d[k] for k in fields} for d in old['decisions']]
        if (row['status'], row['loss_seconds'], row['events'], a) != (
                old['status'], old['loss_seconds'], old['events'], b):
            changed.append([row['date'], row['query'], row['method']])
    return {'source': str(source.relative_to(ROOT)), 'changed_traces': changed,
            'comparison': 'Status, loss, complete events and decision/action/work fields; timing is reported separately.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clock', choices=('logical', 'charged'), default='logical')
    parser.add_argument('--engine', choices=('lazy', 'reference'), default='lazy')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
    parser.add_argument('--dates', nargs='+')
    parser.add_argument('--queries', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or HERE / f'city_{args.engine}_{args.clock}_results.json'
    output = output.resolve()
    data_path = args.input.resolve()
    data, _ = load_city(data_path)
    if args.dates:
        if not set(args.dates) <= set(data['holdout_dates']):
            parser.error('Selected dates must be declared holdouts')
        data['holdout_dates'] = args.dates
    fits = ({day: fit_training(data, retain_no_pickup=True) for day in data['holdout_dates']}
            if data['train_dates'] else {day: past_forecast_fit(day) for day in data['holdout_dates']})
    forecasts = {'dates': {day: {'frames': []} for day in data['holdout_dates']}}
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'output'}
    protocol = freeze_replay_inputs(output, data=data, forecasts=forecasts, training=fits,
                                    settings=settings, sources=[str(data_path), str(Path(__file__)),
                                    str(HERE / 'runtime_reference_evolving.py'),
                                    'src/certified_router.py', 'src/certified_policy_search.py',
                                    'src/sparse_certificate.py', 'src/sparse_certificate_cutoff.py',
                                    'data/route_priors_temporal_corrected.pkl'],
                                    implementation=VERSION)
    folder = output.with_name(output.stem + '_cases')
    folder.mkdir(exist_ok=True)
    rows, pending = [], []
    for day in data['holdout_dates']:
        for qi in args.queries:
            for method in args.methods:
                spec = day, qi, method
                path = folder / f'{day}_{qi}_{method}.json'
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                else:
                    pending.append((spec, path))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(execute, spec, data, forecasts, fits[spec[0]], settings): (spec, path)
                for spec, path in pending}
        for future in as_completed(jobs):
            spec, path = jobs[future]
            row = future.result()
            path.write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
            print(*spec, row['status'], row['loss_seconds'], row['end_to_end_ms'], flush=True)
    result = {'implementation': VERSION, 'settings': settings,
              'frozen_protocol': str(protocol.relative_to(ROOT)),
              'scope': 'Complete dated city candidates, fixed static OD queries, past-only fits; see frozen inputs for dates.',
              'clock': 'Decision processing including initial routing is rounded up to seconds and paid before executing each choice.'
                       if args.clock == 'charged' else 'Computation is measured separately from the passenger event clock.',
              'training': fits, 'summary': summarize(rows, METHODS), 'paired': paired_summary(rows, METHODS),
              'rows': sorted(rows, key=lambda r: (r['date'], r['query'], r['method']))}
    if args.clock == 'logical':
        result['archived_trace_comparison'] = compare_archived(rows)
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
