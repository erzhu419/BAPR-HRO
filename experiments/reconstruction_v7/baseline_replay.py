"""Existing city journeys with the exact optimized strict-native port."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import ReplayRouter, fit_training, run_one, summarize
from src.optimized_native_router import OptimizedNativeEventRouter, build_native_kernel, load_native_kernel

KERNEL = None


def initialize(path):
    global KERNEL
    KERNEL = load_native_kernel(path)


class OptimizedNativeReplayRouter(ReplayRouter, OptimizedNativeEventRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, native_kernel=KERNEL, **kwargs)

    def select_connection(self, *args, **kwargs):
        result = super().select_connection(*args, **kwargs)
        prediction = self.predictions[-1]
        self.replay_decisions[-1].update({key: prediction[key] for key in
            ('native_objective', 'native_work', 'native_selected', 'native_model')})
        return result


def execute(spec, data, forecasts, fit, clock):
    before = process_time()
    row = run_one(spec, data, forecasts, fit, 4, 64, 'gtfs_station_max', True,
        algorithm_override='uct', router_type=OptimizedNativeReplayRouter,
        charge_computation=clock == 'charged')
    row.update(implementation='v7_optimized_native_same_recursion', clock=clock,
        transition_budget_per_decision=None, native_objective='capped_root_strict_mean_labels',
        cpu_seconds=process_time() - before)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clock', choices=('logical', 'charged'), required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    forecasts = {'dates': {day: {'frames': []} for day in data['holdout_dates']}}
    output = HERE / f'baseline_native_{args.clock}.json'
    protocol = dict(scope='Existing V6 development inputs and strict-native objective; implementation comparison, not independent confirmation.',
        clock=args.clock, workers=args.workers, fit=fit, input=str(source.relative_to(ROOT)),
        methods=['native_optimized'], native_objective='Same V6 capped root/strict conditional-mean recursion.',
        sources={name: (ROOT / name).read_text() for name in (
            'src/optimized_native_router.py', 'src/native_order.cpp',
            'experiments/reconstruction_v7/baseline_replay.py')})
    output.with_name(output.stem + '_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    library = HERE / 'build/native_order_replay.so'
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    before = perf_counter()
    build_native_kernel(library)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    compilation = dict(wall_seconds=perf_counter() - before,
        compiler_cpu_seconds=child_after.ru_utime + child_after.ru_stime - child_before.ru_utime - child_before.ru_stime)
    rows, todo = [], []
    folder = output.with_name(output.stem + '_cases')
    folder.mkdir(exist_ok=True)
    for day in data['holdout_dates']:
        for qi in range(len(data['od_queries'])):
            spec = day, qi, 'native_optimized'
            path = folder / f'{day}_{qi}.json'
            if path.exists():
                rows.append(json.loads(path.read_text()))
            else:
                todo.append((spec, path))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(str(library),)) as pool:
        futures = {pool.submit(execute, spec, data, forecasts, fit, args.clock): (spec, path)
                   for spec, path in todo}
        for future in as_completed(futures):
            spec, path = futures[future]
            row = future.result()
            path.write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
            print(*spec, row['status'], row['loss_seconds'], round(row['end_to_end_ms'] / 1000, 3), flush=True)
    old = json.loads((ROOT / 'experiments/reconstruction_v6/native_city_interned_results.json').read_text())
    comparisons = []
    old_cases = {(r['date'], r['query']): r for r in old['rows'] if r['clock'] == args.clock}
    for row in rows:
        reference = old_cases[row['date'], row['query']]
        action_equality = [d['action'] for d in row['decisions']] == [d['action'] for d in reference['decisions']]
        if args.clock == 'logical' and (not action_equality or row['loss_seconds'] != reference['loss_seconds']):
            raise AssertionError('An exact implementation change altered the logical native policy')
        comparisons.append(dict(date=row['date'], query=row['query'], old_loss=reference['loss_seconds'],
            optimized_loss=row['loss_seconds'], same_action_sequence=action_equality,
            original_seconds=reference['end_to_end_ms'] / 1000, optimized_seconds=row['end_to_end_ms'] / 1000))
    output.write_text(json.dumps(dict(protocol=output.with_name(output.stem + '_protocol.json').name,
        compilation=compilation, summary=summarize(rows, ('native_optimized',)), comparisons=comparisons, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
