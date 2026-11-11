"""Eight old-date journeys using strict-native labels and capped root scoring."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from time import process_time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import ReplayRouter, fit_training, run_one, freeze_replay_inputs, summarize
from src.native_event_router import NativeEventRouter


class NativeReplayRouter(ReplayRouter, NativeEventRouter):
    def select_connection(self, *args, **kwargs):
        result = super().select_connection(*args, **kwargs)
        prediction = self.predictions[-1]
        self.replay_decisions[-1].update({key: prediction[key] for key in
            ('native_objective', 'native_work', 'native_selected', 'native_model')})
        return result


def execute(spec, data, frames, fit, clock):
    before = process_time()
    row = run_one(spec, data, frames, fit, 4, 64, 'gtfs_station_max', True,
                  algorithm_override='uct', router_type=NativeReplayRouter,
                  charge_computation=clock == 'charged')
    row.update(implementation='v6_native_event_capped_interned_secondary', clock=clock,
               transition_budget_per_decision=None, native_objective='capped_root_strict_mean_labels',
               cpu_seconds=process_time() - before)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    frames = {'dates': {day: {'frames': []} for day in data['holdout_dates']}}
    output = HERE / 'native_city_interned_results.json'
    protocol = freeze_replay_inputs(output, data=data, forecasts=frames, training=fit,
        settings={'workers': args.workers, 'methods': ['native_capped'], 'clocks': ['logical', 'charged'],
                  'scope': 'Secondary July26/27; new July28/29 primary protocol unchanged',
                  'objective': 'Capped root score; original conditional-mean strict labels unchanged',
                  'inputs': 'All connections and same Evolving public event history; physical seconds execution',
                  'quantization': 'Event clocks nearest minute <=30s; query source floor <60s; MCT ceil <60s',
                  'root_execution': 'Exact-second catch mass; MCT/12min/pickup; nominal WAIT minute phase'},
        sources=[str(source), str(Path(__file__)), 'src/native_event_router.py', 'src/durner/native.py'],
        implementation='v6_native_event_capped_interned_secondary')
    folder = HERE / 'native_city_interned_cases'
    folder.mkdir(exist_ok=True)
    rows, pending = [], []
    for clock in ('logical', 'charged'):
        for day in data['holdout_dates']:
            for qi in range(len(data['od_queries'])):
                path = folder / f'{clock}_{day}_{qi}.json'
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                else:
                    pending.append(((day, qi, 'native_capped'), clock, path))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(execute, spec, data, frames, fit, clock): path for spec, clock, path in pending}
        for future in as_completed(jobs):
            path = jobs[future]
            row = future.result()
            path.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
            rows.append(row)
            print(row['clock'], row['date'], row['query'], row['status'], row['loss_seconds'],
                  round(row['end_to_end_ms'] / 1000, 3), flush=True)
    by_clock, comparisons = {}, []
    for clock in ('logical', 'charged'):
        baseline = json.loads((HERE / f'city_itinerary_{clock}.json').read_text())
        methods = ('evolving_uct', 'compiled_nominal', 'analytic_recourse')
        references = [r for r in baseline['rows'] if r['method'] in methods]
        joined = {(r['date'], r['query'], r['method']): r for r in references}
        current = [r for r in rows if r['clock'] == clock]
        for row in current:
            for method in methods:
                previous = joined[row['date'], row['query'], method]
                if any(row[key] != previous[key] for key in ('od', 'graph_connections', 'known_world_events', 'transfer_rule')):
                    raise AssertionError('Native comparison geometry or actual coverage changed')
                comparisons.append(dict(clock=clock, date=row['date'], query=row['query'], reference=method,
                    native_status=row['status'], reference_status=previous['status'],
                    native_loss=row['loss_seconds'], reference_loss=previous['loss_seconds']))
        by_clock[clock] = summarize(current + references, ('native_capped', *methods))
    output.write_text(json.dumps(dict(implementation='v6_native_event_capped_interned_secondary',
        frozen_protocol=str(protocol.relative_to(ROOT)), by_clock=by_clock,
        comparisons=comparisons, rows=rows), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
