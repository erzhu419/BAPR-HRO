"""Secondary old-date comparison of a one-second transfer gate with UCT64."""
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
from src.transfer_gate_router import GateUCTRouter

VERSION = 'v6_secondary_one_second_transfer_gate_uct64'


class GateReplayRouter(ReplayRouter, GateUCTRouter):
    def select_connection(self, *args, **kwargs):
        result = super().select_connection(*args, **kwargs)
        p = self.predictions[-1]
        self.replay_decisions[-1].update({key: p[key] for key in (
            'gate_used', 'gate_cached', 'gate_checked', 'gate_check_ms', 'gate_work',
            'gate_issue_clock', 'gate_certificate_at_issue', 'gate_rejected_certificate',
            'decision_ordinal', 'fallback_planning_ms') if key in p})
        return result


def execute(spec, data, frames, fit, clock):
    before = process_time()
    row = run_one(spec, data, frames, fit, 4, 64, 'gtfs_station_max', True,
                  algorithm_override='uct', router_type=GateReplayRouter,
                  charge_computation=clock == 'charged')
    row.update(implementation=VERSION, clock=clock, cpu_seconds=process_time() - before)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clock', choices=('logical', 'charged'), required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    frames = {'dates': {day: {'frames': []} for day in data['holdout_dates']}}
    output = HERE / f'city_gate_{args.clock}_results.json'
    protocol = freeze_replay_inputs(output, data=data, forecasts=frames, training=fit,
        settings={'clock': args.clock, 'workers': args.workers, 'methods': ['gate_uct'],
                  'scope': 'Secondary old July26/27 dates; new July28/29 primary protocol is not amended.',
                  'gate_tolerance_seconds': 1, 'fallback_budget': 64, 'samples': 4,
                  'query_seed': 'Every actual selector call occupies a query ordinal, including issued macro legs.',
                  'transfer_contract': 'Existing graph GTFS max-stated or assumed120s, not validated platform minima.'},
        sources=[str(source), str(Path(__file__)), 'src/risk_bounds.py', 'src/transfer_gate_router.py'],
        implementation=VERSION)
    folder = output.with_name(output.stem + '_cases')
    folder.mkdir(exist_ok=True)
    rows, pending = [], []
    for day in data['holdout_dates']:
        for qi in range(len(data['od_queries'])):
            path = folder / f'{day}_{qi}.json'
            if path.exists():
                rows.append(json.loads(path.read_text()))
            else:
                pending.append(((day, qi, 'gate_uct'), path))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(execute, spec, data, frames, fit, args.clock): path for spec, path in pending}
        for future in as_completed(jobs):
            path = jobs[future]
            row = future.result()
            path.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
            rows.append(row)
            print(row['date'], row['query'], row['status'], row['loss_seconds'],
                  round(row['end_to_end_ms'] / 1000, 3), flush=True)
    baseline_path = HERE / f'city_itinerary_{args.clock}.json'
    baseline = json.loads(baseline_path.read_text())
    references = [r for r in baseline['rows'] if r['method'] in ('evolving_uct', 'compiled_nominal')]
    joined = {(r['date'], r['query'], r['method']): r for r in references}
    comparisons = []
    for row in rows:
        for method in ('evolving_uct', 'compiled_nominal'):
            previous = joined[row['date'], row['query'], method]
            if any(row[key] != previous[key] for key in ('od', 'graph_connections', 'known_world_events', 'transfer_rule')):
                raise AssertionError('Gate comparison input geometry or world coverage changed')
            comparisons.append({'date': row['date'], 'query': row['query'], 'reference': method,
                'gate_status': row['status'], 'reference_status': previous['status'],
                'gate_loss': row['loss_seconds'], 'reference_loss': previous['loss_seconds'],
                'gate_compute_ms': row['end_to_end_ms'], 'reference_compute_ms': previous['end_to_end_ms']})
    decisions = [d for row in rows for d in row['decisions']]
    result = {'implementation': VERSION, 'frozen_protocol': str(protocol.relative_to(ROOT)),
              'baseline_source': str(baseline_path.relative_to(ROOT)),
              'summary': summarize(rows + references, ('gate_uct', 'evolving_uct', 'compiled_nominal')),
              'gate_counts': {'checked': sum(d['gate_checked'] for d in decisions),
                              'newly_accepted': sum(d['gate_used'] and not d['gate_cached'] for d in decisions),
                              'issued_lookups': sum(d['gate_cached'] for d in decisions),
                              'all_fallback_work_leq64': all(d['model_transitions'] <= 64 for d in decisions)},
              'comparisons': comparisons, 'rows': sorted(rows, key=lambda r: (r['date'], r['query']))}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
