"""Isolate support centering on all eight archived V10 world tapes."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from src.continuation_confidence import continuation_lower
from src.temporal_corridor import TemporalCorridor
from src.temporal_scan import TemporalScan, build_temporal_kernel


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    source = ROOT / 'experiments/reconstruction_v10/matched_gate_results.json'
    archived = json.loads(source.read_text())
    data, _ = load_city(ROOT / archived['source'])
    fit = fit_training(data, retain_no_pickup=True)
    journeys = {(r['date'], r['query']): r for r in json.loads(
        (ROOT / 'experiments/reconstruction_v4/city_replay_results.json').read_text())['rows']
        if r['method'] == 'evolving_rollout'}
    witnesses = {(r['date'], r['query'], r['decision']): r for r in json.loads(
        (ROOT / 'experiments/reconstruction_v9/witness_reference_results.json').read_text())['rows']}
    rows, day = [], None
    with TemporaryDirectory(prefix='bapr_v11_floor_') as tmp:
        kernel = build_temporal_kernel(Path(tmp) / 'scan.so')
        for old in archived['rows']:
            key = old['date'], old['query'], old['decision']
            if day != key[0]:
                day = key[0]
                graph, _, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
                scan = TemporalScan(graph, kernel)
                transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
                priors = fit['route_priors']
                parameters = {c.route: priors.get(c.route, priors['pooled']) for c in graph.connections}
            belief, state = context(graph, metadata, journeys[key[:2]], key[2], fit)
            assert state.time == old['clock']
            cap = old['caps']['station_corridor']
            before = perf_counter()
            corridor = TemporalCorridor(scan)
            connections, floor = corridor.connections_and_floor(
                belief, parameters, state, witnesses[key]['destination'], state.time + cap,
                transfer=transfers)
            seconds = perf_counter() - before
            losses = np.asarray(old['losses']['station_corridor'])
            # A violated support bound invalidates the certificate, not just its timing.
            assert np.all(losses >= floor)
            alpha = archived['protocol']['root_alpha']
            values = {}
            for n in (32, 128):
                zero = continuation_lower(losses[:n], cap, alpha)
                centered = continuation_lower(losses[:n], cap, alpha, floor=floor)
                assert abs(zero - old['values']['station_corridor'][str(n)]['lower']) < 1e-5
                values[str(n)] = dict(zero_floor_lower=zero, centered_lower=centered,
                    zero_floor_gap=old['upper'] - zero, centered_gap=old['upper'] - centered,
                    rescue_gap=old['tail_rescue_upper'] - centered)
            row = dict(date=key[0], query=key[1], decision=key[2], floor=floor,
                upper=old['upper'], rescue_upper=old['tail_rescue_upper'], cap=cap,
                deterministic_accept=old['upper'] - floor <= 60,
                corridor_seconds=seconds, connections=len(connections), values=values)
            rows.append(row)
            print(key, 'floor', floor, 'gap128', round(values['128']['centered_gap'], 2), flush=True)
    summary = {str(n): dict(
        old_accepts=sum(r['values'][str(n)]['zero_floor_gap'] <= 60 for r in rows),
        centered_accepts=sum(r['values'][str(n)]['centered_gap'] <= 60 for r in rows),
        rescue_accepts=sum(r['values'][str(n)]['rescue_gap'] <= 60 for r in rows),
        mean_old_gap=float(np.mean([r['values'][str(n)]['zero_floor_gap'] for r in rows])),
        mean_centered_gap=float(np.mean([r['values'][str(n)]['centered_gap'] for r in rows])))
        for n in (32, 128)}
    result = dict(protocol=dict(source=str(source.relative_to(ROOT)), alpha=alpha,
        epsilon=60, cpu=sorted(os.sched_getaffinity(0)),
        scope='All eight development histories; same archived V10 losses, cutoff and U. '
              'Deterministic support floor recomputed from each full frozen posterior. '
              'No new independence or runtime savings inferred from reused tapes.'),
        rows=rows, summary=summary)
    (HERE / 'floor_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
