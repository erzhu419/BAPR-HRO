"""Matched station-transfer interval corridor check on the V9 histories."""

import json
from math import ceil
from pathlib import Path
from statistics import median
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
from src.temporal_corridor import TemporalCorridor
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map, window_trips


def main():
    data, _ = load_city(ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
    fit = fit_training(data, retain_no_pickup=True)
    archive = json.loads((ROOT / 'experiments/reconstruction_v4/city_replay_results.json').read_text())['rows']
    journeys = {(j['date'], j['query']): j for j in archive if j['method'] == 'evolving_rollout'}
    v9 = ROOT / 'experiments/reconstruction_v9'
    witnesses = {(r['date'], r['query'], r['decision']): r['best_upper_seconds']
                 for r in json.loads((v9 / 'witness_reference_results.json').read_text())['rows']}
    n, rows = 16, []
    with TemporaryDirectory(prefix='bapr_corridor_') as tmp:
        kernel = build_temporal_kernel(Path(tmp) / 'scan.so')
        day = None
        for row in json.loads((v9 / 'sample_results.json').read_text())['rows']:
            key = row['date'], row['query'], row['decision']
            if day != key[0]:
                day = key[0]
                graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
                parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                              for c in graph.connections}
                scan = TemporalScan(graph, kernel)
                transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
            journey = journeys[key[:2]]
            destination = stops[journey['od']['destination']]
            original_deadline = journey['od']['deadline_seconds']
            belief, state = context(graph, metadata, journey, key[2], fit)
            cap = min(original_deadline, state.time + ceil(witnesses[key] + 60))
            started = perf_counter()
            corridor = TemporalCorridor(scan)
            connections = corridor.connections(belief, parameters, state, destination, cap, transfer=transfers)
            kept = {belief._trip_by_cid[cid] for cid in connections}
            cold_seconds = perf_counter() - started
            started = perf_counter()
            full_deadline_trips = corridor.trips(belief, parameters, state, destination, original_deadline,
                                                 transfer=transfers)
            full_deadline_warm_seconds = perf_counter() - started
            started = perf_counter()
            clipped = window_trips(belief, parameters, state.time, cap)
            window_seconds = perf_counter() - started
            assert kept <= clipped

            before = perf_counter()
            templates = template_trip_map(belief, parameters)
            template_seconds = perf_counter() - before
            arms = [('window', clipped), ('corridor', kept)]
            if key[2] % 2:
                arms.reverse()
            elapsed = {}
            losses = {}
            for label, allowed in arms:
                fresh, _ = context(graph, metadata, journey, key[2], fit)
                tape = np.random.default_rng(np.random.SeedSequence([*row['random_seed'], 10, n]))
                start = perf_counter()
                worlds = sample_clocks(fresh, parameters, tape, n, trips=allowed, templates=templates)
                losses[label] = scan.arrivals(*worlds, state, destination, cap, transfers) - state.time
                elapsed[label] = perf_counter() - start
            np.testing.assert_array_equal(losses['window'], losses['corridor'])
            rows.append(dict(date=day, query=key[1], decision=key[2], cap_seconds=cap - state.time,
                window_trips=len(clipped), corridor_trips=len(kept), corridor_connections=len(connections),
                original_deadline_corridor_trips=len(full_deadline_trips),
                window_support_seconds=window_seconds, corridor_cold_seconds=cold_seconds,
                original_deadline_corridor_warm_seconds=full_deadline_warm_seconds, template_seconds=template_seconds,
                window_draw_and_scan_seconds=elapsed['window'], corridor_draw_and_scan_seconds=elapsed['corridor'],
                paired_losses=losses['corridor'].tolist()))
            print(day, key[1:], 'window', len(clipped), 'corridor', len(kept),
                  'cold', round(cold_seconds, 3), 'draw', {k: round(v, 3) for k, v in elapsed.items()}, flush=True)
    summary = {key: median(r[key] for r in rows) for key in (
        'window_trips', 'corridor_trips', 'corridor_connections', 'original_deadline_corridor_trips',
        'window_support_seconds', 'corridor_cold_seconds', 'original_deadline_corridor_warm_seconds',
        'template_seconds', 'window_draw_and_scan_seconds', 'corridor_draw_and_scan_seconds')}
    result = dict(scope='Same eight V9 development histories. The shortened cap is fixed from the archived public '
                        'witness upper plus 60 seconds. Both arms have identical caps, MCT, templates, and random tapes.',
                  draws_per_root=n, summary=summary, rows=rows)
    (HERE / 'corridor_results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
