"""Independent fixed-size precision check on the closest development root."""

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
from experiments.reconstruction_v9.search_reference import cold_search
from src.temporal_scan import (TemporalScan, bounded_mean_lower, build_temporal_kernel, sample_clocks,
                               template_trip_map, window_trips)


def main():
    n, alpha = 128, .05
    pilot = json.loads((HERE / 'sample_results.json').read_text())['rows']
    case = min(pilot, key=lambda row: row['certified_gap'])
    key = case['date'], case['query'], case['decision']
    witnesses = json.loads((HERE / 'witness_reference_results.json').read_text())['rows']
    witness = next(row for row in witnesses if (row['date'], row['query'], row['decision']) == key)
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    archive = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    journeys = json.loads(archive.read_text())['rows']
    journey = next(r for r in journeys if r['method'] == 'evolving_rollout' and (r['date'], r['query']) == key[:2])
    graph, _, stops, _, metadata, _ = dated_graph(data, key[0], 'gtfs_station_max', True)
    destination = stops[journey['od']['destination']]
    deadline = journey['od']['deadline_seconds']
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                  for c in graph.connections}
    started = perf_counter()
    belief, state = context(graph, metadata, journey, key[2], fit)
    context_seconds = perf_counter() - started
    if (state.stop, state.time, destination) != (witness['stop'], witness['clock'], witness['destination']):
        raise AssertionError('Unmatched city history')
    # A separate seed stream and fixed budget after selection on the n=32 pilot.
    seed = [*case['random_seed'], 9, n]
    with TemporaryDirectory(prefix='bapr_temporal_closest_') as tmp:
        started = perf_counter()
        kernel = build_temporal_kernel(Path(tmp) / 'temporal.so')
        scan = TemporalScan(graph, kernel)
        setup_seconds = perf_counter() - started
        started = perf_counter()
        kept = window_trips(belief, parameters, state.time, deadline)
        clocks = sample_clocks(belief, parameters, np.random.default_rng(np.random.SeedSequence(seed)), n, trips=kept)
        losses = scan.arrivals(*clocks, state, destination, deadline) - state.time
        cold_seconds = perf_counter() - started
        template_belief, _ = context(graph, metadata, journey, key[2], fit)
        started = perf_counter()
        retained = window_trips(template_belief, parameters, state.time, deadline)
        templates = template_trip_map(template_belief, parameters)
        reused = sample_clocks(template_belief, parameters, np.random.default_rng(np.random.SeedSequence(seed)), n,
                               trips=retained, templates=templates)
        reused_losses = scan.arrivals(*reused, state, destination, deadline) - state.time
        template_seconds = perf_counter() - started
        if any(not np.array_equal(a, b) for a, b in zip(clocks, reused)) or not np.array_equal(losses, reused_losses):
            raise AssertionError('Translated trip templates changed the conditional sample')
    search_timing = {}
    for budget in (64, 128):
        search = cold_search(graph, metadata, stops, fit, journey, key[2], budget)
        if (search['clock'], search['stop'], search['destination']) != (state.time, state.stop, destination):
            raise AssertionError('Search timing used a different history')
        search_timing[budget] = dict(model_setup=search['model_setup'], planning=search['planning'],
                                    prediction=search['prediction'])
    confidence = bounded_mean_lower(losses, deadline - state.time, alpha)
    upper = witness['best_upper_seconds']
    result = dict(date=key[0], query=key[1], decision=key[2], cpu_affinity=sorted(os.sched_getaffinity(0)),
                  protocol='One development history selected by minimum n=32 simultaneous gap. '
                           'Independent fixed n=128 random stream, zero-MCT conditional model only.',
                  seed=seed, alpha=alpha, draws=n, trips_retained=len(kept), confidence=confidence,
                  upper_seconds=upper, pilot_gap_seconds=case['certified_gap'],
                  sample_mean_seconds=float(losses.mean()), model_oracle_losses=losses.tolist(),
                  confidence_gap_seconds=upper - confidence['lower'],
                  context_seconds=context_seconds, setup_seconds=setup_seconds, cold_lower_seconds=cold_seconds,
                  cold_template_lower_seconds=template_seconds, same_core_search=search_timing)
    (HERE / 'closest_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print({key: value for key, value in result.items() if key != 'model_oracle_losses'}, flush=True)


if __name__ == '__main__':
    main()
