"""Fixed-n model lower confidence diagnostic on the eight V8 roots."""

import json
from pathlib import Path
from statistics import mean, median
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
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_oracle import earliest_arrival
from src.temporal_scan import (TemporalScan, bounded_mean_lower, build_temporal_kernel,
                               sample_clocks, window_trips)


def rng(row):
    return np.random.default_rng(np.random.SeedSequence(row['random_seed']))


def packed_scene(graph, clocks, world):
    dep, arr, canceled = (column[world] for column in clocks)
    return JourneyScenario({c.id: VehicleEvent(int(dep[c.id]) - c.dep_time, bool(canceled[c.id]),
                          int(dep[c.id]), int(arr[c.id])) for c in graph.connections},
                          minute=60, publications={})


def run_root(row, graph, metadata, journey, fit, parameters, scan, protocol):
    goal = row['destination']
    deadline = journey['od']['deadline_seconds']
    before = perf_counter()
    belief, state = context(graph, metadata, journey, row['decision'], fit)
    context_seconds = perf_counter() - before
    if (state.stop, state.time) != (row['stop'], row['clock']):
        raise AssertionError('Reconstructed context differs from frozen V8 root')

    n = protocol['draws_per_root']
    before = perf_counter()
    clocks = sample_clocks(belief, parameters, rng(row), n)
    draw_seconds = perf_counter() - before
    before = perf_counter()
    losses = scan.arrivals(*clocks, state, goal, deadline) - state.time
    scan_seconds = perf_counter() - before
    cap = deadline - state.time
    for world in range(2):
        actual = earliest_arrival(graph, packed_scene(graph, clocks, world), state, goal, deadline)[0]
        if actual - state.time != losses[world]:
            raise AssertionError('Packed n=32 world differs from temporal reference')

    window_belief, _ = context(graph, metadata, journey, row['decision'], fit)
    before = perf_counter()
    kept = window_trips(window_belief, parameters, state.time, deadline)
    window_seconds = perf_counter() - before
    before = perf_counter()
    clipped = sample_clocks(window_belief, parameters, rng(row), n, trips=kept)
    window_draw_seconds = perf_counter() - before
    before = perf_counter()
    clipped_losses = scan.arrivals(*clipped, state, goal, deadline) - state.time
    window_scan_seconds = perf_counter() - before
    kept_ids = [slot.cid for trip in kept for slot in belief._trips[trip][::2]]
    if (not np.array_equal(losses, clipped_losses) or
            any(not np.array_equal(a[:, kept_ids], b[:, kept_ids]) for a, b in zip(clocks, clipped))):
        raise AssertionError('Time-window sampling changed a retained clock or an arrival')

    transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
    station_losses = scan.arrivals(*clipped, state, goal, deadline, transfers) - state.time
    if np.any(station_losses < losses):
        raise AssertionError('Positive transfers improved a relaxed oracle arrival')

    # Recreate both V8 worlds with a fresh lazy sampler, at the original n=2.
    reference_belief, _ = context(graph, metadata, journey, row['decision'], fit)
    before = perf_counter()
    reference_worlds = reference_belief.sample(parameters, rng(row), 2)
    lazy_setup_seconds = perf_counter() - before
    before = perf_counter()
    reference = [earliest_arrival(graph, world, state, goal, deadline) for world in reference_worlds]
    lazy_reference_seconds = perf_counter() - before
    stored = [(item['full_temporal_seconds'] + state.time, tuple(item['full_path']))
              for item in row['paired_worlds']]
    if reference != stored:
        raise AssertionError('Lazy reference differs from frozen V8 clocks or path')
    before = perf_counter()
    original_clocks = sample_clocks(belief, parameters, rng(row), 2)
    warm_draw_two_seconds = perf_counter() - before
    before = perf_counter()
    original_arrivals = scan.arrivals(*original_clocks, state, goal, deadline)
    scan_two_seconds = perf_counter() - before
    before = perf_counter()
    original_scenes = [packed_scene(graph, original_clocks, w) for w in range(2)]
    pack_to_object_seconds = perf_counter() - before
    before = perf_counter()
    packed_reference = [earliest_arrival(graph, world, state, goal, deadline) for world in original_scenes]
    warm_reference_seconds = perf_counter() - before
    if packed_reference != stored or list(original_arrivals) != [a for a, _ in stored]:
        raise AssertionError('Packed scan differs from frozen V8 world')

    point = float(losses.mean())
    confidence = bounded_mean_lower(losses, cap, protocol['familywise_alpha'] / 8)
    lower = max(row['lower'], confidence['lower'])
    upper = row['upper']
    gap = upper - lower
    return dict(date=row['date'], query=row['query'], decision=row['decision'], clock=state.time,
        connection_count=len(graph.connections), cap=cap, random_seed=row['random_seed'],
        witness=row['witness'], upper=upper, v8_lower=row['lower'], v8_gap=row['gap'],
        model_oracle_losses=losses.tolist(), sample_mean=point, prefix8_mean=float(losses[:8].mean()),
        point_gap=upper - point, confidence=confidence, lower=lower, certified_gap=gap,
        station_transfer_losses=station_losses.tolist(), station_transfer_mean=float(station_losses.mean()),
        station_transfer_changed_worlds=int(np.sum(station_losses != losses)),
        trips_full=len(belief._trips), trips_retained=len(kept), connections_retained=len(kept_ids),
        passes={str(t): gap <= t for t in protocol['threshold_seconds']},
        frozen_two_losses=(original_arrivals - state.time).tolist(),
        timings=dict(context_seconds=context_seconds, cold_draw_32_seconds=draw_seconds,
            scan_32_seconds=scan_seconds, lazy_setup_2_seconds=lazy_setup_seconds,
            lazy_reference_2_seconds=lazy_reference_seconds, warm_draw_2_seconds=warm_draw_two_seconds,
            scan_2_seconds=scan_two_seconds, pack_to_object_2_seconds=pack_to_object_seconds,
            warm_reference_2_seconds=warm_reference_seconds,
            window_support_seconds=window_seconds, window_draw_32_seconds=window_draw_seconds,
            window_scan_32_seconds=window_scan_seconds,
            window_total_seconds=window_seconds + window_draw_seconds + window_scan_seconds,
            v8_witness_seconds=row['witness_seconds']))


def main():
    protocol = json.loads((HERE / 'sample_protocol.json').read_text())
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    before = perf_counter()
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    training_seconds = perf_counter() - before
    v8 = json.loads((ROOT / 'experiments/reconstruction_v8/opportunity_city_results.json').read_text())['rows']
    assert len(v8) == 8
    archive = json.loads((ROOT / 'experiments/reconstruction_v4/city_replay_results.json').read_text())
    journeys = {(j['date'], j['query']): j for j in archive['rows'] if j['method'] == 'evolving_rollout'}
    rows, setups = [], {}
    with TemporaryDirectory(prefix='bapr_temporal_scan_') as tmp:
        before = perf_counter()
        kernel = build_temporal_kernel(Path(tmp) / 'temporal.so')
        compile_seconds = perf_counter() - before
        day = None
        for old in v8:
            if old['date'] != day:
                day = old['date']
                before = perf_counter()
                graph, _, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
                parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                              for c in graph.connections}
                graph_seconds = perf_counter() - before
                before = perf_counter()
                scan = TemporalScan(graph, kernel)
                setups[day] = dict(graph_seconds=graph_seconds, scan_index_seconds=perf_counter() - before)
            row = run_root(old, graph, metadata, journeys[day, old['query']], fit, parameters,
                           scan, protocol)
            rows.append(row)
            print(day, row['query'], row['decision'], 'L', round(row['lower'], 2),
                  'point gap', round(row['point_gap'], 2), 'bound gap', round(row['certified_gap'], 2),
                  'full seconds', round(row['timings']['cold_draw_32_seconds'] +
                                        row['timings']['scan_32_seconds'], 4),
                  'window seconds', round(row['timings']['window_total_seconds'], 4), flush=True)
    values = ('v8_lower', 'sample_mean', 'station_transfer_mean', 'point_gap', 'lower',
              'certified_gap', 'v8_gap')
    timing = rows[0]['timings']
    summary = dict(mean_seconds={key: mean(row[key] for row in rows) for key in values},
        median_runtime_seconds={key: median(row['timings'][key] for row in rows) for key in timing},
        pass_counts={str(t): sum(row['passes'][str(t)] for row in rows)
                     for t in protocol['threshold_seconds']},
        matched_reference_worlds=4 * len(rows), world_count=protocol['draws_per_root'] * len(rows),
        station_transfer_changed_worlds=sum(row['station_transfer_changed_worlds'] for row in rows),
        mean_retained_trip_fraction=mean(row['trips_retained'] / row['trips_full'] for row in rows),
        mean_retained_connection_fraction=mean(row['connections_retained'] / row['connection_count']
                                               for row in rows))
    result = dict(protocol=protocol, source=str(source.relative_to(ROOT)), training_dates=fit['train_dates'],
        training_seconds=training_seconds, compile_seconds=compile_seconds, setups=setups,
        rows=rows, summary=summary)
    (HERE / 'sample_results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
