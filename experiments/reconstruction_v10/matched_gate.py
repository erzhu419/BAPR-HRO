"""Frozen-history comparison of transfer assumptions, confidence and exact pruning."""

import json
from math import ceil
import os
from pathlib import Path
from statistics import mean, median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from experiments.reconstruction_v9.search_reference import cold_search
from src.continuation_confidence import continuation_lower
from src.temporal_corridor import TemporalCorridor
from src.temporal_scan import (TemporalScan, bounded_mean_lower, build_temporal_kernel,
                               sample_clocks, template_trip_map, window_trips)


SIZES = (32, 128)
FAMILY_ALPHA = .05
EPSILON = 60


def timed(function, *args, **kwargs):
    wall, cpu = perf_counter(), process_time()
    value = function(*args, **kwargs)
    return value, dict(wall_seconds=perf_counter() - wall, cpu_seconds=process_time() - cpu)


def key(row):
    return row['date'], row['query'], row['decision']


def draw(belief, parameters, seed, trips):
    templates, template_time = timed(template_trip_map, belief, parameters)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    clocks, sample_time = timed(sample_clocks, belief, parameters, rng, max(SIZES),
                                trips=trips, templates=templates)
    return clocks, dict(template=template_time, draw=sample_time)


def summarize(losses, deadline_loss, upper, tail_upper, alpha):
    results = {}
    for n in SIZES:
        sample = losses[:n]
        old, _ = timed(bounded_mean_lower, sample, deadline_loss, alpha)
        lower, confidence = timed(continuation_lower, sample, deadline_loss, alpha)
        results[str(n)] = dict(point=float(sample.mean()), point_gap=upper - float(sample.mean()),
                               old_lower=old['lower'], old_gap=upper - old['lower'], lower=lower,
                               gap=upper - lower, passes=upper - lower <= EPSILON,
                               tail_rescue_gap=tail_upper - lower, tail_rescue_passes=tail_upper - lower <= EPSILON,
                               confidence_timing=confidence)
    return results


def evaluate(row, witness, tail, graph, scan, metadata, journey, fit, parameters, alpha):
    day, query, decision = key(row)
    upper = witness['best_upper_seconds']
    goal = witness['destination']
    tail_upper = tail['upper']
    deadline = journey['od']['deadline_seconds']
    (belief, state), history_time = timed(context, graph, metadata, journey, decision, fit)
    if (state.time, state.stop) != (row['clock'], witness['stop']):
        raise AssertionError('Unmatched V9 context')
    if state.current_trip is None or state.time != state.last_arrival:
        raise AssertionError('This declared station-MCT panel uses fresh onboard contexts')
    seed = [*row['random_seed'], 10, 202611]
    transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
    trips, window_time = timed(window_trips, belief, parameters, state.time, deadline)
    full, timing = draw(belief, parameters, seed, trips)
    zero, zero_time = timed(scan.arrivals, *full, state, goal, deadline)
    station, station_time = timed(scan.arrivals, *full, state, goal, deadline, transfers)
    if np.any(station < zero):
        raise AssertionError('Positive transfer duration improved a same-world oracle')

    clipped_deadline = min(deadline, state.time + ceil(upper + EPSILON))
    (clipped_belief, _), _ = timed(context, graph, metadata, journey, decision, fit)
    corridor, corridor_setup = timed(TemporalCorridor, scan)
    corridor_trips, corridor_time = timed(corridor.trips, clipped_belief, parameters, state,
                                         goal, clipped_deadline, transfer=transfers)
    clipped, clipped_timing = draw(clipped_belief, parameters, seed, corridor_trips)
    corridor_arrivals, corridor_scan = timed(scan.arrivals, *clipped, state, goal, clipped_deadline, transfers)
    np.testing.assert_array_equal(corridor_arrivals, np.minimum(station, clipped_deadline))
    losses = dict(zero=zero - state.time, station=station - state.time,
                  station_corridor=corridor_arrivals - state.time)
    caps = dict(zero=deadline - state.time, station=deadline - state.time,
                station_corridor=clipped_deadline - state.time)
    values = {label: summarize(loss, caps[label], upper, tail_upper, alpha) for label, loss in losses.items()}
    wall_sum = lambda stages: sum(stage['wall_seconds'] for stage in stages)
    full_cost = wall_sum((window_time, *timing.values(), station_time,
                          values['station'][str(max(SIZES))]['confidence_timing']))
    corridor_cost = wall_sum((corridor_setup, corridor_time, *clipped_timing.values(), corridor_scan,
                              values['station_corridor'][str(max(SIZES))]['confidence_timing']))
    return dict(date=day, query=query, decision=decision, clock=state.time, seed=seed, upper=upper,
                tail_rescue_upper=tail_upper,
                retained_window_trips=len(trips), retained_corridor_trips=len(corridor_trips), caps=caps,
                timings=dict(history=history_time, window=window_time, **timing, zero_scan=zero_time,
                             station_scan=station_time, corridor_setup=corridor_setup, corridor=corridor_time,
                             corridor_sampling=clipped_timing, corridor_scan=corridor_scan,
                             full_cold_wall_seconds=full_cost, corridor_cold_wall_seconds=corridor_cost),
                losses={label: x.tolist() for label, x in losses.items()}, values=values)


def accepted_search_controls(rows, data, fit, journeys):
    controls = []
    for row in rows:
        if not row['values']['station_corridor'][str(max(SIZES))]['passes']:
            continue
        day, query, decision = key(row)
        graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        for budget in (64, 128):
            control = cold_search(graph, metadata, stops, fit, journeys[day, query], decision, budget)
            if control['clock'] != row['clock']:
                raise AssertionError('Search control did not reproduce the gate clock')
            controls.append(control)
    return controls


def main():
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    v9 = ROOT / 'experiments/reconstruction_v9'
    samples = json.loads((v9 / 'sample_results.json').read_text())['rows']
    witnesses = {key(r): r for r in json.loads((v9 / 'witness_reference_results.json').read_text())['rows']}
    tails = {key(r): r for r in json.loads((HERE / 'tail_rescue_results.json').read_text())['rows']}
    archive_path = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    archive = json.loads(archive_path.read_text())['rows']
    journeys = {(r['date'], r['query']): r for r in archive if r['method'] == 'evolving_rollout'}
    (data, _), training_io = timed(load_city, source)
    fit, training = timed(fit_training, data, retain_no_pickup=True)
    alpha = FAMILY_ALPHA / len(samples)
    rows, setups = [], {}
    with TemporaryDirectory(prefix='bapr_matched_v10_') as tmp:
        kernel, compile_time = timed(build_temporal_kernel, Path(tmp) / 'scan.so')
        day = None
        for row in samples:
            if day != row['date']:
                day = row['date']
                (graph, _, _, _, metadata, _), graph_time = timed(dated_graph, data, day, 'gtfs_station_max', True)
                scan, scan_time = timed(TemporalScan, graph, kernel)
                setups[day] = dict(graph=graph_time, scan=scan_time)
                priors = fit['route_priors']
                parameters = {c.route: priors.get(c.route, priors['pooled']) for c in graph.connections}
            witness, journey = witnesses[key(row)], journeys[key(row)[:2]]
            result = evaluate(row, witness, tails[key(row)], graph, scan, metadata, journey, fit, parameters, alpha)
            rows.append(result)
            print(key(row), 'station gaps', [round(result['values']['station'][str(n)]['gap'], 2) for n in SIZES],
                  'corridor/full seconds', *(round(result['timings'][f'{arm}_cold_wall_seconds'], 3)
                                             for arm in ('corridor', 'full')), flush=True)
    summary = {}
    for label in ('zero', 'station', 'station_corridor'):
        summary[label] = {}
        for n in SIZES:
            cases = [r['values'][label][str(n)] for r in rows]
            summary[label][str(n)] = dict(mean_point=mean(x['point'] for x in cases),
                mean_gap=mean(x['gap'] for x in cases), min_gap=min(x['gap'] for x in cases),
                mean_old_gap=mean(x['old_gap'] for x in cases), passes=sum(x['passes'] for x in cases),
                old_passes=sum(x['old_gap'] <= EPSILON for x in cases),
                mean_tail_rescue_gap=mean(x['tail_rescue_gap'] for x in cases),
                tail_rescue_passes=sum(x['tail_rescue_passes'] for x in cases))
    summary['cold_wall_seconds'] = {arm: median(r['timings'][f'{arm}_cold_wall_seconds'] for r in rows)
                                    for arm in ('full', 'corridor')}
    protocol = dict(history_source=str(v9.relative_to(ROOT) / 'sample_results.json'),
                    witness_source=str(v9.relative_to(ROOT) / 'witness_reference_results.json'),
                    tail_rescue_source='tail_rescue_results.json',
                    draws=SIZES, family_alpha=FAMILY_ALPHA, root_alpha=alpha, epsilon=EPSILON,
                    scope='Frozen city development histories under the same independent ordered-trip belief, '
                          'with a fresh independent V10 seed at every root. Evaluate two prefixes of the '
                          'same world stream. Family inference uses only the station-MCT e-mixture. '
                          'Zero-MCT and old formulas are descriptive.',
                    corridor='Cap at min(deadline, t+ceil(U+60)), then prune only trips unable to arrive '
                             'before that cap in a hard-support interval relaxation. The clipped sample '
                             'is pointwise <= the primary sample, so its e-mixture also bounds the '
                             'unclipped station-MCT mean.',
                    timing='Cold wall by phase, with a fresh belief in each arm. Both prefixes derive '
                           'from the single measured 128-world arm per transfer/corridor setting.')
    result = dict(protocol=protocol, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  source=str(source.relative_to(ROOT)), training_dates=fit['train_dates'],
                  setup=dict(data=training_io, training=training, compile=compile_time, dates=setups),
                  summary=summary, rows=rows, accepted_search_controls=accepted_search_controls(rows, data, fit, journeys))
    (HERE / 'matched_gate_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
