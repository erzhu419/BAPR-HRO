"""Matched full-support temporal comparators on the eight frozen V8 contexts."""

from collections import defaultdict
from math import log
from pathlib import Path
from statistics import mean, median
from time import perf_counter, process_time
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from src.evolving_trip_belief import logsumexp
from src.temporal_support import IntervalOperatingSupport, JointOperatingSupport, TemporalReachability


def route_atom(support, path, start):
    """Evaluate the selected route's exact clocks without adding grid anchors."""
    pins = defaultdict(dict)
    departures, ready = [], start
    for cid in path:
        dep, _ = support.clocks(cid)
        departure, arrival = int(dep[dep.searchsorted(ready)]), support(cid, ready)
        trip, slot = support.belief._positions[cid, 'departure']
        pins[trip].update({slot: departure, slot + 1: arrival})
        departures.append(departure)
        ready = arrival
    log_probability = 0.
    for trip, clocks in pins.items():
        law = support.belief._law(trip, support.parameters)
        for i, emission in enumerate(law.emissions):
            prefix = (law.initial if i == 0 else
                      logsumexp(weights[:, None] + law._transition(i - 1), axis=0))
            weights = prefix + emission
            if i in clocks:
                weights = np.where((law.times[i] == clocks[i]) & ~law.canceled[i], weights, -np.inf)
        log_probability += float(logsumexp(weights) - logsumexp(law.forward[-1]))
    return dict(path_departures=departures, path_distinct_trips=len(pins),
                path_clock_atom_log10_probability=(log_probability / log(10)
                                                   if path and np.isfinite(log_probability) else None))


def evaluate(reachability, build_support, belief, parameters, state, destination, deadline):
    started, cpu = perf_counter(), process_time()
    support = build_support(belief, parameters)
    label = reachability.solve(state, destination, deadline, support)
    wall_seconds, cpu_seconds = perf_counter() - started, process_time() - cpu
    arrivals, ready = [], state.time
    for cid in label.path:
        ready = support(cid, ready)
        arrivals.append(ready)
    result = dict(lower_seconds=label.arrival - state.time, path=label.path, path_arrivals=arrivals,
                  edge_queries=label.edge_queries, settled_labels=label.settled_labels,
                  wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)
    if isinstance(support, JointOperatingSupport):
        result.update(trips_built=support.trips_built, grid_atoms=support.grid_atoms,
                      **route_atom(support, label.path, state.time))
    return result


def main():
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    archive_path = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    frozen_path = ROOT / 'experiments/reconstruction_v8/opportunity_city_results.json'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    archive = json.loads(archive_path.read_text())
    journeys = {(r['date'], r['query']): r for r in archive['rows'] if r['method'] == 'evolving_rollout'}
    frozen = json.loads(frozen_path.read_text())['rows']
    rows, setup = [], {}
    for day in dict.fromkeys(case['date'] for case in frozen):
        graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                      for c in graph.connections}
        started = perf_counter()
        reachability = TemporalReachability(graph)
        setup[day] = dict(graph_connections=len(graph.connections), index_wall_seconds=perf_counter() - started)
        for case in (c for c in frozen if c['date'] == day):
            journey = journeys[day, case['query']]
            destination = stops[journey['od']['destination']]
            deadline = journey['od']['deadline_seconds']
            started = perf_counter()
            belief, state = context(graph, metadata, journey, case['decision'], fit)
            context_wall_seconds = perf_counter() - started
            assert (state.stop, state.time, destination) == (case['stop'], case['clock'], case['destination'])
            results = {key: evaluate(reachability, method, belief, parameters, state, destination, deadline)
                       for key, method in [('interval', IntervalOperatingSupport), ('joint', JointOperatingSupport)]}
            interval, joint = (results[key]['lower_seconds'] for key in ('interval', 'joint'))
            paired_worlds = [sample['full_temporal_seconds'] for sample in case['paired_worlds']]
            assert interval <= joint <= min(paired_worlds) and joint <= case['upper'] + 1e-8
            combined = max(joint, case['lower'])
            row = dict(date=day, query=case['query'], decision=case['decision'], stop=state.stop,
                       clock=state.time, destination=destination, deadline=deadline, upper_seconds=case['upper'],
                       v8_first_exit_lower_seconds=case['lower'], combined_lower_seconds=combined,
                       combined_gap_seconds=case['upper'] - combined, v8_full_temporal_world_seconds=paired_worlds,
                       context_wall_seconds=context_wall_seconds, **results)
            rows.append(row)
            print(f'{day}/{case["query"]}/{case["decision"]} interval {interval}s, joint {joint}s, '
                  f'V8 {case["lower"]:.2f}s, joint {results["joint"]["wall_seconds"]:.3f}s', flush=True)
    summary = {key: dict(mean_lower_seconds=mean(r[key]['lower_seconds'] for r in rows),
                         median_wall_seconds=median(r[key]['wall_seconds'] for r in rows),
                         max_wall_seconds=max(r[key]['wall_seconds'] for r in rows))
               for key in ('interval', 'joint')}
    summary.update(v8_mean_lower_seconds=mean(r['v8_first_exit_lower_seconds'] for r in rows),
                   mean_combined_lower_seconds=mean(r['combined_lower_seconds'] for r in rows),
                   min_combined_gap_seconds=min(r['combined_gap_seconds'] for r in rows),
                   mean_combined_gap_seconds=mean(r['combined_gap_seconds'] for r in rows),
                   joint_improves_v8=sum(r['joint']['lower_seconds'] > r['v8_first_exit_lower_seconds'] for r in rows),
                   positive_joint_route_atoms=sum(r['joint']['path_clock_atom_log10_probability'] is not None
                                                  for r in rows),
                   combined_gaps_within_seconds={t: sum(r['combined_gap_seconds'] <= t for r in rows)
                                                 for t in (1, 30, 60)})
    protocol = dict(source=str(source.relative_to(ROOT)), archive=str(archive_path.relative_to(ROOT)),
                    frozen_cases_and_upper=str(frozen_path.relative_to(ROOT)), training_dates=fit['train_dates'],
                    scope='Probability-free all-root-action temporal lower comparator, combined with the frozen V8 bound.',
                    assumptions='Same EvolvingTripBelief grid and observations, zero transfer and no positive runtime floor, full graph with no footpaths. Edge times need only individual positive support, not a shared trip realization along a route.',
                    timing='Adjacency setup once per date. Both method timings include cold support construction per public context, exclude replay, and share no warmed probabilistic inference.',
                    route_atom='For the joint-support path, separately compute the probability of its selected operating clock vector on the original frozen grids, in log10 scale. This is one path and clock vector, not a probability of optimal arrival by the bound. The check is outside the comparator timing.')
    result = dict(protocol=protocol, setup_by_date=setup, summary=summary, rows=rows)
    (HERE / 'support_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
