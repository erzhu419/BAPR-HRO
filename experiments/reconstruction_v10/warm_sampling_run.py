"""Profile exact posterior reuse across successive V9 public histories."""

import json
from math import ceil
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_chain_boundaries import apply_receipt
from experiments.reconstruction_v5.run_sparse_certificates import context
from src.temporal_scan import sample_clocks, template_trip_map, window_trips


def sample(belief, parameters, state, cap, seed, n):
    start = perf_counter()
    templates = template_trip_map(belief, parameters)
    trips = window_trips(belief, parameters, state.time, cap)
    support = perf_counter() - start
    clocks = sample_clocks(belief, parameters, np.random.default_rng(np.random.SeedSequence(seed)), n,
                           trips=trips, templates=templates)
    return clocks, dict(seconds=perf_counter() - start, support_template_seconds=support, trips=len(trips))


def main():
    data, _ = load_city(ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
    fit = fit_training(data, retain_no_pickup=True)
    archive = json.loads((ROOT / 'experiments/reconstruction_v4/city_replay_results.json').read_text())['rows']
    journeys = {(j['date'], j['query']): j for j in archive if j['method'] == 'evolving_rollout'}
    v9 = ROOT / 'experiments/reconstruction_v9'
    witnesses = {(r['date'], r['query'], r['decision']): r['best_upper_seconds']
                 for r in json.loads((v9 / 'witness_reference_results.json').read_text())['rows']}
    day, n, rows = None, 32, []
    live = {}
    for row in json.loads((v9 / 'sample_results.json').read_text())['rows']:
        key = row['date'], row['query'], row['decision']
        if day != key[0]:
            day = key[0]
            graph, _, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
            parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                          for c in graph.connections}
            live.clear()
        journey = journeys[key[:2]]
        cold, state = context(graph, metadata, journey, key[2], fit)
        previous = live.get(key[1])
        warm = previous is not None
        if warm:
            start, last_arrival, forward = previous
            for decision in journey['decisions'][start:key[2]]:
                last_arrival = apply_receipt(forward, graph, metadata, decision, last_arrival)
            assert last_arrival == state.last_arrival
        else:
            forward, _ = context(graph, metadata, journey, key[2], fit)
        live[key[1]] = key[2], state.last_arrival, forward
        cap = min(journey['od']['deadline_seconds'], state.time + ceil(witnesses[key] + 180))
        seed = [*row['random_seed'], 10, 18, n]
        arms = [('cold', cold), ('forward', forward)]
        if key[2] % 2:
            arms.reverse()
        values = {name: sample(b, parameters, state, cap, seed, n) for name, b in arms}
        for reference, actual in zip(values['cold'][0], values['forward'][0]):
            np.testing.assert_array_equal(actual, reference)
        rows.append(dict(date=day, query=key[1], decision=key[2], warm=warm, cap_seconds=cap - state.time,
                         **{k: v[1] for k, v in values.items()}))
        print(day, key[1:], 'warm', warm, {k: round(v[1]['seconds'], 3) for k, v in values.items()}, flush=True)
    summary = {label: {arm: median(r[arm]['seconds'] for r in rows if r['warm'] == warm)
                       for arm in ('cold', 'forward')} for label, warm in [('first', False), ('subsequent', True)]}
    result = dict(scope='Rebuild the cold belief at each V9 history. Warm arms apply only newly observed receipts to '
                        'the last charged belief in the same voyage. Use fresh independent tapes across histories, '
                        'identical between arms. Time includes window, templates, and cold population of caches.',
                  draws_per_root=n, summary=summary, rows=rows)
    (HERE / 'warm_sampling_results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
