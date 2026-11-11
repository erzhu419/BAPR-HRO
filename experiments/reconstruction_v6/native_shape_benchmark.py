"""Compare exact PMF-array interning against the original full native query."""
import json
from pathlib import Path
import sys
from time import perf_counter, process_time
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from src.durner.native import native_topocsa
from src.evolving_trip_belief import EvolvingTripBelief
from src.native_event_router import NativeEventRouter, minute_snapshot


def main():
    data, _ = load_city(ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
    fit = fit_training(data, retain_no_pickup=True)
    graph, _, stops, _, _, _ = dated_graph(data, '2023-07-26', 'gtfs_station_max', True)
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    pars = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    temporary, work = minute_snapshot(belief, pars)
    query = data['od_queries'][0]
    source, goal = stops[query['origin']], stops[query['destination']]
    records, results = [], []
    for variant in ('separate_arrays', 'interned_exact_arrays'):
        before = perf_counter()
        if variant.startswith('interned'):
            shapes = {}
            for c in temporary.connections:
                for pmf in (c.dep_distribution, c.arr_distribution):
                    pmf.probs = shapes.setdefault(pmf.probs.tobytes(), pmf.probs)
            shape_count = len(shapes)
        else:
            # The production adapter now interns; reconstruct its original
            # separate-array representation for a repeat of this comparison.
            for c in temporary.connections:
                for pmf in (c.dep_distribution, c.arr_distribution):
                    pmf.probs = pmf.probs.copy()
            shape_count = 2 * len(temporary.connections)
        coalesce_seconds = perf_counter() - before
        before, cpu = perf_counter(), process_time()
        result = native_topocsa(temporary, source, goal, query['ready_seconds'] // 60,
                                (query['deadline_seconds'] + 59) // 60)
        elapsed, cpu = perf_counter() - before, process_time() - cpu
        records.append(dict(variant=variant, exact_arrays=shape_count,
            coalesce_seconds=coalesce_seconds, query_seconds=elapsed, query_cpu_seconds=cpu))
        results.append(result)
        print(records[-1], flush=True)
    a, b = results
    identical = a.cuts == b.cuts and a.n_connections_processed == b.n_connections_processed
    for stop in a.stop_labels:
        first, second = a.stop_labels[stop], b.stop_labels[stop]
        identical &= len(first) == len(second)
        for x, y in zip(first, second):
            identical &= (x.connection_id == y.connection_id and x.feasibility == y.feasibility
                and x.mean_dest_arrival == y.mean_dest_arrival and x.dest_arrival.offset == y.dest_arrival.offset
                and np.array_equal(x.dest_arrival.probs, y.dest_arrival.probs))
    choices = []
    for result in results:
        router = NativeEventRouter(graph, pars, event_belief=belief, menu='full', algorithm='uct',
            work_budget=64, samples=4, max_time=query['deadline_seconds'] - query['ready_seconds'])
        router.online_mean = router.online_cancel = False
        router._initial_cancels = {r: p['cancel_rate'] for r, p in pars.items()}
        router.route(source, goal, query['ready_seconds'])
        router.set_information_context(source, query['ready_seconds'], None, query['ready_seconds'], set(), set(), set())
        with patch('src.native_event_router.minute_snapshot', return_value=(temporary, {})), \
             patch('src.native_event_router.native_topocsa', return_value=result):
            router.select_connection(source, query['ready_seconds'], np.random.default_rng(0))
        choices.append(router.predictions[-1]['native_selected'])
    output = dict(scope='Same cold posterior snapshot, no source pruning or probability approximation',
        snapshot=work, records=records, all_native_labels_cuts_values_pmfs_bitwise_equal=bool(identical),
        capped_root_choices=choices, capped_root_choice_equal=choices[0] == choices[1])
    (HERE / 'native_shape_results.json').write_text(json.dumps(output, indent=2) + '\n')
    print({k:v for k,v in output.items() if k not in ('snapshot', 'records')}, flush=True)
    assert identical and choices[0] == choices[1]


if __name__ == '__main__':
    main()
