"""Cold public first query and retrospective probability check for fixed fits."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, ReplayRouter
from experiments.reconstruction_v5.runtime_transition_bias import catch_metrics, profile_trip
from src.evolving_trip_belief import EvolvingTripBelief


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('grid60', 'cdf20'), required=True)
    args = parser.parse_args()
    fit_record = json.loads((HERE / 'ordered_duration_training_fit.json').read_text())
    fit = fit_record['candidate_training' if args.model == 'grid60' else 'continuous_training']
    if args.model == 'cdf20':
        from src.ordered_duration_belief import OrderedDurationBelief
        belief_type = OrderedDurationBelief
    else:
        belief_type = EvolvingTripBelief
    data, _ = load_city(ROOT / fit_record['input'])
    day, qi = '2023-07-27', 0
    graph, actual, stops, source_map, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
    query = data['od_queries'][qi]
    priors = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    belief = belief_type(graph, minute=60, public_occurrences=False, **fit['increments'])
    router = ReplayRouter(graph, priors, event_belief=belief, menu='full', algorithm='uct', samples=4,
                          work_budget=64, max_time=query['deadline_seconds'] - query['ready_seconds'],
                          seed=20230727, frames=(), source_to_local=source_map,
                          forecast_sd={c.id: priors[c.route]['std'] * 60 for c in graph.connections})
    router.online_mean = router.online_cancel = False
    router._initial_cancels = {r: p['cancel_rate'] for r, p in priors.items()}
    origin, clock = stops[query['origin']], query['ready_seconds']
    wall, cpu = perf_counter(), process_time()
    router.route(origin, stops[query['destination']], clock)
    router.set_execution_context(None, clock, set())
    router.set_information_context(origin, clock, None, clock, set(), set(), set())
    selected = router.select_connection(origin, clock, np.random.default_rng(8000))
    timing = {'wall_seconds': perf_counter() - wall, 'cpu_seconds': process_time() - cpu}
    # Realized clocks are used only below, after selection, for retrospective diagnosis.
    scenes = belief.sample(priors, np.random.default_rng(np.random.SeedSequence([20230727, 0])), 4)
    catches = [catch_metrics(cid, clock, belief, priors, scenes, graph, actual) for cid in (7588, 8341)]
    trips = [profile_trip(graph.connections[cid].trip_id, fit, graph, metadata, priors, belief, False)
             for cid in (7588, 8341)]
    result = {'scope': 'Cold initial public UCT64 query; actual endpoint comparisons computed only after action.',
              'duration_model': args.model, 'fit': 'ordered_duration_training_fit.json', 'date': day,
              'query': qi, 'timing': timing, 'selected': None if selected is None else selected[0].connection_id,
              'planning': router.predictions[-1], 'posterior_and_bank': catches, 'trip_comparison': trips}
    output = HERE / f'runtime_fit_probe_{args.model}.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
