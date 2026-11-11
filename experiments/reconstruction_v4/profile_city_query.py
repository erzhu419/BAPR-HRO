"""Profile one full-city decision at the main experiment's sample/work budget."""
import cProfile
import io
import json
from pathlib import Path
import pstats
import resource
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import fit_training, dated_graph, ReplayRouter, ROOT, HERE
from src.evolving_trip_belief import EvolvingTripBelief


def main():
    data, _ = load_city(ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
    fit = fit_training(data, retain_no_pickup=True)
    graph, _, stops, sources, _, _ = dated_graph(data, '2023-07-26', 'gtfs_station_max', True)
    query = data['od_queries'][0]
    priors = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    router = ReplayRouter(graph, priors, event_belief=belief, menu='full', algorithm='rollout',
                          samples=4, work_budget=512, max_time=3600, seed=20230726,
                          source_to_local=sources)
    router.online_mean = router.online_cancel = False
    router._initial_cancels = {route: p['cancel_rate'] for route, p in priors.items()}
    origin, destination = stops[query['origin']], stops[query['destination']]
    router.route(origin, destination, query['ready_seconds'])
    router.set_information_context(origin, query['ready_seconds'], None, query['ready_seconds'], set(), set(), set())
    profile = cProfile.Profile()
    started = perf_counter()
    profile.runcall(router.select_connection, origin, query['ready_seconds'], np.random.default_rng(8000))
    wall = perf_counter() - started
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(18)
    output = {'scope': 'One cold initial public decision on the complete July26city graph, with the same4draws/512transition limit as the main panel. Profiling adds runtime overhead.',
              'wall_seconds': wall, 'worker_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'decision': router.replay_decisions[0], 'search': router.predictions[0],
              'profile_top18_cumulative': stream.getvalue()}
    path = HERE / 'city_root_profile.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(path, flush=True)


if __name__ == '__main__':
    main()
