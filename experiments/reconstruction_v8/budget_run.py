"""Matched city work-budget and charged-execution curves on two used dates."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import ReplayRouter, dated_graph, fit_training, run_one, summarize
from experiments.reconstruction_v6.run_itinerary_replay import ItineraryReplayRouter
from src.evolving_trip_belief import EvolvingTripBelief
from src.indexed_nominal_tail import IndexedNominalTailRouter


METHODS = ('compiled_nominal', 'analytic_recourse', 'uct64', 'uct128', 'uct512')
SOURCE = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
_REPLAY_INPUTS = None


class IndexedReplayRouter(ReplayRouter, IndexedNominalTailRouter):
    def select_connection(self, *args, **kwargs):
        selected = super().select_connection(*args, **kwargs)
        prediction = self.predictions[-1]
        self.replay_decisions[-1].update({name: prediction[name] for name in (
            'value', 'completed_rollouts', 'scored_root_actions', 'tree_nodes')})
        self.replay_decisions[-1]['nominal_index'] = self.nominal_index_stats
        return selected


class CompiledNominalReplayRouter(ItineraryReplayRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, optimize=False, max_itineraries=8, **kwargs)


class AnalyticReplayRouter(ItineraryReplayRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, optimize=True, max_itineraries=8, **kwargs)


def pilot_one(data, fit, day, qi, budget, indexed):
    started = perf_counter()
    graph, _, stops, _, _, _ = dated_graph(data, day, 'gtfs_station_max', True)
    query = data['od_queries'][qi]
    priors = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    router_type = IndexedReplayRouter if indexed else ReplayRouter
    router = router_type(graph, priors, algorithm='uct', work_budget=budget, samples=4,
        seed=int(day.replace('-', '')) + qi * 1000,
        max_time=query['deadline_seconds'] - query['ready_seconds'], event_belief=belief)
    router.online_mean = router.online_cancel = False
    router._initial_cancels = {route: row['cancel_rate'] for route, row in priors.items()}
    origin, destination = stops[query['origin']], stops[query['destination']]
    clock = query['ready_seconds']
    router.route(origin, destination, clock)
    router.set_execution_context(None, clock, set())
    router.set_information_context(origin, clock, None, clock, set(), set(), set())
    setup_seconds = perf_counter() - started
    wall, cpu = perf_counter(), process_time()
    router.select_connection(origin, clock, np.random.default_rng(8000 + qi))
    return dict(date=day, query=qi, budget=budget, indexed=indexed,
        setup_seconds=setup_seconds, decision_wall_seconds=perf_counter() - wall,
        decision_cpu_seconds=process_time() - cpu, prediction=router.predictions[-1],
        nominal_index=getattr(router, 'nominal_index_stats', None))


def execute(spec, data, forecasts, fit):
    day, qi, method, clock = spec
    if method.startswith('uct'):
        router_type, algorithm = IndexedReplayRouter, 'uct'
        budget = int(method[3:])
    elif method == 'compiled_nominal':
        router_type, algorithm, budget = CompiledNominalReplayRouter, None, 64
    else:
        router_type, algorithm, budget = AnalyticReplayRouter, None, 64
    started = process_time()
    row = run_one((day, qi, method), data, forecasts, fit, 4, budget,
        'gtfs_station_max', True, algorithm, router_type=router_type,
        charge_computation=clock == 'charged')
    row.update(clock=clock, cpu_seconds=process_time() - started,
        transition_budget_per_decision=budget if algorithm == 'uct' else None,
        candidate_itinerary_limit=8 if method == 'analytic_recourse' else None)
    return row


def initialize_worker(data, forecasts, fit):
    global _REPLAY_INPUTS
    _REPLAY_INPUTS = data, forecasts, fit


def replay_case(spec):
    return execute(spec, *_REPLAY_INPUTS)


def metadata(data, fit, source_files, settings):
    return dict(scope='Already-exposed July26/27 development and V6 committed contract.',
        source=str(SOURCE.relative_to(ROOT)), days=data['holdout_dates'],
        training=fit, queries=data['od_queries'], settings=settings, python=platform.python_version(),
        numpy=np.__version__, cpu_affinity=sorted(os.sched_getaffinity(0)),
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources={f: (ROOT / f).read_text() for f in source_files})


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    data, _ = load_city(SOURCE)
    started = perf_counter()
    fit = fit_training(data, retain_no_pickup=True)
    fit_seconds = perf_counter() - started
    source_files = [str(Path(__file__).relative_to(ROOT)), 'src/indexed_nominal_tail.py',
        'src/causal_planning_router.py', 'src/continuation_router.py', 'src/evolving_trip_belief.py',
        'src/committed_transit.py', 'src/simulate_bandit.py', 'src/recourse_itinerary_router.py',
        'src/risk_bounds.py', 'experiments/reconstruction_v4/run_dated_replay.py']
    if args.pilot:
        protocol = metadata(data, fit, source_files, dict(mode='cold root equality and timing', budgets=[64, 512]))
        write_json(HERE / 'budget_pilot_protocol.json', protocol)
        rows = []
        for qi, budget in ((0, 64), (1, 512)):
            pair = [pilot_one(data, fit, data['holdout_dates'][0], qi, budget, indexed)
                    for indexed in (False, True)]
            key = lambda row: {k: v for k, v in row['prediction'].items() if k != 'planning_ms'}
            if key(pair[0]) != key(pair[1]):
                raise AssertionError(f'Optimized UCT changes root prediction in {pair}')
            rows.extend(pair)
            print(qi, budget, [(r['indexed'], round(r['decision_cpu_seconds'], 3)) for r in pair], flush=True)
        write_json(HERE / 'budget_pilot_results.json', rows)
        return
    protocol = metadata(data, fit, source_files, dict(workers=args.workers, methods=list(METHODS),
        clocks=['logical', 'charged'], samples=4, max_itineraries=8, transfer_rule='gtfs_station_max',
        transfer_time_scope='V6 operational station buffer', thread_environment={name: os.environ.get(name) for name in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        timing='Wall and CPU include graph/world/router construction and all decision calls. Common past-only fit separate.'))
    write_json(HERE / 'budget_protocol.json', protocol)
    forecasts = {'dates': {day: {'frames': []} for day in data['holdout_dates']}}
    folder = HERE / 'budget_cases'
    folder.mkdir(exist_ok=True)
    specs = [(day, qi, method, clock) for clock in ('logical', 'charged')
             for method in METHODS for day in data['holdout_dates'] for qi in range(len(data['od_queries']))]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(data, forecasts, fit)) as pool:
        jobs = {pool.submit(replay_case, spec): spec for spec in specs}
        for future in as_completed(jobs):
            spec = jobs[future]
            row = future.result()
            write_json(folder / ('_'.join(map(str, spec)) + '.json'), row)
            rows.append(row)
            print(*spec, row['status'], row['loss_seconds'], round(row['end_to_end_ms'] / 1000, 3), flush=True)
    rows.sort(key=lambda r: (r['clock'], r['date'], r['query'], r['method']))
    summaries = {clock: summarize([r for r in rows if r['clock'] == clock], METHODS)
                 for clock in ('logical', 'charged')}
    write_json(HERE / 'budget_results.json', dict(protocol='budget_protocol.json', fit_seconds=fit_seconds,
               summaries=summaries, rows=rows))


if __name__ == '__main__':
    main()
