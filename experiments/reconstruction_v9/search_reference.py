"""Cold indexed UCT on the V8 archived intermediate histories."""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from experiments.reconstruction_v8.budget_run import SOURCE, write_json
from src.causal_planning_router import BudgetedPolicySearch
from src.committed_transit import build_committed_model
from src.indexed_nominal_tail import IndexedNominalTailRouter

ARCHIVE = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
HISTORIES = ROOT / 'experiments/reconstruction_v8/opportunity_city_results.json'


def elapsed(wall, cpu):
    return {'wall_seconds': perf_counter() - wall, 'cpu_seconds': process_time() - cpu}


def cold_search(graph, metadata, stops, fit, journey, number, budget):
    wall, cpu = perf_counter(), process_time()
    belief, state = context(graph, metadata, journey, number, fit)
    priors = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
              for c in graph.connections}
    query, day = journey['query'], journey['date']
    od = journey['od']
    departure, deadline = od['ready_seconds'], od['deadline_seconds']
    seed = int(day.replace('-', '')) + query * 1000
    router = IndexedNominalTailRouter(graph, priors, algorithm='uct', work_budget=budget,
        samples=4, seed=seed, max_time=deadline - departure, event_belief=belief)
    router.online_mean = router.online_cancel = False
    router._initial_cancels = {route: row['cancel_rate'] for route, row in priors.items()}
    router.route(stops[od['origin']], stops[od['destination']], departure)
    history_setup = elapsed(wall, cpu)

    # Same sequence as select_connection, split before tail/index construction.
    wall, cpu = perf_counter(), process_time()
    parameters = router._parameters()
    rng = np.random.default_rng(np.random.SeedSequence([seed, number]))
    scenes = belief.sample(parameters, rng, router.samples)
    belief.mean_events(parameters)
    model = build_committed_model(graph, [(1 / router.samples, scene) for scene in scenes],
        router.origin, router.destination, departure, deadline, candidate_ids=router.menu,
        lookahead=router.lookahead, initial_state=state)
    model_setup = elapsed(wall, cpu)

    wall, cpu = perf_counter(), process_time()
    planner = BudgetedPolicySearch(model, belief, router._nominal_tail(parameters),
        router._advance_belief, rng, budget, router.max_time)
    result = planner.plan(router.algorithm)
    planning = elapsed(wall, cpu)
    return dict(date=day, query=query, decision=number, budget=budget, clock=state.time,
        stop=state.stop, destination=router.destination, seed=[seed, number],
        history_setup=history_setup, model_setup=model_setup, planning=planning,
        prediction=asdict(result), nominal_index=router.nominal_index_stats)


def main():
    data, _ = load_city(SOURCE)
    wall, cpu = perf_counter(), process_time()
    fit = fit_training(data, retain_no_pickup=True)
    shared_fit = elapsed(wall, cpu)
    archived = json.loads(ARCHIVE.read_text())['rows']
    journeys = {(r['date'], r['query']): r for r in archived if r['method'] == 'evolving_rollout'}
    histories = json.loads(HISTORIES.read_text())['rows']
    graphs, shared_graph = {}, {}
    for day in sorted({row['date'] for row in histories}):
        wall, cpu = perf_counter(), process_time()
        graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        graphs[day] = graph, metadata, stops
        shared_graph[day] = elapsed(wall, cpu)
    for budget in (64, 128):
        rows = []
        for row in histories:
            day, query, number = row['date'], row['query'], row['decision']
            result = cold_search(*graphs[day], fit, journeys[day, query], number, budget)
            if any(result[key] != row[key] for key in ('clock', 'stop', 'destination')):
                raise AssertionError('The reconstructed root differs from the V8 archive')
            rows.append(result)
            print(day, query, number, budget, round(result['planning']['cpu_seconds'], 3),
                  result['prediction']['action'], flush=True)
        stages = ('history_setup', 'model_setup', 'planning')
        summary = {stage: {clock: median(r[stage][clock] for r in rows)
                           for clock in ('wall_seconds', 'cpu_seconds')} for stage in stages}
        write_json(HERE / f'search_reference_{budget}.json', dict(
            scope='Previously exposed city histories with frozen four-world banks and fresh belief/router caches per case.',
            archive=str(ARCHIVE.relative_to(ROOT)), histories=str(HISTORIES.relative_to(ROOT)),
            source=str(SOURCE.relative_to(ROOT)), budget=budget, train_dates=fit['train_dates'],
            model='V8 committed station execution contract', cpu_affinity=sorted(os.sched_getaffinity(0)),
            samples=4, lookahead_seconds=1500, shared_fit=shared_fit, shared_graph=shared_graph,
            median_timing=summary, rows=rows))


if __name__ == '__main__':
    main()
