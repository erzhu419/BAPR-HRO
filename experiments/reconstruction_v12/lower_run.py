"""Bounded full-action causal-prefix diagnostics on the three frozen roots."""
from dataclasses import asdict
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
from experiments.reconstruction_v8.budget_run import SOURCE
from experiments.reconstruction_v9.search_reference import ARCHIVE, HISTORIES
from experiments.reconstruction_v12.transfer_inventory import CASES, key
from src.causal_prefix import causal_prefix_lower, PrefixBudgetExceeded
from src.continuation_confidence import continuation_lower
from src.platform_transit import PlatformState, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map

BANKS, BANK_SIZE, HORIZONS, MAX_NODES = 16, 4, (60, 180), 20000


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    protocol = dict(cases=CASES, banks=BANKS, worlds_per_bank=BANK_SIZE, horizons=HORIZONS,
        node_budget=MAX_NODES, cpu=cpu, seed_rule='[20261112, int(date), query, decision]',
        bank='Disjoint IID groups from the full frozen independent-trip posterior; no corridor pruning.',
        prefix='All physical platform actions; stop at first ridden connection or fixed elapsed horizon.',
        tail='Full-graph clairvoyant station-MCT scan respecting current transfer phase.',
        interpretation='Each bank optimum is downward biased for the population causal-prefix optimum. '
            'The paired increase also includes delaying physical relaxations, not only information value. '
            'Singleton support measures accidental identification in a sampled bank.',
        budget='A node-budget failure yields no value. Stop further banks for that root/horizon; '
            'retain the failed work and do not compute population summaries for incomplete panels.',
        inference='Optional e-mixture lower bounds use independent bank optima and disjoint world losses; '
            'alpha=.05/(3 roots * 2 horizons * 2 estimators). These are not claims about real traffic.')
    write('lower_protocol.json', protocol)
    before = perf_counter()
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    shared_training_seconds = perf_counter() - before
    histories = {key(r): r for r in json.loads(HISTORIES.read_text())['rows']}
    journeys = {(r['date'], r['query']): r for r in json.loads(ARCHIVE.read_text())['rows']
                if r['method'] == 'evolving_rollout'}
    rows, setups = [], []
    with TemporaryDirectory(prefix='bapr_v12_lower_') as tmp:
        before = perf_counter()
        kernel = build_temporal_kernel(Path(tmp) / 'scan.so')
        compile_seconds = perf_counter() - before
        for day, query, decision in CASES:
            before = perf_counter()
            graph, _, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
            history, journey = histories[day, query, decision], journeys[day, query]
            belief, state = context(graph, metadata, journey, decision, fit)
            goal, deadline = history['destination'], journey['od']['deadline_seconds']
            incoming = journey['decisions'][decision - 1]['action']
            initial = PlatformState(state.stop, state.time, state.current_trip, state.last_arrival,
                last_connection=incoming, used_connections=state.unavailable_ids, local_since=state.time)
            parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                          for c in graph.connections}
            scan = TemporalScan(graph, kernel)
            transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
            setup_seconds = perf_counter() - before
            before = perf_counter()
            packed = sample_clocks(belief, parameters,
                np.random.default_rng(np.random.SeedSequence([20261112, int(day.replace('-', '')), query, decision])),
                BANKS * BANK_SIZE, templates=template_trip_map(belief, parameters))
            draw_seconds = perf_counter() - before
            setups.append(dict(date=day, query=query, decision=decision,
                               setup_seconds=setup_seconds, draw_seconds=draw_seconds))
            stopped = set()
            for bank_index in range(BANKS):
                start = bank_index * BANK_SIZE
                clocks = tuple(x[start:start + BANK_SIZE] for x in packed)
                before = perf_counter()
                scenes = [(1. / BANK_SIZE, JourneyScenario({c.id: VehicleEvent(
                    int(dep[c.id]) - c.dep_time, bool(canceled[c.id]), int(dep[c.id]), int(arr[c.id]))
                    for c in graph.connections}, minute=60, publications={}))
                    for dep, arr, canceled in zip(*clocks)]
                model = build_platform_model(graph, scenes, state.stop, goal, state.time, deadline,
                                             initial_state=initial, poll_interval=1)
                model_seconds = perf_counter() - before

                def oracle(index, public):
                    one = tuple(x[index:index + 1] for x in clocks)
                    return float(scan.platform_arrivals(*one, public, goal, deadline, transfers)[0] - public.time)

                for horizon in HORIZONS:
                    if horizon in stopped:
                        continue
                    before = perf_counter()
                    row = dict(date=day, query=query, decision=decision, bank_index=bank_index,
                               horizon=horizon, cap=deadline-state.time, model_seconds=model_seconds)
                    try:
                        value = causal_prefix_lower(model, oracle, elapsed_horizon=horizon, max_nodes=MAX_NODES)
                        assert value.value >= value.clairvoyant_root_mean - 1e-7
                        row.update(status='complete', **asdict(value),
                                   paired_increment=value.value-value.clairvoyant_root_mean)
                    except PrefixBudgetExceeded as error:
                        row.update(status='budget_exceeded', information_nodes=error.nodes,
                                   transition_evaluations=error.transitions)
                        stopped.add(horizon)
                    row['solver_seconds'] = perf_counter() - before
                    rows.append(row)
                    with (HERE / 'lower_progress.jsonl').open('a') as log:
                        log.write(json.dumps(row) + '\n')
                    print(day, decision, bank_index, horizon, row['status'],
                          round(row.get('paired_increment', 0.), 3), round(row['solver_seconds'], 3), flush=True)
                if len(stopped) == len(HORIZONS):
                    break
    summary = []
    alpha = .05 / (len(CASES) * len(HORIZONS) * 2)
    for case in CASES:
        for horizon in HORIZONS:
            arm = [r for r in rows if key(r) == case and r['horizon'] == horizon]
            done = [r for r in arm if r['status'] == 'complete']
            item = dict(date=case[0], query=case[1], decision=case[2], horizon=horizon,
                        completed_banks=len(done), complete=len(done) == BANKS,
                        total_solver_seconds=sum(r['solver_seconds'] for r in arm))
            if item['complete']:
                values = np.array([r['value'] for r in done])
                roots = np.array([r['clairvoyant_root_mean'] for r in done])
                item.update(mean_prefix=float(values.mean()), mean_clairvoyant=float(roots.mean()),
                    mean_increment=float((values-roots).mean()),
                    prefix_lower=continuation_lower(values, done[0]['cap'], alpha),
                    root_bank_mean_lower=continuation_lower(roots, done[0]['cap'], alpha), alpha=alpha,
                    mean_singleton_before_boundary=float(np.mean([
                        r['singleton_before_boundary_probability'] for r in done])),
                    mean_singleton_at_boundary=float(np.mean([r['singleton_boundary_probability'] for r in done])))
            summary.append(item)
    write('lower_results.json', dict(protocol=protocol, rows=rows, summary=summary,
        shared_training_seconds=shared_training_seconds, compile_seconds=compile_seconds, setups=setups))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
