"""Eight frozen-root V11 publication/cost comparisons; execution sensitivity only."""

from concurrent.futures import ProcessPoolExecutor
import argparse
from functools import partial
from itertools import permutations
import json
from math import pi
import os
from pathlib import Path
from statistics import mean, median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from experiments.reconstruction_v8.budget_run import SOURCE, write_json
from experiments.reconstruction_v8.issued_replay import unresolved_check
from experiments.reconstruction_v9.search_reference import ARCHIVE, HISTORIES
from experiments.reconstruction_v10.platform_sensitivity import METHODS, adapter, case_arms
from experiments.reconstruction_v10 import issued_run
from src.background_policy import execute_background
from src.continuation_gate import check_continuation
from src.platform_transit import PlatformState, build_platform_model
from src.platform_witness import platform_witness_conditions
from src.tail_rescue import TailRescueBounds, TailRescuePlan
from src.temporal_scan import build_temporal_kernel


def incumbent_replay(graph, actual, metadata, history, journey, fit, goal):
    """Realized marginal value reference: same issued policy, no replacement."""
    decision = history['decision']
    belief, state = context(graph, metadata, journey, decision, fit)
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                  for c in graph.connections}
    bounds = TailRescueBounds(belief, parameters, graph, goal, journey['od']['deadline_seconds'])
    before = perf_counter()
    conditions = platform_witness_conditions(bounds, state, TailRescuePlan(tuple(history['witness'])))
    witness_check_seconds = perf_counter() - before
    assert not conditions, conditions
    incoming = journey['decisions'][decision - 1]['action']
    initial = PlatformState(stop=state.stop, time=state.time, current_trip=state.current_trip,
        last_arrival=state.last_arrival, last_connection=incoming,
        used_connections=state.unavailable_ids, local_since=state.time)
    deadline = journey['od']['deadline_seconds']
    model = build_platform_model(graph, [(1., actual)], state.stop, goal, state.time, deadline,
                                 poll_interval=1, initial_state=initial)
    policy, targets, _ = adapter(graph, tuple(history['witness']), initial, 'incumbent')
    unresolved = unresolved_check(graph, metadata, actual, {'incumbent': targets})
    result = execute_background(model, model.scenarios[0], policy, policy, initial.time,
                                unresolved=unresolved)
    final = result['state']
    censored = result['cost'] is None
    lo = final.time - state.time if censored else result['cost']
    return dict(loss_lower=lo, loss_upper=deadline-state.time if censored else lo,
        witness_check_seconds=witness_check_seconds,
        status='censored' if censored else 'arrived' if final.stop == goal else 'timeout',
        rides=[r for r in result['trace'] if r['ridden_connection'] is not None])


def run_cell(data, fit, cell, indexed_histories, journey, library, cpu, sequential):
    os.sched_setaffinity(0, {cpu})
    if sequential:
        issued_run.SAMPLES = 512
        # One query per fresh Worker: undo its first-call schedule to allocate .05/8.
        issued_run.ALPHA = (.05 / 8) * pi**2 / 6
        issued_run.check_continuation = partial(check_continuation, sample_sizes=(32, 128, 512))
    started = perf_counter()
    graph, actual, stops, _, metadata, _ = dated_graph(data, cell[0], 'gtfs_station_max', True)
    graph_seconds = perf_counter() - started
    started = perf_counter()
    kernel = build_temporal_kernel(library.with_name(f'platform_{cell[0]}_{cell[1]}.so'))
    kernel_seconds = perf_counter() - started
    orders = tuple(permutations(METHODS))
    rows, references = [], []
    for index, history in indexed_histories:
        # Eight roots cannot exactly balance three positions; counts differ by at most one.
        order = orders[(0, 1, 2, 3, 4, 5, 3, 4)[index]]
        current = case_arms(graph, actual, metadata, stops, history, journey, fit, kernel, order)
        reference = incumbent_replay(graph, actual, metadata, history, journey, fit,
                                     stops[journey['od']['destination']])
        references.append(dict(date=cell[0], query=cell[1], decision=history['decision'], **reference))
        for row in current:
            work = row['work']
            row.update(order=list(order), order_index=order.index(row['method']), cpu=cpu,
                setup_plus_worker_seconds=row['setup_seconds'] + row['worker_seconds'],
                search_called=work['search_seconds'] > 0,
                marginal_loss_lower=row['loss_lower']-reference['loss_upper'],
                marginal_loss_upper=row['loss_upper']-reference['loss_lower'],
                useful_published=bool(row['published'] and row['loss_upper'] < reference['loss_lower']),
                harmful_published=bool(row['published'] and row['loss_lower'] > reference['loss_upper']))
        rows.extend(current)
    return dict(date=cell[0], query=cell[1], graph_seconds=graph_seconds,
                native_compile_seconds=kernel_seconds, rows=rows, incumbent_references=references)


def summarize(rows):
    by_key = {(r['date'], r['query'], r['decision'], r['method']): r for r in rows}
    pairs, summary = {}, {}
    for method in METHODS:
        arm = [r for r in rows if r['method'] == method]
        assert len(arm) == 8, 'The frozen protocol includes every one of the eight roots'
        comparisons = []
        for row in arm:
            ref = by_key[row['date'], row['query'], row['decision'], 'off64']
            gate = row['work'].get('gate')
            comparisons.append(dict(date=row['date'], query=row['query'], decision=row['decision'],
                loss_difference_lower=row['loss_lower']-ref['loss_upper'],
                loss_difference_upper=row['loss_upper']-ref['loss_lower'],
                setup_plus_worker_difference=row['setup_plus_worker_seconds']-ref['setup_plus_worker_seconds'],
                worker_difference=row['worker_seconds']-ref['worker_seconds'],
                saved_search=int(ref['search_called'])-int(row['search_called']),
                gate_saved_search=bool(gate and gate['accepted'] and ref['search_called'] and not row['search_called']),
                publication_preserved=row['published'] == ref['published']))
        pairs[method] = comparisons
        summary[method] = dict(roots=len(arm), mean_loss_lower=mean(r['loss_lower'] for r in arm),
            mean_loss_upper=mean(r['loss_upper'] for r in arm), arrived=sum(r['status']=='arrived' for r in arm),
            censored=sum(r['status']=='censored' for r in arm),
            mean_setup_plus_worker_seconds=mean(r['setup_plus_worker_seconds'] for r in arm),
            median_setup_plus_worker_seconds=median(r['setup_plus_worker_seconds'] for r in arm),
            mean_worker_seconds=mean(r['worker_seconds'] for r in arm),
            mean_paired_total_difference=mean(p['setup_plus_worker_difference'] for p in comparisons),
            median_paired_total_difference=median(p['setup_plus_worker_difference'] for p in comparisons),
            search_calls=sum(r['search_called'] for r in arm),
            actually_saved_searches=sum(p['saved_search'] for p in comparisons),
            gate_saved_searches=sum(p['gate_saved_search'] for p in comparisons),
            published=sum(r['published'] for r in arm), rejected=sum(r['rejected'] for r in arm),
            useful_published=sum(r['useful_published'] for r in arm),
            harmful_published=sum(r['harmful_published'] for r in arm),
            mean_marginal_loss_lower=mean(r['marginal_loss_lower'] for r in arm),
            mean_marginal_loss_upper=mean(r['marginal_loss_upper'] for r in arm))
    return summary, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequential', action='store_true')
    args = parser.parse_args()
    output_prefix = 'sequential_platform' if args.sequential else 'platform'
    started = perf_counter()
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    shared_training_seconds = perf_counter() - started
    archive = json.loads(ARCHIVE.read_text())['rows']
    journeys = {(r['date'], r['query']): r for r in archive if r['method']=='evolving_rollout'}
    histories = sorted(json.loads(HISTORIES.read_text())['rows'], key=lambda r:(r['date'], r['query'], r['decision']))
    assert len(histories) == 8
    cells = sorted({(r['date'], r['query']) for r in histories})
    cpus = sorted(os.sched_getaffinity(0))[-len(cells):]
    protocol = dict(histories=str(HISTORIES.relative_to(ROOT)), archive=str(ARCHIVE.relative_to(ROOT)),
        methods=METHODS, order='All six permutations, then permutations 3 and 4; each arm position count differs by at most one, fixed before results.',
        contract='Committed planner and certificate; platform execution sensitivity, not a platform near-optimality guarantee.',
        adapter='Same V10 primary-only policy: visible-departure boarding, explicit alighting, physical onboard inertia; no tail rescue.',
        release='Root+ceil(measured worker wall), same public pose/prefix/availability revalidation for every arm.',
        accounting='Per-arm context/Worker/scan setup plus worker charged; shared graph, fitting and native compilation recorded separately.',
        marginal_reference='Same archived incumbent replay without worker; realized difference diagnoses useful publications, not expected optimality.',
        sampling=('Current shared trip sampler and deterministic floor; cumulative 32/128/512 worlds, stop only at acceptance, same alpha across prefixes.'
                  if args.sequential else 'Current shared trip sampler and deterministic floor; fixed 128 worlds.'),
        tolerance_seconds=60,
        alpha=('Family .05 across eight frozen roots; each root uses .05/8=.00625 for every sequential prefix.'
               if args.sequential else 'V10 regression: each fresh Worker first call uses .05*6/pi^2; no simultaneous eight-root95% claim.'),
        scope='All eight existing development roots, once; no new dates or claim of G2 success.')
    write_json(HERE/f'{output_prefix}_protocol.json', protocol)
    bundles = []
    with TemporaryDirectory(prefix='bapr_v11_platform_') as tmp, ProcessPoolExecutor(max_workers=len(cpus)) as pool:
        tasks = [pool.submit(run_cell, data, fit, cell,
            [(j,h) for j,h in enumerate(histories) if (h['date'],h['query'])==cell], journeys[cell],
            Path(tmp)/'temporal.so', cpus[i], args.sequential) for i,cell in enumerate(cells)]
        for task in tasks:
            bundles.append(task.result())
    rows = [r for b in bundles for r in b['rows']]
    summary, pairs = summarize(rows)
    result = dict(protocol=protocol, shared_training_seconds=shared_training_seconds,
        shared_costs=[{k:v for k,v in b.items() if k not in ('rows','incumbent_references')} for b in bundles],
        rows=rows, incumbent_references=[r for b in bundles for r in b['incumbent_references']],
        summary=summary, paired=pairs)
    write_json(HERE/f'{output_prefix}_results.json', result)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
