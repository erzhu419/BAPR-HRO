"""Replay V10 plans on the V8 local-event publisher at the eight frozen roots."""

from concurrent.futures import ProcessPoolExecutor
from math import ceil
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from experiments.reconstruction_v8.budget_run import SOURCE, write_json
from experiments.reconstruction_v8.issued_replay import path_targets, unresolved_check
from experiments.reconstruction_v9.search_reference import ARCHIVE, HISTORIES
from experiments.reconstruction_v10.issued_run import Worker
from src.background_policy import IssuedPolicy, execute_background
from src.platform_transit import PlatformState, build_platform_model
from src.tail_rescue import freeze_tail_plan
from src.temporal_scan import TemporalScan, build_temporal_kernel


METHODS = ('off64', 'terminal64', 'middle64')


def adapter(graph, path, root, name, *, wait_until=None):
    targets = path_targets(graph, path, (), root.time)
    release_check = {}

    def action(state):
        options = targets(state)
        if wait_until is not None and state.used_connections == root.used_connections and state.time < wait_until:
            return ('wait',)
        if options and state.current_trip is not None and all(
                graph.connections[cid].trip_id != state.current_trip for cid in options):
            return ('alight',)
        return next((('board', cid) for cid in options if cid in state.visible_connections), ('wait',))

    def applicable(state):
        same_pose = (state.stop, state.current_trip, state.last_connection, state.last_arrival) == (
            root.stop, root.current_trip, root.last_connection, root.last_arrival)
        same_prefix = state.used_connections == root.used_connections
        gone = {cid for clock, cid, event in state.public_log if event == 'published_cancel'
                or (event == 'departure' and clock < state.time)}
        available = any(cid not in gone for cid in targets(state))
        release_check.update(clock=state.time, stop=state.stop, same_pose=same_pose,
                             same_prefix=same_prefix, first_publicly_available=available)
        return same_pose and same_prefix and available

    return IssuedPolicy(action, applicable, name), targets, release_check


def case_arms(graph, actual, metadata, stops, history, journey, fit, kernel, methods):
    decision = history['decision']
    od = journey['od']
    origin, goal = stops[od['origin']], stops[od['destination']]
    start, deadline = od['ready_seconds'], od['deadline_seconds']
    seed = int(history['date'].replace('-', '')) + 1000 * history['query']
    previous = journey['decisions'][decision - 1]
    if previous['outcome'] != 'board':
        raise AssertionError('The archived continuation root must follow a boarding')
    incoming = previous['action']
    path = tuple(history['witness'])
    rows = []
    for method in methods:
        before = perf_counter()
        belief, state = context(graph, metadata, journey, decision, fit)
        if (state.stop, state.time, state.last_arrival, goal) != (
                history['stop'], history['clock'], history['clock'], history['destination']):
            raise AssertionError('Unmatched fresh-onboard continuation state')
        scan = TemporalScan(graph, kernel) if method == 'middle64' else None
        worker = Worker(graph, fit, origin, goal, start, deadline, seed, method, scan)
        worker.belief = worker.router.trip_belief = belief
        incumbent_plan = freeze_tail_plan(worker.bounds(), path)
        setup_seconds = perf_counter() - before
        before, cpu = perf_counter(), process_time()
        proposal = worker.update(state, incumbent_plan, decision)
        wall_seconds, cpu_seconds = perf_counter() - before, process_time() - cpu
        ready_at = state.time + ceil(wall_seconds)

        initial = PlatformState(stop=state.stop, time=state.time, current_trip=state.current_trip,
                                last_arrival=state.last_arrival, last_connection=incoming,
                                used_connections=state.unavailable_ids, local_since=state.time)
        model = build_platform_model(graph, [(1., actual)], initial.stop, goal, initial.time, deadline,
                                     poll_interval=1, initial_state=initial)
        incumbent, incumbent_targets, _ = adapter(graph, path, initial, 'incumbent')
        has_update = not proposal.retain and proposal.plan is not None and bool(proposal.plan.primary)
        proposed_path = proposal.plan.primary if has_update else ()
        if has_update:
            candidate, candidate_targets, release_check = adapter(graph, proposed_path, initial, method,
                wait_until=state.time + 60 if proposal.defer else None)
        else:
            # The incumbent is unchanged while a retain/no-plan result returns.
            candidate, candidate_targets, release_check = incumbent, incumbent_targets, {}
        unresolved = unresolved_check(graph, metadata, actual,
                                      {'incumbent': incumbent_targets, method: candidate_targets})
        before = perf_counter()
        result = execute_background(model, model.scenarios[0], incumbent, candidate, ready_at,
                                    unresolved=unresolved)
        execution_seconds = perf_counter() - before
        final = result['state']
        censored = result['cost'] is None
        lo = final.time - state.time if censored else result['cost']
        hi = deadline - state.time if censored else lo
        status = 'censored' if censored else 'arrived' if final.stop == goal else 'timeout'
        row = dict(date=history['date'], query=history['query'], decision=decision, method=method,
                   status=status, loss_lower=lo, loss_upper=hi, root_clock=state.time, ready_at=ready_at,
                   worker_seconds=wall_seconds, worker_cpu_seconds=cpu_seconds, setup_seconds=setup_seconds,
                   replay_seconds=execution_seconds, retained=proposal.retain, deferred=proposal.defer,
                   replacement_requested=has_update, published=has_update and result['published'],
                   rejected=has_update and result['rejected'], release_check=release_check,
                   incumbent_path=path, replacement_path=proposed_path,
                   missing_connections=result['missing_connections'], work=proposal.work,
                   incumbent_moves=result['incumbent_actions_during_compute'],
                   rides=[r for r in result['trace'] if r['ridden_connection'] is not None])
        rows.append(row)
        print(row['date'], row['query'], decision, method, status, lo, hi, round(wall_seconds, 3), flush=True)
    return rows


def pair(data, fit, cell, histories, journey, library, cpu):
    os.sched_setaffinity(0, {cpu})
    graph, actual, stops, _, metadata, _ = dated_graph(data, cell[0], 'gtfs_station_max', True)
    kernel = build_temporal_kernel(library.with_name(f'platform_{cell[0]}_{cell[1]}.so'))
    return [row for i, history in enumerate(histories) for row in case_arms(
        graph, actual, metadata, stops, history, journey, fit, kernel, METHODS if i % 2 == 0 else METHODS[::-1])]


def main():
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    archive = json.loads(ARCHIVE.read_text())['rows']
    journeys = {(r['date'], r['query']): r for r in archive if r['method'] == 'evolving_rollout'}
    histories = json.loads(HISTORIES.read_text())['rows']
    cells = sorted({(r['date'], r['query']) for r in histories})
    cpus = sorted(os.sched_getaffinity(0))[:len(cells)]
    rows = []
    with TemporaryDirectory(prefix='bapr_v10_platform_') as tmp, ProcessPoolExecutor(max_workers=len(cpus)) as pool:
        tasks = [pool.submit(pair, data, fit, cell,
                             [r for r in histories if (r['date'], r['query']) == cell], journeys[cell],
                             Path(tmp) / 'temporal.so', cpus[i % len(cpus)]) for i, cell in enumerate(cells)]
        for task in tasks:
            rows.extend(task.result())
    summary = {}
    for method in METHODS:
        arm = [r for r in rows if r['method'] == method]
        summary[method] = dict(mean_lower=mean(r['loss_lower'] for r in arm),
            mean_upper=mean(r['loss_upper'] for r in arm), arrived=sum(r['status'] == 'arrived' for r in arm),
            censored=sum(r['status'] == 'censored' for r in arm),
            median_worker_seconds=median(r['worker_seconds'] for r in arm),
            replacement_requests=sum(r['replacement_requested'] for r in arm),
            published=sum(r['published'] for r in arm), rejected=sum(r['rejected'] for r in arm),
            gate_accepts=sum(bool(r['work'].get('gate') and r['work']['gate']['accepted']) for r in arm),
            incumbent_moves=sum(r['incumbent_moves'] for r in arm))
    result = dict(rows=rows, summary=summary, cpus=cpus,
        protocol=dict(histories=str(HISTORIES.relative_to(ROOT)), archive=str(ARCHIVE.relative_to(ROOT)),
            model='Same frozen committed planner and archived primary incumbent per root; V8 local-event replay is a separate execution sensitivity.',
            adapter='Only the primary path is executed. All arms board on visible target departure and explicitly alight for a transfer. WAIT retains platform physical inertia. A planned wait delays target boarding to at least root+60, when platform inertia permits.',
            release='One charged background update at root+ceil(measured worker wall). Check current stop, onboard trip, incoming leg and arrival phase, unchanged boarded prefix and no public departure/cancellation of the next target.',
            unresolved='V8 issued-target and next same-service audit; unresolved loss kept as elapsed/deadline interval. No V8-platform guarantee is inferred from the committed certificate.'))
    write_json(HERE / 'platform_sensitivity_results.json', result)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
