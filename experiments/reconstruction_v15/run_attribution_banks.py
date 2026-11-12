"""Measure population-bound endpoints on two independent-bank constructions."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import json
import os
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v13 import factored_run as v13
from experiments.reconstruction_v13 import platform_run as platform_v13
from experiments.reconstruction_v14.run_bank_traces import oracle_path
from experiments.reconstruction_v15.run_finite_attribution import pathwise_witness
from src.background_policy import execute_background
from src.causal_prefix import PrefixBudgetExceeded
from src.event_platform_solver import EventPlatform
from src.factored_lower import product_trip_bank
from src.platform_transit import PlatformState, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map


V13 = HERE.parent / 'reconstruction_v13'
V14 = HERE.parent / 'reconstruction_v14'
NODE_BUDGET = 50_000
POLICY_THRESHOLD = (900, 500, 850)
WHOLE_BANK_SIZE = 8


def read(path):
    return json.loads(path.read_text())


def rng(seed):
    return np.random.default_rng(np.random.SeedSequence(seed))


def prepare(case, *, conditioning_cost=None):
    day, query, decision = case
    data, _ = v13.load_city(v13.SOURCE)
    fit = v13.fit_training(data, retain_no_pickup=True)
    graph, _, stops, _, meta, _ = v13.dated_graph(data, day, 'gtfs_station_max', True)
    record = next(r for r in read(v13.INVENTORY)['rows'] if v13.key(r) == case)
    historical = next(r for r in read(v13.V10)['rows'] if v13.key(r) == case)
    journey = next(r for r in read(v13.ARCHIVE)['rows']
                   if (r['date'], r['query'], r['method']) == (day, query, 'evolving_rollout'))
    prefix = tuple(record['next_transfer_boundary']['prefix'])
    if conditioning_cost is not None:
        started, cpu = perf_counter(), process_time()
    belief, initial = v13.prepare_waiting_context(graph, meta, journey, decision, fit, prefix,
                                                  record['recorded_incumbent']['rides'])
    if conditioning_cost is not None:
        conditioning_cost.update(wall=perf_counter()-started, cpu=process_time()-cpu)
    goal, deadline = stops[journey['od']['destination']], journey['od']['deadline_seconds']
    plan = v13.load_plan(historical['fixed_primary_tail_rescue']['plan'], prefix)
    pars = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
            for c in graph.connections}
    return graph, belief, initial, goal, deadline, plan, pars


def scene_from_clocks(graph, clocks):
    dep, arr, cnl = clocks
    return JourneyScenario({c.id: VehicleEvent(int(dep[c.id])-c.dep_time, bool(cnl[c.id]),
                          int(dep[c.id]), int(arr[c.id])) for c in graph.connections},
                          minute=60, publications={})


def pack_product(graph, clocks, factors):
    common = scene_from_clocks(graph, clocks)
    bank, labels = product_trip_bank(graph, common, factors)
    packed = tuple(np.repeat(x[None, :], len(bank), axis=0) for x in clocks)
    for trip, atoms in factors.items():
        for i, atom in enumerate(labels[trip]):
            for cid, event in atoms[atom].items():
                packed[0][i,cid], packed[1][i,cid], packed[2][i,cid] = (
                    event.departure, event.arrival, event.canceled)
    return bank, packed, common


def scanner(scan, packed, initial, goal, deadline, transfers):
    @lru_cache(None)
    def cached(index, stop, time, trip, last_arrival, last_connection, used):
        state = replace(initial, stop=stop, time=time, current_trip=trip, last_arrival=last_arrival,
                        last_connection=last_connection, used_connections=used)
        return float(scan.platform_arrivals(*(x[index:index+1] for x in packed), state, goal,
                                            deadline, transfers)[0]-time)

    def oracle(index, state):
        return cached(index, state.stop, state.time, state.current_trip, state.last_arrival,
                      state.last_connection, state.used_connections)
    return oracle


def executable_oracle(graph, scene, state, goal, deadline, path):
    """Validate a relaxed path against boarding, transfer and WAIT inertia."""
    stop, time, trip = state.stop, state.time, state.current_trip
    used = set(state.used_connections)
    ready = (max(time, state.last_arrival+graph.stops[stop].min_transfer_time)
             if trip is None and state.last_connection is not None else time)
    for cid in path:
        c, event = graph.connections[cid], scene.events[cid]
        if (c.dep_stop != stop or cid in used or event.canceled
                or not time <= event.departure <= event.arrival <= deadline):
            return None
        if trip is not None and c.trip_id == trip:
            forced = any(other.trip_id == trip and other.id not in used
                         and not scene.events[other.id].canceled
                         and time <= scene.events[other.id].departure < event.departure
                         for other in graph.get_connections_from(stop))
            if forced and (not c.pickup_allowed
                           or event.departure < time+graph.stops[stop].min_transfer_time):
                return None
        else:
            if trip is not None:
                ready = time+graph.stops[stop].min_transfer_time
            if not c.pickup_allowed or event.departure < ready:
                return None
        stop, time, trip = c.arr_stop, event.arrival, c.trip_id or f'connection:{cid}'
        used.add(cid)
    return time-state.time if stop == goal else None


def singleton_witness(graph, bank, oracle, goal, deadline):
    @lru_cache(None)
    def exact(index, stop, time, trip, last_arrival, last_connection, used):
        state = PlatformState(stop, time, trip, last_arrival, last_connection, used)
        lower = oracle(index, state)
        if lower == deadline-time or stop == goal:
            return lower
        scene = bank[index][1]
        _, path = oracle_path(graph, scene, state, goal, deadline)
        upper = executable_oracle(graph, scene, state, goal, deadline, path)
        return lower if upper == lower else None

    def bound(index, state):
        return exact(index, state.stop, state.time, state.current_trip, state.last_arrival,
                     state.last_connection, state.used_connections)
    return bound


def solve(platform, oracle, *, full=False, fallback, singleton_exact=None, witness=False, trace=None):
    start = perf_counter()
    work = {}
    if witness:
        support = tuple(range(len(platform.model.scenarios)))
        exact, nodes, transitions = pathwise_witness(platform, oracle, support)
        work.update(witness_nodes=nodes, witness_transitions=transitions)
        if exact:
            value = mean(oracle(i, platform.initial) for i in support)
            return dict(status='complete', value=value, population_lower_sample=value,
                        nodes=nodes, transitions=transitions, method='pathwise_witness',
                        seconds=perf_counter()-start, **work)
    try:
        value = platform.solve(oracle, elapsed_horizon=None if full else v13.HORIZON,
                               stop_after_first_ride=not full, max_nodes=NODE_BUDGET,
                               singleton_exact=singleton_exact, trace=trace,
                               prune_with_oracle=full and trace is None)
        return dict(status='complete', value=value.value, population_lower_sample=value.value,
                    nodes=value.information_nodes, transitions=value.transition_evaluations,
                    action_branches_pruned=value.action_branches_pruned,
                    method='event_dp', seconds=perf_counter()-start, **work)
    except PrefixBudgetExceeded as error:
        return dict(status='budget_exceeded', value=None, population_lower_sample=fallback,
                    nodes=error.nodes-1, transitions=error.transitions,
                    action_branches_pruned=error.action_branches_pruned,
                    seconds=perf_counter()-start, **work)


def first_visible_tail_upper(oracle, graph, root, trace, deadline):
    """A feasible policy for the V13 relaxation: run first_visible to the V13 stop."""
    values = []
    for i, world in enumerate(trace['worlds']):
        route = world['first_visible_route']
        cutoff = root.time + v13.HORIZON
        if route and route[0]['departure'] < cutoff:
            first = route[0]
            if first['arrival'] >= deadline:
                values.append(deadline-root.time)
                continue
            cid = first['cid']
            state = replace(root, time=first['arrival'], stop=first['destination'],
                            current_trip=graph.connections[cid].trip_id, last_arrival=first['arrival'],
                            last_connection=cid, used_connections=root.used_connections | {cid},
                            local_since=first['arrival'])
        else:
            state = replace(root, time=cutoff)
        cost = state.time-root.time+oracle(i, state)
        assert cost <= world['first_visible_loss'] + 1e-7
        values.append(cost)
    return values


def known_rest_upper(common, graph, root, goal, deadline, hidden, threshold, archived):
    """Freeze a policy rule which reads the revealed rest but never A/B draws."""
    masked = replace(common, events={c.id: replace(common.events[c.id], canceled=True)
                   if c.trip_id in hidden else common.events[c.id] for c in graph.connections})
    arrival, path = oracle_path(graph, masked, root, goal, deadline)
    cap = deadline-root.time
    if not path:
        return archived, None, cap
    policy, _, _ = platform_v13.adapter(graph, path, root, 'rest_only')
    model = build_platform_model(graph, [(1., masked)], root.stop, goal, root.time, deadline,
                                 initial_state=root, poll_interval=1)
    replay = execute_background(model, model.scenarios[0], policy, policy, root.time)
    if replay['state'].stop != goal or replay['cost'] != arrival-root.time:
        return archived, None, cap
    # The issued path and its whole trip chains belong to revealed rest; A/B
    # can add observations but cannot change the path interpreter's actions.
    assert all(graph.connections[cid].trip_id not in hidden for cid in path)
    cost = replay['cost']
    return ([cost]*len(archived) if cost < threshold else archived), path, cost


def run_v13(case, graph, belief, root, goal, deadline, plan, pars, scan, transfers):
    source = {r['bank']: r for r in read(V13/'factored_results.json')['rows']
              if v13.key(r) == case and r['arm'] == 'product'}
    trace = next(x for x in read(V14/'bank_traces.json')['roots'] if v13.key(x) == case)['banks']
    hidden = tuple(graph.connections[cid].trip_id for cid in (plan.primary[0], plan.rescue[0]))
    assert len(set(hidden)) == 2
    day, query, decision = case
    seed = [20261113, int(day.replace('-', '')), query, decision, 13]
    rest = sample_clocks(belief, pars, rng(seed+[0]), v13.BANKS, templates=template_trip_map(belief, pars))
    original = {trip: v13.hidden_draws(belief, pars, trip, seed+[1, j])
                for j,trip in enumerate(hidden)}
    fresh = {trip: v13.hidden_draws(belief, pars, trip, seed+[15, 1, j])
             for j,trip in enumerate(hidden)}
    rows = []
    for i in range(v13.BANKS):
        start = perf_counter()
        clocks = tuple(x[i] for x in rest)
        sl = slice(i*v13.ATOMS, (i+1)*v13.ATOMS)
        factors = {trip: atoms[sl] for trip, atoms in original.items()}
        bank, packed, common = pack_product(graph, clocks, factors)
        oracle = scanner(scan, packed, root, goal, deadline, transfers)
        reference = source[i]
        j, z = reference['policies']['first_visible']['mean_loss'], reference['value']
        oracle_mean = mean(oracle(k, root) for k in range(len(bank)))
        assert abs(oracle_mean-reference['clairvoyant_root_mean']) < 1e-7
        model = EventPlatform(graph, bank, root, goal, deadline)
        fast_prefix = solve(model, oracle, fallback=oracle_mean)
        assert fast_prefix['status'] == 'complete' and abs(fast_prefix['value']-z) < 1e-7
        upper_tau = first_visible_tail_upper(oracle, graph, root, trace[i], deadline)
        upper_r, path, rest_only = known_rest_upper(common, graph, root, goal, deadline, hidden,
                                          POLICY_THRESHOLD[v13.CASES.index(case)],
                                          reference['policies']['first_visible']['losses'])
        if abs(z-min(j, mean(upper_r), rest_only)) < 1e-7:
            full = dict(status='complete', value=z, population_lower_sample=z, nodes=0,
                        transitions=0, method='feasible_policy_sandwich', seconds=0.)
        else:
            full = solve(model, oracle, full=True, fallback=z, witness=True,
                         singleton_exact=singleton_witness(graph, bank, oracle, goal, deadline))
        if full['value'] is not None:
            assert z-1e-7 <= full['value'] <= min(j, mean(upper_r))+1e-7
        assert z-1e-7 <= mean(upper_tau) <= j+1e-7
        del model
        bank8, packed8, _ = pack_product(graph, clocks,
            {trip: factors[trip] + fresh[trip][sl] for trip in hidden})
        oracle8 = scanner(scan, packed8, root, goal, deadline, transfers)
        fallback8 = mean(oracle8(k, root) for k in range(len(bank8)))
        large = solve(EventPlatform(graph, bank8, root, goal, deadline), oracle8, fallback=fallback8)
        rows.append(dict(bank=i, j=j, z=z, clairvoyant_mean=oracle_mean, fast_prefix=fast_prefix,
            v_r_empirical=full['value'], v_r_lower_sample=full['population_lower_sample'],
            v_r_tau_policy=mean(upper_tau), v_r_tau_policy_rows=upper_tau,
            v_r_policy=mean(upper_r), v_r_policy_rows=upper_r, v_r_only_path=path,
            v_r_rest_only=rest_only,
            z8_empirical=large['value'], z8_lower_sample=large['population_lower_sample'],
            full=full, larger_prefix=large, wall_seconds=perf_counter()-start))
        print(case, 'v13', i, z, full['value'], large['value'], flush=True)
    return rows


def run_iid(case, graph, belief, root, goal, deadline, pars, scan, transfers):
    day, query, decision = case
    rng0 = rng([20261113, int(day.replace('-', '')), query, decision])
    count = platform_v13.WORLDS
    clocks = sample_clocks(belief, pars, rng0, count, templates=template_trip_map(belief, pars))
    receipts = {r['world']: r['loss_upper'] for r in read(V13/'platform_results.json')['rows']
                if v13.key(r) == case and r['source'] == 'posterior' and r['method'] == 'first_visible'}
    rows = []
    for start in range(0, count, WHOLE_BANK_SIZE):
        before = perf_counter()
        indices = range(start,start+WHOLE_BANK_SIZE)
        packed = tuple(x[indices.start:indices.stop] for x in clocks)
        oracle = scanner(scan, packed, root, goal, deadline, transfers)
        bank = [(1/WHOLE_BANK_SIZE, scene_from_clocks(graph, tuple(x[i] for x in clocks))) for i in indices]
        oracle_mean = mean(oracle(i, root) for i in range(len(bank)))
        known = []

        def record(state, indices, _, boundary):
            if boundary == 'exact_continuation':
                known.append(dict(world=indices[0]+start, elapsed=state.time-root.time,
                                  before_first_ride=state.used_connections == root.used_connections))

        full = solve(EventPlatform(graph, bank, root, goal, deadline), oracle, full=True,
                     fallback=oracle_mean, singleton_exact=singleton_witness(graph, bank, oracle, goal, deadline),
                     trace=record)
        j = mean(receipts[i] for i in indices)
        if full['value'] is not None:
            assert oracle_mean-1e-7 <= full['value'] <= j+1e-7
        rows.append(dict(bank=len(rows), original_indices=list(indices), j=j, clairvoyant_mean=oracle_mean,
                         v_star_empirical=full['value'], v_star_lower_sample=full['population_lower_sample'],
                         full=full, singleton_continuations=known, wall_seconds=perf_counter()-before))
        print(case, 'iid', len(rows)-1, full['status'], full['value'], flush=True)
    return rows


def evaluate(case, cpu, panel):
    os.sched_setaffinity(0, {cpu})
    start = perf_counter()
    graph, belief, root, goal, deadline, plan, pars = prepare(case)
    with TemporaryDirectory(prefix='bapr_v15_bank_') as tmp:
        scan = TemporalScan(graph, build_temporal_kernel(Path(tmp)/'scan.so'))
        transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
        setup = perf_counter()-start
        if panel == 'v13':
            banks = run_v13(case, graph, belief, root, goal, deadline, plan, pars, scan, transfers)
        else:
            banks = run_iid(case, graph, belief, root, goal, deadline, pars, scan, transfers)
    return dict(case=case, cap=deadline-root.time, setup_seconds=setup, banks=banks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', choices=('v13', 'iid'))
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    panel = options.panel
    output = options.output or HERE / ('v13_exact_banks.json' if panel == 'v13' else 'iid_causal_banks.json')
    results = dict(status='running', panel=panel, node_budget=NODE_BUDGET, roots=[])
    cpus = sorted(os.sched_getaffinity(0), reverse=panel=='iid')
    with ProcessPoolExecutor(max_workers=min(len(v13.CASES),len(cpus))) as pool:
        pending = {pool.submit(evaluate, case, cpus[i % len(cpus)], panel): case
                   for i,case in enumerate(v13.CASES)}
        for task in as_completed(pending):
            results['roots'].append(task.result())
            results['roots'].sort(key=lambda r: v13.CASES.index(tuple(r['case'])))
            output.write_text(json.dumps(results, indent=2)+'\n')
    results['status'] = 'complete'
    output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
