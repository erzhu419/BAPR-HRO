"""Product versus shared-index lower bounds at the three public waiting states."""
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
from experiments.reconstruction_v8.budget_run import SOURCE
from experiments.reconstruction_v9.search_reference import ARCHIVE
from experiments.reconstruction_v12.transfer_inventory import CASES, key
from experiments.reconstruction_v13.platform_run import (
    INVENTORY, V10, METHODS, load_plan, prepare_waiting_context, compile_arms, replay_arm)
from src.causal_prefix import causal_prefix_lower, PrefixBudgetExceeded
from src.continuation_confidence import continuation_lower
from src.factored_lower import product_trip_bank, factored_prefix_lower
from src.platform_transit import build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map

BANKS, ATOMS, HORIZON, MAX_NODES = 8, 4, 720, 50000


def read(path):
    return json.loads(path.read_text())


def save(value):
    (HERE / 'factored_results.json').write_text(json.dumps(value, indent=2) + '\n')


def hidden_draws(belief, parameters, trip, seed):
    law, slots = belief._law(trip, parameters), belief._trips[trip]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    states = law.draw_uniforms_batch(rng.random((BANKS * ATOMS, len(slots)))[:, ::-1])
    draws = []
    for row in states:
        draws.append({slots[i].cid: VehicleEvent(int(law.times[i][row[i]])-slots[i].scheduled,
            bool(law.canceled[i][row[i]]), int(law.times[i][row[i]]), int(law.times[i+1][row[i+1]]))
            for i in range(0, len(slots), 2)})
    return draws


def summarize(rows):
    result = []
    alpha = .05 / (len(CASES) * 2)
    for case in CASES:
        for arm in ('product', 'shared_index'):
            group = [r for r in rows if key(r) == case and r['arm'] == arm]
            complete = len(group) == BANKS and all(r['status'] == 'complete' for r in group)
            item = dict(date=case[0], query=case[1], decision=case[2], arm=arm,
                complete=complete, completed_banks=sum(r['status'] == 'complete' for r in group),
                solver_seconds=sum(r['solver_seconds'] for r in group))
            if complete:
                values = np.array([r['value'] for r in group])
                clairvoyant = np.array([r['clairvoyant_root_mean'] for r in group])
                item.update(mean_lower_estimator=float(values.mean()),
                    mean_clairvoyant=float(clairvoyant.mean()),
                    mean_nonanticipativity_increment=float(np.mean(values-clairvoyant)),
                    lower_confidence=continuation_lower(values, group[0]['cap'], alpha), alpha=alpha,
                    singleton_before_boundary=float(np.mean([
                        r['singleton_before_boundary_probability'] for r in group])),
                    mean_information_nodes=float(np.mean([r['information_nodes'] for r in group])))
                if arm == 'product':
                    item['policy_mean_loss'] = {method: float(np.mean([
                        r['policies'][method]['mean_loss'] for r in group])) for method in METHODS}
                    item['policy_mean_gap'] = {method: float(np.mean([
                        r['policies'][method]['mean_loss']-r['value'] for r in group])) for method in METHODS}
            result.append(item)
    return result


def main():
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    protocol = dict(cases=CASES, banks=BANKS, atoms_per_hidden_trip=ATOMS, horizon=HORIZON,
        node_budget=MAX_NODES, cpu=cpu,
        context='V12 public prefix receipts, then immediate ALIGHT. Full waiting-state platform menu.',
        hidden='Whole next-primary trip and whole first-rescue trip; independent empirical supports. '
               'All other trips share one revealed rest realization per outer bank.',
        seeds='rest:[20261113,date,query,decision,13,0]; hidden j:[20261113,date,query,decision,13,1,j]',
        comparison='Same rest and factor atoms; product enumerates4x4, shared_index pairs equal indices. '
                   'A single shared-index result need not be below its paired product result.',
        stop='First boarding after ALIGHT or720 elapsed seconds, then full-graph clairvoyant continuation.',
        failure='Stop later banks only for the failed root/arm; retain cost and no partial bound. No budget increase.',
        inference='Eight independent outer banks, not128 independent product rows. '
                  'Per arm/root alpha=.05/6. Confidence concerns the declared posterior only.',
        policies='Same four preissued rules and publication clock as platform_run; evaluated on every product row. '
                 'Point gaps are not near-optimality certificates. Product/DP evaluation overhead reported separately.')
    (HERE / 'factored_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    before = perf_counter()
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    result = dict(protocol=protocol, training_seconds=perf_counter()-before, setups=[], rows=[])
    inventory = {key(r): r for r in read(INVENTORY)['rows']}
    old = {key(r): r for r in read(V10)['rows']}
    journeys = {(r['date'], r['query']): r for r in read(ARCHIVE)['rows'] if r['method']=='evolving_rollout'}
    with TemporaryDirectory(prefix='bapr_v13_factors_') as tmp:
        before = perf_counter()
        kernel = build_temporal_kernel(Path(tmp)/'scan.so')
        result['compile_seconds'] = perf_counter()-before
        for case in CASES:
            day, query, decision = case
            before = perf_counter()
            graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
            record, journey = inventory[case], journeys[day, query]
            prefix = tuple(record['next_transfer_boundary']['prefix'])
            belief, initial = prepare_waiting_context(graph, metadata, journey, decision, fit,
                prefix, record['recorded_incumbent']['rides'])
            goal, deadline = stops[journey['od']['destination']], journey['od']['deadline_seconds']
            parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                          for c in graph.connections}
            plan = load_plan(old[case]['fixed_primary_tail_rescue']['plan'], prefix)
            hidden = (graph.connections[plan.primary[0]].trip_id, graph.connections[plan.rescue[0]].trip_id)
            assert len(set(hidden)) == 2
            scan = TemporalScan(graph, kernel)
            transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
            setup_seconds = perf_counter()-before
            base_seed = [20261113, int(day.replace('-','')), query, decision, 13]
            before = perf_counter()
            rest = sample_clocks(belief, parameters, np.random.default_rng(np.random.SeedSequence(base_seed+[0])),
                                 BANKS, templates=template_trip_map(belief, parameters))
            factors = {trip: hidden_draws(belief, parameters, trip, base_seed+[1,j]) for j,trip in enumerate(hidden)}
            draw_seconds = perf_counter()-before
            compiled = compile_arms(graph, plan, initial, deadline)
            result['setups'].append(dict(date=day, query=query, decision=decision, hidden_trips=hidden,
                root_time=initial.time, root_stop=initial.stop, setup_seconds=setup_seconds, draw_seconds=draw_seconds))
            stopped = set()
            for index in range(BANKS):
                before = perf_counter()
                dep, arr, cnl = (x[index] for x in rest)
                common = JourneyScenario({c.id: VehicleEvent(int(dep[c.id])-c.dep_time, bool(cnl[c.id]),
                    int(dep[c.id]), int(arr[c.id])) for c in graph.connections}, minute=60, publications={})
                atoms = {trip: draws[index*ATOMS:(index+1)*ATOMS] for trip,draws in factors.items()}
                bank, labels = product_trip_bank(graph, common, atoms)
                packed = tuple(np.repeat(x[None,:], len(bank), axis=0) for x in (dep,arr,cnl))
                for trip in hidden:
                    for row, atom in enumerate(labels[trip]):
                        for cid,event in atoms[trip][atom].items():
                            packed[0][row,cid],packed[1][row,cid],packed[2][row,cid] = event.departure,event.arrival,event.canceled
                packing_seconds = perf_counter()-before
                diagonal = [j*ATOMS+j for j in range(ATOMS)]
                for arm in ('product','shared_index'):
                    if arm in stopped:
                        continue
                    use = list(range(len(bank))) if arm == 'product' else diagonal
                    worlds = [(1./len(use), bank[i][1]) for i in use]
                    clocks = tuple(np.ascontiguousarray(x[use]) for x in packed)
                    selected_labels = {trip: tuple(labels[trip][i] for i in use) for trip in hidden}
                    before = perf_counter()
                    model = build_platform_model(graph, worlds, initial.stop, goal, initial.time, deadline,
                                                 initial_state=initial, poll_interval=1)
                    model_seconds = perf_counter()-before
                    def oracle(i, public):
                        return float(scan.platform_arrivals(*(x[i:i+1] for x in clocks), public,
                                                            goal, deadline, transfers)[0]-public.time)
                    row = dict(date=day, query=query, decision=decision, bank=index, arm=arm,
                               cap=deadline-initial.time, packing_seconds=packing_seconds, model_seconds=model_seconds)
                    before = perf_counter()
                    try:
                        if arm == 'product':
                            value = factored_prefix_lower(model, oracle, graph=graph, factor_labels=selected_labels,
                                                          elapsed_horizon=HORIZON, max_nodes=MAX_NODES)
                        else:
                            value = causal_prefix_lower(model, oracle, elapsed_horizon=HORIZON,
                                                        max_nodes=MAX_NODES, factor_labels=selected_labels)
                        assert value.value >= value.clairvoyant_root_mean-1e-7
                        row.update(status='complete', **asdict(value))
                    except PrefixBudgetExceeded as error:
                        row.update(status='budget_exceeded', information_nodes=error.nodes,
                                   transition_evaluations=error.transitions)
                        stopped.add(arm)
                    row['solver_seconds'] = perf_counter()-before
                    if arm == 'product' and row['status'] == 'complete':
                        before = perf_counter()
                        policies = {}
                        for method in METHODS:
                            runs = [replay_arm(graph, plan, initial, goal, deadline, model, world,
                                               method, compiled[method]) for world in model.scenarios]
                            loss = float(np.mean([r['loss_upper'] for r in runs]))
                            assert row['value'] <= loss+1e-7
                            policies[method] = dict(mean_loss=loss,
                                mean_pipeline_seconds=float(np.mean([r['total_rule_pipeline_seconds'] for r in runs])),
                                losses=[r['loss_upper'] for r in runs])
                        row.update(policies=policies, policy_evaluation_seconds=perf_counter()-before)
                    result['rows'].append(row)
                    save(result)
                    print(case,index,arm,row['status'],round(row.get('value',0),3),
                          round(row.get('value',0)-row.get('clairvoyant_root_mean',0),3),
                          round(row['solver_seconds'],3),flush=True)
                if len(stopped)==2:
                    break
    result['summary'] = summarize(result['rows'])
    save(result)
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()
