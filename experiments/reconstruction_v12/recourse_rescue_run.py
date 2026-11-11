"""Restore the existing full V10 rescue as a frozen boundary-family control."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from experiments.reconstruction_v8.budget_run import SOURCE, write_json
from experiments.reconstruction_v9.search_reference import ARCHIVE
from experiments.reconstruction_v12.transfer_inventory import CASES, key, state_spec
from src.boundary_recourse import BoundaryRecourseBounds, choose_boundary_tail
from src.tail_rescue import TailRescueBounds, TailRescuePlan


ORIGINAL = HERE / 'recourse_results.json'
INVENTORY = HERE / 'transfer_inventory.json'
V10 = ROOT / 'experiments/reconstruction_v10/tail_rescue_results.json'


def load_plan(row):
    return TailRescuePlan(tuple(row['primary']), row['failure_index'], tuple(row['rescue']),
                         row['first_action_now'], tuple(map(tuple,row['departures'])))


def observed_choice(graph, belief, parameters, state, prefix, tails, goal, deadline, previous):
    """A separate posthoc action diagnostic using public receipts only."""
    posterior = belief.copy()
    observed = state
    receipts = previous['recorded_incumbent']['rides'][:len(prefix)]
    assert [r['ridden_connection'] for r in receipts] == list(prefix)
    for receipt in receipts:
        cid = receipt['ridden_connection']
        c = graph.connections[cid]
        posterior.observe_connection_event(cid,'departure',receipt['time']-c.dep_time)
        posterior.observe_connection_event(cid,'arrival',receipt['next_time']-c.arr_time)
        observed = replace(observed,stop=receipt['next_stop'],time=receipt['next_time'],
            last_arrival=receipt['next_time'],current_trip=c.trip_id,
            unavailable_ids=observed.unavailable_ids | {cid},learned_trips=observed.learned_trips | {c.trip_id})
    bounds = TailRescueBounds(posterior,parameters,graph,goal,deadline)
    choice = choose_boundary_tail(bounds,observed,tails)
    return dict(state=state_spec(observed), **asdict(choice), selected_plan=asdict(tails[choice.selected]),
        source='Existing public incumbent ride receipts applied to a fresh posterior after all root evaluation; no actual future tail outcome is read.',
        scope='Operational action re-evaluation only; not reuse or extension of the original root finite-grid certificate.')


def run_root(identity, graph, metadata, stops, fit, journey, original, previous, archived):
    started = perf_counter()
    belief,state = context(graph,metadata,journey,identity[2],fit)
    goal,deadline = stops[journey['od']['destination']],journey['od']['deadline_seconds']
    parameters = {c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
    bounds = BoundaryRecourseBounds(belief,parameters,graph,goal,deadline)
    setup_seconds = perf_counter()-started
    prefix = tuple(original['prefix'])
    started = perf_counter()
    tails = tuple(load_plan(row['plan']) for row in original['candidates'])
    full = load_plan(archived['fixed_primary_tail_rescue']['plan'])
    assert full.primary[:len(prefix)] == prefix and full.failure_index >= len(prefix)
    restored = replace(full,primary=full.primary[len(prefix):],
                       failure_index=full.failure_index-len(prefix),first_action_now=False)
    tails += (restored,)
    added_candidate_seconds = perf_counter()-started
    cpu = process_time()
    plan,value = bounds.evaluate(state,prefix,tails,departures=full.departures)
    phase_cpu_seconds = process_time()-cpu
    old_error = max(abs(a-b) for a,b in zip(value.fixed_tail_uppers,original['exact']['fixed_tail_uppers']))
    rescue_error = value.fixed_tail_uppers[-1]-archived['fixed_primary_tail_rescue']['upper']
    row = dict(date=identity[0],query=identity[1],decision=identity[2],prefix=list(prefix),
        root=state_spec(state),candidates=[asdict(t) for t in tails],
        old_candidate_count=len(tails)-1,restored_candidate_index=len(tails)-1,
        setup_seconds=setup_seconds,added_candidate_seconds=added_candidate_seconds,
        reused_original_candidate_generation_seconds=original['generation']['seconds'],
        exact=asdict(value),phase_cpu_seconds=phase_cpu_seconds,
        old_fixed_family_max_error=old_error,restored_v10_fixed_upper_error=rescue_error,
        archived_v10_fixed_upper=archived['fixed_primary_tail_rescue']['upper'],
        archived_v10_best_across_root_actions=archived['upper'],
        historical_original_adaptive_upper=original['adaptive_upper'])
    if old_error > 1e-7 or abs(rescue_error) > 1e-7:
        return dict(row,status='fixed_control_mismatch')
    static_index = int(np.argmin(value.fixed_tail_uppers))
    static = value.fixed_tail_uppers[static_index]
    observed = observed_choice(graph,belief,parameters,state,prefix,tails,goal,deadline,previous)
    observed['on_original_grid'] = any(p.arrival == observed['state']['clock'] for p in value.phases)
    row.update(status='complete',static_best_index=static_index,static_best_upper=static,
        adaptive_upper=value.upper,arrival_information_improvement_seconds=static-value.upper,
        improvement_vs_initial_adaptive_seconds=original['adaptive_upper']-value.upper,
        improvement_vs_v10_fixed_seconds=archived['fixed_primary_tail_rescue']['upper']-value.upper,
        incremental_construction_and_phase_seconds=added_candidate_seconds+value.seconds,
        full_construction_and_phase_seconds=original['generation']['seconds']+added_candidate_seconds+value.seconds,
        selected_probability_by_candidate=[sum(p.probability for p in value.phases if p.selected == i)
                                           for i in range(len(tails))],
        prefix_failure_probability=max(0.,1.-value.prefix_success_probability),
        prefix_failure_cap=deadline-state.time,observed_boundary_diagnostic=observed)
    return row


def main():
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),cases=[list(c) for c in CASES],
        rationale='The initial family used the V10 rescue as a direct route and omitted its successful failure-contingent mechanism. Restore that existing baseline, preserving all initial outputs.',
        family='Reload every initial frozen tail unchanged; append the full archived V10 fixed-primary rescue with primary sliced by prefix length, failure_index shifted by prefix length, departures and rescue unchanged, first_action_now=False. No new nominal routing.',
        controls='Integrated unchanged family must match initial values; appended fixed candidate must match V10 fixed-primary upper, not V10 best across different root actions. Absolute tolerance1e-7.',
        policy='Same prefix and frozen grid. Choose minimum conditional expected tail upper after arrival. Cap prefix failure at the root deadline loss.',
        observed_diagnostic='After integration, update a fresh posterior using existing public prefix departure/arrival receipts and choose among the same frozen tails. This also handles observed off-grid seconds without reverting to candidate0; it does not renew the root certificate.',
        execution='All three already-declared roots once. No sample or performance replay. If a fixed-control equality fails, save that result and stop.',
        inputs=[str(p.relative_to(ROOT)) for p in (SOURCE,ARCHIVE,ORIGINAL,INVENTORY,V10)])
    write_json(HERE / 'recourse_rescue_protocol.json',protocol)
    original = {key(r):r for r in json.loads(ORIGINAL.read_text())['rows']}
    inventory = {key(r):r for r in json.loads(INVENTORY.read_text())['rows']}
    archived = {key(r):r for r in json.loads(V10.read_text())['rows']}
    journeys = {(r['date'],r['query']):r for r in json.loads(ARCHIVE.read_text())['rows']
                if r['method']=='evolving_rollout'}
    started = perf_counter()
    data,_ = load_city(SOURCE)
    fit = fit_training(data,retain_no_pickup=True)
    result = dict(protocol=protocol,shared_fit_seconds=perf_counter()-started,shared_graphs=[],rows=[])
    graphs = {}
    for identity in CASES:
        day,query,_ = identity
        if day not in graphs:
            started = perf_counter()
            graph,_,stops,_,metadata,_ = dated_graph(data,day,'gtfs_station_max',True)
            graphs[day] = graph,metadata,stops
            result['shared_graphs'].append(dict(date=day,seconds=perf_counter()-started))
        row = run_root(identity,*graphs[day],fit,journeys[day,query],original[identity],
                       inventory[identity],archived[identity])
        result['rows'].append(row)
        write_json(HERE / 'recourse_rescue_results.json',result)
        print(identity,row['status'],'static',row.get('static_best_upper'),
              'adaptive',row.get('adaptive_upper'),'gain',row.get('arrival_information_improvement_seconds'),flush=True)
        if row['status'] != 'complete':
            result['status'] = 'stopped_fixed_control_mismatch'
            write_json(HERE / 'recourse_rescue_results.json',result)
            return
    result['status'] = 'complete'
    write_json(HERE / 'recourse_rescue_results.json',result)


if __name__ == '__main__':
    main()
