"""Three-root exact boundary-recourse diagnostic with a frozen small tail family."""

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
from experiments.reconstruction_v9.search_reference import ARCHIVE, HISTORIES
from experiments.reconstruction_v12.transfer_inventory import CASES, key, state_spec
from src.boundary_recourse import BoundaryRecourseBounds
from src.recourse_itinerary_router import build_trip_blocks, nominal_itinerary
from src.tail_rescue import freeze_tail_plan


QUANTILES = (0., .25, .5, .75, 1.)
PREFIXES = ((8780,), (9488, 9727), (8780,))
MAX_FAMILY = 7
PILOT_LIMIT_SECONDS = 120.
MAX_PHASE_TAIL_CELLS = 2000


def protocol():
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
        cases=[dict(date=d, query=q, decision=n, prefix=list(p))
               for (d,q,n),p in zip(CASES, PREFIXES)],
        quantiles=list(QUANTILES), max_family=MAX_FAMILY,
        candidates='Original suffix first, five root-posterior success-arrival quantile nominal routes, then V10 fixed-primary rescue if contiguous from this boundary; deduplicate in that order.',
        menu='Full public connection menu. Quantile routes exclude the incoming trip; root posterior means for every other trip. No actual arrival or held-out future event enters generation.',
        policy='Commit the maximum initial same-trip block. On successful arrival choose the frozen tail with minimum conditional expected upper; any prefix or selected-tail failure is deadline capped.',
        thresholds='All first post-boundary attempts use frozen nominal waiting thresholds (first_action_now=False), including the original suffix.',
        static_control='Minimum integrated upper over the exact same frozen tail family, before observing boundary arrival.',
        off_grid='Use original suffix, candidate zero, if an observed arrival is absent from the frozen finite-grid phase table.',
        pilot='First declared root is the pilot and is retained as its result. Stop before later roots if candidate generation or phase evaluation exceeds 120 seconds. Stop before evaluation if phase count times family exceeds 2000; preserve the partial receipt.',
        scope='Exact finite-grid feasible-policy diagnostic, not a whole-journey platform performance comparison or proof of true causal regret.',
        inputs=[str(p.relative_to(ROOT)) for p in (SOURCE,ARCHIVE,HISTORIES,
            HERE / 'transfer_inventory.json', ROOT / 'experiments/reconstruction_v10/tail_rescue_results.json')])


def generate(bounds, state, prefix, original_suffix, rescue, blocks):
    started, cpu = perf_counter(), process_time()
    incoming = bounds.prefix_arrivals(state, prefix)
    times = np.unique(incoming.arrival_times[incoming.arrival_mass > 0.])
    masses = np.array([incoming.arrival_mass[incoming.arrival_times == t].sum() for t in times])
    cumulative = np.cumsum(masses) / masses.sum()
    incoming_trip = bounds.graph.connections[prefix[-1]].trip_id
    stop = bounds.graph.connections[prefix[-1]].arr_stop
    nominal, menu = bounds.belief.mean_events(bounds.parameters), range(len(bounds.graph.connections))
    paths, origins, attempts = [tuple(original_suffix)], [['original_suffix']], []

    def add(path, label):
        if not path:
            return None
        path = tuple(path)
        if path in paths:
            i = paths.index(path)
            origins[i].append(label)
            return i
        paths.append(path)
        origins.append([label])
        return len(paths)-1

    for quantile in QUANTILES:
        position = len(times)-1 if quantile == 1 else int(np.searchsorted(cumulative, quantile, side='left'))
        clock = int(times[position])
        child = replace(state, stop=stop, time=clock, last_arrival=clock, current_trip=incoming_trip,
                        unavailable_ids=state.unavailable_ids | frozenset(prefix))
        path = nominal_itinerary(bounds.graph, child, bounds.destination, nominal, menu,
            bounds.deadline, 1500, excluded_trips=(incoming_trip,), trip_blocks=blocks)
        index = add(path, f'quantile_{quantile:g}')
        attempts.append(dict(quantile=quantile, arrival=clock, candidate_index=index,
                             path=list(path) if path else [], feasible=bool(path)))
    rescue_contiguous = bool(rescue and bounds.graph.connections[rescue[0]].dep_stop == stop)
    if rescue_contiguous:
        add(rescue, 'v10_rescue_as_direct_tail')
    assert len(paths) <= MAX_FAMILY
    tails = tuple(freeze_tail_plan(bounds, path, first_action_now=False) for path in paths)
    return tails, dict(seconds=perf_counter()-started, cpu_seconds=process_time()-cpu,
        successful_arrival_probability=incoming.success_probability,
        successful_arrival_phases=[dict(arrival=int(t), probability=float(p)) for t,p in zip(times,masses)],
        attempts=attempts, sources=origins, rescue_contiguous=rescue_contiguous,
        phase_tail_cells=len(times)*len(tails))


def run_root(identity, prefix, graph, metadata, stops, fit, journey, history, previous, blocks):
    started = perf_counter()
    belief, state = context(graph, metadata, journey, identity[2], fit)
    goal, deadline = stops[journey['od']['destination']], journey['od']['deadline_seconds']
    path = tuple(history['witness'])
    assert path[:len(prefix)] == prefix
    assert graph.connections[path[len(prefix)]].trip_id != graph.connections[prefix[-1]].trip_id
    parameters = {c.route: fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
    bounds = BoundaryRecourseBounds(belief, parameters, graph, goal, deadline)
    setup_seconds = perf_counter()-started
    rescue = previous['paths']['v10_fixed_tail_rescue']['rescue']
    tails, generation = generate(bounds, state, prefix, path[len(prefix):], rescue, blocks)
    row = dict(date=identity[0], query=identity[1], decision=identity[2], prefix=list(prefix),
        root=state_spec(state), deadline=deadline, destination=goal, setup_seconds=setup_seconds,
        candidates=[dict(index=i, sources=generation['sources'][i], plan=asdict(t)) for i,t in enumerate(tails)],
        generation=generation, historical_upper_seconds=previous['upper_seconds'])
    if generation['phase_tail_cells'] > MAX_PHASE_TAIL_CELLS or generation['seconds'] > PILOT_LIMIT_SECONDS:
        return dict(row, status='stopped_before_phase_evaluation', stop_reason='Declared candidate/phase work bound')
    cpu = process_time()
    plan, value = bounds.evaluate(state, prefix, tails)
    phase_cpu_seconds = process_time()-cpu
    original = previous['upper_seconds']['incumbent']
    if abs(value.fixed_tail_uppers[0] - original) > 1e-7:
        return dict(row, status='original_upper_mismatch', value=asdict(value), expected_original=original)
    static_index = int(np.argmin(value.fixed_tail_uppers))
    static = value.fixed_tail_uppers[static_index]
    selected_mass = [sum(p.probability for p in value.phases if p.selected == i) for i in range(len(tails))]
    observed = previous['next_transfer_boundary']['state']['clock']
    recorded_choice = next((p for p in value.phases if p.arrival == observed), None)
    selected = recorded_choice.selected if recorded_choice else 0
    cap = deadline-state.time
    row.update(status='complete', exact=asdict(value), static_best_index=static_index,
        static_best_upper=static, adaptive_upper=value.upper,
        family_improvement_seconds=original-static,
        arrival_information_improvement_seconds=static-value.upper,
        improvement_vs_original_seconds=original-value.upper,
        improvement_vs_v10_best_seconds=previous['upper_seconds']['v10_best']-value.upper,
        prefix_failure_probability=max(0., 1.-value.prefix_success_probability),
        prefix_failure_cap=cap, prefix_failure_expected_loss=max(0.,1.-value.prefix_success_probability)*cap,
        selected_probability_by_candidate=selected_mass, phase_cpu_seconds=phase_cpu_seconds,
        candidate_plus_phase_seconds=generation['seconds']+value.seconds,
        observed_boundary_diagnostic=dict(arrival=observed, on_grid=recorded_choice is not None,
            selected=selected, tail=list(plan.tail_for(observed).primary),
            conditional_uppers=list(recorded_choice.conditional_uppers) if recorded_choice else None,
            source='Previously recorded incumbent replay arrival; queried only after candidate generation and exact integration.'))
    return row


def main():
    declared = protocol()
    write_json(HERE / 'recourse_protocol.json', declared)
    started = perf_counter()
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    shared_fit_seconds = perf_counter()-started
    journeys = {(r['date'],r['query']):r for r in json.loads(ARCHIVE.read_text())['rows']
                if r['method']=='evolving_rollout'}
    histories = {key(r):r for r in json.loads(HISTORIES.read_text())['rows']}
    inventory = {key(r):r for r in json.loads((HERE / 'transfer_inventory.json').read_text())['rows']}
    graphs, shared, rows = {}, [], []
    result = dict(protocol=declared, shared_fit_seconds=shared_fit_seconds, shared_graphs=shared, rows=rows)
    for identity, prefix in zip(CASES,PREFIXES):
        day, query, _ = identity
        if day not in graphs:
            started = perf_counter()
            graph, _, stops, _, metadata, _ = dated_graph(data,day,'gtfs_station_max',True)
            blocks = build_trip_blocks(graph)
            graphs[day] = graph,metadata,stops,blocks
            shared.append(dict(date=day, graph_and_blocks_seconds=perf_counter()-started))
        graph,metadata,stops,blocks = graphs[day]
        row = run_root(identity,prefix,graph,metadata,stops,fit,journeys[day,query],
                       histories[identity],inventory[identity],blocks)
        rows.append(row)
        write_json(HERE / 'recourse_results.json', result)
        print(identity,row['status'], 'family',len(row['candidates']),
            'static',row.get('static_best_upper'),'adaptive',row.get('adaptive_upper'),
            'seconds',row.get('candidate_plus_phase_seconds'),flush=True)
        if row['status'] != 'complete' or row['exact']['seconds'] > PILOT_LIMIT_SECONDS:
            result['status'] = 'stopped_at_declared_work_or_consistency_bound'
            write_json(HERE / 'recourse_results.json', result)
            return
    result['status'] = 'complete'
    write_json(HERE / 'recourse_results.json', result)


if __name__ == '__main__':
    main()
