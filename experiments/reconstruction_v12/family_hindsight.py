"""Paired attribution within the frozen four-tail family on the existing tape."""

from dataclasses import replace
import json
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
from experiments.reconstruction_v8.budget_run import SOURCE, write_json
from experiments.reconstruction_v9.search_reference import ARCHIVE
from experiments.reconstruction_v12.recourse_rescue_run import load_plan
from experiments.reconstruction_v12.transfer_inventory import CASES, key
from src.boundary_recourse import BoundaryRecoursePlan
from src.committed_transit import build_committed_model
from src.simulator import LazyTripScenario, VehicleEvent
from src.tail_rescue import TailRescueBounds, TailRescuePlan
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map


WORLDS = 64


def paired_summary(values, analytic=None):
    values = np.asarray(values, dtype=float)
    result = dict(mean=float(values.mean()),
                  standard_error=float(values.std(ddof=1) / np.sqrt(len(values))),
                  minimum=float(values.min()), maximum=float(values.max()))
    if analytic is not None:
        result.update(exact_expectation=float(analytic),
                      sample_minus_exact=float(values.mean() - analytic))
    return result


def execute(model, world, graph, active, boundary=None):
    state = model.initial_state
    prefix_live = boundary is not None
    selected = arrival = None
    for _ in range(model.horizon):
        if not model.actions(state) or not active.primary:
            break
        action = active.action(state, graph, minute=60)
        outcome = model.transition(world, state, action)
        caught = any(kind == 'board' for kind, _, _ in
                     outcome.state.feedback[len(state.feedback):])
        active = active.advance(action, caught)
        state = outcome.state
        if prefix_live and action[0] == 'select':
            if not caught:
                break
            if not active.primary:
                arrival = state.time
                choices = dict(boundary.choices)
                if arrival not in choices:
                    raise AssertionError('A sampled successful prefix arrived outside the frozen phase table')
                selected = choices[arrival]
                active = boundary.tail_for(arrival)
                prefix_live = False
        if outcome.terminal:
            break
    else:
        raise AssertionError('A frozen itinerary exhausted the committed execution horizon')
    return (state.time - model.initial_state.time + model.terminal_cost(world, state),
            selected, arrival)


def run_root(identity, graph, metadata, fit, journey, frozen, previous_lower, kernel):
    started = perf_counter()
    belief, state = context(graph, metadata, journey, identity[2], fit)
    goal = graph.connections[frozen['candidates'][0]['primary'][-1]].arr_stop
    deadline = journey['od']['deadline_seconds']
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                  for c in graph.connections}
    prefix = tuple(frozen['prefix'])
    tails = tuple(load_plan(row) for row in frozen['candidates'])
    assert len(tails) == 4
    prefix_departures = tails[frozen['restored_candidate_index']].departures
    prefix_plan = TailRescuePlan(prefix, departures=prefix_departures)
    full_plans = tuple(replace(tail, primary=prefix + tail.primary, first_action_now=True,
        failure_index=None if tail.failure_index is None else len(prefix) + tail.failure_index,
        departures=tuple(dict(prefix_departures + tail.departures).items())) for tail in tails)
    boundary = BoundaryRecoursePlan(prefix_plan, tails,
        tuple((phase['arrival'], phase['selected']) for phase in frozen['exact']['phases']))
    bounds = TailRescueBounds(belief, parameters, graph, goal, deadline)
    analytic = np.array([bounds.evaluate(state, plan).upper for plan in full_plans])
    analytic_error = float(np.max(np.abs(analytic - frozen['exact']['fixed_tail_uppers'])))
    if analytic_error > 1e-7:
        raise AssertionError(f'Concatenated fixed policies changed the exact values: {analytic_error}')
    setup_seconds = perf_counter() - started

    # Identical shape, seed and full-trip template draw as lower_run.py.
    started = perf_counter()
    seed = [20261112, int(identity[0].replace('-', '')), identity[1], identity[2]]
    packed = sample_clocks(belief, parameters,
        np.random.default_rng(np.random.SeedSequence(seed)), WORLDS,
        templates=template_trip_map(belief, parameters))
    draw_seconds = perf_counter() - started
    scan = TemporalScan(graph, kernel)
    transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
    oracle = scan.arrivals(*packed, state, goal, deadline, transfer=transfers) - state.time
    if abs(float(oracle.mean()) - previous_lower['mean_clairvoyant']) > 1e-9:
        raise AssertionError('Regenerated full-graph oracle differs from the saved 64-world tape')

    started = perf_counter()
    needed = sorted({cid for plan in full_plans for cid in plan.primary + plan.rescue})
    scenes = [(1. / WORLDS, LazyTripScenario({cid: VehicleEvent(
        int(dep[cid]) - graph.connections[cid].dep_time, bool(canceled[cid]),
        int(dep[cid]), int(arr[cid])) for cid in needed}, minute=60, publications={}))
        for dep, arr, canceled in zip(*packed)]
    model = build_committed_model(graph, scenes, state.stop, goal, state.time, deadline,
                                  initial_state=state)
    fixed_losses, adaptive_losses, observations = [], [], []
    for number, world in enumerate(model.scenarios):
        fixed = [execute(model, world, graph, plan)[0] for plan in full_plans]
        adaptive, selected, arrival = execute(model, world, graph, prefix_plan, boundary)
        if selected is not None and adaptive != fixed[selected]:
            raise AssertionError('Arrival selection differs from executing the same frozen full plan')
        if selected is None and any(cost != deadline-state.time for cost in fixed):
            raise AssertionError('Prefix failure must cap every member of the common-prefix family')
        if oracle[number] > min(fixed):
            raise AssertionError('Full temporal oracle exceeds a feasible fixed family member')
        fixed_losses.append(fixed)
        adaptive_losses.append(adaptive)
        observations.append(dict(world=number, boundary_arrival=arrival, selected_tail=selected,
            fixed_losses=fixed, arrival_policy_loss=adaptive, hindsight_family_loss=min(fixed),
            hindsight_tail_indices=[i for i, loss in enumerate(fixed) if loss == min(fixed)],
            full_oracle_loss=int(oracle[number])))
    execution_seconds = perf_counter() - started
    fixed = np.array(fixed_losses, dtype=float)
    adaptive = np.array(adaptive_losses, dtype=float)
    best_index = frozen['static_best_index']  # Frozen analytic optimum, never sample-selected.
    old, best, hindsight = fixed[:, 0], fixed[:, best_index], fixed.min(axis=1)
    adaptive_exact = frozen['adaptive_upper']
    components = dict(
        old_minus_best_fixed=paired_summary(old-best, analytic[0]-analytic[best_index]),
        best_fixed_minus_arrival_policy=paired_summary(best-adaptive, analytic[best_index]-adaptive_exact),
        arrival_policy_minus_hindsight_family=paired_summary(adaptive-hindsight),
        hindsight_family_minus_full_oracle=paired_summary(hindsight-oracle))
    return dict(date=identity[0], query=identity[1], decision=identity[2], worlds=WORLDS,
        seed=seed, prefix=list(prefix), family_size=len(tails), static_best_index=best_index,
        analytic_fixed_uppers=list(map(float, analytic)), analytic_replay_max_error=analytic_error,
        original_lower_oracle_mean=previous_lower['mean_clairvoyant'],
        values=dict(old_incumbent=paired_summary(old, analytic[0]),
            best_fixed=paired_summary(best, analytic[best_index]),
            arrival_policy=paired_summary(adaptive, adaptive_exact),
            hindsight_family=paired_summary(hindsight), full_oracle=paired_summary(oracle)),
        components=components, total_gap=paired_summary(old-oracle),
        arrival_policy_vs_static_different_worlds=int(np.count_nonzero(best != adaptive)),
        hindsight_improves_arrival_policy_worlds=int(np.count_nonzero(hindsight < adaptive)),
        prefix_failure_worlds=sum(r['selected_tail'] is None for r in observations),
        setup_seconds=setup_seconds, draw_seconds=draw_seconds, execution_seconds=execution_seconds,
        rows=observations)


def main():
    frozen_path = HERE / 'recourse_rescue_results.json'
    lower_path = HERE / 'lower_results.json'
    frozen = {key(r): r for r in json.loads(frozen_path.read_text())['rows']}
    lower = {key(r): r for r in json.loads(lower_path.read_text())['summary'] if r['horizon'] == 60}
    journeys = {(r['date'], r['query']): r for r in json.loads(ARCHIVE.read_text())['rows']
                if r['method'] == 'evolving_rollout'}
    protocol = dict(cases=CASES, worlds=WORLDS,
        tape='Regenerate lower_run.py single64 draw with seed [20261112,int(date),query,decision], unchanged full posterior and templates; no new confirmation worlds.',
        family=str(frozen_path.relative_to(ROOT)),
        fixed='Same four saved tails, original prefix prepended, rescue index shifted by prefix length, every frozen nominal departure retained.',
        adaptive='Execute original prefix, then use saved exact arrival-phase choice through BoundaryRecoursePlan.tail_for. No policy regeneration or posterior re-evaluation.',
        hindsight='Minimum realized committed loss among the same four full policies, selected after seeing the complete world.',
        oracle='Full temporal station-MCT scan on the same packed world; its mean must reproduce the saved lower run.',
        decomposition='old-oracle = (old-best_fixed)+(best_fixed-arrival_policy)+(arrival_policy-hindsight_family)+(hindsight_family-oracle)',
        inference='Paired Monte Carlo standard errors on the reused64 IID worlds; exact saved analytic expectations shown where available. No new confidence certificate.',
        interpretation='First two components need not be nonnegative in each world. Last component includes frozen family, prefix, commitment and temporal-relaxation differences; neither it nor family hindsight is the full causal optimum.')
    result = dict(protocol=protocol, rows=[])
    write_json(HERE / 'family_hindsight_protocol.json', protocol)
    started = perf_counter()
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    result['shared_training_seconds'] = perf_counter()-started
    graphs = {}
    with TemporaryDirectory(prefix='bapr_v12_family_') as tmp:
        kernel = build_temporal_kernel(Path(tmp) / 'temporal.so')
        for identity in CASES:
            if identity[0] not in graphs:
                graph, _, _, _, metadata, _ = dated_graph(data, identity[0], 'gtfs_station_max', True)
                graphs[identity[0]] = graph, metadata
            row = run_root(identity, *graphs[identity[0]], fit, journeys[identity[:2]],
                           frozen[identity], lower[identity], kernel)
            result['rows'].append(row)
            write_json(HERE / 'family_hindsight_results.json', result)
            print(identity, {name: round(v['mean'], 3) for name, v in row['components'].items()},
                  'seconds', round(row['execution_seconds'], 3), flush=True)
    result['status'] = 'complete'
    write_json(HERE / 'family_hindsight_results.json', result)


if __name__ == '__main__':
    main()
