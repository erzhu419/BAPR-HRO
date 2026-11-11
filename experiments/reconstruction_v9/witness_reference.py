"""Exact fixed-recourse witness checks on the V8 intermediate histories."""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from experiments.reconstruction_v8.budget_run import SOURCE, write_json
from experiments.reconstruction_v9.search_reference import ARCHIVE, HISTORIES, elapsed
from src.first_branch_belief import FirstBranchSampler
from src.recourse_itinerary_router import (
    _ConditionalNominal, build_trip_blocks, expand_trip_blocks, nominal_itinerary, trip_key,
)
from src.risk_bounds import CommitmentBounds

SEARCH = HERE / 'search_reference_128.json'


def evaluate(belief, parameters, graph, state, destination, deadline, blocks, cid, incumbent=None):
    wall, cpu = perf_counter(), process_time()
    bounds = CommitmentBounds(belief, parameters, graph, destination, deadline)
    lookahead = 25 * belief.minute
    nominal = belief.mean_events(parameters)
    menu = set(range(len(graph.connections)))
    risk = bounds.commit(state, cid)
    trip = belief._trip_by_cid[cid]

    def path_from(at, *, events=nominal, excluded=()):
        return nominal_itinerary(graph, at, destination, events, menu, deadline, lookahead,
            minute=belief.minute, excluded_trips=excluded, trip_blocks=blocks)

    primary = incumbent
    if primary is None and risk.catch_probability > 0.:
        first = graph.connections[cid]
        if first.arr_stop == destination:
            primary = (cid,)
        else:
            law = FirstBranchSampler(belief, parameters, cid, risk.ready, deadline).conditional_law(True)
            conditional = _ConditionalNominal(nominal, belief, trip, law)
            arrival = min(deadline, round(conditional[cid].arrival))
            after = replace(state, stop=first.arr_stop, time=arrival, last_arrival=arrival,
                current_trip=trip_key(first), unavailable_ids=state.unavailable_ids | {cid})
            tail = path_from(after, events=conditional)
            if tail:
                primary = expand_trip_blocks(graph, (cid, *tail), destination, state.unavailable_ids, blocks)
    failed = replace(state, time=risk.failure_clock)
    rescue = path_from(failed, excluded=(trip,)) if primary and risk.failure_probability else ()
    generation = elapsed(wall, cpu)
    row = dict(connection=cid, primary=primary or (), rescue=rescue or (), catch_probability=risk.catch_probability,
        failure_probability=risk.failure_probability, failure_clock=risk.failure_clock, generation=generation)
    if not primary:
        return dict(row, upper_seconds=None, status='no_feasible_primary')
    wall, cpu = perf_counter(), process_time()
    value = bounds.with_rescue(state, primary, rescue or (), lookahead=lookahead)
    evaluation = elapsed(wall, cpu)
    return dict(row, status='evaluated', evaluation=evaluation, upper_seconds=value.upper,
        primary_only_upper_seconds=value.primary.upper, primary_completion_probability=value.primary.success_probability,
        rescue_completion_probability=value.rescue.success_probability, weighted_rescue_gain_seconds=(
            value.first.failure_probability * value.rescue.success_gain), work=bounds.stats)


def main():
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    archived = json.loads(ARCHIVE.read_text())['rows']
    journeys = {(r['date'], r['query']): r for r in archived if r['method'] == 'evolving_rollout'}
    histories = json.loads(HISTORIES.read_text())['rows']
    actions = {(r['date'], r['query'], r['decision']): r for r in json.loads(SEARCH.read_text())['rows']}
    graphs, rows = {}, []
    for case in histories:
        day, query, number = (case[k] for k in ('date', 'query', 'decision'))
        if day not in graphs:
            graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
            parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                          for c in graph.connections}
            graphs[day] = graph, metadata, stops, parameters, build_trip_blocks(graph)
        graph, metadata, stops, parameters, blocks = graphs[day]
        journey = journeys[day, query]
        destination = stops[journey['od']['destination']]
        deadline = journey['od']['deadline_seconds']
        wall, cpu = perf_counter(), process_time()
        belief, state = context(graph, metadata, journey, number, fit)
        context_time = elapsed(wall, cpu)
        if (state.stop, state.time, destination) != (case['stop'], case['clock'], case['destination']):
            raise AssertionError('The reconstructed root differs from the V8 archive')
        args = belief, parameters, graph, state, destination, deadline, blocks
        original = tuple(case['witness'])
        same = evaluate(*args, original[0], incumbent=original)
        if abs(same['primary_only_upper_seconds'] - case['upper']) > 1e-8:
            raise AssertionError('The incumbent upper bound did not reproduce')
        search = actions[day, query, number]
        action = search['prediction']['action']
        alternative = None
        if action[0] == 'select' and action[1] != original[0]:
            # A separate fresh belief prevents the incumbent from prewarming this cost control.
            fresh, _ = context(graph, metadata, journey, number, fit)
            alternative = evaluate(fresh, *args[1:], action[1])
            alternative['uct_proposal_wall_seconds'] = sum(
                search[stage]['wall_seconds'] for stage in ('model_setup', 'planning'))
        upper = min(case['upper'], same['upper_seconds'], alternative['upper_seconds']
                    if alternative and alternative['upper_seconds'] is not None else case['upper'])
        rows.append(dict(date=day, query=query, decision=number, clock=state.time, stop=state.stop,
            destination=destination, context_setup=context_time, incumbent_upper_seconds=case['upper'],
            incumbent_v8_generation_and_eval_wall_seconds=case['witness_seconds'],
            same_primary=same, uct128_alternative=alternative, best_upper_seconds=upper))
        print(day, query, number, 'old', round(case['upper'], 2),
              'same+rescue', round(same['upper_seconds'], 2), 'alternative', (
                  alternative['upper_seconds'] if alternative else None), flush=True)
    summary = dict(mean_incumbent_upper_seconds=mean(r['incumbent_upper_seconds'] for r in rows),
        mean_same_primary_upper_seconds=mean(r['same_primary']['upper_seconds'] for r in rows),
        mean_best_upper_seconds=mean(r['best_upper_seconds'] for r in rows),
        improved=sum(r['best_upper_seconds'] < r['incumbent_upper_seconds'] - 1e-8 for r in rows),
        alternative_count=sum(r['uct128_alternative'] is not None for r in rows),
        median_same_primary_wall_seconds=median(sum(r['same_primary'][stage]['wall_seconds']
            for stage in ('generation', 'evaluation')) for r in rows))
    write_json(HERE / 'witness_reference_results.json', dict(rows=rows, summary=summary,
        cpu_affinity=sorted(os.sched_getaffinity(0)), histories=str(HISTORIES.relative_to(ROOT)),
        source=str(SOURCE.relative_to(ROOT)), proposal_results=str(SEARCH.relative_to(ROOT)),
        scope='Frozen V8 committed no-publication contract. At most one rescue on root failure, deadline cap on later misses.',
        timing='Shared graph, trip blocks and training excluded. Each witness starts with a fresh posterior. Incumbent root path reused. Alternative proposal cost logged separately.'))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
