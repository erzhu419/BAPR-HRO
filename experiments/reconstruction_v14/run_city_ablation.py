"""Isolate information changes at the three frozen V13 waiting roots."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
from itertools import product
import json
from math import sqrt
import os
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v13.factored_run import (
    ATOMS, BANKS, HORIZON, MAX_NODES, hidden_draws,
)
from experiments.reconstruction_v13.platform_run import (
    ARCHIVE, CASES, INVENTORY, SOURCE, V10, dated_graph, fit_training, key,
    load_city, load_plan, prepare_waiting_context,
)
from src.causal_prefix import PrefixBudgetExceeded, causal_prefix_lower
from src.factored_lower import product_trip_bank
from src.platform_transit import build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map


EXTENDED = 1080
SUFFIX_DRAWS = 2
EXACT_SUFFIX_LIMIT = 32
V13_RESULTS = ROOT / 'experiments/reconstruction_v13/factored_results.json'
ARMS = (
    'v13', 'tail_720', 'tail_1080', 'r_revealed', 'r_hidden',
    'suffix_labels_revealed', 'suffix_labels_hidden',
)


def read(path):
    return json.loads(path.read_text())


def write(value, path):
    path.write_text(json.dumps(value, indent=2) + '\n')


def random_tape(seed):
    return np.random.default_rng(np.random.SeedSequence(seed))


def make_event(law, slots, states, index):
    departure = int(law.times[index][states[index]])
    return VehicleEvent(departure-slots[index].scheduled, bool(law.canceled[index][states[index]]),
                        departure, int(law.times[index+1][states[index+1]]))


def suffix_draw(law, cut, anchor, uniforms):
    """FFBS on the frozen root grid conditional on one exact cut state."""
    point = np.full(len(law.times[cut]), -np.inf)
    point[anchor] = 0.
    forward = [point]
    for index in range(cut+1, len(law.times)):
        forward.append(logsumexp(forward[-1][:, None] + law._transition(index-1), axis=0)
                       + law.emissions[index])
    chosen = np.zeros(len(law.times), dtype=np.intp)
    chosen[cut] = anchor
    for index in range(len(law.times)-1, cut, -1):
        weights = forward[index-cut]
        if index+1 < len(law.times):
            weights = weights + law._transition(index)[:, chosen[index+1]]
        cumulative = np.cumsum(np.exp(weights-logsumexp(weights)))
        cumulative /= cumulative[-1]
        chosen[index] = np.searchsorted(cumulative, uniforms[index], side='right')
    return chosen


def suffix_support(law, cut):
    """Count valid suffixes per cut state, saturating at the representation cap."""
    counts = np.isfinite(law.emissions[-1]).astype(int)
    for index in range(len(law.times)-2, cut-1, -1):
        counts = np.minimum(EXACT_SUFFIX_LIMIT+1, np.isfinite(law._transition(index)) @ counts)
        counts *= np.isfinite(law.emissions[index])
    return counts


def conditional_factor(belief, parameters, cid, original_seed, suffix_seed):
    trip, cut = belief._positions[cid, 'departure']
    law, slots = belief._law(trip, parameters), belief._trips[trip]
    shape = BANKS*ATOMS, len(slots)
    original = law.draw_uniforms_batch(random_tape(original_seed).random(shape)[:, ::-1])
    uniform = random_tape(suffix_seed).random(shape)
    complete, split, observations = [], [], []
    counts = suffix_support(law, cut)
    next_backward = law._backward_from(len(law.times)-1, cut+1)
    for states, tape in zip(original, uniform):
        anchor = states[cut]
        events = {slots[i].cid: make_event(law, slots, states, i) for i in range(0, len(slots), 2)}
        complete.append(events)
        split.append(events)
        extension = suffix_draw(law, cut, anchor, tape)
        revised = states.copy()
        revised[cut+1:] = extension[cut+1:]
        split.append({**events, **{slots[i].cid: make_event(law, slots, revised, i)
                                 for i in range(cut, len(slots), 2)}})
        logs = law._transition(cut)[anchor] + law.emissions[cut+1] + next_backward
        prob = np.exp(logs-logsumexp(logs))
        times = law.times[cut+1]
        sd = sqrt(max(0., float(prob @ ((times-float(prob @ times))**2))))
        observations.append(dict(departure=int(law.times[cut][anchor]), canceled=bool(law.canceled[cut][anchor]),
                                 original_arrival=events[cid].arrival, split_arrival=split[-1][cid].arrival,
                                 conditional_arrival_sd=sd, valid_suffixes_capped=int(counts[anchor])))
    return complete, split, observations


def public_third(graph, scan, initial, goal, deadline, hidden, transfers):
    """Choose one alternative using only scheduled connections and platform readiness."""
    dep = np.array([[c.dep_time for c in graph.connections]], dtype=np.int32)
    arr = np.array([[c.arr_time for c in graph.connections]], dtype=np.int32)
    canceled = np.zeros(dep.shape, dtype=np.uint8)
    ready = max(initial.time, initial.last_arrival+graph.stops[initial.stop].min_transfer_time)
    ranked = []
    for c in graph.get_connections_from(initial.stop):
        if (not c.pickup_allowed or c.trip_id in hidden or not c.trip_id
                or c.id in initial.used_connections or not ready <= c.dep_time <= ready+HORIZON
                or c.arr_time > deadline):
            continue
        state = replace(initial, time=c.arr_time, stop=c.arr_stop, current_trip=c.trip_id,
                        last_arrival=c.arr_time, last_connection=c.id,
                        used_connections=initial.used_connections | {c.id})
        arrival = int(scan.platform_arrivals(dep, arr, canceled, state, goal, deadline, transfers)[0])
        if arrival < deadline or state.stop == goal:
            ranked.append((arrival, c.dep_time, c.id, c.trip_id))
    if not ranked:
        return None
    arrival, dep, cid, trip = min(ranked)
    return dict(trip=trip, cid=cid, scheduled_departure=dep, scheduled_goal_arrival=arrival,
                scored_connections=len(ranked))


def bank_model(graph, common, clocks, atoms, initial, goal, deadline, scan, transfers):
    bank, labels = product_trip_bank(graph, common, atoms)
    packed = tuple(np.repeat(array[None, :], len(bank), axis=0) for array in clocks)
    for trip in atoms:
        for row, atom in enumerate(labels[trip]):
            for cid, event in atoms[trip][atom].items():
                packed[0][row, cid] = event.departure
                packed[1][row, cid] = event.arrival
                packed[2][row, cid] = event.canceled
    model = build_platform_model(graph, bank, initial.stop, goal, initial.time, deadline,
                                 initial_state=initial, poll_interval=1)

    @lru_cache(None)
    def remaining(world, stop, clock, trip, last_arrival, last_connection, used):
        public = replace(initial, stop=stop, time=clock, current_trip=trip, last_arrival=last_arrival,
                         last_connection=last_connection, used_connections=used)
        return float(scan.platform_arrivals(*(x[world:world+1] for x in packed),
                                           public, goal, deadline, transfers)[0]-clock)

    def oracle(world, public):
        return remaining(world, public.stop, public.time, public.current_trip, public.last_arrival,
                         public.last_connection, public.used_connections)

    return model, oracle, labels


def solve(panel, graph, arm, groups, *, horizon=HORIZON, delayed=False):
    model, oracle, labels = panel
    elapsed = perf_counter()
    nodes, transitions, oracle_evaluations = 0, 0, 0
    values, boundaries = [], Counter()
    root_used = model.initial_state.used_connections
    root_trip = model.initial_state.current_trip

    def first_boarding(state):
        return any((graph.connections[cid].trip_id or f'connection:{cid}') != root_trip
                   for cid in state.used_connections-root_used)

    for indices in groups:
        if nodes >= MAX_NODES:
            return dict(arm=arm, status='budget_exceeded', completed_groups=len(values),
                        information_nodes=nodes, transition_evaluations=transitions,
                        solver_seconds=perf_counter()-elapsed)
        submodel = replace(model, scenarios=tuple(replace(model.scenarios[i], probability=1./len(indices))
                                                  for i in indices))
        try:
            value = causal_prefix_lower(submodel, lambda i, s: oracle(indices[i], s),
                elapsed_horizon=horizon, max_nodes=MAX_NODES-nodes,
                stop_when=(lambda state: False) if delayed else first_boarding,
                factor_labels={trip: tuple(label[i] for i in indices) for trip, label in labels.items()})
        except PrefixBudgetExceeded as error:
            return dict(arm=arm, status='budget_exceeded', completed_groups=len(values),
                        information_nodes=nodes+error.nodes-1, transition_evaluations=transitions+error.transitions,
                        solver_seconds=perf_counter()-elapsed)
        nodes += value.information_nodes
        transitions += value.transition_evaluations
        oracle_evaluations += value.oracle_evaluations
        weight = len(indices)/len(model.scenarios)
        values.append((weight, value))
        for kind, mass in value.boundary_kind_mass.items():
            boundaries[kind] += weight*mass

    def weighted(attribute):
        return sum(p*getattr(value, attribute) for p, value in values)

    return dict(arm=arm, status='complete', world_count=len(model.scenarios), groups=len(groups),
                information_nodes=nodes, transition_evaluations=transitions, oracle_evaluations=oracle_evaluations,
                value=weighted('value'), clairvoyant_root_mean=weighted('clairvoyant_root_mean'),
                expected_boundary_elapsed=weighted('expected_boundary_elapsed'),
                singleton_before_boundary_probability=weighted('singleton_before_boundary_probability'),
                boundary_kind_mass=dict(boundaries), solver_seconds=perf_counter()-elapsed)


def evaluate(case, cpu, deadline_probe=False):
    os.sched_setaffinity(0, {cpu})
    started = perf_counter()
    day, query, decision = case
    base_seed = [20261113, int(day.replace('-', '')), query, decision, 13]
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
    journey = next(r for r in read(ARCHIVE)['rows']
                   if (r['date'], r['query'], r['method']) == (day, query, 'evolving_rollout'))
    previous = next(r for r in read(INVENTORY)['rows'] if key(r) == case)
    archived = next(r for r in read(V10)['rows'] if key(r) == case)
    v13 = {r['bank']: r for r in read(V13_RESULTS)['rows'] if key(r) == case and r['arm'] == 'product'}
    prefix = tuple(previous['next_transfer_boundary']['prefix'])
    belief, initial = prepare_waiting_context(graph, metadata, journey, decision, fit, prefix,
                                               previous['recorded_incumbent']['rides'])
    goal, deadline = stops[journey['od']['destination']], journey['od']['deadline_seconds']
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                  for c in graph.connections}
    plan = load_plan(archived['fixed_primary_tail_rescue']['plan'], prefix)
    target_ids = (plan.primary[0], plan.rescue[0])
    hidden = tuple(graph.connections[cid].trip_id for cid in target_ids)
    with TemporaryDirectory(prefix='bapr_v14_city_ablation_') as tmp:
        scan = TemporalScan(graph, build_temporal_kernel(Path(tmp)/'scan.so'))
        transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
        third = public_third(graph, scan, initial, goal, deadline, hidden, transfers)
        rest = sample_clocks(belief, parameters, random_tape(base_seed+[0]), BANKS,
                             templates=template_trip_map(belief, parameters))
        data_factors = [conditional_factor(belief, parameters, cid, base_seed+[1, j],
                                           base_seed+[14, 1, j]) for j, cid in enumerate(target_ids)]
        r_factors = None if third is None else hidden_draws(belief, parameters, third['trip'], base_seed+[14, 0])
        r_ids = [c.id for c in graph.connections if third is not None and c.trip_id == third['trip']]
        setup_seconds = perf_counter()-started
        banks = []
        for bank in range(1 if deadline_probe else BANKS):
            before = perf_counter()
            dep, arr, cnl = (x[bank] for x in rest)
            common = JourneyScenario({c.id: VehicleEvent(int(dep[c.id])-c.dep_time, bool(cnl[c.id]),
                int(dep[c.id]), int(arr[c.id])) for c in graph.connections}, minute=60, publications={})
            start, end = bank*ATOMS, (bank+1)*ATOMS
            atoms = {trip: row[0][start:end] for trip, row in zip(hidden, data_factors)}
            if third is not None:
                r = third['trip']
                third_atoms = [{cid: common.events[cid] for cid in r_ids}, *r_factors[start+1:end]]
            if deadline_probe:
                arms = []
                if third is not None:
                    panel = bank_model(graph, common, (dep, arr, cnl), {**atoms, r: third_atoms},
                                       initial, goal, deadline, scan, transfers)
                    arms.append(solve(panel, graph, 'r_hidden_full_deadline', [list(range(ATOMS**3))],
                                      horizon=deadline-initial.time, delayed=True))
                banks.append(dict(bank=bank, arms=arms, wall_seconds=perf_counter()-before))
                print(case, bank, arms, flush=True)
                continue
            split = {trip: row[1][start*SUFFIX_DRAWS:end*SUFFIX_DRAWS]
                     for trip, row in zip(hidden, data_factors)}
            suffix = {trip: row[2][start:end] for trip, row in zip(hidden, data_factors)}
            ab = bank_model(graph, common, (dep, arr, cnl), atoms, initial, goal, deadline, scan, transfers)
            full = [list(range(ATOMS**2))]
            arms = [solve(ab, graph, 'v13', full),
                    solve(ab, graph, 'tail_720', full, delayed=True),
                    solve(ab, graph, 'tail_1080', full, horizon=EXTENDED, delayed=True)]
            expected = v13[bank]
            baseline = arms[0]
            if (baseline['status'] != 'complete' or abs(baseline['value']-expected['value']) > 1e-7
                    or abs(baseline['clairvoyant_root_mean']-expected['clairvoyant_root_mean']) > 1e-7):
                raise AssertionError(f'V13 baseline failed to reconstruct for {case}, bank {bank}')
            if third is not None:
                panel = bank_model(graph, common, (dep, arr, cnl), {**atoms, r: third_atoms},
                                   initial, goal, deadline, scan, transfers)
                labels = panel[2][r]
                groups = [[i for i, label in enumerate(labels) if label == j] for j in range(ATOMS)]
                arms.extend((solve(panel, graph, 'r_revealed', groups),
                             solve(panel, graph, 'r_hidden', [list(range(len(labels)))])))
                del panel
            panel = bank_model(graph, common, (dep, arr, cnl), split, initial, goal, deadline, scan, transfers)
            labels = list(zip(*(panel[2][trip] for trip in hidden)))
            groups = [[i for i, (a, b) in enumerate(labels) if (a % SUFFIX_DRAWS, b % SUFFIX_DRAWS) == group]
                      for group in product(range(SUFFIX_DRAWS), repeat=2)]
            arms.extend((solve(panel, graph, 'suffix_labels_revealed', groups),
                         solve(panel, graph, 'suffix_labels_hidden', [list(range(len(labels)))])))
            del panel
            by_name = {r['arm']: r for r in arms}
            for less, more in (('v13', 'tail_720'), ('tail_720', 'tail_1080'),
                               ('r_revealed', 'r_hidden'), ('suffix_labels_revealed', 'suffix_labels_hidden')):
                if less in by_name and all(by_name[k]['status'] == 'complete' for k in (less, more)):
                    a, b = by_name[less], by_name[more]
                    if abs(a['clairvoyant_root_mean']-b['clairvoyant_root_mean']) > 1e-7:
                        raise AssertionError(f'Mismatched realized worlds for {case}, bank {bank}: {less}, {more}')
                    if b['value']+1e-7 < a['value']:
                        raise AssertionError(f'Information ordering failed for {case}, bank {bank}: {less}, {more}')
            banks.append(dict(bank=bank, arms=arms, conditional_suffix=suffix,
                              wall_seconds=perf_counter()-before))
            print(case, bank, [(r['arm'], r['status'], round(r['value'], 2) if 'value' in r else None)
                               for r in arms], flush=True)
    return dict(case=case, root_stop=initial.stop, root_clock=initial.time, goal=goal, deadline=deadline,
                setup_seconds=setup_seconds, third=third, ab_trip_ids=hidden, ab_target_ids=target_ids, banks=banks)


def summarize(case):
    indexed = [{r['arm']: r for r in row['arms']} for row in case['banks']]
    v13 = {r['bank']: r['policies']['first_visible']['mean_loss'] for r in read(V13_RESULTS)['rows']
           if key(r) == tuple(case['case']) and r['arm'] == 'product'}
    result = dict(case=case['case'], arms={}, comparisons={}, v13_first_visible_same_world_gaps={})
    for arm in ARMS:
        rows = [r[arm] for r in indexed if arm in r]
        good = [r for r in rows if r['status'] == 'complete']
        result['arms'][arm] = dict(completed=len(good), attempted=len(rows),
            total_solver_seconds=sum(r['solver_seconds'] for r in rows),
            mean_complete_value=mean(r['value'] for r in good) if good else None)
        if arm in ('v13', 'tail_720', 'tail_1080'):
            gaps = [v13[row['bank']]-arms[arm]['value'] for row, arms in zip(case['banks'], indexed)
                    if arms[arm]['status'] == 'complete']
            result['v13_first_visible_same_world_gaps'][arm] = dict(pairs=len(gaps), gaps=gaps,
                mean_gap=mean(gaps) if gaps else None, mean_gap_beyond_60=mean(gaps)-60 if gaps else None)
    for old, new in (('v13', 'tail_720'), ('v13', 'tail_1080'), ('tail_720', 'tail_1080'),
                     ('r_revealed', 'r_hidden'), ('suffix_labels_revealed', 'suffix_labels_hidden'),
                     ('v13', 'suffix_labels_hidden')):
        paired = [r[new]['value']-r[old]['value'] for r in indexed
                  if old in r and new in r and r[old]['status'] == r[new]['status'] == 'complete']
        result['comparisons'][f'{new}_minus_{old}'] = dict(pairs=len(paired), differences=paired,
            same_realized_worlds=(old, new) != ('v13', 'suffix_labels_hidden'),
            mean_difference=mean(paired) if paired else None)
    observations = [x for r in case['banks'] for group in r['conditional_suffix'].values()
                    for x in group if not x['canceled']]
    original_support = []
    split_support = []
    for bank in case['banks']:
        for group in bank['conditional_suffix'].values():
            by_departure = {}
            for row in group:
                if not row['canceled']:
                    by_departure.setdefault(row['departure'], []).append(row)
            for rows in by_departure.values():
                original = {row['original_arrival'] for row in rows}
                original_support.append(len(original))
                split_support.append(len(original | {row['split_arrival'] for row in rows}))
    result['distinct_operating_departure_conditions'] = len(original_support)
    result['singleton_arrival_conditions_v13'] = original_support.count(1)
    result['singleton_arrival_conditions_split'] = split_support.count(1)
    result['operating_cut_anchors'] = len(observations)
    result['anchors_exceeding_exact_suffix_limit'] = sum(
        r['valid_suffixes_capped'] > EXACT_SUFFIX_LIMIT for r in observations)
    result['mean_original_law_conditional_arrival_sd'] = mean(
        r['conditional_arrival_sd'] for r in observations) if observations else None
    result['same_cut_changed_arrival'] = sum(r['original_arrival'] != r['split_arrival'] for r in observations)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-deadline', action='store_true',
                        help='Attempt only a full-deadline hidden-R solve on bank 0 of each root.')
    probe = parser.parse_args().full_deadline
    path = HERE / ('city_full_deadline_probe.json' if probe else 'city_ablation_results.json')
    protocol = dict(cases=CASES, banks=1 if probe else BANKS, ab_atoms=ATOMS,
        horizon='journey deadline' if probe else HORIZON, extended_horizon=None if probe else EXTENDED,
        node_limit_per_arm_bank=MAX_NODES, exact_suffix_path_limit_per_cut=EXACT_SUFFIX_LIMIT,
        inputs=[str(p.relative_to(ROOT)) for p in (SOURCE, ARCHIVE, INVENTORY, V10, V13_RESULTS)],
        v13='Same rest and A/B draws, frozen local clocks and observations, complete actions. Verify both means per bank.',
        third='Earliest scheduled arrival among legal root departures through transfer-ready+720, excluding A/B. '
              'Break ties by departure then connection ID. Hold the candidate fixed for all banks. The original '
              'rest draw for R and three independent draws form its factor. Reveal its index at the root or hide it.',
        suffix='For each original A/B chain retain all latent event states up to its target departure. Keep the original '
               'suffix as draw zero and use a second FFBS draw conditional on the cut state and root evidence on the '
               'same frozen grid. This preserves trip marginals in expectation. The full product gives identical '
               'realized worlds to the hidden-label arm and the arm that knows both suffix draw indices at the root.',
        tail='Full local execution until the 720/1080 second clock boundary, without stopping at the first ride. '
             'As in V13, an atomic ride may cross the cutoff. All arms then use the V13 clairvoyant tail.',
        cost='Bank wall time includes packing and model construction. Arms sharing the same bank reuse a '
             'memoized oracle; solver times include that shared-work benefit and are not independent timings.',
        inference='Point diagnostics on 8 common outer draws per root, not independent product rows. The suffix '
                  'label comparison diagnoses this nested sampling scheme and is not a population bias estimate. '
                  'No incomplete solve contributes a value. No G2 certification.')
    if probe:
        protocol['tail'] = ('Only bank 0 per root. Hide R and retain local causal decisions up to the normal '
                            'journey deadline. There is no earlier clairvoyant boundary. Other rest trips '
                            'remain known to the finite-bank planner.')
        protocol['inference'] = 'Single-bank feasibility probe. No incomplete solve contributes a value.'
    cases = []
    result = dict(protocol=protocol, status='running', roots=cases)
    write(result, path)
    cpus = sorted(os.sched_getaffinity(0))
    if probe:
        cpus.reverse()
    with ProcessPoolExecutor(max_workers=min(len(cpus), len(CASES))) as pool:
        futures = {pool.submit(evaluate, case, cpus[i % len(cpus)], probe): case for i, case in enumerate(CASES)}
        for future in as_completed(futures):
            try:
                cases.append(future.result())
            except Exception as error:
                cases.append(dict(case=futures[future], error=f'{type(error).__name__}: {error}'))
            cases.sort(key=lambda r: CASES.index(r['case']))
            write(result, path)
    result['status'] = 'failed' if any('error' in case for case in cases) else 'complete'
    if not probe:
        result['summary'] = [summarize(case) for case in cases if 'error' not in case]
    write(result, path)
    if result['status'] == 'failed':
        raise SystemExit('One or more roots failed; see the retained result.')


if __name__ == '__main__':
    main()
