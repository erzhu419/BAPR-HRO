"""Same compiled witnesses: exact full-model integration versus Monte Carlo."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training, freeze_replay_inputs
from experiments.reconstruction_v5.run_sparse_certificates import context, setup
from experiments.reconstruction_v6.itinerary_mc_evaluator import FixedWitness, execute_fixed_witness
from src.committed_transit import build_committed_model
from src.first_branch_bank import FirstBranchBank
from src.first_branch_belief import FirstBranchSampler
from src.recourse_itinerary_router import nominal_itinerary, build_trip_blocks, expand_trip_blocks, _ConditionalNominal, trip_key
from src.risk_bounds import CommitmentBounds

_CASES = None


def thresholds(belief, parameters, graph, paths):
    mean = belief.mean_events(parameters)
    return {cid: (graph.connections[cid].dep_time if mean[cid].departure is None else round(mean[cid].departure))
            for path in paths for cid in path}


def entry_ready(state, cid, first_now, departures, graph, deadline):
    c, clock = graph.connections[cid], state.time
    transfer = 0 if state.current_trip is None or c.trip_id == state.current_trip else graph.stops[state.stop].min_transfer_time
    if not first_now:
        while clock < deadline and (c.dep_time > clock + 1500 or
                departures[cid] > max(clock, state.last_arrival + transfer) + 720):
            clock += 60
    return max(clock, state.last_arrival + transfer)


def construct_case(graph, actual, stops, metadata, fit, archived, number):
    before = perf_counter()
    destination = stops[archived['od']['destination']]
    original = setup(graph, metadata, archived, number, fit, destination)
    belief, state, parameters = original['belief'], original['state'], original['parameters']
    deadline = archived['od']['deadline_seconds']
    blocks, menu = build_trip_blocks(graph), {c.id for c in graph.connections}
    nominal = belief.mean_events(parameters)
    bounds = CommitmentBounds(belief, parameters, graph, destination, deadline)

    def path_from(at, excluded=(), events=nominal):
        return nominal_itinerary(graph, at, destination, events, menu, deadline, 1500,
                                  minute=60, excluded_trips=excluded, trip_blocks=blocks)

    primary = path_from(state)
    if not primary:
        raise ValueError('The declared city diagnostic needs a nonempty nominal itinerary')
    candidates = [primary]
    forced = ([7588 if primary[0] != 7588 else 8341] if number == 0 else
              [c.id for c in sorted(graph.get_connections_from(state.stop),
                                    key=lambda c: (abs(c.dep_time - state.time), c.id))
               if c.id != primary[0] and c.trip_id != state.current_trip and c.dep_time <= state.time + 1500
               and c.id not in state.unavailable_ids and c.pickup_allowed])
    for cid in forced:
        first = bounds.commit(state, cid)
        if not first.catch_probability:
            continue
        c = graph.connections[cid]
        if c.arr_stop == destination:
            alternative = (cid,)
        else:
            sampler = FirstBranchSampler(belief, parameters, cid, first.ready, deadline)
            conditional = _ConditionalNominal(nominal, belief, belief._trip_by_cid[cid], sampler.conditional_law(True))
            time = min(deadline, round(conditional[cid].arrival))
            after = replace(state, stop=c.arr_stop, time=time, last_arrival=time, current_trip=trip_key(c),
                            unavailable_ids=state.unavailable_ids | {cid})
            tail = path_from(after, events=conditional)
            alternative = expand_trip_blocks(graph, (cid,) + tail, destination, state.unavailable_ids, blocks) if tail is not None else None
        if alternative:
            candidates.append(alternative)
            break
    if len(candidates) != 2:
        raise ValueError('No declared representative alternative is supported at this root')
    policies = []
    for label, primary in zip(('nominal', 'alternative'), candidates):
        first = bounds.commit(state, primary[0])
        failed = replace(state, time=first.failure_clock)
        rescue = path_from(failed, (belief._trip_by_cid[primary[0]],)) or ()
        policy = FixedWitness(primary, rescue, True)
        departures = thresholds(belief, parameters, graph, (primary, rescue))
        policies.append({'label': label, 'policy': asdict(policy), 'departures': departures})
    construction = perf_counter() - before
    # A separate fresh context measures analytic evaluation of these fixed
    # paths, including their root threshold reconstruction, without the large
    # graph search needed to discover them.
    analytic = []
    for item in policies:
        started, cpu = perf_counter(), process_time()
        fresh, at = context(graph, metadata, archived, number, fit)
        policy = FixedWitness(**item['policy'])
        freeze = thresholds(fresh, parameters, graph, (policy.primary, policy.rescue))
        if freeze != item['departures']:
            raise ValueError('Fresh analytic context changed the frozen public thresholds')
        index = CommitmentBounds(fresh, parameters, graph, destination, deadline)
        value = index.with_rescue(at, policy.primary, policy.rescue)
        analytic.append({'label': item['label'], 'value': value.upper,
            'catch_probability': value.first.catch_probability,
            'success_probability': value.primary.success_probability,
            'rescue_success_probability': value.rescue.success_probability,
            'wall_seconds': perf_counter() - started, 'cpu_seconds': process_time() - cpu,
            'work': dict(index.stats)})
    state_json = {key: sorted(value) if isinstance(value, frozenset) else value
                  for key, value in asdict(state).items()}
    physical = []
    for item in policies:
        cid = item['policy']['primary'][0]
        event = actual.events.get(cid)
        ready = entry_ready(state, cid, True, item['departures'], graph, deadline)
        physical.append({'label': item['label'], 'cid': cid, 'ready': ready,
                         'actual_departure': None if event is None else event.departure,
                         'actual_caught': None if event is None else
                         (not event.canceled and ready <= event.departure <= min(deadline, ready + 720))})
    return {'date': archived['date'], 'query': archived['query'], 'number': number,
            'state': state_json, 'destination': destination, 'deadline': deadline,
            'parameters': parameters, 'policies': policies, 'analytic': analytic,
            'common_context_path_construction_seconds': construction,
            'posthoc_physical_first_attempt': physical,
            'archived': archived, 'graph': graph, 'metadata': metadata, 'fit': fit}


def initialize(cases):
    global _CASES
    _CASES = cases


def evaluate(spec):
    case_number, seed, n, method = spec
    case = _CASES[case_number]
    started, cpu = perf_counter(), process_time()
    belief, state = context(case['graph'], case['metadata'], case['archived'], case['number'], case['fit'])
    parameters, graph = case['parameters'], case['graph']
    policies = [FixedWitness(**p['policy']) for p in case['policies']]
    freeze = thresholds(belief, parameters, graph,
                        [path for policy in policies for path in (policy.primary, policy.rescue)])
    for item in case['policies']:
        if any(freeze[cid] != value for cid, value in item['departures'].items()):
            raise ValueError('MC and analytic policies use different public thresholds')
    bank = FirstBranchBank(belief, parameters, np.random.default_rng(np.random.SeedSequence([60071, case_number, seed])), n)
    rows = []
    for item, policy in zip(case['policies'], policies):
        ready = entry_ready(state, policy.primary[0], policy.first_action_now, freeze, graph, case['deadline'])
        worlds = ([(1 / n, scene) for scene in bank.original] if method == 'iid' else
                  bank.for_action(policy.primary[0], ready, case['deadline']))
        model = build_committed_model(graph, worlds, state.stop, case['destination'], state.time,
                                      case['deadline'], lookahead=1500, initial_state=state)
        values = [execute_fixed_witness(model, scenario, graph, policy, freeze, case['deadline'])
                  for scenario in model.scenarios]
        estimate = sum(weight * value['cost'] for (weight, _), value in zip(worlds, values))
        rows.append({'label': item['label'], 'estimate': estimate,
                     'costs': [value['cost'] for value in values],
                     'weights': [weight for weight, _ in worlds],
                     'transitions': sum(value['transitions'] for value in values),
                     'successes': sum(value['arrived'] for value in values)})
    exact = {row['label']: row['value'] for row in case['analytic']}
    selected = min(rows, key=lambda row: row['estimate'])['label']
    return {'context': case_number, 'date': case['date'], 'number': case['number'], 'seed': seed,
            'n': n, 'method': method, 'estimates': rows, 'selected': selected,
            'family_regret': exact[selected] - min(exact.values()),
            'ranking_error': exact[selected] > min(exact.values()) + 1e-9,
            'wall_seconds': perf_counter() - started, 'cpu_seconds': process_time() - cpu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path, default=HERE / 'itinerary_mc_results.json')
    args = parser.parse_args()
    output = args.output.resolve()
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    archive_path = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    archive = json.loads(archive_path.read_text())
    specs = [('2023-07-26', 0, 0), ('2023-07-26', 0, 5), ('2023-07-27', 0, 0)]
    prepared = {day: dated_graph(data, day, 'gtfs_station_max', True) for day, _, _ in specs}
    cases = []
    for day, qi, number in specs:
        graph, actual, stops, _, metadata, _ = prepared[day]
        archived = next(r for r in archive['rows'] if r['date'] == day and r['query'] == qi and r['method'] == 'evolving_rollout')
        case = construct_case(graph, actual, stops, metadata, fit, archived, number)
        cases.append(case)
        print('PREPARED', day, number, [(r['label'], round(r['value'], 4)) for r in case['analytic']], flush=True)
    contexts = json.loads(json.dumps([{key: value for key, value in case.items()
                                      if key not in ('graph', 'metadata', 'fit')} for case in cases]))
    frozen_cases = [{key: value for key, value in case.items() if key not in (
                    'analytic', 'common_context_path_construction_seconds', 'posthoc_physical_first_attempt')}
                    for case in contexts]
    settings = {'contexts': frozen_cases, 'sample_counts': [4, 8], 'seeds': list(range(8)),
                'methods': ['iid', 'stratified'], 'workers': args.workers,
                'scope': 'Fixed primary/root-failure-rescue policy family; model MC error, not full-city optimal regret or new-date benefit.'}
    protocol = freeze_replay_inputs(output, data=data, forecasts={'dates': {}}, training=fit,
        settings=settings, sources=[str(source), str(archive_path), str(Path(__file__)),
        'src/risk_bounds.py', 'src/recourse_itinerary_router.py', 'src/first_branch_bank.py',
        'src/first_branch_belief.py', 'experiments/reconstruction_v6/itinerary_mc_evaluator.py',
        'experiments/reconstruction_v5/run_sparse_certificates.py'], implementation='v6_same_itinerary_exact_vs_mc')
    folder = output.with_name(output.stem + '_cases')
    folder.mkdir(exist_ok=True)
    rows, pending = [], []
    trials = [(ci, seed, n, method) for ci in range(len(cases)) for seed in range(8)
              for n in (4, 8) for method in ('iid', 'stratified')]
    for trial in trials:
        path = folder / ('_'.join(map(str, trial)) + '.json')
        if path.exists():
            rows.append(json.loads(path.read_text()))
        else:
            pending.append((trial, path))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(cases,)) as pool:
        jobs = {pool.submit(evaluate, trial): (trial, path) for trial, path in pending}
        for future in as_completed(jobs):
            trial, path = jobs[future]
            row = future.result()
            path.write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
    output.write_text(json.dumps({'settings': settings, 'contexts': contexts,
        'frozen_protocol': str(protocol.relative_to(ROOT)),
        'rows': sorted(rows, key=lambda r: (r['context'], r['seed'], r['n'], r['method']))}, indent=2) + '\n')
    print('COMPLETED', len(rows), 'trials', flush=True)


if __name__ == '__main__':
    main()
