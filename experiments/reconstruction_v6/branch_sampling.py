"""Fixed-public-root first-stratum variance and recovery-cost attribution."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
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
from experiments.reconstruction_v5.run_sparse_certificates import context
from src.causal_planning_router import BudgetedPolicySearch, CausalPlanningRouter
from src.committed_transit import WAIT, build_committed_model
from src.continuation_router import nominal_action
from src.first_branch_belief import FirstBranchSampler
from src.first_branch_bank import FirstBranchBank

_INPUT = None


def initialize(data, fit, archived):
    global _INPUT
    day = archived['date']
    graph, actual, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
    _INPUT = graph, actual, stops, metadata, fit, archived


def evaluate(spec):
    kind, replicate, n, method, cid = spec
    graph, actual, stops, metadata, fit, archived = _INPUT
    wall, cpu = perf_counter(), process_time()
    belief, state = context(graph, metadata, archived, 0, fit)
    priors = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    deadline = archived['od']['deadline_seconds']
    destination = stops[archived['od']['destination']]
    router = CausalPlanningRouter(graph, priors, event_belief=belief, menu='full', algorithm='rollout',
                                  samples=n, work_budget=512,
                                  max_time=deadline - archived['od']['ready_seconds'])
    router.online_mean = router.online_cancel = False
    router._initial_cancels = {route: value['cancel_rate'] for route, value in priors.items()}
    router.route(state.stop, destination, archived['od']['ready_seconds'])
    parameters = router._parameters()
    c = graph.connections[cid]
    transfer = (graph.stops[state.stop].min_transfer_time
                if state.current_trip is not None and state.current_trip != c.trip_id else 0)
    ready = max(state.time, state.last_arrival + transfer)
    sampler = FirstBranchSampler(belief, parameters, cid, ready, deadline)
    seed = ([int(archived['date'].replace('-', '')), 0] if kind == 'archived'
            else [60017, replicate])
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    if method == 'iid':
        scenes = belief.sample(parameters, rng, n)
        bank = [(1 / n, scene) for scene in scenes]
    elif method == 'stratified':
        bank = sampler.stratified(rng, n // 2, n - n // 2)
    else:
        common = FirstBranchBank(belief, parameters, rng, n)
        bank = common.for_action(cid, ready, deadline,
                                  n_catch=n // 2 if method == 'coupled_half' else None)
    model = build_committed_model(graph, bank, state.stop, destination, state.time, deadline,
                                  lookahead=1500, initial_state=state)

    def tail(public, current, actions):
        boardings = {a[1] for a in actions if a[0] == 'select'}
        return (nominal_action(graph, public, destination, current.mean_events(parameters),
                               router.menu, deadline, 1500, root_boardings=boardings, minute=60)
                if boardings else WAIT)

    # Every witness completes. The report then checks whether actual work fits
    # the common512-transition envelope; it never drops slow/failed worlds.
    planner = BudgetedPolicySearch(model, belief, tail, router._advance_belief, rng, 1000000,
                                   deadline - state.time)
    costs, caught, departures = [], [], []
    for world, (_, scene) in zip(model.scenarios, bank):
        event = scene.events[cid]
        caught.append(not event.canceled and ready <= event.departure <= min(deadline, ready + 720))
        departures.append(event.departure)
        costs.append(planner._simulate(world, first=('select', cid)))
    weights = np.array([weight for weight, _ in bank])
    value = float(weights @ costs)
    physical = actual.events[cid]
    return {'kind': kind, 'replicate': replicate, 'n': n, 'method': method, 'cid': cid,
            'seed': seed, 'probability_catch': sampler.probability(True),
            'value': value, 'world_costs': costs, 'caught': caught,
            'departure_clocks': departures, 'weights': weights.tolist(),
            'transitions': planner.work, 'within512': planner.work <= 512,
            'wall_seconds': perf_counter() - wall, 'cpu_seconds': process_time() - cpu,
            'actual_departure': physical.departure,
            'actual_caught': not physical.canceled and ready <= physical.departure <= min(deadline, ready + 720)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=16)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--methods', nargs='+', choices=('iid', 'stratified', 'coupled_half', 'coupled_proportional'),
                        default=['iid', 'stratified'])
    parser.add_argument('--output', type=Path, default=HERE / 'branch_sampling_results.json')
    args = parser.parse_args()
    args.output = args.output.resolve()
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    archive_path = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    archive = json.loads(archive_path.read_text())
    archived = next(r for r in archive['rows'] if r['date'] == '2023-07-27'
                    and r['query'] == 0 and r['method'] == 'evolving_rollout')
    specs = [('archived', 0, 4, method, cid) for method in args.methods for cid in (7588, 8341)]
    specs += [('replicate', seed, n, method, cid) for seed in range(args.replicates)
              for n in (4, 8) for method in args.methods for cid in (7588, 8341)]
    settings = {'context': ['2023-07-27', 0, 0], 'root_actions': [7588, 8341],
                'sample_counts': [4, 8], 'replicate_seeds': [[60017, i] for i in range(args.replicates)],
                'methods': args.methods, 'stratum_allocation': 'Half catch/fail except coupled_proportional rounds n*p; exact masses.',
                'tail': 'Same fixed deterministic public-history nominal policy, original posterior updates.',
                'work_comparison': 'Complete every witness, report actual transitions against512; no selective truncation.',
                'workers': args.workers, 'archived_context': archived,
                'scope': 'Conditional simulation variance on one already examined public root; not independent real-day evidence.'}
    protocol = freeze_replay_inputs(args.output, data=data, forecasts={'dates': {}}, training=fit,
                                    settings=settings, sources=[str(source), str(archive_path),
                                    'src/first_branch_belief.py', 'src/first_branch_bank.py',
                                    'experiments/reconstruction_v5/run_sparse_certificates.py',
                                    str(Path(__file__))], implementation='v6_exact_first_stratum_variance')
    folder = args.output.with_name(args.output.stem + '_cases')
    folder.mkdir(exist_ok=True)
    rows, pending = [], []
    for spec in specs:
        path = folder / ('_'.join(map(str, spec)) + '.json')
        if path.exists():
            rows.append(json.loads(path.read_text()))
        else:
            pending.append((spec, path))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(data, fit, archived)) as pool:
        jobs = {pool.submit(evaluate, spec): (spec, path) for spec, path in pending}
        for future in as_completed(jobs):
            spec, path = jobs[future]
            row = future.result()
            path.write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
            print(*spec, round(row['value'], 2), row['transitions'], round(row['wall_seconds'], 2), flush=True)
    groups = defaultdict(list)
    for row in rows:
        if row['kind'] == 'replicate':
            groups[row['n'], row['method'], row['cid']].append(row)
    summary = {}
    for (n, method, cid), group in sorted(groups.items()):
        conditional = {}
        for caught in (True, False):
            values = [cost for row in group for cost, event in zip(row['world_costs'], row['caught']) if event == caught]
            conditional['catch' if caught else 'fail'] = {'n': len(values), 'mean': float(np.mean(values)) if values else None,
                                                        'variance': float(np.var(values, ddof=1)) if len(values) > 1 else None}
        summary[f'{n}/{method}/{cid}'] = {'n_replicates': len(group),
            'mean': float(np.mean([row['value'] for row in group])),
            'variance': float(np.var([row['value'] for row in group], ddof=1)),
            'mean_transitions': float(np.mean([row['transitions'] for row in group])),
            'max_transitions': max(row['transitions'] for row in group),
            'all_within512': all(row['within512'] for row in group),
            'mean_wall_seconds': float(np.mean([row['wall_seconds'] for row in group])),
            'mean_cpu_seconds': float(np.mean([row['cpu_seconds'] for row in group])),
            'zero_failure_replicates': sum(all(row['caught']) for row in group), 'conditional_costs': conditional}
    result = {'settings': settings, 'frozen_protocol': str(protocol.relative_to(ROOT)), 'summary': summary,
              'rows': sorted(rows, key=lambda r: (r['kind'], r['replicate'], r['n'], r['method'], r['cid']))}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
