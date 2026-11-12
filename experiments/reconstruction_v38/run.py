"""Fixed-stream exactness benchmark and fully computed online control timing."""

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v33.nodes import Ride, Platform, root_ride
from experiments.reconstruction_v33.run import jsonable
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld, Executor
from experiments.reconstruction_v34.forecast import ForecastPlanner
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from experiments.reconstruction_v37.controller import (
    current_diagnostics, has_legal_continue, onboard_state, restore_single_snapshot,
)
from experiments.reconstruction_v38.evaluator import evaluate_actions
from experiments.reconstruction_v38.planner import FastPlanner

HERE = Path(__file__).resolve().parent
V37 = HERE.parent/'reconstruction_v37'
VARIANTS = ('legacy', 'shared', 'fast')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.gz':
        with gzip.open(path, 'xt') as handle:
            json.dump(value, handle, default=jsonable, allow_nan=False)
    else:
        path.write_text(json.dumps(value, default=jsonable, allow_nan=False, indent=2)+'\n')


def ordered(items, index):
    items = list(items)
    offset = index % len(items)
    return items[offset:]+items[:offset]


def restore_node(value):
    if value['kind'] == 'ride':
        return Ride(value['first_cid'], tuple(value['path']), {
            e['cid']: None if e['next'] is None else restore_node(e['next']) for e in value['exits']})
    return Platform(value['stop'], {e['cid']: restore_node(e['next']) for e in value['rides']})


def make_manifest():
    """Only public states and RNG assignment; no old answers or outcomes."""
    path = HERE/'seed_manifest.json'
    if path.exists():
        return read(path)
    rows = []
    for group in ('reference_cases', 'policy_cases'):
        for source in sorted((V37/group).glob('*.json.gz')):
            old = read(source)
            rows.append(dict(source=str(source.relative_to(V37)), state=old['state'],
                seed=old['seed'], count=old['count']))
    if len(rows) != 148:
        raise ValueError('Expected the complete 148-bank V37 ledger')
    save(path, rows)
    return rows


def bank_science(result):
    value = dict(actions=result['actions'], paired=result['paired'],
        selected_action=result['selected_action'], seed=result['seed'], count=result['count'],
        public_history=result['public_history'],
        rows={arm: [{k: row[k] for k in ('loss', 'status', 'trace', 'decisions', 'world_index')}
                    for row in rows] for arm, rows in result['rows'].items()})
    # In-memory non-arrival intervals are tuples; saved receipts use JSON lists.
    return json.loads(json.dumps(value, default=jsonable, allow_nan=False))


def physical_outcome(outcome):
    value = dict(loss=outcome['loss'], status=outcome['status'], decisions=outcome['decisions'],
        trace=[{k: v for k, v in event.items() if k != 'receipt'} for event in outcome['trace']])
    return json.loads(json.dumps(value, default=jsonable, allow_nan=False))


def evaluate(graph, model, parameters, state, node, history, seed, variant):
    result = evaluate_actions(graph, model, parameters, state['deadline'], node,
        {k: state[k] for k in ('kind', 'cid', 'clock')}, history,
        state['actions']['alternative'], state['actions']['base'],
        count=state['count'], seed=seed, variant=variant)
    result['rows'] = dict(alternative=result['rows']['nominal'], base=result['rows']['combined'])
    costs = result['cost']['action_continuations']
    result['cost']['action_continuations'] = dict(alternative=costs['nominal'], base=costs['combined'])
    result['actions'] = state['actions']
    result['paired']['orientation'] = 'alternative action loss minus base action loss; negative favors alternative'
    result['state'] = state
    return result


def benchmark(manifest, data, physical_fit, emission_fit):
    for index, entry in enumerate(manifest):
        state = entry['state']
        name = entry['source'].replace('/', '_').removesuffix('.json.gz')
        graph = context(data, state['day'])[0]
        original = read(V37/entry['source'])
        for variant in ordered(VARIANTS, index):
            path = HERE/'banks'/f'{name}_{variant}.json.gz'
            if path.exists():
                if not read(path)['exact_v37']:
                    raise AssertionError(f'Preserved failed bank: {path}')
                continue
            tick, cpu = perf_counter(), process_time()
            model, parameters = setup(graph, physical_fit)
            node = restore_node(state['node'])
            history = restore_single_snapshot(model, parameters, emission_fit,
                state['public_history'], state['clock'])
            result = evaluate(graph, model, parameters, state, node, history, entry['seed'], variant)
            result['complete_cost'] = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
            result.update(source=entry['source'], variant=variant,
                exact_v37=bank_science(result) == bank_science(original))
            save(path, result)
            print('bank', index, variant, 'exact', result['exact_v37'],
                'CPU', round(result['complete_cost']['process_cpu'], 4), flush=True)
            if not result['exact_v37']:
                raise AssertionError(f'Fixed-stream equivalence failed; preserved {path}')


class OnlineExecutor(Executor):
    def __init__(self, graph, planner, deadline, *, root_index, day, manifest, variant):
        super().__init__(graph, planner, deadline, method='rolling_fixed', waiting='combined')
        self.root_index, self.day, self.variant = root_index, day, variant
        self.manifest = [r for r in manifest if r['state']['root_index'] == root_index]
        self.gates = []

    def choose_onboard(self, ride, cid, clock, history):
        diagnostic = current_diagnostics(self.planner, ride, cid, clock, history)
        if not has_legal_continue(ride, cid):
            return diagnostic['base'] == 'alight', dict(triggered=False, **diagnostic)
        state = onboard_state(ride, cid, clock, history, self.deadline, root_index=self.root_index,
            day=self.day, count=128, diagnostics=diagnostic)
        entry = next((r for r in self.manifest if r['state'] == state), None)
        if entry is None:
            raise ValueError('New public state in an exact replacement; no new bank authorized by protocol')
        gate = evaluate(self.graph, self.planner.model, self.planner.parameters, state, ride,
            history, entry['seed'], self.variant)
        gate.update(source=entry['source'], cheap_receipts=diagnostic['cheap_receipts'])
        self.gates.append(gate)
        return gate['selected_action'] == 'alight', dict(triggered=True, seed=entry['seed'],
            selected_action=gate['selected_action'], paired=gate['paired'], cost=gate['cost'])


def online_one(index, variant, graph, metadata, mapping, root, physical_fit, emission_fit, frames):
    tick, cpu = perf_counter(), process_time()
    # Charged IO contains only the fixed public-state-to-stream assignment.
    manifest = read(HERE/'seed_manifest.json')
    model, parameters = setup(graph, physical_fit)
    ride = root_ride(graph, root)
    planner = (FastPlanner if variant == 'fast' else ForecastPlanner)(graph, model, parameters, root['deadline'])
    history = SingleMessageHistory(model, parameters, fit=emission_fit,
        frames=relevant_frames(frames, ride.trips(model), mapping), source_to_local=mapping)
    executor = OnlineExecutor(graph, planner, root['deadline'], root_index=index,
        day=root['day'], manifest=manifest, variant=variant)
    outcome = executor.run(ride, root['start'], history, ArchivedWorld(metadata))
    return dict(outcome=outcome, gates=executor.gates, cost=dict(
        total_process_cpu=process_time()-cpu, total_wall=perf_counter()-tick,
        planner_stats=dict(planner.stats)), answers_reused=0)


def cheap_one(graph, metadata, mapping, root, physical_fit, emission_fit, frames, method, waiting):
    tick, cpu = perf_counter(), process_time()
    model, parameters = setup(graph, physical_fit)
    ride = root_ride(graph, root)
    planner = FastPlanner(graph, model, parameters, root['deadline'])
    history = SingleMessageHistory(model, parameters, fit=emission_fit,
        frames=relevant_frames(frames, ride.trips(model), mapping), source_to_local=mapping)
    executor = Executor(graph, planner, root['deadline'], method=method, waiting=waiting)
    outcome = executor.run(ride, root['start'], history, ArchivedWorld(metadata))
    return dict(outcome=outcome, cost=dict(total_process_cpu=process_time()-cpu,
        total_wall=perf_counter()-tick, planner_stats=dict(planner.stats)))


def replays(data, messages, physical_fit, emission_fit, selection, protocol):
    methods = protocol['cheap_methods']
    for index, root in enumerate(selection['selected']):
        tick, cpu = perf_counter(), process_time()
        graph, _, _, mapping, metadata, _ = context(data, root['day'])
        preparation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        frames = messages['dates'][root['day']]['frames']
        for name in ordered([*VARIANTS, *methods], index):
            path = HERE/'roots'/f'{index:03}_{name}.json.gz'
            if path.exists():
                if not read(path)['exact_original']:
                    raise AssertionError(f'Preserved failed replay: {path}')
                continue
            if name in VARIANTS:
                row = online_one(index, name, graph, metadata, mapping, root, physical_fit, emission_fit, frames)
                old = read(V37/'roots'/f'{index:03}.json.gz')
                exact = physical_outcome(row['outcome']) == physical_outcome(old['outcome'])
                row['all_banks_exact_v37'] = all(bank_science(g) == bank_science(read(V37/g['source']))
                    for g in row['gates'])
                exact = exact and row['all_banks_exact_v37']
            else:
                row = cheap_one(graph, metadata, mapping, root, physical_fit, emission_fit, frames, *methods[name])
                if name in ('nominal', 'combined_rolling'):
                    old_name = 'rolling_nominal' if name == 'nominal' else name
                    old = read(HERE.parent/'reconstruction_v36/roots'/f'{index:03}.json.gz')['rows']['public_messages'][old_name]
                else:
                    old = read(V37/'cheap_roots'/f'{index:03}.json.gz')['rows']['public_messages'][name]
                exact = physical_outcome(row['outcome']) == physical_outcome(old['outcome'])
            row.update(root_index=index, root_id=root['root_id'], day=root['day'], method=name,
                preparation=preparation, exact_original=exact)
            save(path, row)
            print('root', index, name, 'exact', exact, 'loss', row['outcome']['loss'],
                'CPU', round(row['cost']['total_process_cpu'], 4), flush=True)
            if not exact:
                raise AssertionError(f'Original trajectory or bank changed; preserved {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('banks', 'online'))
    args = parser.parse_args()
    protocol = read(HERE/'protocol.json')
    manifest = make_manifest()
    tick, cpu = perf_counter(), process_time()
    _, data, messages, physical_fit, selection = inputs()
    emission_fit = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    print('common_input_preparation', dict(wall=perf_counter()-tick, process_cpu=process_time()-cpu), flush=True)
    if args.stage == 'banks':
        benchmark(manifest, data, physical_fit, emission_fit)
    else:
        banks = list((HERE/'banks').glob('*.json.gz'))
        if len(banks) != 444 or not all(read(p)['exact_v37'] for p in banks):
            raise ValueError('Finish all 148 x 3 exact reference checks before online replays')
        replays(data, messages, physical_fit, emission_fit, selection, protocol)


if __name__ == '__main__':
    main()
