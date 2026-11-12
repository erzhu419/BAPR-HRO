"""All-state sufficient-state test and fixed-stream online cost comparison."""

import argparse
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld, Executor
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from experiments.reconstruction_v37.controller import (
    current_diagnostics, has_legal_continue, onboard_state, restore_single_snapshot,
)
from experiments.reconstruction_v38.planner import FastPlanner
from experiments.reconstruction_v38.run import save, ordered, restore_node, bank_science, physical_outcome
from experiments.reconstruction_v39.state import DecisionEngine
from experiments.reconstruction_v39.evaluator import evaluate_actions

HERE = Path(__file__).resolve().parent
V38 = HERE.parent/'reconstruction_v38'
VARIANTS = ('v38', 'fullmemo', 'frontier')


def source_bank(source):
    name = source.replace('/', '_').removesuffix('.json.gz')
    return V38/'banks'/f'{name}_fast.json.gz'


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
        original = read(source_bank(entry['source']))
        for variant in ordered(VARIANTS, index):
            path = HERE/'banks'/f'{name}_{variant}.json.gz'
            if path.exists():
                if not read(path)['exact_v38']:
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
                exact_v38=bank_science(result) == bank_science(original))
            save(path, result)
            print('bank', index, variant, 'exact', result['exact_v38'],
                'CPU', round(result['complete_cost']['process_cpu'], 4), flush=True)
            if not result['exact_v38']:
                raise AssertionError(f'Fixed-stream mismatch, preserved {path}')


class DecisionExecutor(Executor):
    """Keep actual observations and execution intact; project only planning."""
    def __init__(self, graph, planner, deadline, *, engine=None, **kwargs):
        super().__init__(graph, planner, deadline, **kwargs)
        self.engine = engine

    def choose_platform(self, platform, entry, ready, history, visible, observed, clock):
        if self.engine is None:
            return super().choose_platform(platform, entry, ready, history, visible, observed, clock)
        return self.engine.platform(platform, entry, ready, history, visible, observed, clock,
                                    self.choose, method=self.method)

    def choose_onboard(self, ride, cid, clock, history):
        if self.engine is None:
            return super().choose_onboard(ride, cid, clock, history)
        return self.engine.onboard(ride, cid, clock, history, method=self.method)


class OnlineExecutor(DecisionExecutor):
    def __init__(self, graph, planner, deadline, *, root_index, day, manifest, variant):
        engine = None if variant == 'v38' else DecisionEngine(planner, variant=variant)
        super().__init__(graph, planner, deadline, engine=engine, method='rolling_fixed', waiting='combined')
        self.root_index, self.day, self.variant = root_index, day, variant
        self.manifest = [r for r in manifest if r['state']['root_index'] == root_index]
        self.gates = []

    def choose_onboard(self, ride, cid, clock, history):
        if self.engine is None:
            diagnostic = current_diagnostics(self.planner, ride, cid, clock, history)
        else:
            base, base_receipt = self.engine.onboard(ride, cid, clock, history, method='rolling_fixed')
            nominal, nominal_receipt = self.engine.onboard(ride, cid, clock, history, method='nominal')
            base, nominal = ('alight' if x else 'continue' for x in (base, nominal))
            diagnostic = dict(base=base, alternative='continue' if base == 'alight' else 'alight',
                nominal_diagnostic=nominal, cheap_agreed=base == nominal,
                cheap_receipts=dict(base=base_receipt, nominal=nominal_receipt))
        if not has_legal_continue(ride, cid):
            return diagnostic['base'] == 'alight', dict(triggered=False, **diagnostic)
        state = onboard_state(ride, cid, clock, history, self.deadline, root_index=self.root_index,
            day=self.day, count=128, diagnostics=diagnostic)
        entry = next((r for r in self.manifest if r['state'] == state), None)
        if entry is None:
            raise ValueError('New public state in exact replacement; no new stream is sampled')
        gate = evaluate(self.graph, self.planner.model, self.planner.parameters, state, ride,
            history, entry['seed'], self.variant)
        gate.update(source=entry['source'], cheap_receipts=diagnostic['cheap_receipts'])
        self.gates.append(gate)
        return gate['selected_action'] == 'alight', dict(triggered=True, seed=entry['seed'],
            selected_action=gate['selected_action'], paired=gate['paired'], cost=gate['cost'])


def one(index, name, graph, metadata, mapping, root, physical_fit, emission_fit, frames, methods):
    tick, cpu = perf_counter(), process_time()
    manifest = read(V38/'seed_manifest.json') if name in VARIANTS else None
    model, parameters = setup(graph, physical_fit)
    ride = root_ride(graph, root)
    planner = FastPlanner(graph, model, parameters, root['deadline'])
    history = SingleMessageHistory(model, parameters, fit=emission_fit,
        frames=relevant_frames(frames, ride.trips(model), mapping), source_to_local=mapping)
    if name in VARIANTS:
        executor = OnlineExecutor(graph, planner, root['deadline'], root_index=index,
            day=root['day'], manifest=manifest, variant=name)
    else:
        implementation, method_name = name.split('__')
        method, waiting = methods[method_name]
        engine = DecisionEngine(planner, variant='frontier') if implementation == 'frontier' else None
        executor = DecisionExecutor(graph, planner, root['deadline'], engine=engine, method=method, waiting=waiting)
    outcome = executor.run(ride, root['start'], history, ArchivedWorld(metadata))
    return dict(outcome=outcome, gates=getattr(executor, 'gates', []), cost=dict(
        total_process_cpu=process_time()-cpu, total_wall=perf_counter()-tick,
        planner_stats=dict(planner.stats),
        decision_stats={} if executor.engine is None else dict(executor.engine.stats)), answers_reused=0)


def replays(data, messages, physical_fit, emission_fit, selection, protocol):
    methods = protocol['cheap_methods']
    arms = [*VARIANTS, *(f'{impl}__{name}' for impl in ('v38', 'frontier') for name in methods)]
    for index, root in enumerate(selection['selected']):
        tick, cpu = perf_counter(), process_time()
        graph, _, _, mapping, metadata, _ = context(data, root['day'])
        preparation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        frames = messages['dates'][root['day']]['frames']
        for name in ordered(arms, index):
            path = HERE/'roots'/f'{index:03}_{name}.json.gz'
            if path.exists():
                if not read(path)['exact_original']:
                    raise AssertionError(f'Preserved failed replay: {path}')
                continue
            row = one(index, name, graph, metadata, mapping, root, physical_fit, emission_fit, frames, methods)
            original_name = 'fast' if name in VARIANTS else name.split('__')[1]
            old = read(V38/'roots'/f'{index:03}_{original_name}.json.gz')
            exact = physical_outcome(row['outcome']) == physical_outcome(old['outcome'])
            if name in VARIANTS:
                row['all_banks_exact_v38'] = all(bank_science(g) == bank_science(read(source_bank(g['source'])))
                    for g in row['gates'])
                exact = exact and row['all_banks_exact_v38']
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
    protocol, manifest = read(HERE/'protocol.json'), read(V38/'seed_manifest.json')
    if len(manifest) != 148:
        raise ValueError('Use the complete fixed V38 cohort')
    tick, cpu = perf_counter(), process_time()
    _, data, messages, physical_fit, selection = inputs()
    emission_fit = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    print('common_input_preparation', dict(wall=perf_counter()-tick, process_cpu=process_time()-cpu), flush=True)
    if args.stage == 'banks':
        benchmark(manifest, data, physical_fit, emission_fit)
    else:
        banks = list((HERE/'banks').glob('*.json.gz'))
        if len(banks) != 444 or not all(read(p)['exact_v38'] for p in banks):
            raise ValueError('Finish the 148 x 3 exact conditional checks before online replays')
        replays(data, messages, physical_fit, emission_fit, selection, protocol)


if __name__ == '__main__':
    main()
