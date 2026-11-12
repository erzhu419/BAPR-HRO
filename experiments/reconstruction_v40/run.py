"""Frozen budget curves and genuinely stopped full-journey policy replays."""

import argparse
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.conditional_reference import summarize_pairs
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld, Executor
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from experiments.reconstruction_v37.controller import current_diagnostics, has_legal_continue, onboard_state, restore_single_snapshot
from experiments.reconstruction_v38.planner import FastPlanner
from experiments.reconstruction_v38.run import save, ordered, restore_node, bank_science, physical_outcome
from experiments.reconstruction_v39.state import DecisionEngine
from experiments.reconstruction_v39.run import source_bank
from experiments.reconstruction_v40.budget import CurrentDecisionEngine, evaluate_actions

HERE = Path(__file__).resolve().parent
V38, V39 = HERE.parent/'reconstruction_v38', HERE.parent/'reconstruction_v39'
VARIANTS = ('v38', 'currentmemo', 'frontier')


def engine_for(planner, variant):
    return (None if variant == 'v38' else CurrentDecisionEngine(planner)
            if variant == 'currentmemo' else DecisionEngine(planner, variant='frontier'))


def diagnostics(planner, engine, ride, cid, clock, history):
    if engine is None:
        return current_diagnostics(planner, ride, cid, clock, history)
    base, base_receipt = engine.onboard(ride, cid, clock, history, method='rolling_fixed')
    nominal, nominal_receipt = engine.onboard(ride, cid, clock, history, method='nominal')
    base, nominal = ('alight' if x else 'continue' for x in (base, nominal))
    return dict(base=base, alternative='continue' if base == 'alight' else 'alight',
        nominal_diagnostic=nominal, cheap_agreed=base == nominal,
        cheap_receipts=dict(base=base_receipt, nominal=nominal_receipt))


def seed_for(state, manifest):
    match = next((entry for entry in manifest if entry['state'] == state), None)
    if match is not None:
        return dict(seed=match['seed'], source=match['source'], seed_origin='original_manifest')
    return dict(seed=int(np.random.SeedSequence([40050, state['root_index'], state['cid']]).generate_state(1)[0]),
                source=None, seed_origin='public_event_rule')


def evaluate(graph, model, parameters, state, node, history, seed, variant, setting, started_cpu):
    result = evaluate_actions(graph, model, parameters, state['deadline'], node,
        {k: state[k] for k in ('kind', 'cid', 'clock')}, history,
        state['actions']['alternative'], state['actions']['base'], seed=seed,
        max_count=setting['max_pairs'], cpu_budget=setting['cpu_seconds'],
        started_cpu=started_cpu, variant=variant)
    result['rows'] = dict(alternative=result['rows']['nominal'], base=result['rows']['combined'])
    costs = result['cost']['action_continuations']
    result['cost']['action_continuations'] = dict(alternative=costs['nominal'], base=costs['combined'])
    result['actions'] = state['actions']
    result['paired']['orientation'] = 'alternative action loss minus base action loss; negative favors alternative'
    result['state'] = state
    return result


def prefix_science(result, count):
    """Compare a saved prefix after measurement; never feed it to control."""
    rows = {arm: values[:count] for arm, values in result['rows'].items()}
    paired = summarize_pairs([r['loss'] for r in rows['alternative']], [r['loss'] for r in rows['base']],
        result['state']['deadline']-result['state']['clock'])
    paired['orientation'] = 'alternative action loss minus base action loss; negative favors alternative'
    selected = result['actions']['alternative'] if paired['full_bank_mean_gain_interval'][1] < 0 else result['actions']['base']
    return bank_science(dict(result, rows=rows, paired=paired, count=count, selected_action=selected))


def benchmark(manifest, data, physical_fit, emission_fit):
    setting = dict(name='pairs512', max_pairs=512, cpu_seconds=None)
    checks = []
    for index, entry in enumerate(manifest):
        state = entry['state']
        stem = entry['source'].replace('/', '_').removesuffix('.json.gz')
        graph = context(data, state['day'])[0]
        original = read(source_bank(entry['source']))
        measured = {}
        for variant in ordered(VARIANTS, index):
            path = HERE/'banks'/f'{stem}_{variant}.json.gz'
            if path.exists():
                result = read(path)
                if not result['anchor128_exact']:
                    raise AssertionError(f'Preserved failed anchor: {path}')
            else:
                tick, cpu = perf_counter(), process_time()
                model, parameters = setup(graph, physical_fit)
                node = restore_node(state['node'])
                history = restore_single_snapshot(model, parameters, emission_fit, state['public_history'], state['clock'])
                planner = FastPlanner(graph, model, parameters, state['deadline'])
                diagnostic = diagnostics(planner, engine_for(planner, variant), node, state['cid'], state['clock'], history)
                if {k: diagnostic[k] for k in state['actions']} != state['actions']:
                    raise AssertionError('The current cheap actions changed at a frozen state')
                result = evaluate(graph, model, parameters, state, node, history, entry['seed'], variant, setting, cpu)
                result['complete_cost'] = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
                result.update(source=entry['source'], variant=variant,
                    anchor128_exact=prefix_science(result, 128) == bank_science(original))
                save(path, result)
                print('bank', index, variant, 'anchor', result['anchor128_exact'],
                    'pairs', result['count'], 'CPU', round(result['complete_cost']['process_cpu'], 4), flush=True)
                if not result['anchor128_exact']:
                    raise AssertionError(f'Original 128 worlds changed; preserved {path}')
            measured[variant] = result
        sciences = [prefix_science(measured[v], 512) for v in VARIANTS]
        same = sciences[0] == sciences[1] == sciences[2]
        checks.append(dict(source=entry['source'], anchor128_exact=all(r['anchor128_exact'] for r in measured.values()),
                           full512_implementations_equal=same))
        save(HERE/'reference_validation.json', dict(expected=148, completed=len(checks), states=checks))
        if not same:
            raise AssertionError(f'Implementation changed the 512-world result at {entry["source"]}')


class TimedExecutor(Executor):
    """Partition full control CPU across decision intervals plus final work.

    A decision interval starts at the preceding action return. The first starts
    before cold setup. This charges incoming physical/public observation work,
    tracing and all planning without hiding them outside a decision's quota.
    """
    def __init__(self, graph, planner, deadline, *, engine, cpu_start, cpu_budget, **kwargs):
        super().__init__(graph, planner, deadline, **kwargs)
        self.engine, self.cpu_budget = engine, cpu_budget
        self.pending_cpu_start = cpu_start
        self.decision_costs = []

    def finish_decision(self, kind, location, clock, gate=False):
        end = process_time()
        elapsed = end-self.pending_cpu_start
        row = dict(kind=kind, clock=clock, process_cpu=elapsed, cpu_budget=self.cpu_budget, gate=gate,
            within_budget=None if self.cpu_budget is None else elapsed <= self.cpu_budget,
            overrun=0. if self.cpu_budget is None else max(0., elapsed-self.cpu_budget))
        row['cid' if kind == 'onboard' else 'stop'] = location
        self.decision_costs.append(row)
        self.pending_cpu_start = end

    def choose_platform(self, platform, entry, ready, history, visible, observed, clock):
        if self.engine is None:
            choice, receipt = super().choose_platform(platform, entry, ready, history, visible, observed, clock)
        else:
            choice, receipt = self.engine.platform(platform, entry, ready, history, visible, observed, clock,
                                                   self.choose, method=self.method)
        self.finish_decision('platform', platform.stop, clock)
        return choice, receipt

    def choose_onboard(self, ride, cid, clock, history):
        choice, receipt = (super().choose_onboard(ride, cid, clock, history) if self.engine is None else
                           self.engine.onboard(ride, cid, clock, history, method=self.method))
        self.finish_decision('onboard', cid, clock)
        return choice, receipt


class OnlineExecutor(TimedExecutor):
    def __init__(self, graph, planner, deadline, *, root_index, day, manifest, variant, setting, cpu_start):
        super().__init__(graph, planner, deadline, engine=engine_for(planner, variant), cpu_start=cpu_start,
            cpu_budget=setting['cpu_seconds'], method='rolling_fixed', waiting='combined')
        self.root_index, self.day, self.variant, self.setting = root_index, day, variant, setting
        self.manifest = [r for r in manifest if r['state']['root_index'] == root_index]
        self.gates = []

    def choose_onboard(self, ride, cid, clock, history):
        diagnostic = diagnostics(self.planner, self.engine, ride, cid, clock, history)
        if not has_legal_continue(ride, cid):
            choice, receipt = diagnostic['base'] == 'alight', dict(triggered=False, **diagnostic)
            self.finish_decision('onboard', cid, clock)
            return choice, receipt
        # Original count=128 belongs only to the seed-assignment context. The
        # evaluator separately records actual generated/committed sample counts.
        state = onboard_state(ride, cid, clock, history, self.deadline, root_index=self.root_index,
            day=self.day, count=128, diagnostics=diagnostic)
        stream = seed_for(state, self.manifest)
        gate = evaluate(self.graph, self.planner.model, self.planner.parameters, state, ride, history,
            stream['seed'], self.variant, self.setting, self.pending_cpu_start)
        gate.update(stream, cheap_receipts=diagnostic['cheap_receipts'])
        self.gates.append(gate)
        receipt = dict(triggered=True, seed=gate['seed'], seed_origin=gate['seed_origin'],
            selected_action=gate['selected_action'], paired=gate['paired'], committed_count=gate['committed_count'],
            executed_count=gate['executed_count'], cpu_budget=self.cpu_budget, cost=gate['cost'])
        self.finish_decision('onboard', cid, clock, gate=True)
        return gate['selected_action'] == 'alight', receipt


def one(index, label, graph, metadata, mapping, root, physical_fit, emission_fit, frames, protocol):
    tick, cpu = perf_counter(), process_time()
    is_cheap = label.startswith('cheap_')
    manifest = None if is_cheap else read(V38/'seed_manifest.json')
    model, parameters = setup(graph, physical_fit)
    ride = root_ride(graph, root)
    planner = FastPlanner(graph, model, parameters, root['deadline'])
    history = SingleMessageHistory(model, parameters, fit=emission_fit,
        frames=relevant_frames(frames, ride.trips(model), mapping), source_to_local=mapping)
    if is_cheap:
        implementation, name = label.removeprefix('cheap_').split('__')
        method, waiting = protocol['cheap_methods'][name]
        executor = TimedExecutor(graph, planner, root['deadline'], engine=engine_for(planner, implementation),
            cpu_start=cpu, cpu_budget=None, method=method, waiting=waiting)
    else:
        variant, name = label.split('__')
        setting = next(b for b in protocol['budgets'] if b['name'] == name)
        executor = OnlineExecutor(graph, planner, root['deadline'], root_index=index, day=root['day'],
            manifest=manifest, variant=variant, setting=setting, cpu_start=cpu)
    outcome = executor.run(ride, root['start'], history, ArchivedWorld(metadata))
    end_cpu, end_wall = process_time(), perf_counter()
    return dict(outcome=outcome, gates=getattr(executor, 'gates', []), decision_costs=executor.decision_costs,
        cost=dict(total_process_cpu=end_cpu-cpu, total_wall=end_wall-tick,
            decision_cpu_sum=sum(r['process_cpu'] for r in executor.decision_costs),
            final_execution_cpu=end_cpu-executor.pending_cpu_start,
            planner_stats=dict(planner.stats), decision_stats={} if executor.engine is None else dict(executor.engine.stats)),
        saved_answers_reused=0)


def replays(data, messages, physical_fit, emission_fit, selection, protocol):
    arms = [f'{v}__{b["name"]}' for b in protocol['budgets'] for v in VARIANTS]
    arms += [f'cheap_{impl}__{name}' for impl in ('v38', 'frontier') for name in protocol['cheap_methods']]
    for index, root in enumerate(selection['selected']):
        tick, cpu = perf_counter(), process_time()
        graph, _, _, mapping, metadata, _ = context(data, root['day'])
        preparation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        frames = messages['dates'][root['day']]['frames']
        for label in ordered(arms, index):
            path = HERE/'roots'/f'{index:03}_{label}.json.gz'
            if path.exists():
                old = read(path)
                if not old.get('anchor128_exact', True) or not old.get('cheap_exact_original', True):
                    raise AssertionError(f'Preserved failed anchor: {path}')
                continue
            result = one(index, label, graph, metadata, mapping, root, physical_fit, emission_fit, frames, protocol)
            result.update(root_index=index, root_id=root['root_id'], day=root['day'],
                loss_cap=root['deadline']-root['start'], method=label, preparation=preparation)
            exact = True
            if label.endswith('__pairs128'):
                original = read(V39/'roots'/f'{index:03}_frontier.json.gz')
                exact = physical_outcome(result['outcome']) == physical_outcome(original['outcome'])
                exact = exact and all(g['source'] is not None and g['count'] == 128 and
                    prefix_science(g,128) == bank_science(read(source_bank(g['source']))) for g in result['gates'])
                result['anchor128_exact'] = exact
            elif label.startswith('cheap_'):
                impl, name = label.removeprefix('cheap_').split('__')
                original = read(V39/'roots'/f'{index:03}_{impl}__{name}.json.gz')
                exact = physical_outcome(result['outcome']) == physical_outcome(original['outcome'])
                result['cheap_exact_original'] = exact
            save(path, result)
            print('root', index, label, result['outcome']['status'], 'loss', result['outcome']['loss'],
                'pairs', sum(g['committed_count'] for g in result['gates']),
                'new_states', sum(g['seed_origin'] == 'public_event_rule' for g in result['gates']),
                'CPU', round(result['cost']['total_process_cpu'],4), 'anchor_ok', exact, flush=True)
            if not exact:
                raise AssertionError(f'Old control or 128-world anchor changed; preserved {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('banks', 'online'))
    args = parser.parse_args()
    protocol, manifest = read(HERE/'protocol.json'), read(V38/'seed_manifest.json')
    if len(manifest) != 148:
        raise ValueError('Use all original 148 public states')
    tick, cpu = perf_counter(), process_time()
    _, data, messages, physical_fit, selection = inputs()
    emission_fit = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    print('common_input_preparation', dict(wall=perf_counter()-tick, process_cpu=process_time()-cpu), flush=True)
    if args.stage == 'banks':
        benchmark(manifest, data, physical_fit, emission_fit)
    else:
        validation = read(HERE/'reference_validation.json')
        if validation['completed'] != 148 or not all(r['anchor128_exact'] and r['full512_implementations_equal'] for r in validation['states']):
            raise ValueError('Complete all reference-stream checks before online replays')
        replays(data, messages, physical_fit, emission_fit, selection, protocol)


if __name__ == '__main__':
    main()
