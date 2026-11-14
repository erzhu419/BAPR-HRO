"""Online matched continuation with full-cost, same-cohort controls."""
import argparse
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v37.controller import has_legal_continue
from experiments.reconstruction_v38.run import ordered, physical_outcome, restore_node, save
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v41.run import normalized, public_seed, restored, score
from experiments.reconstruction_v42.run import gate_science
from experiments.reconstruction_v43.run import StoppingExecutor
from experiments.reconstruction_v44.matched import evaluate, choose

HERE = Path(__file__).resolve().parent
V43 = HERE.parent/'reconstruction_v43'


def validate_reference(result, old):
    """Detect a changed prefix or local target before measuring online use."""
    for key in ('count', 'generated_count', 'seed', 'clock', 'deadline', 'future_messages', 'public_history'):
        if normalized(result[key]) != normalized(old[key]):
            raise ValueError(f'V43 matched reference identity changed: {key}')
    for arm in ('alight', 'continue'):
        if len(result['rows'][arm]) != 128 or len(old['rows'][arm]) != 128:
            raise ValueError('Original 128 paired prefixes required')
        for fresh, saved in zip(result['rows'][arm], old['rows'][arm], strict=True):
            if fresh['world_index'] != saved['world_index'] or normalized(fresh['prefix']) != normalized(saved['prefix']):
                raise ValueError(f'V43 fixed-pi prefix changed: {arm}/{fresh["world_index"]}')
            for key in ('mean', 'lower', 'upper', 'unresolved_probability'):
                a, b = fresh['value'][key], saved['updated'][key]
                if (a is None) != (b is None) or a is not None and not np.isclose(a, b, atol=1e-8, rtol=0):
                    raise ValueError(f'V43 updated value changed: {arm}/{fresh["world_index"]}/{key}')
    return True


def references(manifest, data, fit, emission):
    checks = []
    for index, entry in enumerate(manifest):
        path = HERE/'references'/f'{index:03}.json.gz'
        if path.exists():
            result = read(path)
        else:
            state = entry['state']
            graph = context(data, state['day'])[0]
            tick, cpu = perf_counter(), process_time()
            model, parameters = setup(graph, fit)
            node = restore_node(state['node'])
            history = restored('shallow', model, parameters, emission, state)
            result = evaluate(graph, model, parameters, state['deadline'], node,
                {key: state[key] for key in ('kind', 'cid', 'clock')}, history, count=128, seed=entry['seed'])
            result.update(index=index, root_index=state['root_index'], source=entry['source'],
                complete_cost=dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick))
            save(path, result)
            print('reference', index, 'CPU', round(result['complete_cost']['process_cpu'], 5), flush=True)
        validate_reference(result, read(V43/'references'/f'{index:03}.json.gz'))
        checks.append(dict(index=index, v43_updated_values_and_prefixes_match=True))
    save(HERE/'reference_validation.json', dict(completed=len(checks), expected=148, states=checks))


class OnlineExecutor(StoppingExecutor):
    def __init__(self, *args, wall_start, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_wall = wall_start

    def done(self, kind, clock):
        cpu, wall = process_time(), perf_counter()
        self.decision_costs.append(dict(kind=kind, clock=clock,
            process_cpu=cpu-self.pending_cpu, wall=wall-self.pending_wall))
        self.pending_cpu, self.pending_wall = cpu, wall

    def choose_onboard(self, node, cid, clock, history):
        if self.mode != 'matched128':
            return super().choose_onboard(node, cid, clock, history)
        base, receipt = score(self.engine, node, cid, clock, history)
        compute = None
        if has_legal_continue(node, cid):
            def compute():
                seed, source = public_seed(self.root_index, node, cid, clock, history, self.manifest)
                gate = evaluate(self.graph, self.planner.model, self.planner.parameters,
                    self.deadline, node, dict(kind='onboard', cid=cid, clock=clock), history,
                    count=128, seed=seed)
                gate.update(source=source, cid=cid, clock=clock, base_action=base,
                    base_receipt=receipt, selected_action=choose(gate, base))
                self.gates.append(gate)
                return gate['selected_action']
        selected = self.publish(clock, ('alight', 'continue'), base, compute)
        self.done('onboard', clock)
        return selected == 'alight', dict(base=base, selected=selected, cheap=receipt)


def online(manifest, data, messages, fit, emission, selection, protocol):
    by_root = {i: [e for e in manifest if e['state']['root_index'] == i] for i in range(72)}
    for index, root in enumerate(selection['selected']):
        tick, cpu = perf_counter(), process_time()
        graph, _, _, mapping, metadata, _ = context(data, root['day'])
        preparation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        frames = messages['dates'][root['day']]['frames']
        for label in ordered(protocol['online_configurations'], index):
            path = HERE/'roots'/f'{index:03}_{label}.json.gz'
            if path.exists():
                continue
            tick, cpu = perf_counter(), process_time()
            model, parameters = setup(graph, fit)
            node = root_ride(graph, root)
            planner = MatchedContinuationPlanner(graph, model, parameters, root['deadline'])
            history = PhysicalHistory(model, parameters, fit=emission,
                frames=relevant_frames(frames, node.trips(model), mapping), source_to_local=mapping)
            cold = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
            mode = 'matched128' if label == 'matched128' else 'rollout128' if label == 'full128' else 'cheap'
            method = 'rolling_fixed' if label in ('full128', 'matched128') else label
            executor = OnlineExecutor(graph, planner, root['deadline'], family='shallow', mode=mode,
                root_index=index, manifest=by_root[index], cpu_start=cpu, wall_start=tick, stopping_method=method)
            outcome = executor.run(node, root['start'], history, ArchivedWorld(metadata))
            end_cpu, end_wall = process_time(), perf_counter()
            result = dict(root_index=index, root_id=root['root_id'], day=root['day'],
                origin=root['origin_name'], goal=root['goal_name'], label=label,
                loss_cap=root['deadline']-root['start'], outcome=outcome,
                gates=executor.gates, publications=executor.publications,
                decision_costs=executor.decision_costs, preparation=preparation,
                cost=dict(total_process_cpu=end_cpu-cpu, total_wall=end_wall-tick, cold=cold,
                    decision_cpu_sum=sum(r['process_cpu'] for r in executor.decision_costs),
                    decision_wall_sum=sum(r['wall'] for r in executor.decision_costs),
                    final_execution_cpu=end_cpu-executor.pending_cpu,
                    final_execution_wall=end_wall-executor.pending_wall, planner_stats=dict(planner.stats)),
                saved_answers_reused=0)
            if label != 'matched128':
                old = read(V43/'roots'/f'{index:03}_{label}.json.gz')
                result['v43_physical_exact'] = physical_outcome(outcome) == physical_outcome(old['outcome'])
                result['v43_conditional_exact'] = normalized(gate_science(result)) == normalized(gate_science(old))
            save(path, result)
            print('root', index, label, outcome['status'], 'loss', outcome['loss'],
                'CPU', round(result['cost']['total_process_cpu'], 5),
                'wall', round(result['cost']['total_wall'], 5), flush=True)
            if result.get('v43_physical_exact') is False or result.get('v43_conditional_exact') is False:
                raise AssertionError(f'Existing scientific comparator changed: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('references', 'online'))
    args = parser.parse_args()
    tick, cpu = perf_counter(), process_time()
    protocol = read(HERE/'protocol.json')
    manifest = read(HERE.parent/'reconstruction_v38/seed_manifest.json')
    _, data, messages, fit, selection = inputs()
    emission = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    save(HERE/f'{args.stage}_preparation.json', dict(process_cpu=process_time()-cpu,
        wall=perf_counter()-tick, manifest_states=len(manifest), roots=len(selection['selected']),
        scope='One resident load of common archive, manifest, fit and selection'))
    if args.stage == 'references':
        references(manifest, data, fit, emission)
    else:
        if read(HERE/'reference_validation.json')['completed'] != 148:
            raise ValueError('Validate every original matched target before online deployment')
        online(manifest, data, messages, fit, emission, selection, protocol)


if __name__ == '__main__':
    main()
