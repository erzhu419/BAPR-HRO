"""V43 fixed-pi next-platform attribution and existing stopping baselines."""
import argparse
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v38.run import ordered, physical_outcome, restore_node, save
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory, PhysicalDecisionEngine
from experiments.reconstruction_v41.run import OnlineExecutor as OriginalExecutor, restored, normalized
from experiments.reconstruction_v42.run import gate_science

HERE = Path(__file__).resolve().parent
V41 = HERE.parent/'reconstruction_v41'


class StoppingExecutor(OriginalExecutor):
    """Change only the already implemented stopping method, throughout the menu."""
    def __init__(self, *args, stopping_method, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = stopping_method

    def choose_onboard(self, node, cid, clock, history):
        if self.mode != 'cheap':
            return super().choose_onboard(node, cid, clock, history)
        alight, receipt = self.engine.onboard(node, cid, clock, history, method=self.method)
        base = 'alight' if alight else 'continue'
        selected = self.publish(clock, ('alight', 'continue'), base)
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
            mode = 'rollout128' if label == 'full128' else 'cheap'
            method = 'rolling_fixed' if label == 'full128' else label
            executor = StoppingExecutor(graph, planner, root['deadline'], family='shallow', mode=mode,
                root_index=index, manifest=by_root[index], cpu_start=cpu, stopping_method=method)
            outcome = executor.run(node, root['start'], history, ArchivedWorld(metadata))
            end_cpu, end_wall = process_time(), perf_counter()
            result = dict(root_index=index, root_id=root['root_id'], day=root['day'],
                origin=root['origin_name'], goal=root['goal_name'], label=label,
                loss_cap=root['deadline']-root['start'], outcome=outcome,
                gates=executor.gates, publications=executor.publications,
                decision_costs=executor.decision_costs, preparation=preparation,
                cost=dict(total_process_cpu=end_cpu-cpu, total_wall=end_wall-tick, cold=cold,
                    decision_cpu_sum=sum(r['process_cpu'] for r in executor.decision_costs),
                    final_execution_cpu=end_cpu-executor.pending_cpu, planner_stats=dict(planner.stats)),
                saved_answers_reused=0)
            if label in ('rolling_fixed', 'full128'):
                old_label = 'shallow_cheap' if label == 'rolling_fixed' else 'shallow_rollout128'
                old = read(V41/'roots'/f'{index:03}_{old_label}.json.gz')
                result['v41_physical_exact'] = physical_outcome(outcome) == physical_outcome(old['outcome'])
                result['v41_conditional_exact'] = normalized(gate_science(result)) == normalized(gate_science(old))
            save(path, result)
            print('root', index, label, outcome['status'], 'loss', outcome['loss'],
                'CPU', round(result['cost']['total_process_cpu'], 5), flush=True)
            if result.get('v41_physical_exact') is False or result.get('v41_conditional_exact') is False:
                raise AssertionError(f'Fixed V41 comparator changed: {path}')


def stopping_scores(graph, model, parameters, node, state, history, methods):
    rows = {}
    for method in methods:
        tick, cpu = perf_counter(), process_time()
        planner = MatchedContinuationPlanner(graph, model, parameters, state['deadline'])
        engine = PhysicalDecisionEngine(planner, variant='frontier')
        action, receipt = engine.onboard(node, state['cid'], state['clock'], history, method=method)
        rows[method] = dict(action='alight' if action else 'continue', receipt=receipt,
            process_cpu=process_time()-cpu, wall=perf_counter()-tick)
    return rows


def references(manifest, data, fit, emission, protocol):
    from experiments.reconstruction_v43.reference import evaluate_case
    for index, entry in enumerate(manifest):
        path = HERE/'references'/f'{index:03}.json.gz'
        if path.exists():
            continue
        tick, cpu = perf_counter(), process_time()
        state = entry['state']
        graph = context(data, state['day'])[0]
        model, parameters = setup(graph, fit)
        node = restore_node(state['node'])
        history = restored('shallow', model, parameters, emission, state)
        result = evaluate_case(graph, model, parameters, state['deadline'], node,
            {key: state[key] for key in ('kind', 'cid', 'clock')}, history, count=128, seed=entry['seed'])
        result.update(index=index, root_index=state['root_index'], source=entry['source'],
            stopping_scores=stopping_scores(graph, model, parameters, node, state, history, protocol['strong_methods']),
            complete_cost=dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick))
        save(path, result)
        print('reference', index, 'CPU', round(result['complete_cost']['process_cpu'], 4), flush=True)


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
        references(manifest, data, fit, emission, protocol)
    else:
        online(manifest, data, messages, fit, emission, selection, protocol)


if __name__ == '__main__':
    main()
