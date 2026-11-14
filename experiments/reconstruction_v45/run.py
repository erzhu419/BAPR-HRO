"""V45 fixed-model structural prefix and online quality-cost experiment."""
import argparse
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v37.controller import has_legal_continue
from experiments.reconstruction_v38.run import ordered, physical_outcome, restore_node, save
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v41.run import engine, normalized, public_seed, restored, score
from experiments.reconstruction_v42.run import gate_science
from experiments.reconstruction_v44.matched import evaluate as original, choose
from experiments.reconstruction_v44.run import OnlineExecutor as PreviousExecutor
from experiments.reconstruction_v45.prefix import evaluate

HERE = Path(__file__).resolve().parent
V44 = HERE.parent/'reconstruction_v44'


def matched_equal(fresh, old):
    # Import validation only after the measured computation, outside its cost.
    from experiments.reconstruction_v45.summarize import same_bank
    same_bank(normalized(fresh), normalized(old))
    for key in ('base_action', 'selected_action'):
        if key in old and fresh[key] != old[key]:
            raise ValueError(f'Original matched action changed: {key}')


def references(manifest, data, fit, emission, protocol):
    checks = []
    for index, entry in enumerate(manifest):
        state = entry['state']
        graph = context(data, state['day'])[0]
        results = {}
        for label in ordered(protocol['bank_configurations'], index):
            path = HERE/'banks'/f'{index:03}_{label}.json.gz'
            if path.exists():
                result = read(path)
            else:
                tick, cpu = perf_counter(), process_time()
                model, parameters = setup(graph, fit)
                node = restore_node(state['node'])
                history = restored('shallow', model, parameters, emission, state)
                planner = MatchedContinuationPlanner(graph, model, parameters, state['deadline'])
                base, receipt = score(engine(planner, 'shallow'), node, state['cid'], state['clock'], history)
                kwargs = {} if label == 'matched128' else dict(mode=label)
                result = (original if label == 'matched128' else evaluate)(
                    graph, model, parameters, state['deadline'], node,
                    {key: state[key] for key in ('kind', 'cid', 'clock')}, history,
                    count=128, seed=entry['seed'], **kwargs)
                result.update(index=index, root_index=state['root_index'], source=entry['source'],
                    label=label, base_action=base, base_receipt=receipt, selected_action=choose(result, base),
                    complete_cost=dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick))
                save(path, result)
                print('bank', index, label, 'action', result['selected_action'],
                    'CPU', round(result['complete_cost']['process_cpu'], 5), flush=True)
            results[label] = result
        matched_equal(results['matched128'], read(V44/'references'/f'{index:03}.json.gz'))
        matched_equal(results['markov128'], results['matched128'])
        checks.append(dict(index=index, original_prefixes_values_actions_equal=True))
        save(HERE/'reference_validation.json', dict(completed=len(checks), expected=148, states=checks))


class OnlineExecutor(PreviousExecutor):
    def choose_onboard(self, node, cid, clock, history):
        if self.mode not in ('markov128', 'absorbing'):
            return super().choose_onboard(node, cid, clock, history)
        base, receipt = score(self.engine, node, cid, clock, history)
        compute = None
        if has_legal_continue(node, cid):
            def compute():
                seed, source = public_seed(self.root_index, node, cid, clock, history, self.manifest)
                gate = evaluate(self.graph, self.planner.model, self.planner.parameters,
                    self.deadline, node, dict(kind='onboard', cid=cid, clock=clock), history,
                    count=128, seed=seed, mode=self.mode)
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
            mode = 'cheap' if label == 'rolling_fixed' else 'rollout128' if label == 'full128' else label
            executor = OnlineExecutor(graph, planner, root['deadline'], family='shallow', mode=mode,
                root_index=index, manifest=by_root[index], cpu_start=cpu, wall_start=tick, stopping_method='rolling_fixed')
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
            if label in ('rolling_fixed', 'full128', 'matched128'):
                old = read(V44/'roots'/f'{index:03}_{label}.json.gz')
                result['v44_physical_exact'] = physical_outcome(outcome) == physical_outcome(old['outcome'])
                if label == 'matched128':
                    result['v44_conditional_exact'] = len(result['gates']) == len(old['gates'])
                    for fresh, saved in zip(result['gates'], old['gates'], strict=True):
                        matched_equal(fresh, saved)
                else:
                    result['v44_conditional_exact'] = normalized(gate_science(result)) == normalized(gate_science(old))
            save(path, result)
            print('root', index, label, outcome['status'], 'loss', outcome['loss'],
                'CPU', round(result['cost']['total_process_cpu'], 5),
                'wall', round(result['cost']['total_wall'], 5), flush=True)
            if result.get('v44_physical_exact') is False or result.get('v44_conditional_exact') is False:
                raise AssertionError(f'Original V44 control changed: {path}')


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
        scope='One resident common archive, manifest, fit and selection load'))
    if args.stage == 'references':
        references(manifest, data, fit, emission, protocol)
    else:
        if read(HERE/'reference_validation.json')['completed'] != 148:
            raise ValueError('Validate all148 same-bank structural values first')
        online(manifest, data, messages, fit, emission, selection, protocol)


if __name__ == '__main__':
    main()
