"""Frozen V41 continuation evaluation with event-level dependency projection."""
import argparse
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v37.controller import has_legal_continue
from experiments.reconstruction_v38.run import ordered, physical_outcome, restore_node, save
from experiments.reconstruction_v41.continuation import choose, science
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory, CheckingWorld
from experiments.reconstruction_v41.run import (
    PLANNERS, OnlineExecutor as OriginalOnlineExecutor, engine, normalized,
    public_seed, restored, score,
)
from experiments.reconstruction_v42.event_suffix import evaluate


HERE = Path(__file__).resolve().parent
V41 = HERE.parent/'reconstruction_v41'


def checked_evaluation(graph, model, parameters, node, history, state, family, seed, mode):
    public = history.public_state()
    return evaluate(graph, model, parameters, state['deadline'], node,
        {key: state[key] for key in ('kind', 'cid', 'clock')}, history,
        planner_type=PLANNERS[family], count=128, seed=seed, mode=mode,
        world_type=lambda bank, index: CheckingWorld(bank, index, public))


def references(manifest, data, fit, emission, protocol):
    arms = [(family, mode) for family in protocol['bank_families'] for mode in protocol['bank_modes']]
    validation = []
    for index, entry in enumerate(manifest):
        state = entry['state']
        graph = context(data, state['day'])[0]
        measured = {}
        old = {family: read(V41/'banks'/f'{index:03}_{family}.json.gz')
               for family in protocol['bank_families']}
        for family, mode in ordered(arms, index):
            path = HERE/'banks'/f'{index:03}_{family}_{mode}.json.gz'
            if path.exists():
                result = read(path)
            else:
                tick, cpu = perf_counter(), process_time()
                model, parameters = setup(graph, fit)
                node = restore_node(state['node'])
                history = restored(family, model, parameters, emission, state)
                planner = PLANNERS[family](graph, model, parameters, state['deadline'])
                base, receipt = score(engine(planner, family), node, state['cid'], state['clock'], history)
                result = checked_evaluation(graph, model, parameters, node, history, state,
                                            family, entry['seed'], mode)
                result.update(index=index, source=entry['source'], root_index=state['root_index'],
                    family=family, variant=mode, state=state, base_action=base, base_receipt=receipt,
                    selected_action=choose(result, base),
                    complete_cost=dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick))
                result['v41_science_exact'] = normalized(science(result)) == normalized(science(old[family]))
                result['v41_action_exact'] = (base == old[family]['base_action'] and
                    result['selected_action'] == old[family]['selected_action'])
                save(path, result)
                print('bank', index, family, mode, 'exact', result['v41_science_exact'],
                      'hits', result['meeting_stats']['hits'],
                      'CPU', round(result['complete_cost']['process_cpu'], 5), flush=True)
            if not result['v41_science_exact'] or not result['v41_action_exact']:
                raise AssertionError(f'Frozen V41 equivalence failed: {path}')
            measured[family, mode] = result
        for family in protocol['bank_families']:
            baseline = normalized(science(measured[family, 'full']))
            if any(normalized(science(measured[family, mode])) != baseline for mode in protocol['bank_modes']):
                raise AssertionError(f'Fresh paired equivalence failed: {index} {family}')
        validation.append(dict(index=index, v41_science_exact=True, fresh_science_exact=True))
        save(HERE/'reference_validation.json', dict(completed=len(validation), expected=148, states=validation))


class OnlineExecutor(OriginalOnlineExecutor):
    def choose_onboard(self, node, cid, clock, history):
        base, receipt = score(self.engine, node, cid, clock, history)
        compute = None
        if self.mode != 'cheap' and has_legal_continue(node, cid):
            def compute():
                seed, source = public_seed(self.root_index, node, cid, clock, history, self.manifest)
                state = dict(kind='onboard', cid=cid, clock=clock, deadline=self.deadline)
                gate = checked_evaluation(self.graph, self.planner.model, self.planner.parameters,
                    node, history, state, self.family, seed, self.mode.removesuffix('128'))
                gate.update(source=source, cid=cid, clock=clock, base_action=base,
                            base_receipt=receipt, selected_action=choose(gate, base))
                self.gates.append(gate)
                return gate['selected_action']
        chosen = self.publish(clock, ('alight', 'continue'), base, compute)
        self.done('onboard', clock)
        return chosen == 'alight', dict(base=base, selected=chosen, cheap=receipt)


def gate_science(result):
    return [dict(science(gate), selected_action=gate['selected_action'], base_action=gate['base_action'])
            for gate in result['gates']]


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
            family, mode = label.split('_', 1)
            tick, cpu = perf_counter(), process_time()
            model, parameters = setup(graph, fit)
            node = root_ride(graph, root)
            planner = PLANNERS[family](graph, model, parameters, root['deadline'])
            history = PhysicalHistory(model, parameters, fit=emission,
                frames=relevant_frames(frames, node.trips(model), mapping), source_to_local=mapping)
            cold = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
            executor = OnlineExecutor(graph, planner, root['deadline'], family=family, mode=mode,
                                      root_index=index, manifest=by_root[index], cpu_start=cpu)
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
            old_label = 'shallow_cheap' if mode == 'cheap' else 'shallow_rollout128'
            old = read(V41/'roots'/f'{index:03}_{old_label}.json.gz')
            result['v41_physical_exact'] = physical_outcome(outcome) == physical_outcome(old['outcome'])
            result['v41_conditional_exact'] = normalized(gate_science(result)) == normalized(gate_science(old))
            save(path, result)
            print('root', index, label, 'loss', outcome['loss'],
                'CPU', round(result['cost']['total_process_cpu'], 5),
                'exact', result['v41_physical_exact'] and result['v41_conditional_exact'], flush=True)
            if not result['v41_physical_exact'] or not result['v41_conditional_exact']:
                raise AssertionError(f'Frozen online equivalence failed: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('banks', 'online'))
    args = parser.parse_args()
    tick, cpu = perf_counter(), process_time()
    protocol = read(HERE/'protocol.json')
    manifest = read(HERE.parent/'reconstruction_v38/seed_manifest.json')
    _, data, messages, fit, selection = inputs()
    emission = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    save(HERE/f'{args.stage}_preparation.json', dict(process_cpu=process_time()-cpu,
        wall=perf_counter()-tick, manifest_states=len(manifest), roots=len(selection['selected']),
        scope='One resident load of common archive, manifest, fit and selection'))
    if args.stage == 'banks':
        references(manifest, data, fit, emission, protocol)
    else:
        if read(HERE/'reference_validation.json')['completed'] != 148:
            raise ValueError('Complete all frozen states before online controls')
        online(manifest, data, messages, fit, emission, selection, protocol)


if __name__ == '__main__':
    main()
