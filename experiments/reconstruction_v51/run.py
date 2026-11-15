"""All72 common-wide-menu actual journeys with complete ordinary controls."""
from pathlib import Path
from time import process_time, perf_counter

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v37.controller import has_legal_continue
from experiments.reconstruction_v38.run import ordered, save
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v44.matched import choose
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v50.run import LoggedExecutor, actions
from experiments.reconstruction_v51.baselines import BaselinePolicy, CHEAP_LABELS
from experiments.reconstruction_v51.rollout import evaluate

HERE = Path(__file__).resolve().parent
LABELS = (*CHEAP_LABELS, 'matched128', 'full128')


class PolicyExecutor(LoggedExecutor):
    def __init__(self, *args, label, rollout_limits, **kwargs):
        super().__init__(*args, family='shallow', mode='cheap', manifest=[], **kwargs)
        self.label, self.rollout_limits = label, rollout_limits
        self.control = BaselinePolicy(self.graph, self.planner, self.deadline,
            'cheap' if label in ('matched128', 'full128') else label)

    def choose_onboard(self, node, cid, clock, history):
        state = self.state(node, history, kind='onboard', cid=cid, clock=float(clock),
            legal_actions=['alight', 'continue'], has_continuation=has_legal_continue(node, cid))
        alight, receipt = self.control.onboard(node, cid, clock, history)
        base = 'alight' if alight else 'continue'
        compute = None
        if self.label in ('matched128', 'full128') and has_legal_continue(node, cid):
            def compute():
                gate = evaluate(self.graph, self.planner.model, self.planner.parameters,
                    self.deadline, node, dict(kind='onboard', cid=cid, clock=clock), history,
                    self.observation_contract, root_index=self.root_index, mode=self.label,
                    count=self.rollout_limits['worlds_per_query'],
                    max_process_cpu=self.rollout_limits['process_cpu_seconds'],
                    seed=self.rollout_limits['seed'])
                gate.update(cid=cid, clock=clock, state_index=len(self.states)-1,
                    base_action=base, base_receipt=receipt, selected_action=choose(gate, base))
                self.gates.append(gate)
                state['gate_index'] = len(self.gates)-1
                return gate['selected_action']
        selected = self.publish(clock, ('alight', 'continue'), base, compute)
        self.done('onboard', clock)
        state.update(selected_action=selected, base_action=base)
        return selected == 'alight', dict(policy=self.label, base=base, selected=selected, scoring=receipt)

    def choose_platform(self, node, entry, ready, history, visible, observed, clock):
        state = self.state(node, history, kind='platform', entry=float(entry), ready=float(ready),
            clock=float(clock), visible=list(visible), observed=dict(observed), stop=node.stop,
            legal_actions=[*visible, None])
        choice, receipt = self.control.platform(node, entry, ready, history, visible, observed, clock)
        selected = self.publish(clock, (*visible, None), choice)
        self.done('platform', clock)
        state.update(selected_action=selected, base_action=choice)
        return selected, dict(policy=self.label, scoring=receipt)


def run():
    contract = read(HERE/'contract.json')
    if tuple(contract['labels']) != LABELS:
        raise ValueError('Declared policy panel differs from the implementation')
    tick, cpu = perf_counter(), process_time()
    _, data, messages, fit, selection = inputs()
    emission = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    prepared = dict(archive_process_cpu=process_time()-cpu, archive_wall=perf_counter()-tick,
        scope='Common resident archives and dated graph creation outside individual controller queries', graphs={})
    graphs = {}
    for index, original in enumerate(selection['selected']):
        menu = read(HERE.parent/'reconstruction_v50/menus'/f'{index:03}.json.gz')
        root = menu['wide_root']
        if root['root_id'] != original['root_id'] or menu['root_index'] != index:
            raise ValueError('V50 root population changed')
        if root['day'] not in graphs:
            tick, cpu = perf_counter(), process_time()
            graphs[root['day']] = context(data, root['day'])
            prepared['graphs'][root['day']] = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
            save(HERE/'preparation.json', prepared)
        graph, _, _, mapping, metadata, _ = graphs[root['day']]
        for label in ordered(LABELS, index):
            path = HERE/'roots'/f'{index:03}_{label}.json.gz'
            if path.exists():
                continue
            tick, cpu = perf_counter(), process_time()
            model, parameters = setup(graph, fit)
            node = root_ride(graph, root)
            observations = ObservationContract.from_menu(node)
            frames = relevant_frames(messages['dates'][root['day']]['frames'], node.trips(model), mapping)
            history = PhysicalHistory(model, parameters, fit=emission, frames=frames, source_to_local=mapping)
            planner = MatchedContinuationPlanner(graph, model, parameters, root['deadline'])
            executor = PolicyExecutor(graph, planner, root['deadline'], label=label,
                rollout_limits=contract['rollout_limits'], root_index=index, cpu_start=cpu,
                observation_contract=observations)
            outcome = executor.run(node, root['start'], history, ArchivedWorld(metadata))
            cost = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick, shared_menu=menu['cost'])
            cost.update(process_cpu_including_menu=cost['process_cpu']+menu['cost']['process_cpu'],
                wall_including_menu=cost['wall']+menu['cost']['wall'])
            result = dict(root_index=index, root_id=root['root_id'], day=root['day'],
                origin=root['origin_name'], goal=root['goal_name'], label=label,
                loss_cap=root['deadline']-root['start'], outcome=outcome, actions=actions(outcome),
                states=executor.states, gates=executor.gates, publications=executor.publications,
                decision_costs=executor.decision_costs, control_stats=executor.control.stats,
                planner_stats=planner.stats, observation=observations.describe(),
                cost=cost, menu_source=f'../reconstruction_v50/menus/{index:03}.json.gz',
                messages='Same wide-menu relevant remote frames and local station contract for all policies',
                v50_cheap_validation=None)
            if label == 'cheap':
                old = read(HERE.parent/'reconstruction_v50/roots'/f'{index:03}_wide_common.json.gz')
                result['v50_cheap_validation'] = dict(loss=outcome['loss']==old['outcome']['loss'],
                    status=outcome['status']==old['outcome']['status'], actions=result['actions']==old['actions'])
            save(path, result)
            if result['v50_cheap_validation'] and not all(result['v50_cheap_validation'].values()):
                raise AssertionError(f'Original common-wide cheap changed: {index}')
            print(index, label, outcome['status'], 'loss', outcome['loss'],
                'CPU', round(cost['process_cpu_including_menu'], 5), 'states', len(executor.states),
                'gates', len(executor.gates), flush=True)


if __name__ == '__main__':
    run()
