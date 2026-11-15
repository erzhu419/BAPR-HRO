"""Four-arm separation of action coverage and local observation coverage."""
from pathlib import Path
from time import process_time, perf_counter

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v37.controller import node_structure
from experiments.reconstruction_v38.run import ordered, save
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v41.run import OnlineExecutor, normalized
from experiments.reconstruction_v50.execute import ObservationContract, ObservationMixin
from experiments.reconstruction_v50.menu import wide_root, generation_ledger, missing_labels

HERE = Path(__file__).resolve().parent
LABELS = ('narrow_native','wide_native','narrow_common','wide_common')


def actions(outcome):
    return [dict(kind=r['kind'], **{k:r[k] for k in ('cid','stop','clock','choice') if k in r})
            for r in outcome['trace'] if r['kind']!='observation']


class LoggedExecutor(ObservationMixin, OnlineExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = []

    def state(self, node, history, **fields):
        state = dict(node=node_structure(node),public_history=normalized(history.public_state()),
            deadline=self.deadline,root_index=self.root_index,**fields)
        self.states.append(state)
        return state

    def choose_onboard(self, node, cid, clock, history):
        state = self.state(node,history,kind='onboard',cid=cid,clock=clock)
        choice, receipt = super().choose_onboard(node,cid,clock,history)
        state['selected_action'] = 'alight' if choice else 'continue'
        return choice, receipt

    def choose_platform(self, node, entry, ready, history, visible, observed, clock):
        state = self.state(node,history,kind='platform',entry=entry,ready=ready,
            clock=clock,visible=list(visible),observed=dict(observed),stop=node.stop)
        choice, receipt = super().choose_platform(node,entry,ready,history,visible,observed,clock)
        state['selected_action'] = choice
        return choice, receipt


def run():
    protocol,data,messages,fit,selection=inputs()
    emission=read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    probe=read(HERE.parents[1]/'md/hegel_v10/menu_probe.json')
    graphs={}
    for index,root in enumerate(selection['selected']):
        if root['day'] not in graphs:
            graphs[root['day']]=context(data,root['day'])
        graph,_,_,mapping,metadata,_=graphs[root['day']]
        tick,cpu=perf_counter(),process_time()
        wide, cap_ledger=wide_root(graph,root,protocol['structure'])
        common=ObservationContract.from_menu(root_ride(graph,wide))
        menu_cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
        menu_path=HERE/'menus'/f'{index:03}.json.gz'
        if not menu_path.exists():
            save(menu_path,dict(root_index=index,wide_root=wide,cap_ledger=cap_ledger,
                generation= generation_ledger(graph,root,protocol['structure']),
                labels=missing_labels(graph,metadata,wide),observation_contract=common.describe(),
                cost=menu_cost,original_root_id=root['root_id']))
        else:
            menu_cost=read(menu_path)['cost']
        for label in ordered(LABELS,index):
            path=HERE/'roots'/f'{index:03}_{label}.json.gz'
            if path.exists():continue
            tick,cpu=perf_counter(),process_time()
            model,params=setup(graph,fit)
            node=root_ride(graph,wide if label.startswith('wide') else root)
            frames=relevant_frames(messages['dates'][root['day']]['frames'],
                root_ride(graph,wide).trips(model),mapping)
            history=PhysicalHistory(model,params,fit=emission,frames=frames,source_to_local=mapping)
            planner=MatchedContinuationPlanner(graph,model,params,root['deadline'])
            executor=LoggedExecutor(graph,planner,root['deadline'],family='shallow',mode='cheap',
                root_index=index,manifest=[],cpu_start=cpu,
                observation_contract=common if label.endswith('common') else None)
            outcome=executor.run(node,root['start'],history,ArchivedWorld(metadata))
            cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick,
                shared_menu=menu_cost)
            cost['process_cpu_including_menu']=cost['process_cpu']+menu_cost['process_cpu']
            cost['wall_including_menu']=cost['wall']+menu_cost['wall']
            row=dict(root_index=index,root_id=root['root_id'],day=root['day'],
                origin=root['origin_name'],goal=root['goal_name'],label=label,
                loss_cap=root['deadline']-root['start'],outcome=outcome,actions=actions(outcome),
                states=executor.states,cost=cost,observation=common.describe() if label.endswith('common') else 'native',
                messages='Same wide-menu related remote frames for all four arms',
                supplied_probe_validation=None)
            if label.endswith('native'):
                old=probe['rows'][index]['results']['uncapped' if label.startswith('wide') else 'original']
                row['supplied_probe_validation']=dict(loss_equal=outcome['loss']==old['loss'],
                    status_equal=outcome['status']==old['status'],actions_equal=actions(outcome)==old['actions'])
            save(path,row)
            if row['supplied_probe_validation'] and not all(row['supplied_probe_validation'].values()):
                raise AssertionError(f'Supplied menu probe not reproduced: {index}/{label}')
            print(index,label,outcome['status'],outcome['loss'],round(cost['process_cpu'],4),flush=True)


if __name__=='__main__':run()
