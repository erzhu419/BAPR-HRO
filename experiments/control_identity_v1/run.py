"""All72 expanded-menu journeys, paired across two fixed emission fits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter, process_time

from experiments.decision_gap_v1.panel import read, write
from experiments.decision_gap_v1.prepare import observations
from experiments.reconstruction_v34.discover import context
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.run import relevant_frames, setup
from experiments.reconstruction_v38.run import ordered, restore_node, save
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v50.run import actions
from experiments.reconstruction_v51.run import PolicyExecutor, LABELS as POLICIES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GAP = HERE.parent/'decision_gap_v1'
MODELS = ('old_fit','repaired_fit')
RESIDENT = None


def arm_order(index, policies=POLICIES):
    return [(model, policy) for j,policy in enumerate(ordered(policies,index))
            for model in ordered(MODELS,index+j)]


def check_inputs(directory=HERE):
    directory = Path(directory)
    contract = read(directory/'contract.json')
    if tuple(contract['policies']) != POLICIES or tuple(contract['models']) != MODELS:
        raise ValueError('The declared fixed policy/model panel changed')
    source_contract = read(GAP/'contract.json')
    if contract['rollout_limits'] != source_contract['rollout_limits'] or contract['roots'] != 72:
        raise ValueError('Original budget or root denominator changed')
    for model in MODELS:
        if read(directory/contract['emission_snapshots'][model]) != read(ROOT/contract['emission_sources'][model]):
            raise ValueError(f'The frozen {model} fit changed')
    selected = read(HERE.parent/'reconstruction_v34/selection.json')['selected']
    for item in contract['root_manifest']:
        i = item['root_index']
        menu = read(GAP/'menus'/f'{i:03}.json.gz')
        if menu['root_index'] != i or menu['root']['root_id'] != selected[i]['root_id']:
            raise ValueError('The original root population changed')
        if any(menu['root'][k] != item[k] for k in ('root_id','day','start','deadline')):
            raise ValueError('A frozen root context changed')
    messages = read(directory/'messages.json.gz')
    if set(messages['dates']) != set(contract['control_dates']):
        raise ValueError('The corrected stream does not cover all control dates')
    return contract


def initialize(directory=HERE):
    global RESIDENT
    directory = Path(directory)
    tick, cpu = perf_counter(), process_time()
    contract = read(directory/'contract.json')
    RESIDENT = dict(directory=directory, contract=contract,
        data=read(HERE.parent/'reconstruction_v34/data.json.gz'),
        messages=read(directory/'messages.json.gz'),
        fit=read(HERE.parent/'reconstruction_v34/model_fit.json'),
        emissions={m:read(directory/contract['emission_snapshots'][m]) for m in MODELS}, graphs={})
    RESIDENT['load_cost'] = dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)


def one_arm(index, model_name, policy, menu, graph_context, resident):
    graph, _, _, mapping, metadata, _ = graph_context
    root, contract = menu['root'], resident['contract']
    tick, cpu = perf_counter(), process_time()
    model, parameters = setup(graph,resident['fit'])
    node = restore_node(menu['expanded_node'])
    sight = observations(menu['new_observation_contract'])
    frames = relevant_frames(resident['messages']['dates'][root['day']]['frames'],node.trips(model),mapping)
    history = PhysicalHistory(model,parameters,fit=resident['emissions'][model_name],
                              frames=frames,source_to_local=mapping)
    planner = MatchedContinuationPlanner(graph,model,parameters,root['deadline'])
    executor = PolicyExecutor(graph,planner,root['deadline'],label=policy,
        rollout_limits=contract['rollout_limits'],root_index=index,cpu_start=cpu,observation_contract=sight)
    outcome = executor.run(node,root['start'],history,ArchivedWorld(metadata))
    cost = dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick,shared_menu=menu['cost'])
    cost.update({metric+'_including_menu':cost[metric]+menu['cost'][metric] for metric in ('process_cpu','wall')})
    return dict(root_index=index,root_id=root['root_id'],day=root['day'],
        origin=root['origin_name'],goal=root['goal_name'],label='new_'+policy,policy=policy,
        emission_model=model_name,emission_source=contract['emission_snapshots'][model_name],
        loss_cap=root['deadline']-root['start'],outcome=outcome,actions=actions(outcome),
        states=executor.states,gates=executor.gates,publications=executor.publications,
        decision_costs=executor.decision_costs,control_stats=executor.control.stats,
        planner_stats=planner.stats,observation=sight.describe(),cost=cost,
        menu_source=f'experiments/decision_gap_v1/menus/{index:03}.json.gz',
        relevant_remote_frames=len(frames),
        information='Same repaired external stream and expanded local contract; actual own histories follow actions',
        menu_cost_source='Frozen original V50 construction plus decision_gap_v1 closure receipt')


def run_root(index):
    resident = RESIDENT
    directory = resident['directory']
    menu = read(GAP/'menus'/f'{index:03}.json.gz')
    root = menu['root']
    graph_cost = None
    if root['day'] not in resident['graphs']:
        tick, cpu = perf_counter(), process_time()
        resident['graphs'][root['day']] = context(resident['data'],root['day'])
        graph_cost = dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
    completed = []
    for model_name,policy in arm_order(index):
        path = directory/'models'/model_name/'roots'/f'{index:03}_new_{policy}.json.gz'
        if path.exists():
            continue
        result = one_arm(index,model_name,policy,menu,resident['graphs'][root['day']],resident)
        save(path,result)
        completed.append(dict(model=model_name,policy=policy,status=result['outcome']['status'],
            loss=result['outcome']['loss'],cpu=result['cost']['process_cpu_including_menu']))
        print(index,model_name,policy,result['outcome']['status'],result['outcome']['loss'],
              round(result['cost']['process_cpu_including_menu'],4),flush=True)
    receipt = dict(root_index=index,day=root['day'],archive_load_cost=resident['load_cost'],
        graph_preparation_if_cold=graph_cost,declared_order=arm_order(index),completed=completed)
    write(directory/'preparation_roots'/f'{index:03}.json',receipt)
    return receipt


def run(workers=4, directory=HERE):
    directory = Path(directory)
    contract = check_inputs(directory)
    pending = [i for i in range(contract['roots']) if any(
        not (directory/'models'/m/'roots'/f'{i:03}_new_{p}.json.gz').exists() for m,p in arm_order(i))]
    tick = perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(directory,)) as pool:
        for future in as_completed([pool.submit(run_root,i) for i in pending]):
            print('completed root',future.result()['root_index'],flush=True)
    write(directory/'replay_preparation.json',dict(workers=workers,wall=perf_counter()-tick,
        requested_roots=contract['roots'],executed_roots=pending,
        source='preparation_roots/*.json',
        scope='Common resident/archive and dated graph costs excluded from individual queries; per-arm frozen menu charge included'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    run(parser.parse_args().workers)
