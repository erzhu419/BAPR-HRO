"""P1 ten-arm cold-policy replays, preserving the original V51 controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter, process_time

from experiments.decision_gap_v1.panel import read, write
from experiments.decision_gap_v1.prepare import observations
from experiments.decision_gap_v1.summarize import LABELS, validate_old_replay
from experiments.reconstruction_v34.discover import context
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v38.run import ordered, restore_node, save
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v50.run import actions
from experiments.reconstruction_v51.run import PolicyExecutor

HERE = Path(__file__).resolve().parent
RESIDENT = None


def initialize():
    global RESIDENT
    tick, cpu = perf_counter(), process_time()
    _, data, messages, fit, _ = inputs()
    emission = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    RESIDENT = dict(data=data, messages=messages, fit=fit, emission=emission, graphs={},
        load_cost=dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick))


def run_root(index):
    resident = RESIDENT
    contract = read(HERE/'contract.json')
    menu = read(HERE/'menus'/f'{index:03}.json.gz')
    root = menu['root']
    graph_cost = None
    if root['day'] not in resident['graphs']:
        tick, cpu = perf_counter(), process_time()
        resident['graphs'][root['day']] = context(resident['data'], root['day'])
        graph_cost = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
    graph, _, _, mapping, metadata, _ = resident['graphs'][root['day']]
    completed = []
    for label in ordered(LABELS, index):
        path = HERE/'roots'/f'{index:03}_{label}.json.gz'
        if path.exists():
            continue
        tick, cpu = perf_counter(), process_time()
        model, parameters = setup(graph, resident['fit'])
        node = restore_node(menu['original_node' if label.startswith('old_') else 'expanded_node'])
        sight = observations(menu['old_observation_contract' if label=='old_old_cheap' else 'new_observation_contract'])
        frames = relevant_frames(resident['messages']['dates'][root['day']]['frames'], node.trips(model), mapping)
        history = PhysicalHistory(model, parameters, fit=resident['emission'], frames=frames, source_to_local=mapping)
        planner = MatchedContinuationPlanner(graph, model, parameters, root['deadline'])
        policy = 'cheap' if label.startswith('old_') else label.removeprefix('new_')
        executor = PolicyExecutor(graph, planner, root['deadline'], label=policy,
            rollout_limits=contract['rollout_limits'], root_index=index, cpu_start=cpu,
            observation_contract=sight)
        outcome = executor.run(node, root['start'], history, ArchivedWorld(metadata))
        menu_cost = menu['old_menu_cost' if label=='old_old_cheap' else 'cost']
        cost = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick, shared_menu=menu_cost)
        cost.update({metric+'_including_menu':cost[metric]+menu_cost[metric] for metric in ('process_cpu','wall')})
        result = dict(root_index=index, root_id=root['root_id'], day=root['day'],
            origin=root['origin_name'], goal=root['goal_name'], label=label, policy=policy,
            loss_cap=root['deadline']-root['start'], outcome=outcome, actions=actions(outcome),
            states=executor.states, gates=executor.gates, publications=executor.publications,
            decision_costs=executor.decision_costs, control_stats=executor.control.stats,
            planner_stats=planner.stats, observation=sight.describe(), cost=cost,
            menu_source=f'menus/{index:03}.json.gz',
            messages='Original received remote stream; common expanded local observation contract except old_old',
            v51_cheap_validation=None)
        # Always retain the receipt, including a possible regression, before stopping.
        save(path, result)
        if label=='old_old_cheap':
            validate_old_replay(result, read(HERE.parent/'reconstruction_v51/roots'/f'{index:03}_cheap.json.gz'))
        completed.append(dict(label=label, status=outcome['status'], loss=outcome['loss'],
                              cpu=cost['process_cpu_including_menu']))
        print(index, label, outcome['status'], outcome['loss'], round(cost['process_cpu_including_menu'],4), flush=True)
    return dict(root_index=index, day=root['day'], archive_load_cost=resident['load_cost'],
                graph_preparation_if_cold=graph_cost, completed=completed)


def run(workers):
    contract = read(HERE/'contract.json')
    if tuple(contract['labels']) != LABELS:
        raise AssertionError('Policy panel changed')
    # Complete public menu freeze is required before any policy query.
    if any(not (HERE/'menus'/f'{i:03}.json.gz').exists() for i in range(contract['roots'])):
        raise AssertionError('Freeze all72 public menus first')
    tick = perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        results = list(pool.map(run_root, range(contract['roots'])))
    write(HERE/'replay_preparation.json', dict(workers=workers, wall=perf_counter()-tick,
        roots=results, scope='Shared archive costs repeated per worker, graph costs only where cold; excluded from individual query timing'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.workers)
