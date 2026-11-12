"""Separate all-root public-message replay of the frozen standard rollout probe."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v33.run import jsonable
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.forecast import ForecastPlanner, MessageHistory
from experiments.reconstruction_v34.gated_rollout import GatedExecutor
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup


HERE = Path(__file__).resolve().parent


def run_root(index):
    tick, cpu = perf_counter(), process_time()
    protocol, data, messages, fit, selection = inputs()
    rollout_protocol = read(HERE/'rollout_protocol.json')
    root = selection['selected'][index]
    graph, _, _, mapping, metadata, _ = context(data, root['day'])
    control_tick, control_cpu = perf_counter(), process_time()
    model, parameters = setup(graph, fit)
    ride = root_ride(graph, root)
    frames = relevant_frames(messages['dates'][root['day']]['frames'], ride.trips(model), mapping)
    planner = ForecastPlanner(graph, model, parameters, root['deadline'])
    history = MessageHistory(model, parameters, frames=frames, source_to_local=mapping)
    executor = GatedExecutor(graph, planner, root['deadline'], root_index=index,
        count=rollout_protocol['worlds_per_trigger'], seed_base=rollout_protocol['seed'])
    preparation = dict(wall=perf_counter()-tick, process_cpu=process_time()-cpu)
    outcome = executor.run(ride, root['start'], history, ArchivedWorld(metadata))
    result = dict(root_index=index, root_id=root['root_id'], day=root['day'],
        arm='public_messages', method='standard_conditional_rollout', outcome=outcome,
        gates=executor.gates, cost=dict(preparation=preparation,
            total_wall=perf_counter()-tick, total_process_cpu=process_time()-cpu,
            control_total_wall=perf_counter()-control_tick,
            control_total_process_cpu=process_time()-control_cpu,
            planner_stats=dict(planner.stats)),
        scope='Current-history disagreement trigger; snapshot rollouts do not advance future publications. CPU is charged without shifting the logical action clock.')
    target = HERE/'gated_roots'/f'{index:03}.json.gz'
    target.parent.mkdir(exist_ok=True)
    with gzip.open(target, 'wt') as handle:
        json.dump(result, handle, default=jsonable)
    print(index, root['root_id'], outcome['status'], outcome['loss'], 'gates', len(executor.gates),
          'CPU', round(result['cost']['total_process_cpu'], 3), flush=True)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-index', type=int)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    selection = read(HERE/'selection.json')
    pending = [i for i in range(len(selection['selected']))
               if (args.root_index is None or i == args.root_index)
               and not (HERE/'gated_roots'/f'{i:03}.json.gz').exists()]
    if args.workers == 1:
        for index in pending:
            run_root(index)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(run_root, i): i for i in pending}
            for job in as_completed(jobs):
                job.result()


if __name__ == '__main__':
    main()
