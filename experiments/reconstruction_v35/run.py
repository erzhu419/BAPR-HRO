"""The three fixed controls under the single frozen V35 message model."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v33.run import jsonable
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld, Executor
from experiments.reconstruction_v34.forecast import ForecastPlanner
from experiments.reconstruction_v34.gated_rollout import GatedExecutor
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v35.forecast import FittedMessageHistory


HERE = Path(__file__).resolve().parent


def one(index, graph, metadata, mapping, root, physical_fit, emission_fit, frames, name, method, waiting):
    tick, cpu = perf_counter(), process_time()
    model, parameters = setup(graph, physical_fit)
    ride = root_ride(graph, root)
    local_frames = relevant_frames(frames, ride.trips(model), mapping)
    planner = ForecastPlanner(graph, model, parameters, root['deadline'])
    history = FittedMessageHistory(model, parameters, fit=emission_fit,
        frames=local_frames, source_to_local=mapping)
    if name == 'standard_conditional_rollout':
        executor = GatedExecutor(graph, planner, root['deadline'], root_index=index,
                                 count=128, seed_base=34050)
    else:
        executor = Executor(graph, planner, root['deadline'], method=method, waiting=waiting)
    outcome = executor.run(ride, root['start'], history, ArchivedWorld(metadata))
    return dict(outcome=outcome, gates=getattr(executor, 'gates', []),
        cost=dict(total_wall=perf_counter()-tick, total_process_cpu=process_time()-cpu,
                  relevant_input_frames=len(local_frames), planner_stats=dict(planner.stats)))


def run_root(index):
    tick, cpu = perf_counter(), process_time()
    protocol = read(HERE/'protocol.json')
    emission_fit = read(HERE/'fitted_emission.json')
    # Validation is completed first. Its score cannot change this fixed replay.
    read(HERE/'validation_summary.json')
    _, data, messages, physical_fit, selection = inputs()
    root = selection['selected'][index]
    graph, _, _, mapping, metadata, _ = context(data, root['day'])
    preparation = dict(wall=perf_counter()-tick, process_cpu=process_time()-cpu)
    rows = {}
    for name, (method, waiting) in protocol['paired_replay']['methods'].items():
        rows[name] = one(index, graph, metadata, mapping, root, physical_fit, emission_fit,
                        messages['dates'][root['day']]['frames'], name, method, waiting)
        row = rows[name]
        print(index, name, row['outcome']['status'], row['outcome']['loss'],
              'gates', len(row['gates']), 'CPU', round(row['cost']['total_process_cpu'], 4), flush=True)
    result = dict(root_index=index, root_id=root['root_id'], day=root['day'],
        emission_model='fitted_emission.json', physical_model='../reconstruction_v34/model_fit.json',
        rows=dict(public_messages=rows), preparation=preparation,
        total_wall=perf_counter()-tick, total_process_cpu=process_time()-cpu)
    target = HERE/'roots'/f'{index:03}.json.gz'
    with gzip.open(target, 'wt') as handle:
        json.dump(result, handle, default=jsonable, allow_nan=False)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    selection = read(HERE.parent/'reconstruction_v34/selection.json')
    (HERE/'roots').mkdir(exist_ok=True)
    pending = [i for i in range(len(selection['selected'])) if not (HERE/'roots'/f'{i:03}.json.gz').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_root, index) for index in pending]
        for job in as_completed(jobs):
            job.result()


if __name__ == '__main__':
    main()
