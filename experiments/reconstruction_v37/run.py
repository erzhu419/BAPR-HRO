"""One all-root policy replay, charging saved conditional calculations in full."""

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
from experiments.reconstruction_v34.forecast import ForecastPlanner
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from experiments.reconstruction_v37.controller import OnboardRolloutExecutor, ReceiptStore


HERE = Path(__file__).resolve().parent


def run_root(index):
    tick, cpu = perf_counter(), process_time()
    protocol = read(HERE/'protocol.json')
    emission_fit = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    _, data, messages, physical_fit, selection = inputs()
    root = selection['selected'][index]
    graph, _, _, mapping, metadata, _ = context(data, root['day'])
    preparation = dict(wall=perf_counter()-tick, process_cpu=process_time()-cpu)
    control_tick, control_cpu = perf_counter(), process_time()
    model, parameters = setup(graph, physical_fit)
    ride = root_ride(graph, root)
    frames = relevant_frames(messages['dates'][root['day']]['frames'], ride.trips(model), mapping)
    history = SingleMessageHistory(model, parameters, fit=emission_fit,
                                   frames=frames, source_to_local=mapping)
    planner = ForecastPlanner(graph, model, parameters, root['deadline'])
    store = ReceiptStore(index, HERE/'reference_cases', HERE/'policy_cases')
    executor = OnboardRolloutExecutor(graph, planner, root['deadline'], root_index=index,
        day=root['day'], store=store,
        count=protocol['conditional_evaluation']['worlds_per_state'], seed_base=37050)
    outcome = executor.run(ride, root['start'], history, ArchivedWorld(metadata))
    actual_cpu, actual_wall = process_time()-control_cpu, perf_counter()-control_tick
    result = dict(root_index=index, root_id=root['root_id'], day=root['day'],
        method='onboard_complete_continuation_rollout', outcome=outcome, gates=executor.gates,
        cost=dict(actual_process_cpu=actual_cpu, actual_wall=actual_wall,
            reused_evaluation_cpu=executor.reused_evaluation_cpu,
            reused_evaluation_wall=executor.reused_evaluation_wall,
            total_process_cpu=actual_cpu+executor.reused_evaluation_cpu,
            total_wall=actual_wall+executor.reused_evaluation_wall,
            planner_stats=dict(planner.stats)), preparation=preparation,
        pipeline_actual_process_cpu=process_time()-cpu, pipeline_actual_wall=perf_counter()-tick,
        timing_scope='total fields are charged full control costs; actual fields are executed costs. '
            'Saved conditional evaluation cost is added only for hits. Inputs and graph preparation are separate.')
    with gzip.open(HERE/'roots'/f'{index:03}.json.gz', 'wt') as handle:
        json.dump(result, handle, default=jsonable, allow_nan=False)
    print(index, root['root_id'], outcome['status'], outcome['loss'], 'gates', len(executor.gates),
          'reused', sum(g['reused'] for g in executor.gates),
          'charged CPU', round(result['cost']['total_process_cpu'], 3), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    references = read(HERE/'reference_summary.json')
    expected = read(HERE/'protocol.json')['reference_panel']['eligible_states']
    if references['completed_states'] != expected or references['expected_states'] != expected:
        raise ValueError('Finish the frozen reference banks before the policy replay so they can be reused')
    selection = read(HERE.parent/'reconstruction_v34/selection.json')
    (HERE/'roots').mkdir(exist_ok=True)
    pending = [i for i in range(len(selection['selected'])) if not (HERE/'roots'/f'{i:03}.json.gz').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_root, index) for index in pending]
        for job in as_completed(jobs):
            job.result()


if __name__ == '__main__':
    main()
