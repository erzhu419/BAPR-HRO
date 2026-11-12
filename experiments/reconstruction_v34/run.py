"""Matched physical replays after schedule/message/source selection is frozen."""

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
from experiments.reconstruction_v34.forecast import ForecastPlanner, MessageHistory
from src.evolving_trip_belief import EvolvingTripBelief


HERE = Path(__file__).resolve().parent


def inputs():
    return tuple(read(HERE/name) for name in (
        'protocol.json','data.json.gz','messages.json.gz','model_fit.json','selection.json'))


def setup(graph, fit):
    model = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    params = {c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled'])
              for c in graph.connections}
    return model, params


def relevant_frames(frames, trips, source_to_local):
    result = []
    for frame in frames:
        if frame['trip_id'] not in trips:
            continue
        updates = [u for u in frame['endpoint_updates'] if u[0] in source_to_local]
        result.append(dict(frame, endpoint_updates=updates))
    return result


def actions(outcome):
    return [(r['kind'],r.get('cid',r.get('stop')),r['clock'],r.get('choice')) for r in outcome['trace']]


def one(graph, metadata, mapping, root, fit, frames, method, waiting, *, warm=False):
    began, cpu = perf_counter(), process_time()
    model, params = setup(graph, fit)
    ride = root_ride(graph,root)
    local_frames = relevant_frames(frames, ride.trips(model), mapping)
    planner = ForecastPlanner(graph,model,params,root['deadline'])
    history = MessageHistory(model,params,frames=local_frames,source_to_local=mapping)
    outcome = Executor(graph,planner,root['deadline'],method=method,waiting=waiting).run(
        ride,root['start'],history,ArchivedWorld(metadata))
    cost = dict(total_wall=perf_counter()-began,total_process_cpu=process_time()-cpu,
                relevant_input_frames=len(local_frames),planner_stats=dict(planner.stats))
    if warm:
        tick, tick_cpu = perf_counter(), process_time()
        repeated = Executor(graph,planner,root['deadline'],method=method,waiting=waiting).run(
            ride,root['start'],MessageHistory(model,params,frames=local_frames,source_to_local=mapping),
            ArchivedWorld(metadata))
        cost['identical_history_warm'] = dict(total_wall=perf_counter()-tick,
            total_process_cpu=process_time()-tick_cpu,
            identical_trace_and_outcome=(outcome['trace']==repeated['trace'] and
                (outcome['loss'],outcome['status'])==(repeated['loss'],repeated['status'])))
        if not cost['identical_history_warm']['identical_trace_and_outcome']:
            raise AssertionError('Warm message-aware planning changed the public trace or outcome')
    return dict(outcome=outcome,cost=cost)


def run_root(index):
    tick = perf_counter()
    protocol,data,messages,fit,selection = inputs()
    root = selection['selected'][index]
    graph,_,_,mapping,metadata,_ = context(data,root['day'])
    preparation = perf_counter()-tick
    rows = {}
    for arm in protocol['paired_information_arms']:
        frames = messages['dates'][root['day']]['frames'] if arm=='public_messages' else []
        rows[arm] = {}
        for name,(method,waiting) in protocol['methods'].items():
            row = one(graph,metadata,mapping,root,fit,frames,method,waiting,
                      warm=index==0)
            rows[arm][name] = row
            print(index,root['day'],arm,name,row['outcome']['status'],row['outcome']['loss'],
                  round(row['cost']['total_process_cpu'],3),flush=True)
    result = dict(root_id=root['root_id'],root_index=index,day=root['day'],
        origin=root['origin_name'],goal=root['goal_name'],rows=rows,
        preparation_wall=preparation,total_wall=perf_counter()-tick,
        scope='Logical physical replay with a fixed model and menu; CPU is charged but does not shift the physical decision clock.')
    target = HERE/'roots'/f'{index:03}.json.gz'
    with gzip.open(target,'wt') as handle:
        json.dump(result,handle,default=jsonable)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-index',type=int)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    selection = read(HERE/'selection.json')
    (HERE/'roots').mkdir(exist_ok=True)
    pending = [i for i in range(len(selection['selected']))
               if (args.root_index is None or i==args.root_index)
               and not (HERE/'roots'/f'{i:03}.json.gz').exists()]
    if args.workers == 1:
        for index in pending:
            run_root(index)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(run_root,i):i for i in pending}
            for job in as_completed(jobs):
                print('completed root',job.result(),flush=True)


if __name__ == '__main__':
    main()
