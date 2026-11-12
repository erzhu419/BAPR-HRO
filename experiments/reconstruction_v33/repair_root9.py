"""Replay only root9 actual policies after its departure-only source repair."""

import gzip
import json
from pathlib import Path
from shutil import copy2
from time import perf_counter

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v33.execute import ArchivedWorld, Executor
from experiments.reconstruction_v33.history import ConditionalEngine
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v33.planner import Planner
from experiments.reconstruction_v33.run import setup_day, jsonable


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'


def main():
    started = perf_counter()
    path = HERE/'roots/09.json.gz'
    backup = HERE/'before_endpoint_repair/09.json.gz'
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists():
        copy2(path,backup)
    with gzip.open(path,'rt') as handle:
        result = json.load(handle)
    protocol = json.loads((HERE/'protocol.json').read_text())
    root = json.loads((HERE.parent/'reconstruction_v32/selection_full_hubs.json').read_text())['selected'][9]
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    segment = json.loads((HERE.parent/'reconstruction_v31/segment_fit.json').read_text())
    data,_ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    graph,metadata,model,params,applied = setup_day(data,root['day'],fit,segment)
    ride = root_ride(graph,root)
    actual = {}
    for name,(method,waiting) in protocol['methods'].items():
        planner = Planner(graph,model,params,root['deadline'])
        cold = Executor(graph,planner,root['deadline'],method=method,waiting=waiting).run(
            ride,root['start'],ConditionalEngine(model,params),ArchivedWorld(metadata))
        warm = Executor(graph,planner,root['deadline'],method=method,waiting=waiting).run(
            ride,root['start'],ConditionalEngine(model,params),ArchivedWorld(metadata))
        for key in ('trace','loss','status','decisions'):
            if cold[key] != warm[key]:
                raise AssertionError(('Endpoint repair cold/warm mismatch',name,key))
        actual[name] = dict(cold=cold,warm=warm)
        print(name,cold['status'],cold['loss'],flush=True)
    result['endpoint_repair'] = dict(backup=str(backup.relative_to(HERE)),
        added_endpoint=dict(day=root['day'],cid=10818,kind='departure',clock=29690),
        unresolved_arrival_values=[29794,30157],
        prior_actual={name:dict(status=row['cold']['status'],loss=row['cold']['loss'])
                      for name,row in result['actual'].items()},
        model_worlds_replayed=0, actual_policies_replayed=len(actual),
        actual_repair_wall=perf_counter()-started,
        accounting='Original total/evaluation timing and paired sampled worlds retained; this additional actual-only repair is charged separately.')
    result['actual'],result['supplement_applied'] = actual,applied
    with gzip.open(path,'wt') as handle:
        json.dump(result,handle,default=jsonable)


if __name__ == '__main__':
    main()
