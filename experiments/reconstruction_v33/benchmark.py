"""Fresh-model cold and identical-history warm costs on two static roots.

The fixed roots are original indices zero and one, not selected by outcomes.
This measures implementation costs of the declared controls, not a new-method
claim. Real unsupported histories and unsuccessful journeys remain in output.
"""

import json
import os
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v31.model import SegmentDurationBelief
from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v33.execute import Executor, ArchivedWorld, DrawnWorld
from experiments.reconstruction_v33.history import ConditionalEngine
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v33.planner import Planner
from experiments.reconstruction_v33.run import setup_day, jsonable


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'


def quality(result):
    return {key: result[key] for key in ('loss', 'status', 'trace', 'decisions')}


def measure(graph, fit, segment, identities, parameters, root, world, method, waiting):
    cold_start = perf_counter()
    cold_cpu_start = process_time()
    model = SegmentDurationBelief(graph, duration_fit=fit['duration_fit'], segment_fit=segment,
        stop_identities=identities, public_occurrences=False, grid_step_seconds=20, **fit['increments'])
    constructed = perf_counter()
    constructed_cpu = process_time()
    ride = root_ride(graph, root)
    planner = Planner(graph, model, parameters, root['deadline'])
    prepared = perf_counter()
    prepared_cpu = process_time()
    cold = Executor(graph, planner, root['deadline'], method=method, waiting=waiting).run(
        ride, root['start'], ConditionalEngine(model, parameters), world)
    cold_total = perf_counter()-cold_start
    cold_cpu = process_time()-cold_cpu_start
    cold_stats = dict(planner.stats)
    tick = perf_counter()
    warm_cpu_start = process_time()
    warm = Executor(graph, planner, root['deadline'], method=method, waiting=waiting).run(
        ride, root['start'], ConditionalEngine(model, parameters), world)
    warm_total = perf_counter()-tick
    warm_cpu = process_time()-warm_cpu_start
    agreement = quality(cold) == quality(warm)
    return dict(quality=quality(cold), warm_quality=quality(warm),
        identical_public_trace_and_outcome=agreement,
        cold=dict(total_wall=cold_total, total_process_cpu=cold_cpu,
                  model_constructor_wall=constructed-cold_start,
                  model_constructor_cpu=constructed_cpu-cold_cpu_start,
                  menu_and_planner_constructor_wall=prepared-constructed,
                  menu_and_planner_constructor_cpu=prepared_cpu-constructed_cpu,
                  executor_process_cpu=cold_cpu-(prepared_cpu-cold_cpu_start),
                  executor_compute_wall=cold['compute_wall'], executor_total_wall=cold['total_wall'],
                  planner_stats=cold_stats),
        warm=dict(total_wall=warm_total, total_process_cpu=warm_cpu, executor_compute_wall=warm['compute_wall'],
                  executor_total_wall=warm['total_wall'],
                  planner_stats={key:value-cold_stats[key] for key,value in planner.stats.items()}))


def main():
    tick = perf_counter()
    protocol = json.loads((HERE/'protocol.json').read_text())
    roots = json.loads((HERE.parent/'reconstruction_v32/selection_full_hubs.json').read_text())['selected']
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    segment = json.loads((HERE.parent/'reconstruction_v31/segment_fit.json').read_text())
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    result = dict(complete=False, root_indices=[0,1], data_io_wall=perf_counter()-tick,
        environment=dict(logical_cpus=os.cpu_count(), threads={name:os.environ.get(name)
            for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
        scope='Two prespecified roots; real and original-law world0. Fresh model, nodes, posterior and planner are charged cold; identical-history cache reuse is charged warm.',
        timing='Total includes constructor, menu nodes, posterior, recursive tail preparation, decisions, and physical endpoint lookup. Data/graph preparation and world generation are listed separately.',
        wall_scope='Concurrent V33 work can contend for the host. Wall times are not isolated-host speedup evidence; process CPU is the charged comparison.',
        world_contract='Each root uses the formal evaluation seed and full draw-bank size before selecting world0; physical vehicles are drawn once and never exposed to the planner.',
        days={}, roots=[], measurements=[])
    days = {}
    for index in (0,1):
        root = roots[index]
        day = root['day']
        if day not in days:
            tick = perf_counter()
            days[day] = setup_day(data, day, fit, segment)
            result['days'][day] = dict(graph_endpoint_and_sampling_model_preparation_wall=perf_counter()-tick,
                                       supplement_applied=days[day][-1])
        graph, metadata, sampling_model, parameters, _ = days[day]
        seed = protocol['evaluation_seed']+1000*index
        tick = perf_counter()
        bank = TripWorlds(sampling_model, parameters, root_ride(graph, root).paths(),
                          protocol['evaluation_worlds'], seed)
        result['roots'].append(dict(root_index=index, root_id=root['root_id'], seed=seed,
            bank_worlds=protocol['evaluation_worlds'], executed_world_index=0,
            physical_world_generation_wall=perf_counter()-tick))
        worlds = dict(actual=ArchivedWorld(metadata), model_world0=DrawnWorld(bank, 0))
        for scenario, world in worlds.items():
            for name, (method, waiting) in protocol['methods'].items():
                measurement = measure(graph, fit, segment, sampling_model.stop_identities,
                                      parameters, root, world, method, waiting)
                row = dict(root_index=index, root_id=root['root_id'], scenario=scenario,
                           policy=name, **measurement)
                result['measurements'].append(row)
                (HERE/'benchmark.json').write_text(json.dumps(result, default=jsonable,
                    ensure_ascii=False, indent=2)+'\n')
                print(index, scenario, name, measurement['quality']['status'], measurement['quality']['loss'],
                      'cold', round(measurement['cold']['total_wall'],6),
                      'warm', round(measurement['warm']['total_wall'],6), flush=True)
                if not measurement['identical_public_trace_and_outcome']:
                    raise AssertionError(('Warm replay changed a public action or outcome', index, scenario, name))
    result['complete'] = True
    result['summary'] = dict(cells=len(result['measurements']),
        identical_replays=sum(r['identical_public_trace_and_outcome'] for r in result['measurements']),
        measured_quality='Travel loss and status are retained for every cell; unsupported histories have loss=null.',
        interpretation='Small charged-control benchmark; it does not establish a G2 algorithm contribution.')
    (HERE/'benchmark.json').write_text(json.dumps(result, default=jsonable, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__ == '__main__':
    main()
