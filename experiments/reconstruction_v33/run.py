"""All frozen roots with persistent posterior and later-vehicle adaptation."""

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v31.model import SegmentDurationBelief
from experiments.reconstruction_v32.journey import TripWorlds, paired_statistics
from experiments.reconstruction_v33.execute import Executor, ArchivedWorld, DrawnWorld
from experiments.reconstruction_v33.history import ConditionalEngine
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v33.planner import Planner


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def setup_day(data, day, fit, segment):
    graph, _, ids, _, old_metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
    metadata = deepcopy(old_metadata)
    records = json.loads((HERE/'endpoint_supplement.json').read_text())['records']
    applied = []
    for row in records:
        if row['day'] != day:
            continue
        cid = row['local_city_cid']
        target = metadata[cid]
        if target['trip_id'] != row['trip_id']:
            raise ValueError('Supplement does not address the same physical scheduled trip')
        if (target['dep_sequence'], target['arr_sequence']) != (row['dep_sequence'], row['arr_sequence']):
            raise ValueError('Supplement does not address the same physical event slots')
        for kind in ('departure', 'arrival'):
            if kind not in row:
                continue
            if target[kind]['status'] != 'MISSING':
                if target[kind] != row[kind]:
                    raise ValueError('Supplement would replace an already resolved endpoint')
            else:
                target[kind] = row[kind]
                applied.append((cid, kind))
    model = SegmentDurationBelief(graph, duration_fit=fit['duration_fit'], segment_fit=segment,
        stop_identities={v:k for k,v in ids.items()}, public_occurrences=False,
        grid_step_seconds=20, **fit['increments'])
    params = {c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
    return graph, metadata, model, params, applied


def run_root(graph, metadata, model, params, root, index, protocol, *, pilot=False):
    began = perf_counter()
    ride = root_ride(graph, root)
    actual = {}
    for name, (method, waiting) in protocol['methods'].items():
        planner = Planner(graph, model, params, root['deadline'])
        cold = Executor(graph, planner, root['deadline'], method=method, waiting=waiting).run(
            ride, root['start'], ConditionalEngine(model, params), ArchivedWorld(metadata))
        warm = Executor(graph, planner, root['deadline'], method=method, waiting=waiting).run(
            ride, root['start'], ConditionalEngine(model, params), ArchivedWorld(metadata))
        if (cold['loss'], cold['status']) != (warm['loss'], warm['status']):
            raise AssertionError('Warm conditional-message reuse changed the journey outcome')
        public = lambda result: [(r['kind'], r.get('cid', r.get('stop')), r['clock'], r.get('choice'))
                                 for r in result['trace']]
        if public(cold) != public(warm):
            raise AssertionError('Warm conditional-message reuse changed a public action')
        actual[name] = dict(cold=cold, warm=warm)
        print(index, 'actual', name, cold['status'], cold['loss'], round(cold['compute_wall'], 3), flush=True)
    count = 2 if pilot else protocol['evaluation_worlds']
    seed = protocol['evaluation_seed']+1000*index
    bank = TripWorlds(model, params, ride.paths(), count, seed)
    results, model_wall = {}, {}
    for name, (method, waiting) in protocol['methods'].items():
        tick = perf_counter()
        outcomes = []
        for world in range(count):
            planner = Planner(graph, model, params, root['deadline'])
            result = Executor(graph, planner, root['deadline'], method=method, waiting=waiting, trace=False).run(
                ride, root['start'], ConditionalEngine(model, params), DrawnWorld(bank, world))
            if result['loss'] is None:
                raise AssertionError(('A sampled original-law world is unsupported', index, name, world, result))
            outcomes.append(result)
        results[name] = outcomes
        model_wall[name] = perf_counter()-tick
        print(index, 'worlds', name, count, round(model_wall[name], 2), flush=True)
    primary = protocol['primary']
    paired = {name: paired_statistics([r['loss'] for r in results[primary]], [r['loss'] for r in rows])
              for name, rows in results.items()}
    return dict(root_id=root['root_id'], root_index=index, day=root['day'], actual=actual,
        benchmark_scope='Cold planner and warm same-history replay; underlying fitted-model kernels shared. Separate fresh-model benchmark supplies full cold cost.',
        evaluation_worlds=count, evaluation_seed=seed, sampled=results, paired=paired,
        evaluation_wall=model_wall, total_wall=perf_counter()-began)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-index', type=int)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    protocol = json.loads((HERE/'protocol.json').read_text())
    selection = json.loads((HERE.parent/'reconstruction_v32/selection_full_hubs.json').read_text())
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    segment = json.loads((HERE.parent/'reconstruction_v31/segment_fit.json').read_text())
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    (HERE/'roots').mkdir(exist_ok=True)
    days = {}
    for index, root in enumerate(selection['selected']):
        if args.root_index is not None and index != args.root_index:
            continue
        target = HERE/(f'pilot_{index:02}.json' if args.pilot else f'roots/{index:02}.json.gz')
        if target.exists():
            print('reuse', target.name, flush=True)
            continue
        if root['day'] not in days:
            days[root['day']] = setup_day(data, root['day'], fit, segment)
        graph, metadata, model, params, applied = days[root['day']]
        result = run_root(graph, metadata, model, params, root, index, protocol, pilot=args.pilot)
        result['supplement_applied'] = applied
        if args.pilot:
            target.write_text(json.dumps(result, default=jsonable)+'\n')
        else:
            with gzip.open(target, 'wt') as handle:
                json.dump(result, handle, default=jsonable)


if __name__ == '__main__':
    main()
