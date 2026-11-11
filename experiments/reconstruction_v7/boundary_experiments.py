"""Small-model falsification of intermediate first-exit certificates."""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.information_planner import ExactPlanner
from src.intermediate_boundary import FirstExitEnvelope, evaluate_causal_policy, macro_advantage_reference
from src.platform_transit import OBSERVE, WAIT, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def network(rows):
    graph = TransitGraph()
    for stop in {s for row in rows for s in row[2:4]}:
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for row in rows:
        graph.add_connection(Connection(-1, *row))
    return graph


def scene(graph, publications=None):
    return JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                            for c in graph.connections}, minute=1, publications=publications or {})


def witness_policy(model, issued, graph):
    def choose(node):
        actions = tuple(model.actions(node.state))
        if OBSERVE in actions:
            return OBSERVE
        cid = next((cid for cid in issued if cid not in node.state.used_connections), None)
        if (('alight',) in actions and cid is not None
                and graph.connections[cid].trip_id != node.state.current_trip):
            return ('alight',)
        return ('board', cid) if ('board', cid) in actions else WAIT
    return choose


def cases():
    simple = [('R', 'A', 0, 1, 0, 10), ('R', 'A', 1, 2, 10, 20), ('B', 'B', 0, 2, 25, 30)]
    for name, walks, zero in [('middle_hard_support', (), False),
                              ('walk_shortcut', ((0, 2, 3),), False),
                              ('zero_support_floor', (), True)]:
        graph = network(simple)
        yield dict(name=name, graph=graph, bank=[(1., scene(graph))], macro=(0,), issued=(0, 1),
                   footpaths=walks, runtime={0: 0, 1: 0, 2: 0} if zero else {0: 10, 1: 10, 2: 5},
                   departure={0: 0, 1: 0, 2: 0} if zero else {0: 0, 1: 10, 2: 25})
    graph = network([('R', 'A', 0, 1, 0, 10), ('B', 'B', 1, 2, 10, 11),
                     ('C', 'C', 1, 2, 10, 11), ('N', 'notice', 3, 4, 5, 6)])
    bank = []
    for first_fast in (True, False):
        events = dict(scene(graph).events)
        events[1] = VehicleEvent(0, False, 10, 11 if first_fast else 21)
        events[2] = VehicleEvent(0, False, 10, 21 if first_fast else 11)
        bank.append((.5, JourneyScenario(events, minute=1,
            publications={'3:departure': 5 if first_fast else 15})))
    yield dict(name='correlated_external_notice', graph=graph, bank=bank, macro=(0,), issued=(0, 1),
               footpaths=(), runtime={0: 10, 1: 1, 2: 1, 3: 1}, departure={0: 0, 1: 10, 2: 10, 3: 5})
    graph = network([('R', 'A', 0, 1, 0, 1), ('R', 'A', 1, 2, 3, 4), ('B', 'B', 0, 1, 1, 2)])
    active = scene(graph)
    stopped = dict(active.events)
    stopped[0] = VehicleEvent(999, True, 0, 1)
    yield dict(name='cancellation_then_restore', graph=graph,
        bank=[(.5, active), (.5, JourneyScenario(stopped, minute=1, publications={}))], macro=(0,),
        issued=(0, 1), footpaths=(), runtime={0: 1, 1: 1, 2: 1}, departure={0: 0, 1: 3, 2: 1})
    graph = network([('R', 'A', 0, 1, 0, 10), ('R', 'A', 1, 0, 10, 20), ('R', 'A', 0, 2, 20, 21)])
    yield dict(name='same_trip_loop', graph=graph, bank=[(1., scene(graph))], macro=(0, 1),
               issued=(0, 1, 2), footpaths=(), runtime={0: 10, 1: 10, 2: 1}, departure={0: 0, 1: 10, 2: 20})


def marginal(bank, macro):
    mass, trajectories = defaultdict(float), {}
    for p, world in bank:
        key = tuple(world.events[cid] for cid in macro)
        mass[key] += p
        trajectories[key] = {cid: world.events[cid] for cid in macro}
    return [(mass[key], trajectories[key]) for key in mass]


def main():
    HERE.mkdir(exist_ok=True)
    protocol = dict(scope='Finite physical-model mechanism falsification; no new actual dates.',
        source_floors='Declared synthetic hard physical/support floors; no empirical minimum promoted to a physical floor.',
        repetitions=21, certificate_tolerances=[0, 1, 5],
        fairness='One local-event platform model; positive walks, explicit correlated publications, zero transfer requirements.',
        upper='Full fixed causal witness evaluation only, not optimal subtree expansion; all cost included.',
        exact_reference='Existing solve_compressed, with full same-information optimality and equivalent one-step action merging.',
        stop='A static support envelope plus generic U-L is not novel; zero physical floors or absent net value stop this prototype before independent confirmation.',
        sources={name: (ROOT / name).read_text() for name in ('src/intermediate_boundary.py',
            'src/platform_transit.py', 'src/information_planner.py',
            'experiments/reconstruction_v7/boundary_experiments.py')})
    (HERE / 'boundary_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    records = []
    for case in cases():
        if case['issued'][:len(case['macro'])] != case['macro']:
            raise ValueError('The feasible witness does not extend the proposed fragment')
        samples = []
        for _ in range(protocol['repetitions']):
            before = perf_counter()
            model = build_platform_model(case['graph'], case['bank'], 0, 2, 0, 40,
                                         footpaths=case['footpaths'])
            model_seconds = perf_counter() - before
            before = perf_counter()
            envelope = FirstExitEnvelope(case['graph'], 2, case['runtime'], case['departure'],
                                         footpaths=case['footpaths'])
            preprocess_seconds = perf_counter() - before
            local = marginal(case['bank'], case['macro'])
            before, cpu_before = perf_counter(), process_time()
            upper = evaluate_causal_policy(model, witness_policy(model, case['issued'], case['graph']))
            upper_seconds, upper_cpu = perf_counter() - before, process_time() - cpu_before
            before, cpu_before = perf_counter(), process_time()
            certificate = envelope.certify(local, case['macro'], 0, upper.value, deadline=40, tolerance=1)
            bound_seconds, bound_cpu = perf_counter() - before, process_time() - cpu_before
            planner = ExactPlanner(model)
            before, cpu_before = perf_counter(), process_time()
            optimum = planner.solve_compressed().value
            full_seconds, full_cpu = perf_counter() - before, process_time() - cpu_before
            before = perf_counter()
            planner.solve_compressed()
            warm_identical_seconds = perf_counter() - before
            if not certificate.lower <= optimum + 1e-9 <= upper.value + 1e-9:
                raise AssertionError('Physical exact comparison rejected a claimed bound')
            samples.append(dict(model_setup_seconds=model_seconds, envelope_preprocess_seconds=preprocess_seconds,
                upper_seconds=upper_seconds, upper_cpu_seconds=upper_cpu, boundary_seconds=bound_seconds,
                boundary_cpu_seconds=bound_cpu, full_cold_seconds=full_seconds, full_cold_cpu_seconds=full_cpu,
                full_warm_identical_context_seconds=warm_identical_seconds))
        record = dict(name=case['name'], macro=list(case['macro']), issued=list(case['issued']),
            macro_endpoint=case['graph'].connections[case['macro'][-1]].arr_stop, goal=2,
            world_count=len(case['bank']), local_trajectory_count=len(local),
            inventory=[asdict(e) for e in envelope.inventory(case['macro'], 0)],
            certificate=asdict(certificate), exact_optimum=optimum,
            actual_witness_model_gap=upper.value - optimum,
            witness_transitions=upper.transitions, full_transitions=planner.stats['transition_evaluations'],
            accepted_by_tolerance={str(t): certificate.gap <= t for t in protocol['certificate_tolerances']},
            timing_medians={key: float(np.median([s[key] for s in samples])) for key in samples[0]},
            timing_samples=samples)
        records.append(record)
        print(record['name'], certificate.lower, optimum, upper.value, certificate.gap, flush=True)
    output = dict(protocol='boundary_protocol.json', rows=records,
        interpretation='Toy valid/tight middle cases establish correctness only. All supplied clocks/service mass belong to the same finite model; no claim of real-world risk or algorithmic novelty.')
    (HERE / 'boundary_results.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
