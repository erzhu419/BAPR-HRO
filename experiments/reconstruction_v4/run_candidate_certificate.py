"""Finite-bank candidate certificates and repair on archived Zürich topology."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
import json
from math import isclose
from pathlib import Path
import pickle
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.continuation_v1.run_exact_attribution import PublicStatePlanner
from experiments.continuation_v1.run_multileg_reference import THROUGH, TRANSFER, fragment, make_bank
from src.candidate_certificate import ArrivalRelaxation, certify
from src.committed_transit import CommittedState, OBSERVE, WAIT
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def timed(fn, *args, **kwargs):
    start = perf_counter()
    value = fn(*args, **kwargs)
    return value, (perf_counter() - start) * 1000


def action_name(action, old_ids):
    return str(old_ids[action[1]]) if action[0] == 'select' else action[0]


def cert_row(result, old_ids):
    return {'lower': float(result.lower), 'upper': float(result.upper),
            'gap': float(result.gap), 'transitions': result.transitions,
            'removed': [action_name(a, old_ids) for a in result.removed],
            'action_lower': {action_name(a, old_ids): float(q) for a, q in result.action_lower.items()}}


def witness(graph, destination, through=False):
    def policy(state, actions):
        if OBSERVE in actions:
            return OBSERVE
        ready_departures = {m.event_id for m in state.messages if m.actual_time == state.time}
        candidates = []
        for action in actions:
            if action[0] != 'select':
                continue
            c = graph.connections[action[1]]
            if through:
                eligible = c.trip_id == state.current_trip or f'{c.id}:departure' in ready_departures
            else:
                eligible = c.arr_stop == destination and state.time - 5 <= c.dep_time <= state.time + 12
            if eligible:
                candidates.append((c.arr_time, c.id, action))
        return min(candidates)[-1] if candidates else WAIT
    return policy


def public_roots(relaxation, state):
    model = relaxation.committed_model(state)
    groups = defaultdict(list)
    for i, world in enumerate(model.scenarios):
        groups[model.transition(world, state, OBSERVE).state].append(i)
    return groups.items()


def fragment_case(base, name, ids, start, cap, specs, through=False):
    graph, old_ids = fragment(base, ids)
    destination = graph.connections[-1].arr_stop
    origin = graph.connections[0].dep_stop
    bank = [(p, s) for _, p, s in make_bank(graph, old_ids, specs)]
    relaxation, build_ms = timed(ArrivalRelaxation, graph, bank, destination, start, start + cap)
    start_state = CommittedState(origin, start, None, start)
    ((state, support),) = public_roots(relaxation, start_state)
    if len(support) != len(bank):
        raise AssertionError('Fragment root should have a common initial observation')
    policy = witness(graph, destination, through)
    cert, cert_ms = timed(certify, relaxation, state, policy)
    model = relaxation.committed_model(state)
    full = PublicStatePlanner(model)
    optimum, exact_ms = timed(full.solve)
    if float(cert.lower) > optimum.value + 1e-9 or optimum.value > float(cert.upper) + 1e-9:
        raise AssertionError('Exact reference lies outside certificate')
    if any(float(cert.action_lower[a]) > optimum.action_values[a] + 1e-9 for a in cert.action_lower):
        raise AssertionError('Optimistic per-action cost exceeds exact cost')

    stats = {'calls': 0, 'transitions': 0, 'actions_removed': 0, 'nonroot_removals': 0}
    def menu(info):
        current = certify(relaxation, info.state, policy, support=info.support)
        stats['calls'] += 1
        stats['transitions'] += current.transitions
        stats['actions_removed'] += len(current.removed)
        stats['nonroot_removals'] += len(current.removed) if info.stage else 0
        return tuple(a for a in model.actions(info.state) if a not in current.removed)

    screened = PublicStatePlanner(model, menu)
    restricted, screened_ms = timed(screened.solve)
    if not isclose(optimum.value, restricted.value, abs_tol=1e-9):
        raise AssertionError('Certified H changes the same-information optimum')

    return {'name': name, 'connections': list(ids), 'cap': cap, 'index_labels': relaxation.label_count,
            'index_build_ms': build_ms, 'certificate_ms': cert_ms, **cert_row(cert, old_ids),
            'full': {'value': optimum.value, 'wall_ms': exact_ms, **full.stats},
            'H_every_future_history': {'value': restricted.value, 'wall_ms': screened_ms,
                                       **screened.stats, 'screening_work': stats}}


def city_case(base):
    origin = next(i for i, stop in base.stops.items() if stop.name == 'Zürich, Heuried')
    destination = next(i for i, stop in base.stops.items() if stop.name == 'Zürich, Triemli')
    start, deadline = 533, 563
    bank = []
    for _, weight, trip_shifts in THROUGH:
        shifts = {base.connections[cid].trip_id: shift for cid, shift in trip_shifts.items()}
        events = {}
        for c in base.connections:
            shift = shifts.get(c.trip_id, 0)
            events[c.id] = (VehicleEvent(999, True, c.dep_time, c.arr_time) if shift is None else
                            VehicleEvent(shift, False, c.dep_time + shift, c.arr_time + shift))
        bank.append((weight, JourneyScenario(events)))
    relaxation, build_ms = timed(ArrivalRelaxation, base, bank, destination, start, deadline)
    state = CommittedState(origin, start, None, start)
    policy = witness(base, destination, through=True)
    ids = {c.id: c.id for c in base.connections}

    def screen(index):
        return [{'world_support': support, 'public_event_ids': [m.event_id for m in public.messages],
                 **cert_row(certify(index, public, policy, support=support), ids)}
                for public, support in public_roots(index, state)]

    before, before_ms = timed(screen, relaxation)
    trip = [c for c in base.connections if c.trip_id == base.connections[68].trip_id]
    edits = [('resolve_departure_delay', 1, {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in trip}),
             ('cancel_through_trip', 0, {c.id: VehicleEvent(999, True, c.dep_time, c.arr_time) for c in trip})]
    repairs = []
    for name, world, changes in edits:
        work, repair_ms = timed(relaxation.repair, world, changes)
        after, after_ms = timed(screen, relaxation)
        p, scene = bank[world]
        bank[world] = p, JourneyScenario(dict(scene.events) | changes)
        fresh, rebuild_ms = timed(ArrivalRelaxation, base, bank, destination, start, deadline)
        if any(a.values != b.values for a, b in zip(relaxation.worlds, fresh.worlds)):
            raise AssertionError('The repair changed relaxed labels relative to a fresh build')
        if after != screen(fresh):
            raise AssertionError('The repaired certificate differs from a fresh certificate')
        repairs.append({'name': name, 'changed_world': world, 'changed_connections': sorted(changes),
                        'repair_ms': repair_ms, 'repair': asdict(work), 'fresh_build_ms': rebuild_ms,
                        'after_screen_ms_including_initial_poll': after_ms, 'after': after,
                        'all_labels_and_certificates_match_fresh_build': True})
    if '67' not in before[0]['removed'] or '67' in repairs[0]['after'][0]['removed']:
        raise AssertionError('The repaired screen should restore the tied departure')
    return {'origin': base.stops[origin].name, 'destination': base.stops[destination].name,
            'scope': 'Archived 2000-connection template; four constructed trip-level scenarios, with stochastic shifts/cancellations on two full trips and all other services on time. One OD and horizon; no citywide exact comparison.',
            'index_labels': relaxation.label_count, 'graph_connections': len(base.connections),
            'index_build_ms': build_ms, 'before_screen_ms_including_initial_poll': before_ms,
            'before': before, 'edits': repairs}


def corridor_case():
    stops, deadline = 12, 60
    graph = TransitGraph()
    for stop in range(stops):
        graph.add_stop(Stop(stop, str(stop)))
    for stop in range(stops - 1):
        dep = deadline - stops - 1 + stop
        graph.add_connection(Connection(stop, str(stop), str(stop), stop, stop + 1, dep, dep + 1))
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections})
    index = ArrivalRelaxation(graph, [(1., scene)], stops - 1, 0, deadline)
    final = graph.connections[-1]
    change = {final.id: VehicleEvent(999, True, final.dep_time, final.arr_time)}
    before = index.arrival(0, 0, 0)
    work, elapsed = timed(index.repair, 0, change)
    fresh = ArrivalRelaxation(graph, [(1., JourneyScenario(dict(scene.events) | change))],
                              stops - 1, 0, deadline)
    if index.worlds[0].values != fresh.worlds[0].values:
        raise AssertionError('Long-range dependency propagation differs from fresh rebuild')
    return {'scope': 'Constructed timed corridor with a single downstream failure and no fallback.',
            'index_labels': index.label_count, 'repair': asdict(work), 'repair_ms': elapsed,
            'root_lower_before': before, 'root_lower_after': index.arrival(0, 0, 0)}


def main():
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as file:
        base = pickle.load(file)
    rows = [fragment_case(base, 'through_vehicle', (67, 68, 1303, 1304), 533, 14, THROUGH, True),
            fragment_case(base, 'transfer_with_direct_fallback', (123, 147, 261, 1637), 502, 12, TRANSFER)]
    result = {'scope': 'Fixed local-committed scenario banks with free-transfer optimistic suffix labels. All timings are single-process measurements, not a matched production benchmark.',
              'fragments': rows, 'archived_template': city_case(base), 'corridor': corridor_case()}
    output = Path(__file__).with_name('candidate_certificate_results.json')
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(output)


if __name__ == '__main__':
    main()
