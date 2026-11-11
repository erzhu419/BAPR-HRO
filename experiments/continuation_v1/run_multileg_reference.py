"""Exact multi-leg references on two archived Zurich timetable fragments.

The banks are explicitly constructed persistent regimes, not realized trips.
All comparisons share the committed executor and local publication contract.
Only the action menu or the information disclosed before planning changes.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from math import isclose
from pathlib import Path
import pickle
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.continuation_v1.run_exact_attribution import PublicStatePlanner
from src.committed_transit import OBSERVE, WAIT, build_committed_model
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import TransitGraph


def fragment(base, connection_ids):
    result = TransitGraph()
    for cid in connection_ids:
        c = base.connections[cid]
        for stop in (c.dep_stop, c.arr_stop):
            if stop not in result.stops:
                result.add_stop(base.stops[stop])
        result.add_connection(replace(c, dep_distribution=None, arr_distribution=None))
    return result, dict(enumerate(connection_ids))


def make_bank(graph, new_to_old, specs):
    old_to_new = {old: new for new, old in new_to_old.items()}
    worlds = []
    for regime, weight, offsets in specs:
        shifts = {graph.connections[old_to_new[cid]].trip_id: delay for cid, delay in offsets.items()}
        events = {}
        for c in graph.connections:
            shift = shifts[c.trip_id]
            events[c.id] = (VehicleEvent(999, True, c.dep_time, c.arr_time) if shift is None
                            else VehicleEvent(shift, False, c.dep_time + shift, c.arr_time + shift))
        worlds.append((regime, weight, JourneyScenario(events)))
    return worlds


def menu_rule(model, removed):
    def allowed(info):
        return tuple(a for a in model.actions(info.state) if a[0] != 'select' or a[1] not in removed)
    return allowed


def boarding_leg_rule(graph, planner, worlds, deadline):
    """Choose the smallest conditional next-leg arrival inside H, else wait."""
    def policy(info):
        actions = planner._actions(info)
        if OBSERVE in actions:
            return OBSERVE
        state = info.state
        candidates = []
        for action in actions:
            if action[0] != 'select':
                continue
            c = graph.connections[action[1]]
            transfer = (graph.stops[state.stop].min_transfer_time
                        if state.current_trip is not None and c.trip_id != state.current_trip else 0)
            ready = max(state.time, state.last_arrival + transfer)
            if ready >= deadline:
                continue
            mass, time_sum = 0., 0.
            for i, p in zip(info.support, info.belief):
                event = worlds[i][2].events[c.id]
                if not event.canceled and ready <= event.departure <= min(deadline, ready + 12):
                    mass += p
                    time_sum += p * min(deadline, event.arrival)
            if mass:
                candidates.append((time_sum / mass, c.dep_time, c.id, action))
        return min(candidates)[-1] if candidates else WAIT
    return policy


def action_name(action, old_ids):
    if action is None:
        return 'terminal'
    return str(old_ids[action[1]]) if action[0] == 'select' else action[0]


def solve_model(graph, old_ids, worlds, origin, destination, start, cap, removed, evaluate_policy=False):
    normalizer = sum(w for _, w, _ in worlds)
    model = build_committed_model(graph, [(w / normalizer, scene) for _, w, scene in worlds],
                                  origin, destination, start, start + cap)
    planner = PublicStatePlanner(model, menu_rule(model, removed) if removed else None)
    wall, cpu = perf_counter(), process_time()
    result = planner.solve()
    profile = {'value': result.value, 'solve_wall_ms': 1000 * (perf_counter() - wall),
               'solve_cpu_ms': 1000 * (process_time() - cpu), 'states': dict(planner.stats)}
    root_decisions = []
    for probability, _, child, terminal in planner._branches[planner.root, OBSERVE]:
        if terminal:
            raise AssertionError('Initial public poll unexpectedly terminates the trip')
        choice = planner.solve(child)
        root_decisions.append({'probability': probability,
                               'public_events': [m.event_id for m in child.state.messages],
                               'action': action_name(choice.action, old_ids),
                               'Q': {action_name(a, old_ids): q for a, q in choice.action_values.items()}})
    profile['first_decisions_after_public_poll'] = root_decisions
    if removed:
        # Check the whole visited public value table, not just the root menu.
        if any(a[0] == 'select' and a[1] in removed
               for solution in planner._solutions.values() for a in solution.action_values):
            raise AssertionError('H admitted an excluded connection on a future history')
        profile['restricted_nonroot_states_with_removed_physical_option'] = sum(
            bool(node.stage and any(a[0] == 'select' and a[1] in removed
                                    for a in model.actions(node.state))) for node in planner._solutions)
    if evaluate_policy:
        policy = boarding_leg_rule(graph, planner, worlds, start + cap)
        cost = planner.evaluate(policy)
        loss = planner.decision_loss(policy)
        if not isclose(cost - result.value, loss, abs_tol=1e-9):
            raise AssertionError('Continuation advantage does not telescope to the policy cost gap')
        profile['next_leg_heuristic'] = {'cost': cost, 'within_menu_loss': loss}
    return profile


def comparison(base, name, ids, start, cap, removed_old, world_specs):
    graph, old_ids = fragment(base, ids)
    worlds = make_bank(graph, old_ids, world_specs)
    origin, destination = graph.connections[0].dep_stop, graph.connections[-1].arr_stop
    removed = {new for new, old in old_ids.items() if old in removed_old}

    def evaluate(bank, restricted, heuristic=False):
        return solve_model(graph, old_ids, bank, origin, destination, start, cap,
                           removed if restricted else set(), heuristic)

    rows = {}
    for restricted, label in ((False, 'full'), (True, 'H')):
        causal = evaluate(worlds, restricted, heuristic=restricted)
        current = {}
        for regime in dict.fromkeys(regime for regime, _, _ in worlds):
            subset = [entry for entry in worlds if entry[0] == regime]
            if len({tuple(scene.events.items()) for _, _, scene in subset}) < 2:
                raise AssertionError('Current-regime control must retain future uncertainty')
            current[regime] = {'probability': sum(p for _, p, _ in subset),
                               'future_worlds': len(subset), **evaluate(subset, restricted)}
        current_value = sum(row['probability'] * row['value'] for row in current.values())
        hindsight_value = sum(p * evaluate([(z, p, scene)], restricted)['value'] for z, p, scene in worlds)
        if hindsight_value > current_value + 1e-9 or current_value > causal['value'] + 1e-9:
            raise AssertionError('More information worsened the reference optimum')
        rows[label] = {'same_information': causal, 'current_regime_known': current,
                       'current_regime_value': current_value, 'hindsight_value': hindsight_value}
    full, restricted = (rows[k]['same_information']['value'] for k in ('full', 'H'))
    if restricted < full - 1e-9:
        raise AssertionError('A smaller menu improved the exact same-information optimum')
    return {'name': name, 'origin': graph.stops[origin].name, 'destination': graph.stops[destination].name,
            'start': start, 'cap': cap, 'removed_at_all_histories': removed_old,
            'connections': [{'archive_id': old_ids[c.id], 'local_id': c.id, 'trip': c.trip_id,
                             'from': graph.stops[c.dep_stop].name, 'to': graph.stops[c.arr_stop].name,
                             'departure': c.dep_time, 'arrival': c.arr_time} for c in graph.connections],
            'worlds': [{'regime': z, 'probability': p,
                        'events_by_archive_connection': {old_ids[i]: asdict(e) for i, e in scene.events.items()}}
                       for z, p, scene in worlds], 'values': rows,
            'same_information_candidate_loss': restricted - full,
            'same_information_within_H_loss': rows['H']['same_information']['next_leg_heuristic']['within_menu_loss']}


THROUGH = [('clear', .375, {67: -1, 1303: 3}), ('clear', .125, {67: 2, 1303: 0}),
           ('disrupted', .375, {67: None, 1303: 0}), ('disrupted', .125, {67: 0, 1303: 5})]
TRANSFER = [('good', .4, {123: 0, 147: 0, 261: 6, 1637: 0}),
            ('good', .1, {123: 0, 147: None, 261: 6, 1637: 0}),
            ('bad', .1, {123: 0, 147: 0, 261: 6, 1637: 0}),
            ('bad', .4, {123: 0, 147: None, 261: 6, 1637: 0})]


def main():
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as f:
        base = pickle.load(f)
    through = comparison(base, 'through_vehicle', (67, 68, 1303, 1304), 533, 14, [68], THROUGH)
    transfer_ids = (123, 147, 261, 1637)
    transfer = comparison(base, 'transfer_with_direct_fallback', transfer_ids, 502, 12, [261], TRANSFER)
    if any(not isclose(a, b, abs_tol=1e-9) for a, b in (
            (through['values']['full']['same_information']['value'], 3.625),
            (through['values']['H']['same_information']['value'], 5.75),
            (transfer['values']['full']['same_information']['value'], 6.),
            (transfer['values']['full']['current_regime_value'], 5.6),
            (transfer['values']['full']['hindsight_value'], 5.))):
        raise AssertionError('Multi-leg reference disagrees with the hand-derived finite cases')

    scale = [{'cap': 12, **{k: transfer['values'][k]['same_information'] for k in ('full', 'H')}}]
    graph, old_ids = fragment(base, transfer_ids)
    worlds = make_bank(graph, old_ids, TRANSFER)
    for cap in (20, 30):
        full, restricted = [solve_model(graph, old_ids, worlds, graph.connections[0].dep_stop,
                                       graph.connections[-1].arr_stop, 502, cap, removed)
                            for removed in (set(), {i for i, old in old_ids.items() if old == 261})]
        scale.append({'cap': cap, 'full': full, 'H': restricted})
    output = {
        'scope': 'Two four-connection archived timetable fragments and constructed persistent trip-shift/cancellation banks. Omitted vehicles have no actions or publications.',
        'information': 'Committed 12-minute boarding, per-stop occurred last10/max5 publications, one-minute waiting, and archived min-transfer time. H filters actions on every history while the physical message generator keeps every fragment connection.',
        'information_references': 'Same information sees only the public history. Regime-aware policies know the persistent regime and its conditional law, still with multiple possible future scenes. Hindsight knows each complete scene from the beginning and is a relaxation.',
        'scale_scope': 'One exact run per horizon on the same fragment and bank. Profile is reference-enumeration wall/CPU time and explored states, with no controlled city comparison.',
        'stage_IV': 'Candidate losses can occur in these instances, but a matched practical full-planning bottleneck, cheap certificate and novel structured update are still unestablished.',
        'cases': [through, transfer], 'horizon_scale': scale,
    }
    path = HERE / 'multileg_reference_results.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    for case in output['cases']:
        print(case['name'], {k: [v['same_information']['value'], v['current_regime_value'],
                                 v['hindsight_value']] for k, v in case['values'].items()},
              'support', case['same_information_candidate_loss'], 'within', case['same_information_within_H_loss'])
    print(path)


if __name__ == '__main__':
    main()
