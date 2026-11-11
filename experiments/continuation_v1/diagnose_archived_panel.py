"""Reconstruct three archived decision states, without rerunning the panel.

Probabilities below describe frozen marginal estimates or a stated generated
model. An archived realized outcome is never treated as a causal optimum.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.matched_router import MatchedRouter
from experiments.reconstruction_v2.run_matched_panel import world_distributions

SOURCE = ROOT / 'experiments/reconstruction_v2/matched_corrected_results.json'
METHOD = 'refresh/offline'


def connection_id(event):
    detail = event.get('details', '')
    return int(detail.split(';')[0].split('=')[1]) if detail.startswith('connection=') else None


def main():
    archived = json.loads(SOURCE.read_text())
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as handle:
        base = pickle.load(handle)
    with (ROOT / 'data/per_day_distributions.pkl').open('rb') as handle:
        days = pickle.load(handle)
    identities = json.loads((ROOT / 'experiments/reconstruction_data_audit/graph_identity.json').read_text())['by_trip']
    names = {stop.name: sid for sid, stop in base.stops.items()}
    cache = {}

    def case_at(date, stress, origin, seed):
        return next(case for case in archived['cases']
                    if (case['spec'][0], case['spec'][1], case['spec'][2], case['spec'][4])
                    == (date, stress, origin, seed))

    def router_at(case):
        # Frozen offline labels need only two query reconstructions across the
        # three states; no journey execution or parameter fitting is performed.
        key = (case['spec'][0], case['spec'][3])
        if key not in cache:
            router = MatchedRouter(copy.deepcopy(base), case['priors'])
            router.route(names[case['spec'][2]], names[case['spec'][3]], 490)
            cache[key] = router
        return cache[key]

    def state_at(case, cid, time, previous):
        record = next(row for row in case['records'] if row['method'] == METHOD)
        index, decision = next((i, decision) for i, decision in enumerate(record['decisions'])
                               if decision['selected'] == cid and decision['time'] == time)
        previous_arrival = next(event for event in record['events']
                                if event['event_type'] == 'arrive' and connection_id(event) == previous)
        router = router_at(case)
        router.set_execution_context(base.connections[previous].trip_id, previous_arrival['time'], set())
        stop = decision['stop']
        label = next(label for label in router.cached_result.stop_labels[stop] if label.connection_id == cid)
        reconstructed = router._score(label, time)
        if abs(reconstructed - decision['score']) > 1e-7:
            raise AssertionError('Archived score no longer reproduced by the frozen offline state')
        prior_leg = base.connections[previous]
        delay = previous_arrival['time'] - prior_leg.arr_time
        output = {
            'case_spec': case['spec'], 'method': METHOD, 'decision_index': index,
            'archived_journey_loss': record['loss'],
            'state': {'stop': stop, 'stop_name': base.stops[stop].name, 'time': time,
                      'current_trip': prior_leg.trip_id, 'last_arrival': previous_arrival['time'],
                      'known_current_trip_delay': delay, 'previous_completed_connection': previous},
            'archived_decision': decision,
            'reproduced_score': reconstructed,
            'frozen_label': {'conditional_mean_destination_arrival': label.mean_dest_arrival,
                             'conditional_continuation_success': label.feasibility,
                             'destination_mass_before_current_time': label.dest_arrival.prob_le(time - 1),
                             'selected_leg_unconditioned_arrival_mean': router.graph.connections[cid].arr_distribution.mean()},
            'trace_context': [event for event in record['events'] if event['time'] <= time][:30],
        }
        return output, record, router, label

    def subgraph(case, cids, state, known_delays):
        trips = {base.connections[cid].trip_id for cid in cids}
        selected = [c for c in base.connections if c.trip_id in trips]
        stop_ids = {sid for c in selected for sid in (c.dep_stop, c.arr_stop)}
        routes = {c.route for c in selected}
        generated = world_distributions(days[case['spec'][0]], routes, case['spec'][1])
        return {
            'scope': 'All archived segments of 2-3 named trips; original IDs retained. A new finite submodel requires an explicit observation contract.',
            'initial_state_for_adapter': state,
            'original_departure': 490, 'deadline': 610,
            'original_destination': names[case['spec'][3]],
            'known_trip_delays_from_causal_prefix': known_delays,
            'stops': {sid: {'name': base.stops[sid].name,
                            'min_transfer_time': base.stops[sid].min_transfer_time} for sid in sorted(stop_ids)},
            'connections': [{'original_id': c.id, 'trip_id': c.trip_id, 'route': c.route,
                             'route_id': identities[c.trip_id]['route_id'],
                             'agency_id': identities[c.trip_id]['agency_id'],
                             'dep_stop': c.dep_stop, 'arr_stop': c.arr_stop,
                             'dep_time': c.dep_time, 'arr_time': c.arr_time} for c in selected],
            'original_case_priors_by_route': {route: case['priors'][route] for route in sorted(routes)},
            'generated_trip_factor_laws': {
                trip: {'route': next(c.route for c in selected if c.trip_id == trip),
                       'delay_offset': -5,
                       'delay_probs': generated[next(c.route for c in selected if c.trip_id == trip)]['delay_probs'].tolist(),
                       'cancel_prob': generated[next(c.route for c in selected if c.trip_id == trip)]['cancel_prob'],
                       'shared_within_trip': True, 'independent_between_trips': True}
                for trip in sorted(trips)},
            'archived_realization_for_trace_check_only': {c.id: case['world'][str(c.id)] for c in selected},
            'information_warning': 'Removing other trips changes the local capped publisher. Do not call this the full archived conditional optimum without modeling the receipt history. Message timestamps are event times, not receipt times.',
        }

    results = []
    # A completed same-trip segment identifies the persistent delay exactly.
    case = case_at('2023-10-29', True, 'Zürich, Paradeplatz', 711)
    result, record, router, label = state_at(case, 1422, 523, 1421)
    delay = result['state']['known_current_trip_delay']
    earliest_after_selected_leg = base.connections[1422].arr_time + delay
    result.update({
        'diagnosis': 'Continuation PMF is not conditioned on a causally known shared-trip delay.',
        'exact_generated_model_fact': {'next_segment_departure': base.connections[1422].dep_time + delay,
                                       'next_segment_arrival': earliest_after_selected_leg,
                                       'arrival_at_goal_before_next_segment_arrival_probability': 0.0},
        'frozen_destination_mass_before_known_next_arrival': label.dest_arrival.prob_le(earliest_after_selected_leg - 1),
        'evidence': '1421 was boarded and completed before this decision. In the panel generator every segment of that trip has exactly the same delay and cancellation outcome.',
        'claim_boundary': 'The impossible conditional mass is established under the stated generator; this is not a population calibration estimate or a counterfactual journey-gain estimate.',
    })
    result['subgraph'] = subgraph(case, (1421, 1362), result['state'], {base.connections[1421].trip_id: delay})
    results.append(result)

    # A past departure observed at the origin also identifies another trip's
    # later segment, even after the passenger has moved to another stop.
    case = case_at('2023-10-07', False, 'Zürich Wiedikon, Bahnhof', 711)
    result, record, router, label = state_at(case, 217, 501, 291)
    origin = names[case['spec'][2]]
    initial_messages = []
    for message in record['published_messages']:
        if message['stop_id'] != origin:
            break
        initial_messages.append(message)
    evidence = next(message for message in initial_messages if message['event_id'] == '216:departure')
    known_delay = evidence['actual_time'] - evidence['scheduled_time']
    selected = router.graph.connections[217]
    ready = 501 + base.stops[selected.dep_stop].min_transfer_time
    p_board = selected.dep_distribution.total_prob - selected.dep_distribution.prob_le(ready - 1)
    next_decision = record['decisions'][result['decision_index'] + 1]
    result.update({
        'diagnosis': 'A shared-trip delay observed at another stop is not propagated to the later segment; the score treats a recoverable failed attempt as terminal loss.',
        'causal_evidence_at_initial_origin': evidence,
        'exact_generated_model_fact': {'known_other_trip_delay': known_delay,
                                       'selected_departure': selected.dep_time + known_delay,
                                       'transfer_ready': ready, 'selected_boarding_probability_given_evidence': 0.0},
        'marginal_estimated_boarding_probability': p_board,
        'failure_term_used_by_score': (1.0 - p_board * label.feasibility) * 120,
        'next_archived_decision_after_failed_attempt': next_decision,
        'claim_boundary': 'The exact zero boarding probability follows from a message already received at the origin, not from future world access. Recovery exists in the executor; its expected value requires continuation planning and is not inferred from this one realized journey.',
    })
    known = {base.connections[291].trip_id: result['state']['known_current_trip_delay'],
             base.connections[217].trip_id: known_delay}
    result['subgraph'] = subgraph(case, (217, 944, 291), result['state'], known)
    results.append(result)

    # The five-label cap is applied before the actual loss score, so a lower
    # score already computed by the same model can be outside the offered menu.
    case = case_at('2023-10-07', True, 'Zürich Wiedikon, Bahnhof', 712)
    result, record, router, label = state_at(case, 1097, 499, 705)
    omitted = next(item for item in router.cached_result.stop_labels[result['state']['stop']]
                   if item.connection_id == 706)
    result.update({
        'diagnosis': 'The conditional-mean top-5 screen omits the current-trip continuation with a lower capped-loss score.',
        'omitted_connection': {'original_id': 706, 'conditional_mean_destination_arrival': omitted.mean_dest_arrival,
                               'score_under_same_estimate': router._score(omitted, 499),
                               'known_departure_from_current_trip': base.connections[706].dep_time + result['state']['known_current_trip_delay'],
                               'known_arrival_from_current_trip': base.connections[706].arr_time + result['state']['known_current_trip_delay'],
                               'present_in_archived_candidate_menu': 706 in result['archived_decision']['candidates']},
        'claim_boundary': 'This establishes a pre-score menu effect, not an expected benefit from taking 706, nor a retain-versus-refresh advantage. Both candidate modes use this same cap.',
    })
    result['subgraph'] = subgraph(case, (705, 1097, 293), result['state'],
                                  {base.connections[705].trip_id: result['state']['known_current_trip_delay']})
    results.append(result)

    output = {
        'source': str(SOURCE.relative_to(ROOT)), 'implementation': archived['config']['implementation'],
        'scope': 'Three archived offline decision states and two reconstructed label queries; zero rerun journeys.',
        'results': results,
        'next_exact_panel': 'Use the extracted 2-3-trip subgraphs, condition only on the supplied causal prefix, and keep one delay per trip. Retain missed-attempt recovery. Quantizing the supplied 70-bin laws is an explicit new finite approximation, not the original exact generating model.',
    }
    destination = Path(__file__).with_name('archived_panel_diagnosis.json')
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(destination.relative_to(ROOT)),
                      'reconstructed_queries': len(cache), 'rerun_journeys': 0,
                      'diagnoses': [{'selected': r['archived_decision']['selected'],
                                     'time': r['state']['time'], 'diagnosis': r['diagnosis']}
                                    for r in results]}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
