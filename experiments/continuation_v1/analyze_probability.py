"""Selected-action probability errors under the known generated transit model.

Reconstruct actual public receipt histories and fitted route parameters without
routing, rollout, or hidden-world access. The true reference is the analytic
posterior under the documented generating law, not a realized binary outcome.
"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v2.run_matched_panel import historical_priors
from experiments.continuation_v1.historical_router import HistoricalContinuationRouter
from src.bocd.regime_detector import DelayObservation
from src.trip_belief import TripBelief


def generating_parameters(day, routes, stress):
    """Use the original generator's input moments, before delay discretization."""
    result = {}
    for route in routes:
        item = day.get(route, {'mean': 1., 'std': 2.})
        result[route] = {
            'mean': item['mean'] + (8 if stress else 0),
            'std': max(item['std'], 4 if stress else .5),
            'cancel_rate': .2 if stress else 0.,
        }
    return result


def failure_probability(belief, cid, parameters, ready, deadline):
    """Failure means the selected attempt does not board within its commitment."""
    delays, probabilities = belief.posterior(cid, parameters)
    connection = belief.graph.connections[cid]
    departure = connection.dep_time + delays
    success = ((delays != 999) & (ready < deadline) & (ready <= departure)
               & (departure <= min(deadline, ready + 12)))
    failure = ~success
    active = probabilities > 0
    deterministic = (float(failure[active][0])
                     if np.all(failure[active] == failure[active][0]) else None)
    return float(probabilities[failure].sum()), deterministic


def replay_row(row, graph, priors, true_parameters, *, capture_query=None, on_public_query=None):
    """Replay public feedback, optionally expose states after each poll."""
    if row['variant'] == 'legacy':
        return []
    router = HistoricalContinuationRouter(graph, priors, legacy=True)
    # The same initialization occurs in the actor's initial _assign; no route
    # or TopoCSA call is needed to reconstruct these sufficient statistics.
    for route in {connection.route for connection in graph.connections}:
        router._get_belief(route)
    history = router.trip_belief
    marginal = TripBelief(graph)
    learned, seen, unavailable = set(), set(), set()
    messages = {item['event_id']: DelayObservation(**item) for item in row['messages']}
    board_times = {
        int(re.search(r'connection=(\d+)', event['details'])[1]): event['time']
        for event in row['events'] if event['event_type'] == 'board'}
    polls = row['polls']
    poll_index, current_trip = 0, None
    departure = next(event['time'] for event in row['events'] if event['event_type'] == 'depart')
    last_arrival = departure
    deadline = departure + row['spec'][5]
    conditioned = row['variant'].startswith('conditioned')
    points = []

    def learn(cid, canceled, delay):
        router.observe_vehicle_event(cid, canceled, delay)
        connection = graph.connections[cid]
        key = connection.trip_id or f'connection:{cid}'
        if key not in learned:
            learned.add(key)
            if canceled:
                router.observe_cancel(connection.route, kind='true')
            else:
                router.observe_delay(connection.route, delay)

    for prediction in row['predictions']:
        query = prediction['private_model_query']
        for poll in polls[poll_index:query + 1]:
            batch = [messages[event_id] for event_id in poll['event_ids']]
            # Receipt order, including empty/full capped polls, is evidence.
            router.observe_public_messages(poll['stop'], poll['receipt_time'], batch, seen.copy())
            for message in batch:
                cid = int(message.event_id.split(':', 1)[0])
                learn(cid, message.is_canceled, message.actual_delay)
                seen.add(message.event_id)
                if (message.is_canceled or
                        (message.event_id.endswith(':departure') and message.actual_time < poll['receipt_time'])):
                    unavailable.add(cid)
        poll_index = query + 1
        if (polls[query]['stop'], polls[query]['receipt_time']) != (prediction['stop'], prediction['time']):
            raise AssertionError('Prediction and public receipt indices are misaligned')
        cid = prediction['connection']
        connection = graph.connections[cid]
        same_trip = bool(current_trip and connection.trip_id == current_trip)
        ready = max(prediction['time'], last_arrival + (
            0 if current_trip is None or same_trip else graph.stops[connection.dep_stop].min_transfer_time))
        known = history._trip_by_cid[cid] in history.known
        fitted_parameters = router._parameters()
        if query == capture_query or on_public_query is not None:
            snapshot = {
                'prediction': prediction, 'history': history.copy(),
                'fitted_parameters': fitted_parameters, 'current_trip': current_trip,
                'last_arrival': last_arrival, 'seen': seen.copy(),
                'learned': learned.copy(), 'unavailable': unavailable.copy(),
            }
            if query == capture_query:
                return snapshot
            on_public_query(snapshot)
        p_true, true_point = failure_probability(history, cid, true_parameters, ready, deadline)
        fitted_belief = history if conditioned else marginal
        p_fitted, fitted_point = failure_probability(fitted_belief, cid, fitted_parameters, ready, deadline)
        p_sample = prediction.get('first_attempt_failure_probability')
        if p_sample is not None:
            # A finite-bank estimate of a deterministic model fact must agree
            # exactly. A discrepancy means reconstruction/model timing is wrong.
            if fitted_point is not None and abs(p_sample - fitted_point) > 1e-12:
                raise AssertionError(f'Deterministic fitted probability mismatch: {row["index"]}/{row["variant"]}/query{query}')
            if known and conditioned and (true_point is None or
                                          abs(p_fitted - p_true) > 1e-12 or
                                          abs(p_sample - p_true) > 1e-12):
                raise AssertionError('Known public trip outcome was not reproduced')
            points.append({
                'index': row['index'], 'variant': row['variant'], 'spec': row['spec'],
                'private_model_query': query, 'receipt_time': polls[query]['receipt_time'],
                'time': prediction['time'], 'connection': cid, 'route': connection.route,
                'action_type': 'initial' if current_trip is None else 'same_trip' if same_trip else 'transfer',
                'trip_knowledge': 'known' if known else 'unknown',
                'ready_time': ready, 'deadline': deadline,
                'p_fail_sample': p_sample, 'p_fail_fitted': p_fitted, 'p_fail_true': p_true,
                'sampling_signed_error': p_sample - p_fitted,
                'model_signed_error': p_fitted - p_true,
                'total_signed_error': p_sample - p_true,
                'fitted_support_is_deterministic': fitted_point is not None,
                'selected_route_fitted_parameters': fitted_parameters[connection.route],
                'selected_route_generating_parameters': true_parameters[connection.route],
            })
        # Update only after the scored action. An event time earlier than a
        # later receipt must never make that event visible at this prediction.
        kind, end = prediction.get('actual_kind'), prediction.get('action_end')
        if kind == 'board':
            learn(cid, False, board_times[cid] - connection.dep_time)
            current_trip = connection.trip_id or f'connection:{cid}'
            last_arrival = end
            unavailable.add(cid)
        elif kind == 'cancel':
            learn(cid, True, 0)
            unavailable.add(cid)
        elif kind in ('missed', 'wait_late'):
            history.observe_attempt(cid, 'missed' if kind == 'missed' else 'timeout', end)
            if kind == 'missed':
                unavailable.add(cid)
        elif kind != 'deadline':
            raise AssertionError(f'Missing or unsupported public attempt feedback: {kind}')
    if capture_query is not None:
        raise ValueError(f'Query {capture_query} was not present in the public log')
    return points


def metrics(points):
    if not points:
        return {'n_selected_states': 0}
    total = np.array([p['total_signed_error'] for p in points])
    sampling = np.array([p['sampling_signed_error'] for p in points])
    model = np.array([p['model_signed_error'] for p in points])
    return {
        'n_selected_states': len(points),
        'n_journeys': len({p['index'] for p in points}),
        'total_mae': float(np.abs(total).mean()),
        'model_mae': float(np.abs(model).mean()),
        'sampling_mae': float(np.abs(sampling).mean()),
        'total_signed_bias': float(total.mean()),
        'model_signed_bias': float(model.mean()),
        'sampling_signed_bias': float(sampling.mean()),
        'max_signed_decomposition_residual': float(np.abs(total - sampling - model).max()),
    }


def analyze_rows(rows, graph, days):
    routes = {connection.route for connection in graph.connections}
    priors, truths = {}, {}
    points = []
    for row in rows:
        if row['variant'] == 'legacy':
            continue
        date, stress = row['spec'][:2]
        if date not in priors:
            priors[date] = historical_priors(days, date, routes)
        if (date, stress) not in truths:
            truths[date, stress] = generating_parameters(days[date], routes, stress)
        points.extend(replay_row(row, graph, priors[date], truths[date, stress]))
    summaries = {}
    for variant in sorted({point['variant'] for point in points}):
        selected = [point for point in points if point['variant'] == variant]
        summaries[variant] = {
            'all': metrics(selected),
            'by_action_type': {kind: metrics([p for p in selected if p['action_type'] == kind])
                               for kind in ('initial', 'same_trip', 'transfer')},
            'by_trip_knowledge': {kind: metrics([p for p in selected if p['trip_knowledge'] == kind])
                                  for kind in ('known', 'unknown')},
            'by_action_and_knowledge': {
                f'{action}/{knowledge}': metrics([p for p in selected
                                                  if p['action_type'] == action and p['trip_knowledge'] == knowledge])
                for action in ('initial', 'same_trip', 'transfer') for knowledge in ('known', 'unknown')},
        }
    return {'n_input_journeys': len(rows), 'legacy_journeys_skipped': sum(r['variant'] == 'legacy' for r in rows),
            'summary': summaries, 'points': points}


def main():
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as handle:
        graph = pickle.load(handle)
    with (ROOT / 'data/per_day_distributions.pkl').open('rb') as handle:
        days = pickle.load(handle)
    output = {
        'scope': 'Descriptive probability errors at each method\'s own selected states under a known generated model.',
        'truth': 'Analytic independent-trip generating-law posterior conditioned on the actual public receipt/attempt history; no hidden realized-world access.',
        'fitted': 'The historical router parameters with the original historical_priors and once-per-trip public updates; history posterior for conditioned variants, empty trip belief for marginal variants.',
        'sampling': 'Recorded selected-action eight-world first-attempt failure estimates.',
        'decomposition': 'sample-true = (sample-fitted) + (fitted-true); the corresponding absolute errors do not add.',
        'limits': 'These are different policy-visited states. No common-state superiority, real-data probability calibration, independent-state inference, or predictive guarantee is claimed. Selection uses the same finite bank, so selected-state sampling error is not an unbiased independent validation estimate.',
    }
    for name, filename in (('primary', 'matched_results.json'), ('independent_generated', 'independent_results.json')):
        source = HERE / filename
        rows = json.loads(source.read_text())['methods']
        output[name] = {'source': str(source.relative_to(ROOT)), **analyze_rows(rows, graph, days)}
    destination = HERE / 'probability_results.json'
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    for panel in ('primary', 'independent_generated'):
        print(panel, {variant: {
            'known_n': data['by_trip_knowledge']['known']['n_selected_states'],
            'unknown_n': data['by_trip_knowledge']['unknown']['n_selected_states'],
            'known_total_mae': data['by_trip_knowledge']['known'].get('total_mae'),
            'unknown_total_mae': data['by_trip_knowledge']['unknown'].get('total_mae'),
            'unknown_model_mae': data['by_trip_knowledge']['unknown'].get('model_mae'),
            'unknown_sampling_mae': data['by_trip_knowledge']['unknown'].get('sampling_mae')}
            for variant, data in output[panel]['summary'].items()})


if __name__ == '__main__':
    main()
