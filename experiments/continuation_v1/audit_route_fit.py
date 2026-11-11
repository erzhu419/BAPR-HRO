"""Compare four route fits at the same archived public states and actions."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.continuation_v1.analyze_probability import (
    failure_probability, generating_parameters, replay_row,
)
from experiments.continuation_v1.historical_router import HistoricalContinuationRouter
from experiments.reconstruction_v2.run_matched_panel import historical_priors

FITTERS = ('online_mean_cancel', 'frozen_prior', 'mean_only', 'cancel_only')


def collect_points(row, graph, priors, frozen, truth):
    departure = next(event['time'] for event in row['events'] if event['event_type'] == 'depart')
    deadline = departure + row['spec'][5]
    date, stress, _, _, seed, _, _ = row['spec']
    world = (date, stress, seed)
    points = []

    def collect(snapshot):
        prediction = snapshot['prediction']
        history = snapshot['history']
        fitted = snapshot['fitted_parameters']
        parameters = {
            'online_mean_cancel': fitted,
            'frozen_prior': frozen,
            'mean_only': {route: params | {'cancel_rate': frozen[route]['cancel_rate']}
                          for route, params in fitted.items()},
            'cancel_only': {route: params | {'cancel_rate': fitted[route]['cancel_rate']}
                            for route, params in frozen.items()},
        }
        for raw_id in prediction['candidates']:
            cid = int(raw_id)
            if history._trip_by_cid[cid] in history.known:
                continue
            connection = graph.connections[cid]
            current_trip = snapshot['current_trip']
            transfer = 0 if current_trip is None or connection.trip_id == current_trip else graph.stops[connection.dep_stop].min_transfer_time
            ready = max(prediction['time'], snapshot['last_arrival'] + transfer)
            true_probability, point_value = failure_probability(history, cid, truth, ready, deadline)
            estimates = {name: failure_probability(history, cid, params, ready, deadline)[0]
                         for name, params in parameters.items()}
            points.append({
                'world': world, 'state': (row['index'], prediction['private_model_query']),
                'nondegenerate': point_value is None,
                'error': {name: p - true_probability for name, p in estimates.items()},
            })

    replay_row(row, graph, priors, truth, on_public_query=collect)
    return points


def summarize(points):
    if not points:
        return {'n_candidate_actions': 0}
    by_world = defaultdict(list)
    for point in points:
        by_world[point['world']].append(point)
    per_world = []
    for world, group in sorted(by_world.items()):
        error = {name: np.array([point['error'][name] for point in group]) for name in FITTERS}
        per_world.append({
            'world': world, 'n_candidate_actions': len(group),
            'mae': {name: float(np.abs(values).mean()) for name, values in error.items()},
            'signed_bias': {name: float(values.mean()) for name, values in error.items()},
        })
    fitters = {}
    for name in FITTERS:
        candidate_errors = np.array([point['error'][name] for point in points])
        deltas = [unit['mae'][name] - unit['mae']['online_mean_cancel'] for unit in per_world]
        fitters[name] = {
            'candidate_mae': float(np.abs(candidate_errors).mean()),
            'equal_world_mae': float(np.mean([unit['mae'][name] for unit in per_world])),
            'equal_world_signed_bias': float(np.mean([unit['signed_bias'][name] for unit in per_world])),
            'equal_world_mae_difference_vs_online': float(np.mean(deltas)),
            'worlds_with_smaller_mae_than_online': sum(delta < -1e-12 for delta in deltas),
            'worlds_with_larger_mae_than_online': sum(delta > 1e-12 for delta in deltas),
        }
    return {'n_candidate_actions': len(points), 'n_states': len({p['state'] for p in points}),
            'n_worlds': len(per_world), 'fitters': fitters, 'worlds': per_world}


def groups(points):
    subsets = {'all': points}
    for stress in (False, True):
        label = 'stress' if stress else 'no_cancel'
        subsets[label] = [p for p in points if p['world'][1] == stress]
    for date in sorted({p['world'][0] for p in points}):
        subsets[date] = [p for p in points if p['world'][0] == date]
        for label in ('stress', 'no_cancel'):
            subsets[f'{date}/{label}'] = [p for p in subsets[label] if p['world'][0] == date]
    return {key: {
        'all_unknown_trip_actions': summarize(group),
        'nondegenerate_unknown_trip_actions': summarize([p for p in group if p['nondegenerate']]),
    } for key, group in subsets.items()}


def panel(directory, graph, days):
    routes = {c.route for c in graph.connections}
    priors, frozen, truths = {}, {}, {}
    source_paths = sorted(directory.glob('case_*_conditioned_rescue.json'))
    points = []
    for path in source_paths:
        row = json.loads(path.read_text())
        date, stress = row['spec'][:2]
        if date not in priors:
            priors[date] = historical_priors(days, date, routes)
            frozen[date] = HistoricalContinuationRouter(graph, priors[date], legacy=True)._parameters()
        if (date, stress) not in truths:
            truths[date, stress] = generating_parameters(days[date], routes, stress)
        points.extend(collect_points(row, graph, priors[date], frozen[date], truths[date, stress]))
    return {'source': str(directory.relative_to(ROOT)), 'n_journeys': len(source_paths),
            'strictly_earlier_prior_dates': {date: sorted(d for d in days if d < date) for date in sorted(priors)},
            'groups': groups(points)}


def main():
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as handle:
        graph = pickle.load(handle)
    with (ROOT / 'data/per_day_distributions.pkl').open('rb') as handle:
        days = pickle.load(handle)
    result = {
        'scope': 'Every recorded offered unknown-trip action on the same archived conditioned_rescue public states, evaluated under four fixed fitters.',
        'training': 'Original route priors pool historical dates strictly earlier than each case. Online updates use only that journey’s received public data. The independent generated 2023-10-15 seed-913 panel is reported separately.',
        'frozen_prior': 'Same initialized route mean/std and cancellation pseudo-count as the online fitter before its first observation.',
        'truth': 'Analytic first-attempt failure under the generating trip law given the same public history. Nondegenerate actions have both success and failure in the generating posterior support.',
        'pairing': 'Each fitter sees identical candidate actions and public histories. Equal-world summaries first average candidate errors within (date, stress, seed), then weight each world equally.',
        'decision_scope': 'These are probability errors on the archived state bank. The route fits do not generate new policy trajectories in this audit.',
        'primary': panel(HERE / 'matched_results_methods', graph, days),
        'independent_generated': panel(HERE / 'independent_results_methods', graph, days),
    }
    path = HERE / 'route_fit_results.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for name in ('primary', 'independent_generated'):
        for group in ('all', 'no_cancel', 'stress'):
            summary = result[name]['groups'][group]['nondegenerate_unknown_trip_actions']
            print(name, group, summary['n_candidate_actions'], {
                fitter: round(item['equal_world_mae'], 5)
                for fitter, item in summary.get('fitters', {}).items()})
    print(path)


if __name__ == '__main__':
    main()
