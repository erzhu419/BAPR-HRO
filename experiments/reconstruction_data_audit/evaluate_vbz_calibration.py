"""Past-day calibration on the separate measured municipal tram/bus sample."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def fit(data, basis):
    samples = defaultdict(list)
    for day in data['train_dates']:
        for leg in data['dates'][day]['legs']:
            for kind in ('departure', 'arrival'):
                delay = leg[f'actual_{kind}_seconds'] - leg[f'{basis}scheduled_{kind}_seconds']
                samples[leg['route_id'], kind].append(delay)
                samples[leg['mode'], kind].append(delay)
    return {key: np.asarray(values) for key, values in samples.items()}


def score(legs, fitted, basis, mode):
    rows = []
    fallback = 0
    for leg in legs:
        if leg['mode'] != mode:
            continue
        for kind in ('departure', 'arrival'):
            key = leg['route_id'], kind
            if key not in fitted:
                key = mode, kind
                fallback += 1
            sample = fitted[key]
            lo, hi = np.quantile(sample, [.05, .95])
            actual = leg[f'actual_{kind}_seconds'] - leg[f'{basis}scheduled_{kind}_seconds']
            rows.append((actual, float(sample.mean()) - actual, actual < lo, actual > hi))
    actual, error, below, above = np.asarray(rows).T
    return {'n': len(rows), 'n_routes': len({leg['route_id'] for leg in legs if leg['mode'] == mode}),
            'route_fallback_endpoints': fallback, 'schedule_mae_seconds': float(np.mean(np.abs(actual))),
            'prior_mae_seconds': float(np.mean(np.abs(error))), 'prior_signed_error_seconds': float(np.mean(error)),
            'prior_signed_p10_p90_seconds': np.quantile(error, [.1, .9]).tolist(),
            'central90_covered': len(rows) - int(below.sum()) - int(above.sum()),
            'below': int(below.sum()), 'above': int(above.sum())}


def replay_runs(day):
    legs = {leg['id']: leg for leg in day['legs']}
    outcomes = []
    for query in day['fixed_measured_itineraries']:
        itinerary = [legs[cid] for cid in query['connections']]
        boarded = query['ready_seconds'] <= itinerary[0]['actual_departure_seconds']
        final = itinerary[-1]['actual_arrival_seconds']
        connected = all(first['actual_arrival_seconds'] <= second['actual_departure_seconds']
                        for first, second in zip(itinerary, itinerary[1:]))
        outcomes.append({'connections': query['connections'], 'origin': query['origin'], 'destination': query['destination'],
                         'ready_seconds': query['ready_seconds'], 'boarded': boarded, 'same_vehicle_continuity': connected,
                         'arrival_seconds': final, 'deadline_seconds': query['deadline_seconds'],
                         'arrived': boarded and connected and final <= query['deadline_seconds'],
                         'elapsed_seconds': final - query['ready_seconds'] if boarded else None})
    return outcomes


def main():
    with gzip.open(HERE / 'vbz_measured_sample.json.gz', 'rt', encoding='utf-8') as handle:
        data = json.load(handle)
    results = {}
    for day in data['holdout_dates']:
        legs = data['dates'][day]['legs']
        results[day] = {'coverage': data['dates'][day]['coverage'],
                        'endpoint_calibration': {basis: {mode: score(legs, fit(data, basis), basis, mode)
                                                        for mode in ('tram', 'bus')}
                                                  for basis in ('', 'provider_')},
                        'fixed_measured_run_replay': replay_runs(data['dates'][day])}
    output = {'scope': 'Earlier-day route/event-kind empirical fits within a partial municipal sample. '
              'GTFS and exact provider scheduled-time bases are scored separately. '
              'Fixed measured multileg rides evaluate execution, with incomplete competing actions.',
              'train_dates': data['train_dates'], 'dates': results}
    (HERE / 'vbz_calibration.json').write_text(json.dumps(output, indent=2) + '\n')
    for day, row in results.items():
        print(day, row['endpoint_calibration'], 'fixed_rides', len(row['fixed_measured_run_replay']))


if __name__ == '__main__':
    main()
