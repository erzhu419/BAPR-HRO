"""Evaluate fixed earlier-date delay and boarding models on measured holdouts."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import pickle

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KINDS = ('departure', 'arrival')


def fit_actuals(data):
    delays, status = defaultdict(list), defaultdict(Counter)
    for day in data['train_dates']:
        for c in data['dates'][day]['connections']:
            route = c['route_id']
            for kind in KINDS:
                observation = c[kind]
                if observation['status'] in ('REAL', 'CANCELED'):
                    status[route, kind][observation['status']] += 1
                if observation['status'] == 'REAL':
                    delay = observation['time_seconds'] - c[f'scheduled_{kind}_seconds']
                    delays[route, kind].append(delay)
                    delays['all', kind].append(delay)
    return {key: np.asarray(values) for key, values in delays.items()}, status


def fit_forecasts(data, old):
    samples = defaultdict(list)
    for day in data['train_dates']:
        for route, row in old[day].items():
            if row['n_obs']:
                samples[route].append(60 * row['delays_min'])
    return {route: np.concatenate(values) for route, values in samples.items()}


def summarize_points(rows):
    summary = {'n': len(rows)}
    for name in ('schedule', 'measured_prior', 'forecast_prior'):
        values = [(actual, stats[name]) for actual, stats in rows if name in stats]
        if not values:
            continue
        errors = [model['mean'] - actual for actual, model in values]
        out = {'n': len(values), 'mae_seconds': float(np.mean(np.abs(errors))),
               'signed_error_seconds': float(np.mean(errors)),
               'signed_p10_p90_seconds': np.quantile(errors, [.1, .9]).tolist()}
        if name != 'schedule':
            below = sum(actual < model['lower'] for actual, model in values)
            above = sum(actual > model['upper'] for actual, model in values)
            out.update(central90_covered=len(values) - below - above, below=below, above=above)
        summary[name] = out
    return summary


def point_audit(connections, measured, forecasts):
    def point_stats(sample):
        lo, hi = np.quantile(sample, [.05, .95])
        return {'mean': float(sample.mean()), 'lower': float(lo), 'upper': float(hi)}

    fitted = {key: point_stats(sample) for key, sample in measured.items()}
    predicted = {route: point_stats(sample) for route, sample in forecasts.items()}
    rows = {kind: [] for kind in KINDS}
    fallbacks = 0
    for c in connections:
        route = c['route_id']
        for kind in KINDS:
            if c[kind]['status'] != 'REAL':
                continue
            actual = c[kind]['time_seconds'] - c[f'scheduled_{kind}_seconds']
            fallbacks += (route, kind) not in fitted
            stats = {'schedule': {'mean': 0}, 'measured_prior': fitted.get((route, kind), fitted['all', kind])}
            if route in predicted:
                stats['forecast_prior'] = predicted[route]
            rows[kind].append((actual, stats))
    combined = [row for kind in KINDS for row in rows[kind]]
    return {'all_real_endpoints': summarize_points(combined),
            'common_forecast_supported_endpoints': summarize_points([row for row in combined if 'forecast_prior' in row[1]]),
            'by_event_kind': {kind: summarize_points(row) for kind, row in rows.items()},
            'measured_prior_pool_fallbacks': fallbacks}


def transfer_audit(connections, stops, measured, status):
    departures = defaultdict(list)
    for c in connections:
        if c['pickup_allowed']:
            departures[c['dep_stop']].append(c)
    rows = []
    omitted = 0
    for incoming in connections:
        if not incoming['dropoff_allowed'] or incoming['outcome'] != 'MEASURED':
            continue
        station = incoming['arr_stop']
        buffer = stops[station]['transfer_seconds']
        ready = incoming['arrival']['time_seconds'] + buffer
        for outgoing in departures[station]:
            if incoming['trip_id'] == outgoing['trip_id']:
                continue
            planned = outgoing['scheduled_departure_seconds']
            slack = planned - incoming['scheduled_arrival_seconds'] - buffer
            if not 0 <= slack <= 8 * 60:
                continue
            observed = outgoing['departure']
            if observed['status'] not in ('REAL', 'CANCELED'):
                omitted += 1
                continue
            caught = observed['status'] == 'REAL' and ready <= observed['time_seconds'] <= ready + 12 * 60
            key = outgoing['route_id'], 'departure'
            delays = measured.get(key, measured['all', 'departure'])
            counts = status.get(key, Counter())
            p_operating = counts['REAL'] / counts.total() if counts.total() else 1.0
            p_board = p_operating * np.mean((ready <= planned + delays) & (planned + delays <= ready + 12 * 60))
            nominal = float(ready <= planned <= ready + 12 * 60)
            rows.append((float(caught), float(p_board), nominal, observed['status'] == 'CANCELED'))
    truth, p, nominal, canceled = np.asarray(rows).T
    return {'n': len(rows), 'excluded_unknown_departures': omitted, 'observed_boardable': int(truth.sum()),
            'explicit_canceled_departure_pairs': int(canceled.sum()), 'mean_model_probability': float(p.mean()),
            'measured_prior_brier': float(np.mean((p - truth) ** 2)),
            'schedule_only_brier': float(np.mean((nominal - truth) ** 2))}


def issued_forecast_audit(connections, frames, measured):
    rows = {kind: [] for kind in KINDS}
    declared_canceled = 0
    before_actual = 0
    for connection in connections:
        route = connection['route_id']
        for kind in KINDS:
            observed = connection[kind]
            if observed['status'] != 'REAL':
                continue
            cutoff = connection[f'scheduled_{kind}_seconds'] - 5 * 60
            latest, trip_status = None, None
            for frame in frames:
                if frame['receipt_seconds'] > cutoff:
                    break
                for tid, status in frame['trip_statuses']:
                    if tid == connection['trip_id']:
                        trip_status = status
                for cid, event_kind, status, time in frame['endpoint_updates']:
                    if cid == connection['id'] and event_kind == kind:
                        latest = frame['receipt_seconds'], status, time
            if trip_status == 'CANCELED':
                declared_canceled += 1
                continue
            if latest is None or latest[1] != 'FORECAST':
                continue
            receipt, _, prediction = latest
            before_actual += receipt < observed['time_seconds']
            planned = connection[f'scheduled_{kind}_seconds']
            prior = measured.get((route, kind), measured['all', kind])
            actual_delay = observed['time_seconds'] - planned
            rows[kind].append((actual_delay, {'schedule': {'mean': 0},
                                             'issued_forecast': {'mean': prediction - planned},
                                             'measured_prior': {'mean': float(prior.mean())}}))
    results = {}
    for kind, values in rows.items():
        results[kind] = {'n': len(values), 'mae_seconds': {
            model: float(np.mean([abs(stats[model]['mean'] - actual) for actual, stats in values])) if values else None
            for model in ('schedule', 'issued_forecast', 'measured_prior')}}
    return {'cutoff_before_scheduled_seconds': 300, 'by_event_kind': results,
            'receipts_before_realized_event': before_actual,
            'event_endpoints_excluded_by_cancel_notice': declared_canceled}


def main():
    with gzip.open(HERE / 'dated_replay.json.gz', 'rt', encoding='utf-8') as handle:
        data = json.load(handle)
    with (ROOT / 'data/per_day_distributions_corrected.pkl').open('rb') as handle:
        forecast_archive = pickle.load(handle)
    with gzip.open(HERE / 'dated_public_forecasts.json.gz', 'rt', encoding='utf-8') as handle:
        public = json.load(handle)
    measured, status = fit_actuals(data)
    forecasts = fit_forecasts(data, forecast_archive)
    results = {}
    for day in data['holdout_dates']:
        day_data = data['dates'][day]
        connections = day_data['connections']
        results[day] = {'coverage': day_data['coverage'],
                        'point_and_interval': point_audit(connections, measured, forecasts),
                        'transfer_probability': transfer_audit(connections, data['stops'], measured, status),
                        'issued_precutoff_forecasts': issued_forecast_audit(
                            connections, public['dates'][day]['frames'], measured)}
    output = {'scope': 'Fixed earlier-date empirical fits scored on later REAL rail endpoints. '
              'Observed-status departure ratios and measured departure shifts form a route-level boarding model. '
              'Transfer pairs have scheduled slack 0–8 minutes and use the incoming realized arrival as public traveler feedback. '
              'Pairs and endpoint errors cluster within trips and dates.',
              'train_dates': data['train_dates'], 'dates': results}
    (HERE / 'dated_calibration.json').write_text(json.dumps(output, indent=2) + '\n')
    for day, values in results.items():
        print(day, values['point_and_interval']['common_forecast_supported_endpoints'], values['transfer_probability'])


if __name__ == '__main__':
    main()
