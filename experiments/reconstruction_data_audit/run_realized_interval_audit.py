"""Check empirical past-only route forecast intervals on measured rail endpoints."""
from collections import defaultdict
import json
from pathlib import Path
import pickle
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def summarize(rows):
    errors = [row['mean_prediction_error_min'] for row in rows]
    covered = sum(row['interval_position'] == 'inside' for row in rows)
    return {'n': len(rows), 'inside': covered,
            'below': sum(row['interval_position'] == 'below' for row in rows),
            'above': sum(row['interval_position'] == 'above' for row in rows),
            'coverage': covered / len(rows), 'signed_mean_error_min': mean(errors),
            'signed_error_p10_p90_min': np.quantile(errors, (.1, .9)).tolist(),
            'absolute_error_p90_min': float(np.quantile(np.abs(errors), .9))}


def main():
    with (ROOT / 'data/per_day_distributions_corrected.pkl').open('rb') as handle:
        predictions = pickle.load(handle)
    measured = json.loads((HERE / 'realized_subset.json').read_text())
    dates = {}
    for date, subset in measured['dates'].items():
        training_dates = subset['past_training_dates']
        by_route = defaultdict(list)
        for day in training_dates:
            for route, stats in predictions[day].items():
                if stats['n_obs']:
                    by_route[route].append(stats['delays_min'])
        fitted = {}
        rows = []
        for cid, leg in subset['legs'].items():
            for kind, event in leg.items():
                route = event['route_id']
                if route not in by_route:
                    continue
                if route not in fitted:
                    sample = np.concatenate(by_route[route])
                    lower, upper = np.quantile(sample, (.05, .95))
                    fitted[route] = {'n_training_forecasts': len(sample), 'mean_min': float(sample.mean()),
                                     'central90_min': [float(lower), float(upper)]}
                actual = event['delay_seconds'] / 60
                lower, upper = fitted[route]['central90_min']
                position = 'below' if actual < lower else 'above' if actual > upper else 'inside'
                rows.append({'connection_id': int(cid), 'kind': kind, 'route_id': route,
                             'actual_delay_min': actual, 'interval_position': position,
                             'mean_prediction_error_min': fitted[route]['mean_min'] - actual})
        dates[date] = {'train_dates': training_dates, 'measured_complete_leg_endpoints': 2 * len(subset['legs']),
                       'overall': summarize(rows),
                       'by_event_kind': {kind: summarize([row for row in rows if row['kind'] == kind])
                                         for kind in ('departure', 'arrival')},
                       'route_forecast_intervals': fitted, 'rows': rows}
    output = {'scope': 'Central 90% empirical intervals of earlier unique route forecast delays, '
              'checked against REAL endpoints on complete measured SBB legs. Small clustered sample; '
              'signed error is route forecast mean minus realized delay.', 'dates': dates}
    (HERE / 'realized_interval_audit.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({date: {key: values[key] for key in ('overall', 'by_event_kind')}
                      for date, values in dates.items()}))


if __name__ == '__main__':
    main()
