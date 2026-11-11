"""Past-only endpoint/service calibration of the ordered trip model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_dated_replay import DATA_PATH, dated_graph, fit_training, read_inputs
from src.evolving_trip_belief import EvolvingTripBelief


def interval_and_mean(times, weights):
    cdf = np.cumsum(weights)
    bounds = [int(times[min(np.searchsorted(cdf, p), len(times) - 1)]) for p in (.05, .95)]
    return bounds, float(times @ weights)


def aggregate(rows):
    operating = [row for row in rows if row['actual'] is not None]
    return {
        'measured_endpoints': len(operating),
        'central_90_covered': sum(r['lower'] <= r['actual'] <= r['upper'] for r in operating),
        'central_90_mean_width_seconds': float(np.mean([r['upper'] - r['lower'] for r in operating])),
        'predicted_mean_mae_seconds': float(np.mean([abs(r['mean'] - r['actual']) for r in operating])),
        'schedule_mae_seconds': float(np.mean([abs(r['scheduled'] - r['actual']) for r in operating])),
        'known_service_endpoints': len(rows),
        'service_cancellation_brier': float(np.mean([
            (r['cancel_probability'] - (r['actual'] is None)) ** 2 for r in rows])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DATA_PATH)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('evolving_calibration.json'))
    args = parser.parse_args()
    data, _ = read_inputs(data_path=args.data)
    fit = fit_training(data)
    dates, measured = {}, []
    for day in data['holdout_dates']:
        graph, _, _, _, source, _ = dated_graph(data, day)
        priors = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                  for c in graph.connections}
        model = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
        rows = []
        for c in graph.connections:
            for kind in ('departure', 'arrival'):
                actual = source[c.id][kind]
                if actual['status'] not in ('REAL', 'CANCELED'):
                    continue
                clocks, weights, cancel = model.event_distribution(c.id, kind, priors)
                if not len(clocks):
                    continue
                (lower, upper), mean = interval_and_mean(clocks, weights)
                rows.append({'source_connection': source[c.id]['id'], 'kind': kind,
                             'actual': actual['time_seconds'] if actual['status'] == 'REAL' else None,
                             'scheduled': c.dep_time if kind == 'departure' else c.arr_time,
                             'lower': lower, 'upper': upper, 'mean': mean,
                             'cancel_probability': cancel})
        dates[day] = aggregate(rows)
        for row in rows:
            row['date'] = day
        measured.extend(rows)
    output = {'data': str(args.data), 'scope': 'One pre-event predictive law per held-out dated graph. No held-out event or forecast is conditioned on.',
              'training': fit, 'by_date': dates, 'pooled': aggregate(measured),
              'dependence': 'Endpoints of the same trip and date are correlated. Counts are endpoint diagnostics, without an independent-endpoint confidence claim.',
              'interval': 'Equal-tail conditional-operating 90% intervals of the finite-grid law. Cancellation is scored separately.',
              'rows': measured}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'by_date': dates, 'pooled': output['pooled']}, indent=2))


if __name__ == '__main__':
    main()
