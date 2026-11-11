"""Route, event-kind, clock and public-signal diagnosis of frozen predictions."""
from collections import defaultdict
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_run import METHODS, data_inputs, load, summarize_endpoints, write
from src.calibration_model import PublicForecasts


def group_summary(rows, offsets):
    output = {'n': len(rows)}
    output['methods'] = {method: summarize_endpoints(rows, method, offsets[method]) for method in METHODS}
    return output


def diagnose(data, results, frames):
    output = {}
    dates = results.get('frozen_protocol', results.get('protocol'))['confirmation_dates' if 'confirmation' in results else 'heldout_dates']
    offsets = results['offset_seconds']
    for day in dates:
        rows = [row for row in results['endpoint_rows'][day] if row['actual'] is not None]
        groups = {key: defaultdict(list) for key in ('kind', 'hour', 'route', 'own_forecast')}
        signals = []
        public = PublicForecasts(data['dates'][day]['connections'], frames[day]['frames'])
        connections = public.connections
        for row in rows:
            groups['kind'][row['kind']].append(row)
            groups['hour'][str(row['scheduled'] // 3600)].append(row)
            groups['route'][row['route_id']].append(row)
            groups['own_forecast'][str(row['features']['has_own_forecast'])].append(row)
            own = public.at(row['prediction_clock']).values.get((row['source_connection'], row['kind']))
            c = connections[row['source_connection']]
            signals.append({'residual': row['actual'] - row['methods']['evolving_prior']['mean'],
                            'peer_shift': row['features']['peer_shift_seconds'],
                            'peer_trips': row['features']['peer_trips'],
                            'own_error': row['actual'] - own[0] if own else None,
                            'actual_delay': row['actual'] - c[f"scheduled_{row['kind']}_seconds"]})
        grouped = {name: {key: group_summary(values, offsets) for key, values in collection.items()}
                   for name, collection in groups.items()}
        residual = np.array([row['residual'] for row in signals])
        shift = np.array([row['peer_shift'] for row in signals])
        own_errors = np.array([row['own_error'] for row in signals if row['own_error'] is not None])
        output[day] = {'groups': grouped, 'signals': {
            'n_endpoints': len(rows), 'n_own_forecasts': len(own_errors),
            'actual_minus_prior_mean_seconds': float(residual.mean()),
            'median_actual_minus_prior_seconds': float(np.median(residual)),
            'p90_actual_minus_prior_seconds': float(np.quantile(residual, .9)),
            'mean_public_peer_shift_seconds': float(shift.mean()),
            'median_peer_trips': float(np.median([row['peer_trips'] for row in signals])),
            'actual_minus_issued_forecast_mean_seconds': float(own_errors.mean()) if len(own_errors) else None,
            'issued_forecast_mae_seconds': float(np.abs(own_errors).mean()) if len(own_errors) else None,
            'correlation_peer_shift_and_prior_residual': float(np.corrcoef(shift, residual)[0, 1])
                    if shift.std() > 0 and residual.std() > 0 else None}}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    data = data_inputs()
    name = 'calibration_results.json.gz'
    frame_name = 'calibration_public_frames.json.gz'
    output_name = 'calibration_diagnosis.json'
    if args.confirmation:
        data['dates'].update(load(HERE / 'calibration_confirmation_actuals.json.gz')['dates'])
        name = 'calibration_confirmation_results.json.gz'
        frame_name = 'calibration_confirmation_public_frames.json.gz'
        output_name = 'calibration_confirmation_diagnosis.json'
    result = diagnose(data, load(HERE / name), load(HERE / frame_name)['dates'])
    write(HERE / output_name, result)
    for day, row in result.items():
        print(day, row['signals'])
        for axis in ('kind', 'hour'):
            print(axis, {key: {'n': group['n'], 'prior': round(group['methods']['evolving_prior']['coverage'], 4),
                              'mixture': round(group['methods']['public_peer_mixture']['coverage'], 4),
                              'actual_minus_prior': round(-group['methods']['evolving_prior']['signed_mean_error_seconds'], 1)}
                         for key, group in row['groups'][axis].items()})


if __name__ == '__main__':
    main()
