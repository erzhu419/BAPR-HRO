"""Evaluate predeclared new dates without refitting any frozen component."""
from pathlib import Path
import gzip
import json
import pickle
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_run import (METHODS, data_inputs, evaluate_day, evaluate_transfers, load,
                             probability_summary, summarize_endpoints, write)
from calibration_probability import predict_isotonic


def recalibrated_probabilities(rows, mappings):
    result = {}
    truth = np.asarray([row['caught'] for row in rows], dtype=float)
    for method, mapping in mappings.items():
        p = np.asarray([predict_isotonic(mapping, row['probabilities'][method]) for row in rows])
        bins = []
        for lo, hi in zip((0., .2, .4, .6, .8), (.2, .4, .6, .8, 1.000001)):
            mask = (p >= lo) & (p < hi)
            if mask.any():
                bins.append({'n': int(mask.sum()), 'predicted': float(p[mask].mean()),
                             'observed': float(truth[mask].mean())})
        result[method + '_isotonic'] = {'n': len(rows), 'brier': float(np.mean((p - truth) ** 2)),
                                       'signed_probability_error': float(np.mean(p - truth)), 'reliability_bins': bins}
    return result


def main():
    frozen = json.loads((HERE / 'calibration_frozen_fit.json').read_text())
    offsets = json.loads((HERE / 'calibration_frozen_offsets.json').read_text())['offset_seconds']
    mappings = json.loads((HERE / 'calibration_frozen_probability.json').read_text())['models']
    protocol, fit = frozen['protocol'], frozen['evolving_fit']
    with (HERE / 'calibration_frozen_model.pkl').open('rb') as handle:
        model = pickle.load(handle)
    # No fit function is called below. Confirmation actuals first enter here.
    data = data_inputs()
    actuals = load(HERE / 'calibration_confirmation_actuals.json.gz')
    data['dates'].update(actuals['dates'])
    frames = load(HERE / 'calibration_confirmation_public_frames.json.gz')['dates']
    endpoints, transfers, summary = {}, {}, {}
    for day in protocol['confirmation_dates']:
        rows = evaluate_day(data, day, fit, frames[day]['frames'], model, protocol)
        endpoints[day] = rows
        values = {method: {'raw': summarize_endpoints(rows, method),
                           'date_calibrated': summarize_endpoints(rows, method, offsets[method])}
                  for method in METHODS}
        pairs, unknown = evaluate_transfers(data, day, fit, frames[day]['frames'], model, protocol)
        probabilities = probability_summary(pairs)
        probabilities.update(recalibrated_probabilities(pairs, mappings))
        transfers[day] = {'excluded_unknown_departures': unknown, 'methods': probabilities, 'rows': pairs}
        summary[day] = {'source_coverage': actuals['dates'][day]['coverage'], 'endpoints': values,
                        'transfer_probability': probabilities, 'excluded_unknown_departures': unknown}
        print('CONFIRMATION', day, len(rows), len(pairs), flush=True)
        print('COVERAGE', {name: round(value['date_calibrated']['coverage'], 4) for name, value in values.items()}, flush=True)
        print('BRIER', {name: round(value['brier'], 5) for name, value in probabilities.items()}, flush=True)
    output = {'scope': 'Independent dates predeclared before any of their actual outcomes were inspected. '
              'Model parameters, mixture weights, interval offsets and probability maps were frozen first. '
              'Three confirmation dates are separate from the previously examined development dates.',
              'frozen_protocol': protocol, 'offset_seconds': offsets, 'confirmation': summary,
              'endpoint_rows': endpoints, 'transfer_probability': transfers}
    with gzip.open(HERE / 'calibration_confirmation_results.json.gz', 'wt') as handle:
        json.dump(output, handle, separators=(',', ':'), allow_nan=False)
    write(HERE / 'calibration_confirmation_summary.json', {key: output[key] for key in ('scope', 'confirmation')})


if __name__ == '__main__':
    main()
