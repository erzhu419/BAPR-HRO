"""Keep the historical three-training-date Evolving fit as a fixed reference."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_run import (ROOT, ForecastEvaluator, data_inputs, law_summary, load,
                             summarize_endpoints, write)
from src.calibration_model import PublicForecasts


def main():
    historical = ROOT / 'experiments/reconstruction_v4/evolving_calibration.json'
    fit = json.loads(historical.read_text())['training']
    data = data_inputs()
    data['dates'].update(load(HERE / 'calibration_confirmation_actuals.json.gz')['dates'])
    frames = load(HERE / 'calibration_confirmation_public_frames.json.gz')['dates']
    confirmation = load(HERE / 'calibration_confirmation_results.json.gz')
    results = {}
    for day, rows in confirmation['endpoint_rows'].items():
        public = PublicForecasts(data['dates'][day]['connections'], frames[day]['frames'])
        evaluator = ForecastEvaluator(data, day, fit, public)
        endpoints = []
        for row in rows:
            methods = {}
            for use_public, name in ((False, 'historical_prior'), (True, 'historical_public_forecasts')):
                law = evaluator.evolving(row['source_connection'], row['kind'], row['prediction_clock'], use_public)
                methods[name] = law_summary(law, row['actual'])
            endpoints.append({**row, 'methods': methods})
        pairs = confirmation['transfer_probability'][day]['rows']
        probabilities = {}
        for use_public, name in ((False, 'historical_prior'), (True, 'historical_public_forecasts')):
            p = np.asarray([evaluator.evolving(row['outgoing'], 'departure', row['clock'], use_public).board_probability(row['ready'])
                            for row in pairs])
            y = np.asarray([row['caught'] for row in pairs], dtype=float)
            probabilities[name] = {'n': len(p), 'brier': float(np.mean((p - y) ** 2)),
                                   'signed_probability_error': float(np.mean(p - y))}
        results[day] = {'raw_endpoints': {name: summarize_endpoints(endpoints, name) for name in probabilities},
                        'transfer_probability': probabilities}
        print(day, results[day], flush=True)
    write(HERE / 'calibration_secondary_reference.json', {
        'scope': 'Historical fixed Oct2/6/9 fit, evaluated at the same new-date endpoint and critical-transfer states. '
                 'These are raw central-90% intervals. The Oct1/2 nine-date-calibrated envelope is not transferred to this fit.',
        'fit_source': str(historical.relative_to(ROOT)), 'training_dates': fit['train_dates'], 'confirmation': results})


if __name__ == '__main__':
    main()
