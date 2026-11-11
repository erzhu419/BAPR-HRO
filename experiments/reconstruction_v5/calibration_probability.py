"""Freeze date-balanced monotone boarding calibration on October 3--11."""
from pathlib import Path
import gzip
import json
import pickle
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_run import data_inputs, evaluate_transfers, load, probability_summary, write

BASES = ('evolving_prior', 'public_peer_mixture')


def fit_isotonic(probabilities, outcomes, weights):
    """Weighted least-squares PAVA; equal predictions remain one block."""
    order = np.argsort(probabilities, kind='stable')
    x = np.asarray(probabilities)[order]
    y = np.asarray(outcomes)[order]
    w = np.asarray(weights)[order]
    unique, group = np.unique(x, return_inverse=True)
    group_weight = np.bincount(group, weights=w)
    group_success = np.bincount(group, weights=w * y)
    blocks = []
    for value, total, success in zip(unique, group_weight, group_success):
        blocks.append([float(value), float(total), float(success)])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][1] > blocks[-1][2] / blocks[-1][1]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([right[0], left[1] + right[1], left[2] + right[2]])
    return {'upper': [row[0] for row in blocks], 'probability': [row[2] / row[1] for row in blocks]}


def predict_isotonic(model, probability):
    index = min(int(np.searchsorted(model['upper'], probability, side='left')), len(model['upper']) - 1)
    return model['probability'][index]


def main():
    frozen = json.loads((HERE / 'calibration_frozen_fit.json').read_text())
    protocol, fit = frozen['protocol'], frozen['evolving_fit']
    data = data_inputs()
    frames = load(HERE / 'calibration_public_frames.json.gz')['dates']
    with (HERE / 'calibration_frozen_model.pkl').open('rb') as handle:
        model = pickle.load(handle)
    pairs = {}
    for day in protocol['calibration_dates']:
        rows, unknown = evaluate_transfers(data, day, fit, frames[day]['frames'], model, protocol)
        pairs[day] = {'rows': rows, 'excluded_unknown_departures': unknown}
        print(day, len(rows), flush=True)
    calibration = {}
    for base in BASES:
        p, y, weights = [], [], []
        for day, value in pairs.items():
            for row in value['rows']:
                p.append(row['probabilities'][base])
                y.append(float(row['caught']))
                weights.append(1 / len(value['rows']))
        calibration[base] = fit_isotonic(p, y, weights)
    output = {'training_dates': protocol['calibration_dates'],
              'scope': 'Weighted isotonic least squares on earlier critical-transfer pairs only. '
                       'Each date has equal total weight. Prediction is a monotone step function with endpoint extrapolation. '
                       'This fits a boarding probability; it does not alter endpoint CDFs or the interval envelope. '
                       'No finite-sample probability calibration guarantee is asserted.',
              'n_pairs_by_date': {day: len(value['rows']) for day, value in pairs.items()},
              'models': calibration}
    write(HERE / 'calibration_frozen_probability.json', output)
    with gzip.open(HERE / 'calibration_probability_training.json.gz', 'wt') as handle:
        json.dump(pairs, handle, separators=(',', ':'), allow_nan=False)
    print('PROBABILITY MAPS FROZEN', {name: len(value['upper']) for name, value in calibration.items()}, flush=True)


if __name__ == '__main__':
    main()
