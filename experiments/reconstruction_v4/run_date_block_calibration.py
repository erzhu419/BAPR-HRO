"""Fit before calibration, then score a complete later date as one block."""
from copy import deepcopy
import argparse
import gzip
import json
import math
from pathlib import Path

from evaluate_evolving_calibration import aggregate, interval_and_mean
from run_dated_replay import DATA_PATH, ROOT, dated_graph, fit_training, read_inputs
from src.evolving_trip_belief import EvolvingTripBelief


def endpoint_predictions(data, day, fit):
    graph, _, _, _, source, _ = dated_graph(data, day)
    priors = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
              for c in graph.connections}
    model = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    rows = []
    for c in graph.connections:
        for kind in ('departure', 'arrival'):
            actual = source[c.id][kind]
            if actual['status'] != 'REAL':
                continue
            times, weights, cancel = model.event_distribution(c.id, kind, priors)
            (lower, upper), mean = interval_and_mean(times, weights)
            rows.append({'actual': actual['time_seconds'], 'lower': lower, 'upper': upper,
                         'mean': mean, 'scheduled': c.dep_time if kind == 'departure' else c.arr_time,
                         'cancel_probability': cancel})
    return rows


def conformal_offset(scores, alpha=.1):
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return max(0., sorted(scores)[rank - 1]) if rank <= len(scores) else math.inf


def interval_stats(rows):
    summary = aggregate(rows)
    return {key: value for key, value in summary.items()
            if key not in ('known_service_endpoints', 'service_cancellation_brier')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DATA_PATH)
    parser.add_argument('--earlier', type=Path, default=ROOT / 'experiments/reconstruction_data_audit/rail_calibration_dates_v2.json.gz')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('date_block_calibration.json'))
    args = parser.parse_args()
    data, _ = read_inputs(data_path=args.data)
    with gzip.open(args.earlier, 'rt') as handle:
        earlier = json.load(handle)
    data = deepcopy(data)
    data['dates'].update(earlier['dates'])
    data['train_dates'] = ['2023-10-01', '2023-10-02']
    calibration_dates = [f'2023-10-{day:02}' for day in range(3, 12)]
    fit = fit_training(data)
    scores = {}
    for day in calibration_dates:
        rows = endpoint_predictions(data, day, fit)
        errors = sorted(max(r['lower'] - r['actual'], r['actual'] - r['upper']) for r in rows)
        scores[day] = {'observed_endpoints': len(rows),
                      'score_seconds': errors[math.ceil(.9 * len(errors)) - 1],
                      'maximum_endpoint_score_seconds': errors[-1]}
    offset = conformal_offset([row['score_seconds'] for row in scores.values()])
    tests = {}
    for day in data['holdout_dates']:
        rows = endpoint_predictions(data, day, fit)
        raw = interval_stats(rows)
        for row in rows:
            row['lower'] -= offset
            row['upper'] += offset
        calibrated = interval_stats(rows)
        tests[day] = {'raw': raw, 'calibrated': calibrated,
                      'all_observed_endpoints_covered': calibrated['central_90_covered'] == len(rows)}
    output = {'data': str(args.data), 'earlier_data': str(args.earlier),
              'parameter_training_dates': data['train_dates'], 'calibration_dates': calibration_dates,
              'heldout_dates': data['holdout_dates'], 'alpha': .1, 'quantile_rank': 9,
              'date_scores': scores, 'additive_offset_seconds': offset, 'training': fit,
              'all_endpoints_simultaneous_offset_seconds': conformal_offset([
                  row['maximum_endpoint_score_seconds'] for row in scores.values()]),
              'by_heldout_date': tests,
              'interpretation': 'Each calibration date contributes its empirical 90th-percentile endpoint nonconformity score. The split-conformal guarantee is that a new date reaches at least 90% observed-endpoint coverage with probability at least 90%, assuming exchangeable calibration/test date blocks conditional on earlier training. This differs from a per-endpoint conditional guarantee. Calendar shifts, temporal dependence and disruptions can violate exchangeability. Missing endpoints and cancellation times are outside the observed-REAL score. The separately reported maximum-score offset supports simultaneous coverage of every observed endpoint under the same assumption.',
              'distribution_scope': 'This calibrates an interval envelope. It supplies neither a corrected event PMF nor a boarding-failure probability. The earlier three-training-date replay is a separate experiment.'}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'offset_seconds': offset, 'date_scores': scores, 'by_heldout_date': tests}, indent=2))


if __name__ == '__main__':
    main()
