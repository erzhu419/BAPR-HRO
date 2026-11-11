"""Matched past-only marginal forecast and boarding calibration comparison."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V4 = ROOT / 'experiments/reconstruction_v4'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V4))
from run_dated_replay import dated_graph, fit_training
from src.calibration_model import PublicCalibrationModel, PublicForecasts
from src.evolving_trip_belief import EvolvingTripBelief

AUDIT = ROOT / 'experiments/reconstruction_data_audit'
METHODS = ('evolving_prior', 'evolving_public_forecasts', 'route_empirical', 'public_peer_mixture')


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def data_inputs():
    data = load(AUDIT / 'dated_replay_v2.json.gz')
    data['dates'].update(load(AUDIT / 'rail_calibration_dates_v2.json.gz')['dates'])
    data['station_transfer_bounds'] = json.loads((AUDIT / 'station_transfer_bounds.json').read_text())['stations']
    return data


@dataclass
class DiscreteLaw:
    values: np.ndarray
    weights: np.ndarray
    cancel_probability: float

    @property
    def mean(self):
        return float(self.values @ self.weights)

    def cdf(self, time):
        return float(self.weights[self.values <= time].sum())

    def quantile(self, probability):
        index = min(np.searchsorted(np.cumsum(self.weights), probability), len(self.values) - 1)
        return float(self.values[index])

    def board_probability(self, ready, patience=720):
        return (1 - self.cancel_probability) * float(self.weights[(self.values >= ready) & (self.values <= ready + patience)].sum())


class ForecastEvaluator:
    def __init__(self, data, day, fit, public):
        graph, world, _, mapping, source, _ = dated_graph(data, day, 'gtfs_station_max')
        self.graph, self.world, self.source = graph, world, source
        self.mapping = mapping
        self.priors = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
        self.base = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
        self.public = public
        self.frame_models = {}
        self._predictions = {}

    def evolving(self, source_id, kind, clock, use_public):
        c = self.graph.connections[self.mapping[source_id]]
        snapshot = self.public.at(clock)
        index = snapshot.receipt if use_public else -1
        key = source_id, kind, index
        if key not in self._predictions:
            if index not in self.frame_models:
                model = self.base.copy()
                if use_public:
                    for (cid, event_kind), (estimate, _) in snapshot.values.items():
                        if cid in self.mapping:
                            local = self.mapping[cid]
                            route = self.graph.connections[local].route
                            sd = max(30., self.priors[route]['std'] * 60)
                            model.observe_forecast(local, event_kind, estimate, sd)
                self.frame_models[index] = model
            model = self.frame_models[index]
            times, weights, cancel = model.event_distribution(c.id, kind, self.priors)
            self._predictions[key] = DiscreteLaw(times, weights, cancel)
        return self._predictions[key]


def summarize_endpoints(rows, method, offset=0.):
    operating = [row for row in rows if row['actual'] is not None]
    covered, below, above, scores, widths, errors = [], [], [], [], [], []
    for row in operating:
        estimate = row['methods'][method]
        lower, upper = estimate['lower'] - offset, estimate['upper'] + offset
        actual = row['actual']
        covered.append(lower <= actual <= upper)
        below.append(actual < lower)
        above.append(actual > upper)
        width = upper - lower
        widths.append(width)
        scores.append(width + 20 * max(lower - actual, actual - upper, 0))
        errors.append(estimate['mean'] - actual)
    return {'n': len(operating), 'covered': sum(covered), 'coverage': float(np.mean(covered)),
            'below': sum(below), 'above': sum(above), 'mean_width_seconds': float(np.mean(widths)),
            'mean_interval_score': float(np.mean(scores)), 'mae_seconds': float(np.mean(np.abs(errors))),
            'signed_mean_error_seconds': float(np.mean(errors))}


def law_summary(law, actual):
    return {'lower': law.quantile(.05), 'upper': law.quantile(.95), 'mean': law.mean,
            'cancel_probability': law.cancel_probability, 'pit': law.cdf(actual) if actual is not None else None}


def evaluate_day(data, day, fit, frames, calibrated_model, protocol):
    connections = data['dates'][day]['connections']
    public = PublicForecasts(connections, frames)
    evaluator = ForecastEvaluator(data, day, fit, public)
    included = {row['id']: row for row in connections if row['id'] in evaluator.mapping}
    rows = []
    for cid, c in included.items():
        for kind in ('departure', 'arrival'):
            outcome = c[kind]
            if outcome['status'] not in ('REAL', 'CANCELED'):
                continue
            clock = c[f'scheduled_{kind}_seconds'] - protocol['endpoint_cutoff_before_schedule_seconds']
            actual = outcome['time_seconds'] if outcome['status'] == 'REAL' else None
            mixture, features = calibrated_model.predict(c, kind, clock, public)
            laws = {'evolving_prior': evaluator.evolving(cid, kind, clock, False),
                    'evolving_public_forecasts': evaluator.evolving(cid, kind, clock, True),
                    'route_empirical': calibrated_model.prior(c, kind), 'public_peer_mixture': mixture}
            rows.append({'date': day, 'source_connection': cid, 'trip_id': c['trip_id'], 'route_id': c['route_id'],
                         'route_name': c['route_name'], 'kind': kind, 'scheduled': c[f'scheduled_{kind}_seconds'],
                         'prediction_clock': clock, 'actual': actual, 'features': features,
                         'methods': {name: law_summary(law, actual) for name, law in laws.items()}})
    return rows


def evaluate_transfers(data, day, fit, frames, model, protocol):
    connections = data['dates'][day]['connections']
    public = PublicForecasts(connections, frames)
    evaluator = ForecastEvaluator(data, day, fit, public)
    source = {c['id']: c for c in connections}
    graph = evaluator.graph
    rows = []
    unknown = 0
    for incoming in graph.connections:
        before = evaluator.source[incoming.id]
        if before['outcome'] != 'MEASURED':
            continue
        now = before['arrival']['time_seconds']
        ready = now + graph.stops[incoming.arr_stop].min_transfer_time
        for outgoing in graph.get_connections_from(incoming.arr_stop):
            if outgoing.trip_id == incoming.trip_id or not outgoing.pickup_allowed:
                continue
            if abs(outgoing.dep_time - ready) > protocol['transfer_critical_scheduled_margin_seconds']:
                continue
            after = evaluator.source[outgoing.id]
            cid = after['id']
            if after['departure']['status'] not in ('REAL', 'CANCELED'):
                unknown += 1
                continue
            realized = after['departure']
            caught = (realized['status'] == 'REAL' and ready <= realized['time_seconds'] <= ready + protocol['transfer_patience_seconds'])
            mixture, features = model.predict(after, 'departure', now, public)
            laws = {'evolving_prior': evaluator.evolving(cid, 'departure', now, False),
                    'evolving_public_forecasts': evaluator.evolving(cid, 'departure', now, True),
                    'route_empirical': model.prior(after, 'departure'), 'public_peer_mixture': mixture}
            rows.append({'date': day, 'incoming': before['id'], 'outgoing': cid, 'clock': now, 'ready': ready,
                         'trip_id': after['trip_id'], 'route_id': after['route_id'], 'caught': bool(caught),
                         'features': features, 'probabilities': {name: law.board_probability(ready) for name, law in laws.items()},
                         'scheduled_probability': float(ready <= outgoing.dep_time <= ready + protocol['transfer_patience_seconds'])})
    return rows, unknown


def probability_summary(rows):
    result = {}
    for method in (*METHODS, 'schedule'):
        probabilities = np.asarray([row['scheduled_probability'] if method == 'schedule' else row['probabilities'][method]
                                    for row in rows])
        truth = np.asarray([row['caught'] for row in rows], dtype=float)
        bins = []
        for lo, hi in zip((0., .2, .4, .6, .8), (.2, .4, .6, .8, 1.000001)):
            mask = (probabilities >= lo) & (probabilities < hi)
            if mask.any():
                bins.append({'n': int(mask.sum()), 'predicted': float(probabilities[mask].mean()),
                             'observed': float(truth[mask].mean())})
        result[method] = {'n': len(rows), 'brier': float(np.mean((probabilities - truth) ** 2)),
                          'signed_probability_error': float(np.mean(probabilities - truth)), 'reliability_bins': bins}
    return result


def block_offsets(calibration, dates, alpha):
    methods, detail = {}, {}
    rank = math.ceil((len(dates) + 1) * (1 - alpha))
    for method in METHODS:
        scores = {}
        for day in dates:
            errors = sorted(max(row['methods'][method]['lower'] - row['actual'],
                                row['actual'] - row['methods'][method]['upper'])
                            for row in calibration[day] if row['actual'] is not None)
            scores[day] = errors[math.ceil((1 - alpha) * len(errors)) - 1]
        methods[method] = max(0., sorted(scores.values())[rank - 1])
        detail[method] = scores
    return methods, detail


def main():
    protocol = json.loads((HERE / 'calibration_protocol.json').read_text())
    data = data_inputs()
    data['train_dates'] = protocol['parameter_training_dates']
    frames = load(HERE / 'calibration_public_frames.json.gz')['dates']
    fit = fit_training(data)
    training = {day: [c for c in data['dates'][day]['connections'] if c['pickup_allowed'] and c['dropoff_allowed']]
                for day in data['train_dates']}
    snapshots = {day: PublicForecasts(data['dates'][day]['connections'], frames[day]['frames']) for day in training}
    model = PublicCalibrationModel(training, snapshots, protocol)
    frozen = {'protocol': protocol, 'evolving_fit': fit,
              'public_mixture_weights': model.mixture_weights, 'weight_fit_counts': model.weight_fit_counts}
    write(HERE / 'calibration_frozen_fit.json', frozen)
    model._law_cache.clear()
    with (HERE / 'calibration_frozen_model.pkl').open('wb') as handle:
        pickle.dump(model, handle)
    print('FIT FROZEN', model.mixture_weights, model.weight_fit_counts, flush=True)
    all_rows = {}
    for day in protocol['calibration_dates'] + protocol['heldout_dates']:
        all_rows[day] = evaluate_day(data, day, fit, frames[day]['frames'], model, protocol)
        print('ENDPOINTS', day, len(all_rows[day]), flush=True)
    offsets, scores = block_offsets(all_rows, protocol['calibration_dates'], protocol['nominal_miscoverage'])
    write(HERE / 'calibration_frozen_offsets.json', {'offset_seconds': offsets, 'date_scores': scores})
    tests, transfers = {}, {}
    for day in protocol['heldout_dates']:
        rows = all_rows[day]
        tests[day] = {method: {'raw': summarize_endpoints(rows, method),
                               'date_calibrated': summarize_endpoints(rows, method, offsets[method])}
                      for method in METHODS}
        paired, unknown = evaluate_transfers(data, day, fit, frames[day]['frames'], model, protocol)
        transfers[day] = {'excluded_unknown_departures': unknown, 'methods': probability_summary(paired), 'rows': paired}
        print('HELDOUT', day, tests[day], 'PROBABILITIES', {name: row['brier'] for name, row in transfers[day]['methods'].items()}, flush=True)
    output = {'scope': 'Fixed past-only fit, nine-date interval calibration and three previously examined development test dates. '
              'Numeric forecasts and peer summaries use receipts before each prediction clock. '
              'Transfer predictions use the traveler-known arrival clock, with unknown outgoing outcomes excluded.',
              'protocol': protocol, 'frozen_fit': frozen, 'offset_seconds': offsets, 'date_scores': scores,
              'heldout': tests, 'transfer_probability': transfers, 'endpoint_rows': all_rows}
    with gzip.open(HERE / 'calibration_results.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(output, handle, separators=(',', ':'), allow_nan=False)
    write(HERE / 'calibration_summary.json', {key: output[key] for key in ('scope', 'offset_seconds', 'heldout')})


if __name__ == '__main__':
    main()
