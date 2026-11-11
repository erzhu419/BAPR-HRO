"""Past-only marginal delay laws using time-stamped public forecasts."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


@dataclass
class ForecastSnapshot:
    values: dict
    receipt: int


class PublicForecasts:
    """Reconstruct a query's public forecast state from receipts only."""

    def __init__(self, connections, frames):
        self.connections = {c['id']: c for c in connections}
        self.frames = frames
        self.by_trip = defaultdict(list)
        for c in connections:
            self.by_trip[c['trip_id']].append(c['id'])
        self._cache = {}

    def at(self, clock):
        index = sum(frame['receipt_seconds'] <= clock for frame in self.frames)
        if index not in self._cache:
            values = {}
            for frame in self.frames[:index]:
                receipt = frame['receipt_seconds']
                for trip, status in frame['trip_statuses']:
                    if status == 'CANCELED':
                        for cid in self.by_trip.get(trip, ()):
                            for kind in ('departure', 'arrival'):
                                values.pop((cid, kind), None)
                for cid, kind, status, predicted in frame['endpoint_updates']:
                    if cid not in self.connections:
                        continue
                    if status == 'FORECAST':
                        values[cid, kind] = (predicted, receipt)
                    elif status in ('SKIPPED', 'NO_DATA'):
                        values.pop((cid, kind), None)
            receipt = self.frames[index - 1]['receipt_seconds'] if index else -1
            self._cache[index] = ForecastSnapshot(values, receipt)
        return self._cache[index]


@dataclass
class SmoothedLaw:
    values: np.ndarray
    weights: np.ndarray
    kernel_sd: float
    cancel_probability: float = 0.0

    @property
    def mean(self):
        return float(self.values @ self.weights)

    def cdf(self, time):
        return float(ndtr((time - self.values) / self.kernel_sd) @ self.weights)

    def quantile(self, probability):
        lo = float(self.values.min() - 9 * self.kernel_sd)
        hi = float(self.values.max() + 9 * self.kernel_sd)
        for _ in range(35):
            middle = (lo + hi) / 2
            if self.cdf(middle) < probability:
                lo = middle
            else:
                hi = middle
        return (lo + hi) / 2

    def shifted(self, location):
        return SmoothedLaw(self.values + location, self.weights, self.kernel_sd, self.cancel_probability)

    def board_probability(self, ready, patience=720):
        return (1 - self.cancel_probability) * (self.cdf(ready + patience) - self.cdf(ready))


class PublicCalibrationModel:
    """An empirical prior/forecast-error mixture, with an observed peer shift.

    The class supplies marginal event distributions. A trajectory model requires
    an additional coupling that preserves chronological service constraints.
    """

    def __init__(self, training, snapshots, protocol):
        self.protocol = protocol
        self.sigma = protocol['kernel_sd_seconds']
        self.pseudo = protocol['route_shrinkage_pseudocount']
        self.delays = defaultdict(list)
        self.forecast_errors = defaultdict(list)
        self.forecast_levels = defaultdict(list)
        self.cancel_counts = defaultdict(lambda: [0, 0])
        self.mixture_weights = {'departure': .5, 'arrival': .5}
        self._law_cache = {}
        for day, connections in training.items():
            public = snapshots[day]
            for c in connections:
                dep_status = c['departure']['status']
                if dep_status in ('REAL', 'CANCELED'):
                    self.cancel_counts[c['route_id']][0] += dep_status == 'CANCELED'
                    self.cancel_counts[c['route_id']][1] += 1
                    self.cancel_counts['pooled'][0] += dep_status == 'CANCELED'
                    self.cancel_counts['pooled'][1] += 1
                for kind in ('departure', 'arrival'):
                    if c[kind]['status'] != 'REAL':
                        continue
                    scheduled = c[f'scheduled_{kind}_seconds']
                    delay = c[kind]['time_seconds'] - scheduled
                    for route in (c['route_id'], 'pooled'):
                        self.delays[route, kind].append(delay)
                    cutoff = scheduled - protocol['endpoint_cutoff_before_schedule_seconds']
                    issued = public.at(cutoff).values.get((c['id'], kind))
                    if issued is not None:
                        prediction, _ = issued
                        for route in (c['route_id'], 'pooled'):
                            self.forecast_errors[route, kind].append(c[kind]['time_seconds'] - prediction)
                            self.forecast_levels[route, kind].append(prediction - scheduled)
        self._fit_mixture_weights(training, snapshots)
        self.cancel_counts = dict(self.cancel_counts)

    def _empirical(self, source, route, kind):
        cache_key = id(source), route, kind
        if cache_key not in self._law_cache:
            pooled = np.asarray(source['pooled', kind], dtype=float)
            own = np.asarray(source[route, kind], dtype=float)
            if len(own):
                values = np.r_[own, pooled]
                weights = np.r_[np.ones(len(own)), np.full(len(pooled), self.pseudo / len(pooled))]
            else:
                values = pooled
                weights = np.ones(len(pooled))
            self._law_cache[cache_key] = SmoothedLaw(values, weights / weights.sum(), self.sigma)
        return self._law_cache[cache_key]

    def peer_shift(self, c, clock, public):
        window_start, window_end = self.protocol['peer_forecast_schedule_window_seconds']
        peers = {}
        for (cid, kind), (forecast, _) in public.at(clock).values.items():
            if kind != 'departure':
                continue
            peer = public.connections[cid]
            scheduled = peer['scheduled_departure_seconds']
            if peer['trip_id'] == c['trip_id'] or not clock + window_start <= scheduled <= clock + window_end:
                continue
            route = peer['route_id']
            values = self.forecast_levels[route, kind] or self.forecast_levels['pooled', kind]
            residual = forecast - scheduled - float(np.median(values))
            old = peers.get(peer['trip_id'])
            if old is None or scheduled < old[0]:
                peers[peer['trip_id']] = scheduled, residual
        if not peers:
            return 0., 0
        shift = float(np.median([value[1] for value in peers.values()]))
        n = len(peers)
        return shift * n / (n + self.protocol['peer_shrinkage_pseudocount']), n

    def prior(self, c, kind):
        law = self._empirical(self.delays, c['route_id'], kind).shifted(c[f'scheduled_{kind}_seconds'])
        canceled, count = self.cancel_counts.get(c['route_id'], self.cancel_counts['pooled'])
        law.cancel_probability = canceled / count
        return law

    def predict(self, c, kind, clock, public):
        prior = self.prior(c, kind)
        shift, peers = self.peer_shift(c, clock, public)
        prior = prior.shifted(shift)
        own = public.at(clock).values.get((c['id'], kind))
        if own is None:
            return prior, {'peer_shift_seconds': shift, 'peer_trips': peers, 'has_own_forecast': False}
        estimate, receipt = own
        residual = self._empirical(self.forecast_errors, c['route_id'], kind).shifted(estimate)
        weight = self.mixture_weights[kind]
        combined = SmoothedLaw(np.r_[prior.values, residual.values],
                               np.r_[(1 - weight) * prior.weights, weight * residual.weights],
                               self.sigma, prior.cancel_probability)
        return combined, {'peer_shift_seconds': shift, 'peer_trips': peers, 'has_own_forecast': True,
                          'own_forecast_receipt': receipt, 'own_forecast_weight': weight}

    def _fit_mixture_weights(self, training, snapshots):
        moments = {kind: [0., 0., 0] for kind in ('departure', 'arrival')}
        for day, connections in training.items():
            public = snapshots[day]
            for c in connections:
                for kind in ('departure', 'arrival'):
                    if c[kind]['status'] != 'REAL':
                        continue
                    clock = c[f'scheduled_{kind}_seconds'] - self.protocol['endpoint_cutoff_before_schedule_seconds']
                    own = public.at(clock).values.get((c['id'], kind))
                    if own is None:
                        continue
                    shift, _ = self.peer_shift(c, clock, public)
                    prior = self.prior(c, kind).mean + shift
                    forecast = own[0] + self._empirical(self.forecast_errors, c['route_id'], kind).mean
                    difference = forecast - prior
                    moments[kind][0] += difference * (c[kind]['time_seconds'] - prior)
                    moments[kind][1] += difference ** 2
                    moments[kind][2] += 1
        for kind, (cross, square, _) in moments.items():
            self.mixture_weights[kind] = float(np.clip(cross / square, 0, 1)) if square else .5
        self.weight_fit_counts = {kind: values[2] for kind, values in moments.items()}
