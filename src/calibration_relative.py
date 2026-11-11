"""Past-only relative departure-delay law after a rider-observed arrival."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .calibration_model import SmoothedLaw


@dataclass
class RelativeClockLaw:
    residual: SmoothedLaw
    location: float
    cancel_probability: float
    quantiles: dict

    @property
    def mean(self):
        return self.residual.mean + self.location

    def cdf(self, time):
        return self.residual.cdf(time - self.location)

    def quantile(self, probability):
        if probability not in self.quantiles:
            self.quantiles[probability] = self.residual.quantile(probability)
        return self.quantiles[probability] + self.location

    def board_probability(self, ready, patience=720):
        return (1 - self.cancel_probability) * (self.cdf(ready + patience) - self.cdf(ready))


class RelativeDelayModel:
    """A fixed-coefficient common-location model, conditional on operating.

    The shared additive delay component cancels in departure minus incoming
    arrival delay. The remaining relative residual need not be independent of
    the observed incoming delay; this is not arbitrary conditional coverage.
    """

    coefficient = 1.

    def __init__(self, marginal_model, training_pairs, pseudo=20.):
        self.marginal = marginal_model
        self.pseudo = pseudo
        self.samples = defaultdict(list)
        self._cache = {}
        self._quantiles = defaultdict(dict)
        self.training_counts = defaultdict(int)
        for day, pairs in training_pairs.items():
            for incoming, outgoing in pairs:
                if incoming['arrival']['status'] != 'REAL' or outgoing['departure']['status'] != 'REAL':
                    continue
                left = incoming['arrival']['time_seconds'] - self.marginal.prior(incoming, 'arrival').mean
                right = outgoing['departure']['time_seconds'] - self.marginal.prior(outgoing, 'departure').mean
                residual = right - self.coefficient * left
                self.samples[self.key(incoming, outgoing)].append(residual)
                self.samples['pooled'].append(residual)
                self.training_counts[day] += 1

    @staticmethod
    def key(incoming, outgoing):
        return incoming['route_id'], outgoing['route_id']

    def route_difference(self, incoming, outgoing):
        before = self.marginal.prior(incoming, 'arrival').mean - incoming['scheduled_arrival_seconds']
        after = self.marginal.prior(outgoing, 'departure').mean - outgoing['scheduled_departure_seconds']
        return after - before

    def residual_law(self, incoming, outgoing):
        key = self.key(incoming, outgoing)
        if key not in self._cache:
            own = np.asarray(self.samples[key], dtype=float)
            pooled = np.asarray(self.samples['pooled'], dtype=float)
            if len(own):
                values = np.r_[own, pooled]
                weights = np.r_[np.ones(len(own)), np.full(len(pooled), self.pseudo / len(pooled))]
            else:
                values, weights = pooled, np.ones(len(pooled))
            self._cache[key] = SmoothedLaw(values, weights / weights.sum(), self.marginal.sigma)
        return self._cache[key]

    def predict(self, incoming, outgoing, observed_arrival):
        """Only the supplied observed arrival enters; target truth is unused."""
        incoming_mean = self.marginal.prior(incoming, 'arrival').mean
        location = self.marginal.prior(outgoing, 'departure').mean + self.coefficient * (observed_arrival - incoming_mean)
        cancellation = self.marginal.prior(outgoing, 'departure').cancel_probability
        return RelativeClockLaw(self.residual_law(incoming, outgoing), location, cancellation,
                                self._quantiles[self.key(incoming, outgoing)])


class ShrunkRelativeDelayModel(RelativeDelayModel):
    """A bounded zero-intercept slope fitted on earlier timetable pairs."""

    def __init__(self, marginal_model, training_pairs, pseudo=20.):
        numerator = denominator = 0.
        for pairs in training_pairs.values():
            for incoming, outgoing in pairs:
                if incoming['arrival']['status'] != 'REAL' or outgoing['departure']['status'] != 'REAL':
                    continue
                x = incoming['arrival']['time_seconds'] - marginal_model.prior(incoming, 'arrival').mean
                y = outgoing['departure']['time_seconds'] - marginal_model.prior(outgoing, 'departure').mean
                numerator += x * y
                denominator += x * x
        self.coefficient = float(np.clip(numerator / denominator, 0., 1.)) if denominator else 0.
        super().__init__(marginal_model, training_pairs, pseudo)
