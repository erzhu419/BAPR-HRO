"""A local incoming/outgoing model with a heavy-tailed common day location."""
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
from scipy.special import ndtr

from .calibration_model import SmoothedLaw


class NoiseTable:
    """One-second interpolation of the fixed Gaussian empirical noise law."""

    def __init__(self, values, weights, sigma):
        self.values, self.weights, self.sigma = values, weights, sigma
        self.grid = np.arange(math.floor(values.min() - 9 * sigma), math.ceil(values.max() + 9 * sigma) + 1)
        self.cdf_values = np.empty(len(self.grid))
        self.pdf_values = np.empty(len(self.grid))
        normalizer = sigma * math.sqrt(2 * math.pi)
        for start in range(0, len(self.grid), 128):
            stop = min(start + 128, len(self.grid))
            z = (self.grid[start:stop, None] - values[None, :]) / sigma
            self.cdf_values[start:stop] = ndtr(z) @ weights
            self.pdf_values[start:stop] = np.exp(-z * z / 2) @ weights / normalizer
        self.mean = float(values @ weights)

    def cdf(self, value):
        return np.interp(value, self.grid, self.cdf_values, left=0., right=1.)

    def pdf(self, value):
        return np.interp(value, self.grid, self.pdf_values, left=0., right=0.)


@dataclass
class DayEffectLaw:
    noise: NoiseTable
    delta: np.ndarray
    weights: np.ndarray
    location: float
    cancel_probability: float
    quantiles: dict

    @property
    def mean(self):
        return self.location + self.noise.mean + float(self.delta @ self.weights)

    def cdf(self, time):
        return float(self.noise.cdf(time - self.location - self.delta) @ self.weights)

    def quantile(self, probability):
        if probability not in self.quantiles:
            lower = float(self.delta.min() + self.noise.grid[0])
            upper = float(self.delta.max() + self.noise.grid[-1])
            for _ in range(35):
                middle = (lower + upper) / 2
                value = float(self.noise.cdf(middle - self.delta) @ self.weights)
                if value < probability:
                    lower = middle
                else:
                    upper = middle
            self.quantiles[probability] = (lower + upper) / 2
        return self.location + self.quantiles[probability]

    def board_probability(self, ready, patience=720):
        return (1 - self.cancel_probability) * (self.cdf(ready + patience) - self.cdf(ready))


class HeavyTailDayModel:
    """X=delta+epsilon_in, Y=delta+epsilon_out, noises independent by role.

    Route-centered role-noise KDEs and the Cauchy scale are fitted on earlier
    selected timetable pairs. Conditioning uses one rider-observed X. It does
    not observe other actual endpoints from the same date.
    """

    def __init__(self, marginal_model, training_pairs, pseudo=20., delta_step=15.):
        self.marginal, self.pseudo, self.delta_step = marginal_model, pseudo, delta_step
        self.sigma = marginal_model.sigma
        self.samples = defaultdict(list)
        for pairs in training_pairs.values():
            for incoming, outgoing in pairs:
                if incoming['arrival']['status'] != 'REAL' or outgoing['departure']['status'] != 'REAL':
                    continue
                for role, c, kind in (('incoming', incoming, 'arrival'), ('outgoing', outgoing, 'departure')):
                    residual = c[kind]['time_seconds'] - marginal_model.prior(c, kind).mean
                    self.samples[role, c['route_id']].append(residual)
                    self.samples[role, 'pooled'].append(residual)
        incoming = np.asarray(self.samples['incoming', 'pooled'])
        self.scale = float(np.median(np.abs(incoming - np.median(incoming))))
        self._tables, self._posteriors, self._quantiles = {}, {}, defaultdict(dict)
        self.diagnostics = {}

    def clear_caches(self):
        self._tables, self._posteriors, self._quantiles = {}, {}, defaultdict(dict)
        self.diagnostics = {}

    def table(self, role, route):
        key = role, route
        if key not in self._tables:
            own = np.asarray(self.samples[key], dtype=float)
            pooled = np.asarray(self.samples[role, 'pooled'], dtype=float)
            if len(own):
                values = np.r_[own, pooled]
                weights = np.r_[np.ones(len(own)), np.full(len(pooled), self.pseudo / len(pooled))]
            else:
                values, weights = pooled, np.ones(len(pooled))
            self._tables[key] = NoiseTable(values, weights / weights.sum(), self.sigma)
        return self._tables[key]

    def posterior(self, route, observed_residual):
        key = route, observed_residual
        if key not in self._posteriors:
            noise = self.table('incoming', route)
            # Include every training-KDE likelihood center plus nine bandwidths.
            # The base support and observed anchor remain present on every query.
            lower = min(-300., observed_residual - noise.values.max() - 9 * self.sigma)
            upper = max(3840., observed_residual - noise.values.min() + 9 * self.sigma)
            lower = math.floor(lower / self.delta_step) * self.delta_step
            upper = math.ceil(upper / self.delta_step) * self.delta_step
            delta = np.unique(np.r_[np.arange(lower, upper + self.delta_step / 2, self.delta_step), observed_residual])
            edges = np.r_[lower, (delta[:-1] + delta[1:]) / 2, upper]
            prior_cdf = .5 + np.arctan(edges / self.scale) / math.pi
            prior_mass = np.diff(prior_cdf)
            weights = prior_mass * noise.pdf(observed_residual - delta)
            evidence = float(weights.sum())
            weights /= evidence
            omitted = float(prior_cdf[0] + 1 - prior_cdf[-1])
            unnormalized_tail_bound = omitted * math.exp(-81 / 2) / (self.sigma * math.sqrt(2 * math.pi))
            self.diagnostics[key] = {'observed_residual': observed_residual, 'lower_delta': lower, 'upper_delta': upper,
                'n_delta_nodes': len(delta), 'prior_mass_outside_grid': omitted, 'grid_evidence': evidence,
                'unnormalized_likelihood_tail_bound': unnormalized_tail_bound,
                'tail_bound_over_grid_evidence': unnormalized_tail_bound / evidence,
                'posterior_edge_mass': float(weights[0] + weights[-1]),
                'posterior_mean_delta': float(delta @ weights)}
            self._posteriors[key] = delta, weights
        return self._posteriors[key]

    def predict(self, incoming, outgoing, observed_arrival):
        x = observed_arrival - self.marginal.prior(incoming, 'arrival').mean
        delta, weights = self.posterior(incoming['route_id'], x)
        key = incoming['route_id'], x, outgoing['route_id']
        target = self.marginal.prior(outgoing, 'departure')
        return DayEffectLaw(self.table('outgoing', outgoing['route_id']), delta, weights,
                            target.mean, target.cancel_probability, self._quantiles[key])
