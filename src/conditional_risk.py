"""Observable conditional risk laws and same-chain marginal replacement."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import stdtr, stdtrit

from .evolving_trip_belief import _ChainLaw


@dataclass(frozen=True)
class StudentClockLaw:
    location: float
    scale: float
    cancel_probability: float
    df: float = 3.

    @property
    def mean(self):
        return self.location

    def cdf(self, time):
        return float(stdtr(self.df, (time - self.location) / self.scale))

    def quantile(self, probability):
        return self.location + self.scale * float(stdtrit(self.df, probability))

    def board_probability(self, ready, patience=720):
        return (1 - self.cancel_probability) * (self.cdf(ready + patience) - self.cdf(ready))


class ConditionalStudentModel:
    """A training-only robust conditional location and surprise-dependent scale.

    The prior event location remains route specific. The incoming residual is
    first-hand observed, while the target's actual clock is never read.
    Each training date has equal objective mass; pairs are not treated as
    independent observations for any uncertainty statement.
    """

    def __init__(self, marginal, training_pairs):
        self.marginal = marginal
        x, y, weights = [], [], []
        self.training_counts = {}
        for day, pairs in training_pairs.items():
            eligible = [(a, b) for a, b in pairs
                        if a['arrival']['status'] == b['departure']['status'] == 'REAL']
            self.training_counts[day] = len(eligible)
            for incoming, outgoing in eligible:
                x.append(incoming['arrival']['time_seconds'] - marginal.prior(incoming, 'arrival').mean)
                y.append(outgoing['departure']['time_seconds'] - marginal.prior(outgoing, 'departure').mean)
                weights.append(1 / (len(training_pairs) * len(eligible)))
        x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)

        def objective(parameters):
            beta, log_scale, slope = parameters
            base = np.exp(log_scale)
            scale = base + slope * np.abs(x)
            residual = (y - beta * x) / scale
            loss = np.log(scale) + 2 * np.log1p(residual ** 2 / 3)
            common = 1 - 4 * residual ** 2 / (3 + residual ** 2)
            gradient = np.array([
                np.sum(weights * (-4 * residual * x / (scale * (3 + residual ** 2)))),
                np.sum(weights * common * base / scale),
                np.sum(weights * common * np.abs(x) / scale)])
            return float(weights @ loss), gradient

        fits = []
        for beta in (0., .2, 1.):
            initial_scale = max(1., float(np.median(np.abs(y - beta * x))) / float(stdtrit(3, .75)))
            fit = minimize(objective, (beta, np.log(initial_scale), .1), jac=True,
                           method='L-BFGS-B', bounds=((0., 1.), (0., None), (0., None)))
            fits.append(fit)
        best = min(fits, key=lambda fit: fit.fun)
        self.beta, log_scale, self.scale_slope = map(float, best.x)
        self.base_scale = float(np.exp(log_scale))
        self.fit_diagnostics = [{'success': bool(fit.success), 'objective': float(fit.fun),
                                 'parameters': fit.x.tolist(), 'message': str(fit.message)} for fit in fits]

    def predict(self, incoming, outgoing, observed_arrival):
        residual = observed_arrival - self.marginal.prior(incoming, 'arrival').mean
        target = self.marginal.prior(outgoing, 'departure')
        return StudentClockLaw(target.mean + self.beta * residual,
                               self.base_scale + self.scale_slope * abs(residual),
                               target.cancel_probability)


def observable_group(row):
    return 'late' if row['incoming_delay'] > 180 else 'ordinary'


def simultaneous_group_envelope(rows_by_date, method, groups=('ordinary', 'late')):
    """Date-block rank score for every present, predefined observable group.

    Each group contributes its ceil(.9*n)-th score; the date score is their
    maximum. With nine exchangeable calibration dates, their maximum covers
    the next date score with probability at least9/10. Empty groups are
    vacuous on that date, and counts are retained rather than inflated.
    """
    scores, detail = {}, {}
    for day, rows in rows_by_date.items():
        group_scores = {}
        for group in groups:
            values = []
            for row in rows:
                if row['actual'] is None or observable_group(row) != group:
                    continue
                prediction = row['methods'][method]
                halfwidth = (prediction['upper'] - prediction['lower']) / 2
                values.append(max(prediction['lower'] - row['actual'],
                                  row['actual'] - prediction['upper']) / halfwidth)
            group_scores[group] = {'n': len(values), 'score':
                sorted(values)[math.ceil(.9 * len(values)) - 1] if values else None}
        scores[day] = max(value['score'] for value in group_scores.values() if value['n'])
        detail[day] = group_scores
    return max(0., max(scores.values())), {'date_scores': scores, 'groups_by_date': detail}


def retarget_event_chain(base, index, law):
    """Replace one event's marginal and keep its conditional complete chain.

    CDF cell masses are projected onto the original event's operating grid.
    Impossible states remain impossible. The ratio target/base is an emission
    on the same joint chain, so all future marginals and FFBS draws change
    consistently. Callers obtain catch probability from the returned discrete
    chain, including boardable equality, rather than reusing continuous p.
    """
    original = base.marginal(index)
    operating = ~base.canceled[index]
    clocks = base.times[index][operating]
    target = np.zeros_like(original)
    operating_mass = original[operating].sum()
    if len(clocks) and operating_mass:
        edges = np.r_[-np.inf, (clocks[:-1].astype(float) + clocks[1:]) / 2, np.inf]
        cumulative = np.r_[0., [law.cdf(t) for t in edges[1:-1]], 1.]
        proposed = np.diff(cumulative)
        proposed[original[operating] == 0] = 0
        if not proposed.sum():
            raise ValueError('The target clock law gives zero mass to the observed operating support')
        target[operating] = operating_mass * proposed / proposed.sum()
    # Calibrating operating-clock risk does not identify cancellation rates.
    target[~operating] = original[~operating]
    if not target.sum():
        raise ValueError('The target event law gives zero mass to the observed history support')
    target /= target.sum()
    ratio = np.zeros_like(original)
    supported = original > 0
    with np.errstate(divide='ignore'):
        ratio[supported] = np.log(target[supported]) - np.log(original[supported])
    emissions = list(base.emissions)
    emissions[index] = emissions[index] + ratio
    updated = _ChainLaw(base.times, base.initial, base._transition, emissions,
                        base.canceled, base.stochastic)
    return updated, target
