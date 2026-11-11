"""A shared-day mixture of ordered event-clock trip models.

On the current common grids, mixture weights, marginals and sampling use the
same finite joint law. EvolvingTripBelief augments grids at off-grid observed
clocks; that remains a continuous-clock posterior approximation, rather than
Bayesian conditioning of one immutable global finite support.
"""
from collections.abc import Mapping
from copy import copy
import math

import numpy as np

from .evolving_trip_belief import EvolvingTripBelief, logsumexp
from .trip_belief import NominalEvent


TRAINING_DAY_SCALE_SECONDS = 43.52903486620926


def cauchy_bin_weights(deltas, scale):
    """Midpoint bins; the two endpoint values receive both exterior tails."""
    edges = np.r_[-np.inf, (deltas[:-1] + deltas[1:]) / 2, np.inf]
    return np.diff(.5 + np.arctan(edges / scale) / math.pi)


class _JointNominalEvents(Mapping):
    def __init__(self, belief, parameters):
        self.belief, self.parameters = belief, parameters

    def __getitem__(self, cid):
        return self.belief._mean_event(cid, self.parameters)

    def __iter__(self):
        return iter(self.belief.components[0]._cid_order)

    def __len__(self):
        return len(self.belief.graph.connections)


class JointDayBelief:
    """One delta per world; trips are conditionally independent given delta.

    Day scale is the frozen Oct1--2 incoming-residual MAD. Delta changes only
    route means. Cancellation, increments, grids and observation definitions
    remain the same in all components. Full per-trip evidence replaces its
    previous contribution whenever a revision occurs.
    """

    def __init__(self, graph, *, minute=60, deltas_seconds=None, prior_weights=None,
                 day_scale_seconds=TRAINING_DAY_SCALE_SECONDS, **increments):
        self.graph, self.minute = graph, minute
        self.deltas_seconds = np.asarray(list(range(-360, 361, 60)) if deltas_seconds is None else deltas_seconds, dtype=float)
        prior = cauchy_bin_weights(self.deltas_seconds, day_scale_seconds) if prior_weights is None else np.asarray(prior_weights, dtype=float)
        self.prior_weights = prior / prior.sum()
        self.deltas_seconds.flags.writeable = False
        self.prior_weights.flags.writeable = False
        self.components = [EvolvingTripBelief(graph, minute=minute, **increments) for _ in self.deltas_seconds]
        self.public_occurrences = self.components[0].public_occurrences
        # Transition kernels contain no route mean and are immutable when made.
        for component in self.components[1:]:
            component._transition_cache = self.components[0]._transition_cache
        self._parameter_key = None
        self._weights = None
        self._shifted_parameters = None
        self._event_cache, self._mean_cache = {}, {}

    def copy(self):
        branch = copy(self)
        branch.components = [component.copy() for component in self.components]
        branch._event_cache = self._event_cache.copy()
        branch._mean_cache = self._mean_cache.copy()
        return branch

    def _observe(self, method, *args):
        previous = self.components[0]._revision.copy()
        for component in self.components:
            getattr(component, method)(*args)
        if previous != self.components[0]._revision:
            self._parameter_key, self._weights = None, None
            self._event_cache, self._mean_cache = {}, {}

    def observe_connection_event(self, cid, kind, delay):
        self._observe('observe_connection_event', cid, kind, delay)

    def observe_forecast(self, cid, kind, predicted_time, sd_seconds):
        self._observe('observe_forecast', cid, kind, predicted_time, sd_seconds)

    def observe_poll(self, stop, time, messages, seen_before):
        self._observe('observe_poll', stop, time, messages, seen_before)

    def observe_attempt(self, cid, kind, time):
        self._observe('observe_attempt', cid, kind, time)

    def observe_wait(self, cid, start, end):
        self._observe('observe_wait', cid, start, end)

    def _ensure(self, parameters):
        routes = self.components[0]._route_cids
        default = {'mean': 1., 'std': 2., 'cancel_rate': .01}
        base = {route: dict(parameters.get(route, default)) for route in routes}
        key = tuple((route, row['mean'], row['std'], row['cancel_rate']) for route, row in base.items())
        if self._parameter_key == key:
            return
        shifted = [{route: dict(row, mean=row['mean'] + delta / self.minute) for route, row in base.items()}
                   for delta in self.deltas_seconds]
        log_weights = np.log(self.prior_weights)
        revised = tuple(self.components[0]._revision)
        for index, component in enumerate(self.components):
            for trip in revised:
                law = component._law(trip, shifted[index])
                log_weights[index] += float(logsumexp(law.forward[-1]))
        weights = np.exp(log_weights - logsumexp(log_weights))
        weights.flags.writeable = False
        self._weights, self._shifted_parameters, self._parameter_key = weights, shifted, key
        self._event_cache, self._mean_cache = {}, {}

    def day_posterior(self, parameters):
        self._ensure(parameters)
        return self.deltas_seconds, self._weights

    def event_distribution(self, cid, kind, parameters):
        self._ensure(parameters)
        key = cid, kind
        if key not in self._event_cache:
            active, cancel = [], 0.
            for weight, component, pars in zip(self._weights, self.components, self._shifted_parameters):
                trip, index = component._positions[cid, kind]
                law = component._law(trip, pars)
                marginal = law.marginal(index)
                operating = ~law.canceled[index]
                cancel += float(weight * marginal[~operating].sum())
                # Do not recover tiny operating mass by subtracting a rounded
                # cancellation probability from one after a long wait.
                mass = weight * marginal[operating]
                if mass.sum() > 0:
                    active.append((law.times[index][operating], mass))
            if not active:
                result = np.empty(0, dtype=int), np.empty(0), 1.
            else:
                times = np.unique(np.concatenate([row[0] for row in active]))
                mass = np.zeros(len(times))
                for clocks, probabilities in active:
                    np.add.at(mass, np.searchsorted(times, clocks), probabilities)
                mass /= mass.sum()
                times.flags.writeable = mass.flags.writeable = False
                result = times, mass, cancel
            self._event_cache[key] = result
        return self._event_cache[key]

    def _mean_event(self, cid, parameters):
        self._ensure(parameters)
        if cid not in self._mean_cache:
            times, probabilities, cancel = self.event_distribution(cid, 'departure', parameters)
            if not len(times) or cancel == 1.:
                event = NominalEvent(None, None, None, cancel)
            else:
                arrival, arrival_weights, _ = self.event_distribution(cid, 'arrival', parameters)
                departure = float(times @ probabilities)
                event = NominalEvent(departure - self.graph.connections[cid].dep_time, departure,
                                     float(arrival @ arrival_weights), cancel)
            self._mean_cache[cid] = event
        return self._mean_cache[cid]

    def mean_events(self, parameters):
        self._ensure(parameters)
        return _JointNominalEvents(self.copy(), {route: dict(row) for route, row in parameters.items()})

    def hard_support(self, cid, kind, parameters):
        self._ensure(parameters)
        bounds = [component.hard_support(cid, kind, pars)
                  for weight, component, pars in zip(self._weights, self.components, self._shifted_parameters) if weight > 0]
        bounds = [bound for bound in bounds if bound is not None]
        return None if not bounds else (min(row[0] for row in bounds), max(row[1] for row in bounds))

    def sample(self, parameters, rng, n):
        self._ensure(parameters)
        selected = rng.choice(len(self.components), size=n, p=self._weights)
        scenes = [None] * n
        for index in np.unique(selected):
            positions = np.flatnonzero(selected == index)
            bank = self.components[index].sample(self._shifted_parameters[index], rng, len(positions))
            for position, scene in zip(positions, bank):
                scenes[position] = scene
        return scenes
