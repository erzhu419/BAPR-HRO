"""Per-trip offset mixture matching the shared model's initial marginals."""
from collections.abc import Mapping

import numpy as np

from .evolving_trip_belief import logsumexp
from .joint_day_belief import JointDayBelief
from .simulator import LazyTripScenario


class _IndependentTripEvents(Mapping):
    def __init__(self, components, banks, selected, world):
        self.index = components[0]
        self.banks, self.selected, self.world = banks, selected, world

    def __getitem__(self, cid):
        trip = self.index._trip_by_cid[cid]
        component = self.selected[trip][self.world]
        return self.banks[component][self.world].events[cid]

    def __iter__(self):
        return iter(self.index._cid_order)

    def __len__(self):
        return len(self.index.graph.connections)


class IndependentDayMixtureBelief(JointDayBelief):
    """One independent offset per trip, persistent across that trip's events.

    Components, mean shifts, offset grid, prior, within-trip observation model
    and adaptive-clock approximation are inherited unchanged. Only offset
    coupling differs: a target trip's posterior uses its own likelihood.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trip_weights = {}

    def copy(self):
        branch = super().copy()
        branch._trip_weights = self._trip_weights.copy()
        return branch

    def _ensure(self, parameters):
        previous = self._parameter_key
        super()._ensure(parameters)
        if previous != self._parameter_key:
            self._trip_weights = {}

    def trip_posterior(self, cid, parameters):
        self._ensure(parameters)
        trip = self.components[0]._trip_by_cid[cid]
        if trip not in self._trip_weights:
            log_weights = np.log(self.prior_weights)
            if trip in self.components[0]._revision:
                for index, component in enumerate(self.components):
                    law = component._law(trip, self._shifted_parameters[index])
                    log_weights[index] += float(logsumexp(law.forward[-1]))
            weights = np.exp(log_weights - logsumexp(log_weights))
            weights.flags.writeable = False
            self._trip_weights[trip] = weights
        return self.deltas_seconds, self._trip_weights[trip]

    def day_posterior(self, parameters):
        self._ensure(parameters)
        return {trip: self.trip_posterior(slots[0].cid, parameters)[1]
                for trip, slots in self.components[0]._trips.items()}

    def event_distribution(self, cid, kind, parameters):
        _, weights = self.trip_posterior(cid, parameters)
        key = cid, kind
        if key not in self._event_cache:
            active, cancel = [], 0.
            for weight, component, pars in zip(weights, self.components, self._shifted_parameters):
                trip, index = component._positions[cid, kind]
                law = component._law(trip, pars)
                raw = law.marginal(index)
                operating = ~law.canceled[index]
                cancel += float(weight * raw[~operating].sum())
                mass = weight * raw[operating]
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

    def hard_support(self, cid, kind, parameters):
        _, weights = self.trip_posterior(cid, parameters)
        bounds = [component.hard_support(cid, kind, pars)
                  for weight, component, pars in zip(weights, self.components, self._shifted_parameters) if weight > 0]
        bounds = [bound for bound in bounds if bound is not None]
        return None if not bounds else (min(row[0] for row in bounds), max(row[1] for row in bounds))

    def sample(self, parameters, rng, n):
        self._ensure(parameters)
        selected = {trip: rng.choice(len(self.components), size=n, p=self.trip_posterior(slots[0].cid, parameters)[1])
                    for trip, slots in self.components[0]._trips.items()}
        banks = [component.sample(pars, rng, n) for component, pars in zip(self.components, self._shifted_parameters)]
        return [LazyTripScenario(_IndependentTripEvents(self.components, banks, selected, world), minute=self.minute,
                    publications=None if self.public_occurrences else {}) for world in range(n)]
