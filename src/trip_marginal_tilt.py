"""A persistent event-marginal tilt on one frozen posterior trip model."""
from collections.abc import Mapping
from copy import copy

import numpy as np

from .conditional_risk import retarget_event_chain
from .evolving_trip_belief import EvolvingTripBelief, _ChainLaw, _SampleBank, _LazyVehicleEvents, _ordered_path_exists
from .simulator import LazyTripScenario


class _Means(Mapping):
    def __init__(self, belief):
        self.belief = belief.copy()
        self.original = belief.root.mean_events(belief.parameters)
        self.cache = {}

    def __getitem__(self, cid):
        trip = self.belief._trip_by_cid[cid]
        if (trip not in self.belief._tilts and
                self.belief.base._revision.get(trip, 0) == self.belief.root._revision.get(trip, 0)):
            return self.original[cid]
        if cid not in self.cache:
            law = self.belief._law(trip, self.belief.parameters)
            self.cache[cid] = self.belief.base._mean_event(cid, law)
        return self.cache[cid]

    def __iter__(self):
        return iter(self.belief._cid_order)

    def __len__(self):
        return len(self.belief._cid_order)


class TripMarginalTilt:
    """Keep lambda and all initial grids fixed while conditioning branches.

    Parameters are fixed for this planning context. Exact evidence outside its
    root grids has zero probability and requests a new public-history context;
    it is not silently added as another prior atom. Creating another tilt at
    the next real query is a new posterior projection, not a claim of one
    globally Bayesian calibrated prior across all queries.
    """

    def __init__(self, base, parameters, cid, target_law=None):
        self.root, self.base = base.copy(), base.copy()
        self.parameters = {route: dict(value) for route, value in parameters.items()}
        for name in ('graph', 'minute', 'public_occurrences', '_trips', '_trip_by_cid', '_positions', '_cid_order'):
            setattr(self, name, getattr(base, name))
        targets = dict(cid) if isinstance(cid, Mapping) else {cid: target_law}
        self.targets, self._tilts = targets, {}
        self.applied_cids, self.skipped_cids = [], {}
        self.projected_marginals = {}
        seen_trips = set()
        for target, prediction in targets.items():
            trip, index = self._positions[target, 'departure']
            if trip in seen_trips:
                raise ValueError('A common posterior accepts one marginal target per trip')
            seen_trips.add(trip)
            root_law = self.root._law(trip, self.parameters)
            original = root_law.marginal(index)
            if (target, 'departure') in base._known_times:
                reason = 'known_departure'
            elif base._known_status.get(target) is True:
                reason = 'known_cancellation'
            elif base._excluded.get((target, 'departure')):
                reason = 'observed_attempt_window'
            elif not original[~root_law.canceled[index]].sum():
                reason = 'zero_operating_support'
            else:
                reason = None
            if reason:
                self.skipped_cids[target] = reason
                self.projected_marginals[target] = original.copy()
                continue
            projected, marginal = retarget_event_chain(root_law, index, prediction)
            ratio = np.zeros(len(root_law.times[index]))
            supported = np.isfinite(root_law.emissions[index])
            ratio[supported] = projected.emissions[index][supported] - root_law.emissions[index][supported]
            ratio.flags.writeable = False
            self._tilts[trip] = target, index, ratio
            self.applied_cids.append(target)
            self.projected_marginals[target] = marginal
        self.applied = bool(self.applied_cids)
        if len(targets) == 1:
            self.cid = next(iter(targets))
            self.target_trip, self.target_index = self._positions[self.cid, 'departure']
            self.target_marginal = self.projected_marginals[self.cid]
            self.log_ratio = self._tilts[self.target_trip][2] if self.target_trip in self._tilts else np.zeros_like(self.target_marginal)
        else:
            self.cid = self.target_trip = self.target_index = self.target_marginal = self.log_ratio = None
        self._laws = {}

    def _check_parameters(self, parameters):
        default = {'mean': 1., 'std': 2., 'cancel_rate': .01}
        for route in self.root._route_cids:
            actual = parameters.get(route, default)
            frozen = self.parameters.get(route, default)
            if any(actual[key] != frozen[key] for key in ('mean', 'std', 'cancel_rate')):
                raise ValueError('TripMarginalTilt fixes route parameters for one planning context')

    @property
    def _revision(self):
        return self.base._revision

    def copy(self):
        branch = copy(self)
        branch.base = self.base.copy()
        branch._laws = self._laws.copy()
        return branch

    def _law(self, trip, parameters):
        if parameters is not self.parameters:
            self._check_parameters(parameters)
        revision = self.base._revision.get(trip, 0)
        if trip in self._laws and self._laws[trip][0] == revision:
            return self._laws[trip][1]
        root = self.root._law(trip, self.parameters)
        if revision == self.root._revision.get(trip, 0):
            emissions = list(root.emissions)
        else:
            current = self.base._law(trip, self.parameters)
            emissions = []
            for times, states, actual_times, actual_states, emission in zip(
                    root.times, root.canceled, current.times, current.canceled, current.emissions):
                mapping = {(int(t), bool(s)): i for i, (t, s) in enumerate(zip(actual_times, actual_states))}
                indices = [mapping[int(t), bool(s)] for t, s in zip(times, states)]
                emissions.append(emission[indices])
        if trip in self._tilts:
            _, index, ratio = self._tilts[trip]
            emissions[index] = emissions[index] + ratio
        slots = self._trips[trip]
        route = self.graph.connections[slots[0].cid].route
        q = self.parameters.get(route, {'cancel_rate': .01})['cancel_rate']
        persistence = self.base.cancel_persistence
        service = np.array([[1 - (1 - persistence) * q, (1 - persistence) * q],
                            [(1 - persistence) * (1 - q), persistence + (1 - persistence) * q]]) > 0
        steps = [np.eye(2, dtype=bool) if slot.kind == 'arrival' else service for slot in slots[1:]]
        if not _ordered_path_exists(root.times, root.canceled, emissions, root.initial, steps):
            raise ValueError('New evidence has zero mass on the frozen planning grid; open a new context')
        law = _ChainLaw(root.times, root.initial, root._transition, emissions, root.canceled, root.stochastic)
        self._laws[trip] = revision, law
        return law

    def _observe(self, method, *args):
        getattr(self.base, method)(*args)

    def observe_connection_event(self, cid, kind, delay):
        self._observe('observe_connection_event', cid, kind, delay)

    def observe_poll(self, stop, time, messages, seen_before):
        self._observe('observe_poll', stop, time, messages, seen_before)

    def observe_forecast(self, cid, kind, predicted_time, sd_seconds):
        self._observe('observe_forecast', cid, kind, predicted_time, sd_seconds)

    def observe_attempt(self, cid, kind, time):
        self._observe('observe_attempt', cid, kind, time)

    def observe_wait(self, cid, start, end):
        self._observe('observe_wait', cid, start, end)

    def event_distribution(self, cid, kind, parameters):
        self._check_parameters(parameters)
        trip, index = self._positions[cid, kind]
        return self.base._event_distribution(self._law(trip, self.parameters), index)

    _event_distribution = staticmethod(EvolvingTripBelief._event_distribution)

    def _mean_event(self, cid, law):
        return self.base._mean_event(cid, law)

    def mean_events(self, parameters):
        self._check_parameters(parameters)
        return _Means(self)

    def sample(self, parameters, rng, n):
        self._check_parameters(parameters)
        for trip in self.base._revision:
            self._law(trip, self.parameters)
        uniforms = {}
        for trip, slots in self._trips.items():
            values = rng.random((n, len(slots)))[:, ::-1]
            values.flags.writeable = False
            uniforms[trip] = values
        bank = _SampleBank(self.copy(), self.parameters, uniforms)
        return [LazyTripScenario(_LazyVehicleEvents(bank, world), minute=self.minute,
                                 publications=None if self.public_occurrences else {}) for world in range(n)]

    def hard_support(self, cid, kind, parameters):
        self._check_parameters(parameters)
        original = self.base.hard_support(cid, kind, self.parameters)
        if original is None:
            return None
        trip, index = self._positions[cid, kind]
        root = self.root._law(trip, self.parameters)
        clocks = root.times[index][~root.canceled[index]]
        low, high = max(original[0], int(clocks.min())), min(original[1], int(clocks.max()))
        return (low, high) if low <= high else None

    def projection_summary(self, ready_by_cid, deadline=None):
        """Report continuous versus this common posterior's projected risk."""
        result = {}
        for cid, prediction in self.targets.items():
            trip, index = self._positions[cid, 'departure']
            law = self._law(trip, self.parameters)
            operating = ~law.canceled[index]
            marginal = law.marginal(index)
            active_mass = float(marginal[operating].sum())
            clocks, weights = law.times[index][operating], marginal[operating]
            ready = ready_by_cid[cid]
            end = ready + 12 * self.minute
            if deadline is not None:
                end = min(end, deadline)
            valid = ready <= end and (deadline is None or ready < deadline)
            projected = float(weights[(clocks >= ready) & (clocks <= end)].sum()) if valid else 0.
            continuous = active_mass * (prediction.cdf(end) - prediction.cdf(ready)) if valid else 0.
            conditional = weights / active_mass if active_mass else np.array([])
            cumulative = conditional.cumsum()
            quantile = lambda p: float(clocks[min(np.searchsorted(cumulative, p), len(clocks) - 1)]) if active_mass else None
            result[cid] = {'applied': cid in self.applied_cids, 'skipped_reason': self.skipped_cids.get(cid),
                'operating_mass': active_mass, 'continuous_catch_with_base_service': continuous,
                'projected_catch': projected, 'catch_difference': projected - continuous,
                'continuous_lower': prediction.quantile(.05), 'continuous_upper': prediction.quantile(.95),
                'projected_lower': quantile(.05), 'projected_upper': quantile(.95)}
        return result
