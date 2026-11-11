"""Second-clock ordered elapsed durations integrated over a20-second core grid.

The event grid retains the original60-second tails and all exact anchors.
Initial clocks use Gaussian Voronoi cell masses. Each ordered transition uses
the cell-integrated normal elapsed law conditional on elapsed>=0. This is a
finite-grid approximation; exact observed anchors still change its grid.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr

from .evolving_trip_belief import EvolvingTripBelief, _ChainLaw, logsumexp


def log_normal_interval(lower, upper, mean=0., sd=1.):
    """Stable log normal probability for finite or infinite interval endpoints."""
    low, high = np.broadcast_arrays((np.asarray(lower) - mean) / sd,
                                    (np.asarray(upper) - mean) / sd)
    # In the positive tail subtract survival probabilities; subtract CDFs in
    # the negative tail. Avoid1-CDF and subtraction of two rounded ones.
    positive = low >= 0
    large = np.where(positive, log_ndtr(-low), log_ndtr(high))
    small = np.where(positive, log_ndtr(-high), log_ndtr(low))
    result = np.full(low.shape, -np.inf)
    valid = (high > low) & np.isfinite(large)
    with np.errstate(divide='ignore', invalid='ignore'):
        difference = small[valid] - large[valid]
        result[valid] = large[valid] + np.log(-np.expm1(np.minimum(0., difference)))
    return result


def initial_clock_log_mass(clocks, mean, sd):
    """Voronoi masses, assigning both exterior tails to endpoint clocks."""
    middle = (clocks[:-1].astype(float) + clocks[1:]) / 2
    lower, upper = np.r_[-np.inf, middle], np.r_[middle, np.inf]
    probability = log_normal_interval(lower, upper, mean, sd)
    return probability - logsumexp(probability)


def elapsed_clock_log_mass(previous, following, mean, sd):
    """Rows of truncated elapsed-normal masses on allowed successor clocks.

    The first allowed cell starts at the predecessor clock, even when that
    clock is not itself a successor-grid point. The last cell ends at infinity.
    """
    previous, following = np.asarray(previous), np.asarray(following)
    middle = (following[:-1].astype(float) + following[1:]) / 2
    lower = np.broadcast_to(np.r_[-np.inf, middle], (len(previous), len(following))).copy()
    upper = np.broadcast_to(np.r_[middle, np.inf], lower.shape)
    first = np.searchsorted(following, previous, side='left')
    first_mask = np.arange(len(following))[None, :] == first[:, None]
    lower = np.where(first_mask, previous[:, None], lower)
    allowed = following[None, :] >= previous[:, None]
    mass = log_normal_interval(lower - previous[:, None], upper - previous[:, None], mean, sd)
    mass -= log_ndtr(mean / sd)  # P(elapsed>=0)
    mass = np.where(allowed, mass, -np.inf)
    normalizer = logsumexp(mass, axis=1, keepdims=True)
    return mass - np.where(np.isfinite(normalizer), normalizer, 0.)


class OrderedDurationBelief(EvolvingTripBelief):
    """Ordered elapsed-duration model; input clock ticks are integer seconds.

    Drift/sd and route mean/std keep the inherited *minute* parameter units.
    The clock conversion is fixed at60ticks/minute. The20second grid is chosen
    from training representability, not optimized on held-out decisions.
    """

    def __init__(self, graph, *, minute=60, **kwargs):
        if minute != 60:
            raise ValueError('OrderedDurationBelief uses second clocks: minute must equal60')
        super().__init__(graph, minute=minute, **kwargs)
        # Base density kernels and new CDF kernels must never share a cache.
        self._duration_transition_cache = {}

    def _grid(self, slot, shifts, anchors, cancellation_possible):
        shifts = np.unique(np.r_[shifts, np.arange(-300, 301, 20)]).astype(int)
        return super()._grid(slot, shifts, anchors, cancellation_possible)

    def _law(self, trip, parameters):
        common = super()._law(trip, parameters)
        if getattr(common, '_ordered_duration_cdf20', False):
            return common
        slots = self._trips[trip]
        route = self.graph.connections[slots[0].cid].route
        pars = parameters.get(route, {'mean': 1., 'std': 2., 'cancel_rate': .01})
        times, canceled, emissions = common.times, common.canceled, common.emissions
        q, persistence = pars['cancel_rate'], self.cancel_persistence
        unique = np.unique(times[0])
        log_mass = initial_clock_log_mass(unique, slots[0].scheduled + pars['mean'] * 60,
                                          max(pars['std'], .5) * 60)
        initial = log_mass[np.searchsorted(unique, times[0])]
        with np.errstate(divide='ignore'):
            initial += np.log(np.where(canceled[0], q, 1 - q))
        steps = []
        for index in range(1, len(slots)):
            travel = slots[index].kind == 'arrival'
            gap = (not travel and self.graph.connections[slots[index - 1].cid].arr_stop
                   != self.graph.connections[slots[index].cid].dep_stop)
            sd = 60 * (self.travel_sd if travel else self.gap_sd if gap else self.dwell_sd)
            drift = 60 * (self.travel_drift if travel else self.gap_drift if gap else self.dwell_drift)
            duration = slots[index].scheduled - slots[index - 1].scheduled
            steps.append((duration, travel, sd, drift))
        observed = any((slot.cid, slot.kind) in self._known_times for slot in slots)
        shared = None if observed else self._duration_transition_cache
        offsets = [tuple(clock - slot.scheduled) for clock, slot in zip(times, slots)]

        def transition_kernel(index):
            current = index + 1
            duration, travel, sd, drift = steps[index]
            key = (duration, travel, sd, drift, offsets[index], offsets[current], q, persistence)
            kernel = shared.get(key) if shared is not None else None
            if kernel is None:
                following = np.unique(times[current])
                clock_mass = elapsed_clock_log_mass(times[index], following, duration + drift, sd)
                kernel = clock_mass[:, np.searchsorted(following, times[current])]
                if travel:
                    service = (canceled[index][:, None] == canceled[current][None, :]).astype(float)
                else:
                    next_cancel = persistence * canceled[index][:, None] + (1 - persistence) * q
                    service = np.where(canceled[current][None, :], next_cancel, 1 - next_cancel)
                with np.errstate(divide='ignore'):
                    kernel += np.log(service)
                kernel.flags.writeable = False
                if shared is not None:
                    shared[key] = kernel
            return kernel

        law = _ChainLaw(times, initial, transition_kernel, emissions, canceled, common.stochastic)
        law._ordered_duration_cdf20 = True
        memo = self._cache[trip]
        memo.law, memo.events = law, {}
        return law
