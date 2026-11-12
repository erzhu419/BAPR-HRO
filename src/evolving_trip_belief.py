"""A causal finite-grid model with distinct departure and arrival delays.

Each trip has ordered potential event clocks and a correlated segment-service
chain. Delays evolve through running, dwell and omitted-station gaps. Exact
receipts, issued forecasts and first-hand wait/ride evidence condition that
joint chain. Service can stop and resume within a trip.
"""
from __future__ import annotations

from copy import copy
from bisect import bisect_right
from collections.abc import Mapping
import ctypes
from dataclasses import dataclass, field
from functools import cached_property
import math

import numpy as np

from .simulator import LazyTripScenario, VehicleEvent
from .trip_belief import DELAYS, NominalEvent


_CHAIN_SAMPLER = None


def install_chain_sampler(kernel):
    """Use one explicitly compiled sampler for every law in this process."""
    global _CHAIN_SAMPLER
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    pointers = np.ctypeslib.ndpointer(dtype=np.uintp, flags='C_CONTIGUOUS')
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    indices = np.ctypeslib.ndpointer(dtype=np.intp, flags='C_CONTIGUOUS')
    kernel.chain_draw.argtypes = [ctypes.c_int, ctypes.c_int, i32, pointers, pointers, f64, indices]
    kernel.chain_draw.restype = None
    _CHAIN_SAMPLER = kernel


def logsumexp(values, axis=None, keepdims=False):
    """Stable reduction of real log weights, including impossible (-inf) rows."""
    peak = np.max(values, axis=axis, keepdims=True)
    anchor = np.where(np.isfinite(peak), peak, 0.)
    with np.errstate(divide='ignore'):
        reduced = np.log(np.exp(values - anchor).sum(axis=axis, keepdims=True)) + anchor
    return reduced if keepdims else np.squeeze(reduced, axis=axis)


def _ordered_path_exists(times, canceled, emissions, initial, service_steps):
    """Positive Gaussian paths need only the earliest clock of each service state."""
    earliest = [math.inf, math.inf]
    for i, (clocks, states, emission) in enumerate(zip(times, canceled, emissions)):
        allowed = np.isfinite(emission)
        if i == 0:
            allowed &= np.isfinite(initial)
        next_earliest = [math.inf, math.inf]
        for state in (0, 1):
            threshold = (-math.inf if i == 0 else
                         min((earliest[previous] for previous in (0, 1)
                              if service_steps[i - 1][previous, state]), default=math.inf))
            reachable = allowed & (states == state) & (clocks >= threshold)
            if reachable.any():
                next_earliest[state] = int(clocks[reachable].min())
        earliest = next_earliest
        if min(earliest) == math.inf:
            return False
    return True


@dataclass(frozen=True)
class _Event:
    cid: int
    kind: str
    scheduled: int


@dataclass
class _ChainLaw:
    times: list
    initial: np.ndarray
    transition_factory: object
    emissions: list
    canceled: list
    stochastic: tuple

    def __post_init__(self):
        self._forward = [self.initial + self.emissions[0]]
        self._transitions = [None] * (len(self.times) - 1)
        self._conditional_forward = {}
        self._backward = {}
        self._probabilities = {}
        self._exact = {}
        for i, emission in enumerate(self.emissions):
            finite = np.flatnonzero(np.isfinite(emission))
            if len(finite) == 1:
                self._exact[i] = int(finite[0])
        self._exact_slots = tuple(self._exact)
        self._last_effect = max([0, *(i for i, emission in enumerate(self.emissions)
                                     if np.any(emission != 0)),
                                *(i + 1 for i, normalized in enumerate(self.stochastic) if not normalized)])

    def _transition(self, index):
        if self._transitions[index] is None:
            self._transitions[index] = self.transition_factory(index)
        return self._transitions[index]

    def _full_forward(self, index):
        while len(self._forward) <= index:
            i = len(self._forward)
            self._forward.append(logsumexp(self._forward[-1][:, None] + self._transition(i - 1), axis=0)
                                 + self.emissions[i])
        return self._forward[index]

    @property
    def forward(self):
        self._full_forward(len(self.times) - 1)
        return self._forward

    @cached_property
    def transitions(self):
        return [self._transition(i) for i in range(len(self.times) - 1)]

    def _forward_from(self, left, index):
        if left is None:
            return self._full_forward(index)
        if left not in self._conditional_forward:
            point = np.full(len(self.times[left]), -np.inf)
            point[self._exact[left]] = 0.
            self._conditional_forward[left] = [point]
        values = self._conditional_forward[left]
        while len(values) <= index - left:
            i = left + len(values)
            values.append(logsumexp(values[-1][:, None] + self._transition(i - 1), axis=0)
                          + self.emissions[i])
        return values[index - left]

    def _backward_from(self, end, index):
        values = self._backward.setdefault(end, [np.zeros(len(self.times[end]))])
        while len(values) <= end - index:
            i = end - len(values)
            values.append(logsumexp(self._transition(i) + self.emissions[i + 1] + values[-1], axis=1))
        return values[end - index]

    def marginal(self, index):
        if index not in self._probabilities:
            if index in self._exact:
                probabilities = np.zeros(len(self.times[index]))
                probabilities[self._exact[index]] = 1.
            else:
                position = bisect_right(self._exact_slots, index)
                left = self._exact_slots[position - 1] if position else None
                end = self._exact_slots[position] if position < len(self._exact_slots) else max(index, self._last_effect)
                weights = self._forward_from(left, index) + self._backward_from(end, index)
                probabilities = np.exp(weights - logsumexp(weights))
            self._probabilities[index] = probabilities
        return self._probabilities[index]

    @cached_property
    def sampling_cdfs(self):
        return {}

    @cached_property
    def marginals(self):
        return [self.marginal(i) for i in range(len(self.times))]

    def draw(self, rng):
        return self.draw_uniforms(rng.random(len(self.times))[::-1])

    def draw_uniforms(self, uniforms):
        return self.draw_uniforms_batch(np.asarray(uniforms)[None, :])[0]

    @cached_property
    def sampling_arrays(self):
        # OrderedDurationBelief's column indexing produces Fortran-order kernels.
        # Retain canonical buffers: pointer tables alone would not own copies.
        forward = [np.ascontiguousarray(a, dtype=np.float64) for a in self.forward]
        transitions = [np.ascontiguousarray(a, dtype=np.float64) for a in self.transitions]
        self._sampling_buffers = forward, transitions
        return (np.asarray([len(t) for t in self.times], dtype=np.int32),
                np.asarray([a.ctypes.data for a in forward], dtype=np.uintp),
                np.asarray([a.ctypes.data for a in transitions], dtype=np.uintp))

    def draw_uniforms_batch(self, uniforms):
        """FFBS with the same supplied tape for scalar and batch consumers."""
        n, slots = uniforms.shape
        if _CHAIN_SAMPLER is not None:
            result = np.empty((n, slots), dtype=np.intp)
            _CHAIN_SAMPLER.chain_draw(n, slots, *self.sampling_arrays,
                                     np.ascontiguousarray(uniforms, dtype=np.float64), result)
            return result
        if n == 1:
            return self._draw_uniforms_python(uniforms[0])[None, :]
        result = np.empty((n, slots), dtype=np.intp)
        for index in range(slots - 1, -1, -1):
            forward = self._full_forward(index)
            if index == slots - 1:
                weights = forward[None, :]
                inverse = np.zeros(n, dtype=np.intp)
            else:
                next_state, inverse = np.unique(result[:, index + 1], return_inverse=True)
                weights = forward + self._transition(index)[:, next_state].T
            probabilities = np.exp(weights - logsumexp(weights, axis=1, keepdims=True))
            cumulative = np.cumsum(probabilities, axis=1)
            cumulative /= cumulative[:, -1:]
            result[:, index] = np.sum(uniforms[:, index, None] >= cumulative[inverse], axis=1)
        return result

    def _draw_uniforms_python(self, uniforms):
        chosen = np.empty(len(self.times), dtype=int)
        next_index = None
        for i in range(len(chosen) - 1, -1, -1):
            key = i, next_index
            cumulative = self.sampling_cdfs.get(key)
            if cumulative is None:
                weights = self._full_forward(i)
                if next_index is not None:
                    weights = weights + self._transition(i)[:, next_index]
                probabilities = np.exp(weights - logsumexp(weights))
                cumulative = probabilities.cumsum()
                cumulative /= cumulative[-1]
                cumulative.flags.writeable = False
                self.sampling_cdfs[key] = cumulative
            next_index = int(cumulative.searchsorted(uniforms[i], side="right"))
            chosen[i] = next_index
        return chosen


@dataclass
class _TripMemo:
    key: tuple | None = None
    law: _ChainLaw | None = None
    events: dict = field(default_factory=dict)
    support: tuple | None = None


class _NominalEvents(Mapping):
    def __init__(self, belief, parameters):
        self._belief, self._parameters = belief, parameters
        self._events = belief._flat_event_cache
        self._trip_events = {}

    def __getitem__(self, cid):
        event = self._events.get(cid)
        if event is None:
            trip = self._belief._trip_by_cid[cid]
            if trip not in self._trip_events:
                law = self._belief._law(trip, self._parameters)
                self._trip_events[trip] = law, self._belief._cache[trip].events
            law, events = self._trip_events[trip]
            event = events.get(cid)
            if event is None:
                event = self._belief._mean_event(cid, law)
                events[cid] = event
            self._events[cid] = event
        return event

    def __iter__(self):
        return iter(self._belief._cid_order)

    def __len__(self):
        return len(self._belief._cid_order)


class _SampleBank:
    def __init__(self, belief, parameters, uniforms):
        self.belief, self.parameters, self.uniforms = belief, parameters, uniforms
        self._sampled = {}

    def trip_events(self, trip):
        if trip not in self._sampled:
            law = self.belief._law(trip, self.parameters)
            slots = self.belief._trips[trip]
            worlds = []
            for indices in law.draw_uniforms_batch(self.uniforms[trip]):
                events = {}
                for i in range(0, len(slots), 2):
                    dep, arr = slots[i:i + 2]
                    canceled = bool(law.canceled[i][indices[i]])
                    departure, arrival = int(law.times[i][indices[i]]), int(law.times[i + 1][indices[i + 1]])
                    if canceled and self.belief.public_occurrences:
                        departure, arrival = dep.scheduled, arr.scheduled
                    events[dep.cid] = VehicleEvent(999 if canceled else departure - dep.scheduled,
                                                  canceled, departure, arrival)
                worlds.append(events)
            self._sampled[trip] = worlds
        return self._sampled[trip]


class _LazyVehicleEvents(Mapping):
    def __init__(self, bank, world):
        self._bank, self._world = bank, world

    def __getitem__(self, cid):
        trip = self._bank.belief._trip_by_cid[cid]
        return self._bank.trip_events(trip)[self._world][cid]

    def __iter__(self):
        return iter(self._bank.belief._cid_order)

    def __len__(self):
        return len(self._bank.belief._cid_order)


class EvolvingTripBelief:
    """Adaptive-grid inference for ordered clocks and segment availability.

    ``public_occurrences=True`` assumes immediate local actual publications.
    Feed absence is informative under that synthetic contract. Delayed or
    incomplete feeds use False, retaining only received facts and ride/wait
    evidence. Off-grid exact observations augment the integration grid, so the
    physical continuous-clock posterior is approximated.

    The default grid uses the historical discrete Gaussian weights. An
    explicit ``grid_step_seconds`` uses cell probabilities from a bounded
    ordered Gaussian reference on the same physical support. Route and
    transition parameters remain in minutes and service laws stay fixed.
    """
    def __init__(self, graph, *, minute=60, travel_sd=2.0, dwell_sd=1.0,
                 travel_drift=0.0, dwell_drift=0.0, public_occurrences=True,
                 cancel_persistence=.9, gap_sd=None, gap_drift=None, grid_step_seconds=None):
        self.graph, self.minute = graph, minute
        self._validate_grid_step(grid_step_seconds)
        self.grid_step_seconds = grid_step_seconds
        self._trip_grid_steps = {}
        self.travel_sd, self.dwell_sd = travel_sd, dwell_sd
        self.travel_drift, self.dwell_drift = travel_drift, dwell_drift
        self.gap_sd = travel_sd if gap_sd is None else gap_sd
        self.gap_drift = travel_drift if gap_drift is None else gap_drift
        self.public_occurrences = public_occurrences
        self.cancel_persistence = cancel_persistence
        self._trip_by_cid = {}
        self._trips = {}
        self._known_status = {}
        self._known_times = {}
        self._forecasts = {}
        self._excluded = {}
        self._cannot_cancel = set()
        self._revision = {}
        self._cache = {}
        self._flat_event_cache = {}
        self._flat_parameters = {}
        self._transition_cache = {}
        self._positions = {}
        self._route_cids = {}
        for c in sorted(graph.connections, key=lambda c: (c.dep_time, c.arr_time, c.id)):
            trip = c.trip_id or f"connection:{c.id}"
            self._trip_by_cid[c.id] = trip
            self._trips.setdefault(trip, []).extend((_Event(c.id, "departure", c.dep_time),
                                                     _Event(c.id, "arrival", c.arr_time)))
            self._route_cids.setdefault(c.route, []).append(c.id)
        for trip, slots in self._trips.items():
            self._positions.update({(event.cid, event.kind): (trip, i) for i, event in enumerate(slots)})
        self._cid_order = tuple(slots[i].cid for slots in self._trips.values()
                                for i in range(0, len(slots), 2))
        self._cache = {trip: _TripMemo() for trip in self._trips}

    def _validate_grid_step(self, step):
        if step is not None and (self.minute != 60 or not isinstance(step, (int, np.integer))
                                 or isinstance(step, bool) or step <= 0):
            raise ValueError('The explicit clock grid needs second clocks and a positive integer step')

    def grid_step_for_trip(self, trip):
        return self._trip_grid_steps.get(trip, self.grid_step_seconds)

    def with_grid_step(self, grid_step_seconds, *, trips=None):
        """Fork the belief with new quadrature on all or specified trip IDs."""
        self._validate_grid_step(grid_step_seconds)
        branch = self.copy()
        selected = set(self._trips) if trips is None else set(trips)
        unknown = selected.difference(self._trips)
        if unknown:
            raise KeyError(f'Unknown trip IDs: {sorted(unknown)}')
        if trips is None:
            branch.grid_step_seconds = grid_step_seconds
            branch._trip_grid_steps = {}
        else:
            branch._trip_grid_steps = dict(self._trip_grid_steps)
            branch._trip_grid_steps.update(dict.fromkeys(selected, grid_step_seconds))
        changed = [trip for trip in selected if self.grid_step_for_trip(trip) != grid_step_seconds]
        if changed:
            branch._flat_event_cache = self._flat_event_cache.copy()
            for trip in changed:
                branch._cache[trip] = _TripMemo()
                for slot in self._trips[trip][::2]:
                    branch._flat_event_cache.pop(slot.cid, None)
        return branch

    def copy(self):
        branch = copy(self)
        for attr in ("_known_status", "_known_times", "_forecasts", "_excluded", "_cannot_cancel", "_revision", "_cache"):
            setattr(branch, attr, getattr(self, attr).copy())
        return branch

    def _touch(self, trip):
        self._revision[trip] = self._revision.get(trip, 0) + 1
        self._cache[trip] = _TripMemo()
        self._flat_event_cache = self._flat_event_cache.copy()
        for slot in self._trips[trip][::2]:
            self._flat_event_cache.pop(slot.cid, None)

    def _exclude(self, cid, kind, start, end):
        if start > end:
            return
        key = cid, kind
        previous = self._excluded.get(key, ())
        if any(lo <= start and end <= hi for lo, hi in previous):
            return
        ranges = []
        for lo, hi in sorted((*previous, (start, end))):
            if ranges and lo <= ranges[-1][1] + 1:
                ranges[-1] = (ranges[-1][0], max(hi, ranges[-1][1]))
            else:
                ranges.append((lo, hi))
        self._excluded[key] = tuple(ranges)
        self._touch(self._trip_by_cid[cid])

    def _rule_out_cancellation(self, cid):
        if cid not in self._cannot_cancel:
            self._cannot_cancel.add(cid)
            self._touch(self._trip_by_cid[cid])

    def observe_connection_event(self, cid, kind, delay):
        trip = self._trip_by_cid[cid]
        canceled = kind == "cancel"
        status = self._known_status.get(cid)
        if status is not None and status != canceled:
            raise ValueError(f"Contradictory service status for connection {cid}")
        if status is None:
            self._known_status[cid] = canceled
            self._touch(trip)
        if canceled:
            return
        _, index = self._positions[cid, kind]
        time = self._trips[trip][index].scheduled + delay
        old = self._known_times.get((cid, kind))
        if old is not None and old != time:
            raise ValueError(f"Conflicting clocks for event {cid}:{kind}")
        if old is None:
            self._known_times[cid, kind] = time
            self._touch(trip)

    def observe_poll(self, stop, time, messages, seen_before):
        received = set()
        order = []
        for message in messages:
            cid_text, kind = message.event_id.split(":", 1)
            cid = int(cid_text)
            self.observe_connection_event(cid, kind, message.actual_delay)
            received.add(message.event_id)
            order.append((message.actual_time, cid, kind))
        if not self.public_occurrences:
            return
        cutoff = max(order) if len(order) == 5 else None
        window_start = time - 10 * self.minute
        accounted = seen_before | received
        for kind, connections in (("departure", self.graph.get_connections_from(stop)),
                                  ("arrival", self.graph.get_connections_to(stop))):
            for c in connections:
                trip = self._trip_by_cid[c.id]
                if self._known_status.get(c.id) is True:
                    continue
                if f"{c.id}:{kind}" not in accounted:
                    upper = time if cutoff is None else cutoff[0] - int((c.id, kind) > cutoff[1:])
                    self._exclude(c.id, kind, window_start, upper)
                if kind == "departure" and f"{c.id}:cancel" not in accounted:
                    key = c.dep_time, c.id, "cancel"
                    if window_start <= c.dep_time <= time and (cutoff is None or key <= cutoff):
                        self._rule_out_cancellation(c.id)

    def observe_forecast(self, cid, kind, predicted_time, sd_seconds):
        key = cid, kind
        value = None if predicted_time is None else (predicted_time, max(sd_seconds, 1.))
        if self._forecasts.get(key) == value:
            return
        if value is None:
            self._forecasts.pop(key, None)
        else:
            self._forecasts[key] = value
        self._touch(self._trip_by_cid[cid])

    def observe_attempt(self, cid, kind, time):
        if kind == "missed":
            self._exclude(cid, "departure", time, math.inf)
            self._rule_out_cancellation(cid)
        elif kind == "timeout":
            self._exclude(cid, "departure", -math.inf, time)
            if self.graph.connections[cid].dep_time <= time:
                self._rule_out_cancellation(cid)
        else:
            raise ValueError("Attempt feedback must be missed or timeout")

    def observe_wait(self, cid, start, end):
        if self.public_occurrences:
            self.observe_attempt(cid, "timeout", end)
        else:
            # No vehicle was seen during this commitment. It could already
            # have departed, still be in the future, or be canceled.
            self._exclude(cid, "departure", start, end)

    def _grid(self, slot, shifts, anchors, cancellation_possible):
        times = np.unique(np.concatenate((slot.scheduled + shifts, anchors)))
        clocks = np.tile(times, 2) if cancellation_possible else times
        canceled = (np.repeat([False, True], len(times)) if cancellation_possible
                    else np.zeros(len(times), dtype=bool))
        allowed = np.ones(len(clocks), dtype=bool)
        key = slot.cid, slot.kind
        for lo, hi in self._excluded.get(key, ()):
            allowed &= canceled | (clocks < lo) | (clocks > hi)
        if key in self._known_times:
            allowed &= ~canceled & (clocks == self._known_times[key])
        status = self._known_status.get(slot.cid)
        if status is not None:
            allowed &= canceled == status
        if slot.cid in self._cannot_cancel:
            allowed &= ~canceled
        forecast = self._forecasts.get(key)
        emission = np.zeros(len(clocks)) if forecast is None else -.5 * ((clocks - forecast[0]) / forecast[1]) ** 2
        return clocks, canceled, np.where(allowed, emission, -np.inf)

    def _law(self, trip, parameters):
        slots = self._trips[trip]
        route = self.graph.connections[slots[0].cid].route
        pars = parameters.get(route, {"mean": 1., "std": 2., "cancel_rate": .01})
        grid_step = self.grid_step_for_trip(trip)
        key = self._revision.get(trip, 0), pars["mean"], pars["std"], pars["cancel_rate"], grid_step
        memo = self._cache[trip]
        if memo.key == key:
            return memo.law
        observed = [(s, self._known_times[s.cid, s.kind]) for s in slots
                    if (s.cid, s.kind) in self._known_times]
        unit = self.minute
        fixed_shifts = DELAYS * unit
        if grid_step is not None:
            from .clock_cells import bounded_clock_log_mass, bounded_elapsed_log_mass

            low, high = fixed_shifts[0], fixed_shifts[-1]
            fixed_shifts = np.r_[np.arange(low, high + 1, grid_step), high]
        shifts = np.unique(np.r_[fixed_shifts, [t - s.scheduled for s, t in observed]]).astype(int)
        anchors = np.asarray([t for _, t in observed], dtype=int)
        q = pars["cancel_rate"]
        times, canceled, emissions = zip(*(self._grid(slot, shifts, anchors, q != 0) for slot in slots))
        delays = [(t - s.scheduled) / unit for t, s in zip(times, slots)]
        if grid_step is None:
            initial = -.5 * ((delays[0] - pars["mean"]) / max(pars["std"], .5)) ** 2
        else:
            clocks = times[0][~canceled[0]]
            mass = bounded_clock_log_mass(clocks, slots[0].scheduled + pars["mean"] * unit,
                                          max(pars["std"], .5) * unit)
            initial = mass[np.searchsorted(clocks, times[0])]
        initial -= logsumexp(initial[~canceled[0]])
        with np.errstate(divide="ignore"):
            initial += np.log(np.where(canceled[0], q, 1 - q))
        steps, service_steps = [], []
        persistence = self.cancel_persistence
        for i in range(1, len(slots)):
            travel = slots[i].kind == "arrival"
            gap = (not travel and self.graph.connections[slots[i - 1].cid].arr_stop
                   != self.graph.connections[slots[i].cid].dep_stop)
            sd = self.travel_sd if travel else self.gap_sd if gap else self.dwell_sd
            drift = self.travel_drift if travel else self.gap_drift if gap else self.dwell_drift
            steps.append((slots[i].scheduled - slots[i - 1].scheduled, travel, sd, drift))
            service_steps.append(np.eye(2, dtype=bool) if travel else np.array([
                [1 - (1 - persistence) * q, (1 - persistence) * q],
                [(1 - persistence) * (1 - q), persistence + (1 - persistence) * q]]) > 0)
        if not _ordered_path_exists(times, canceled, emissions, initial, service_steps):
            raise ValueError(f"Observations leave no ordered trip event path for {trip}")

        # The factory retains fixed grid metadata and a pure kernel dictionary,
        # without retaining the belief or its hypothetical public histories.
        shared_cache = None if observed else self._transition_cache

        def transition_kernel(index):
            i = index + 1
            delta, travel, sd, drift = steps[index]
            kernel_key = (grid_step, delta, travel, sd, drift,
                          tuple(delays[i - 1]), tuple(delays[i]),
                          None if travel else (q, persistence))
            # Shared kernels serve fixed, unobserved grids. Observed-clock
            # kernels belong to their law, so discarded hypothetical paths
            # release their anchor-specific matrices.
            log_step = shared_cache.get(kernel_key) if shared_cache is not None else None
            if log_step is None:
                if grid_step is None:
                    log_step = -.5 * ((delays[i][None, :] - delays[i - 1][:, None] - drift) / sd) ** 2
                    ordered = times[i][None, :] >= times[i - 1][:, None]
                    log_step = np.where(ordered, log_step, -np.inf)
                else:
                    previous = times[i - 1][~canceled[i - 1]]
                    following = times[i][~canceled[i]]
                    mass = bounded_elapsed_log_mass(previous, following, delta + drift * unit, sd * unit)
                    log_step = mass[np.ix_(np.searchsorted(previous, times[i - 1]),
                                           np.searchsorted(following, times[i]))]
                # Normalize potential-clock transitions once per availability state.
                norm = logsumexp(log_step[:, ~canceled[i]], axis=1, keepdims=True)
                log_step -= np.where(np.isfinite(norm), norm, 0.0)
                if travel:
                    status_probability = (canceled[i - 1][:, None] == canceled[i][None, :]).astype(float)
                else:
                    next_cancel = persistence * canceled[i - 1][:, None] + (1 - persistence) * q
                    status_probability = np.where(canceled[i][None, :], next_cancel, 1 - next_cancel)
                with np.errstate(divide="ignore"):
                    log_step += np.log(status_probability)
                log_step.flags.writeable = False
                if shared_cache is not None:
                    shared_cache[kernel_key] = log_step
            return log_step

        stochastic = tuple(bool(times[i].max() >= times[i - 1].max()) for i in range(1, len(slots)))
        law = _ChainLaw(times, initial, transition_kernel, emissions, canceled, stochastic)
        memo.key, memo.law, memo.events = key, law, {}
        return law

    def event_distribution(self, cid, kind, parameters):
        """Conditional operating clock distribution and cancellation mass."""
        trip, i = self._positions[cid, kind]
        law = self._law(trip, parameters)
        return self._event_distribution(law, i)

    @staticmethod
    def _event_distribution(law, i):
        operating = ~law.canceled[i]
        marginal = law.marginal(i)
        weights = marginal[operating]
        cancel = float(marginal[~operating].sum())
        if weights.sum() == 0:
            return np.empty(0, dtype=int), np.empty(0), 1.
        return law.times[i][operating], weights / weights.sum(), cancel

    def hard_support(self, cid, kind, parameters):
        """Conservative operating-clock bounds, including adaptive grid anchors."""
        trip, index = self._positions[cid, kind]
        slots = self._trips[trip]
        route = self.graph.connections[slots[0].cid].route
        if self._known_status.get(cid) is True or parameters.get(route, {}).get("cancel_rate") == 1:
            return None
        memo = self._cache[trip]
        if memo.support is None:
            observed = [(i, slot, self._known_times[slot.cid, slot.kind])
                        for i, slot in enumerate(slots) if (slot.cid, slot.kind) in self._known_times]
            shifts = [clock - slot.scheduled for _, slot, clock in observed]
            anchors = [clock for _, _, clock in observed]
            prefix, suffix = [], [math.inf] * len(slots)
            lower = -math.inf
            for slot in slots:
                lower = max(lower, self._known_times.get((slot.cid, slot.kind), -math.inf))
                prefix.append(lower)
            upper = math.inf
            for i in range(len(slots) - 1, -1, -1):
                slot = slots[i]
                upper = min(upper, self._known_times.get((slot.cid, slot.kind), math.inf))
                suffix[i] = upper
            memo.support = (min([-5 * self.minute, *shifts]), max([64 * self.minute, *shifts]),
                            min(anchors, default=math.inf), max(anchors, default=-math.inf), prefix, suffix)
        shift_low, shift_high, anchor_low, anchor_high, prefix, suffix = memo.support
        scheduled = slots[index].scheduled
        lower = max(min(scheduled + shift_low, anchor_low), prefix[index])
        upper = min(max(scheduled + shift_high, anchor_high), suffix[index])
        for start, end in self._excluded.get((cid, kind), ()):
            if start <= lower <= end:
                lower = end + 1
            if start <= upper <= end:
                upper = start - 1
        return (int(lower), int(upper)) if lower <= upper else None

    def sample(self, parameters, rng, n):
        parameters = {route: dict(row) for route, row in parameters.items()}
        for trip in self._revision:
            self._law(trip, parameters)
        uniforms = {}
        for trip, slots in self._trips.items():
            values = rng.random((n, len(slots)))[:, ::-1]
            values.flags.writeable = False
            uniforms[trip] = values
        bank = _SampleBank(self.copy(), parameters, uniforms)
        return [LazyTripScenario(_LazyVehicleEvents(bank, world), minute=self.minute,
                                 publications=None if self.public_occurrences else {}) for world in range(n)]

    def mean_events(self, parameters):
        parameters = {route: dict(row) for route, row in parameters.items()}
        for trip in self._revision:
            self._law(trip, parameters)
        keys = {}
        for route in self._route_cids:
            row = parameters.get(route, {"mean": 1., "std": 2., "cancel_rate": .01})
            keys[route] = row["mean"], row["std"], row["cancel_rate"]
        changed = [route for route, key in keys.items() if self._flat_parameters.get(route) != key]
        if changed:
            self._flat_event_cache = self._flat_event_cache.copy()
            self._flat_parameters = keys
            for route in changed:
                for cid in self._route_cids[route]:
                    self._flat_event_cache.pop(cid, None)
        return _NominalEvents(self.copy(), parameters)

    def _mean_event(self, cid, law):
        trip, index = self._positions[cid, "departure"]
        dep_times, dep_weights, cancel = self._event_distribution(law, index)
        arr_times, arr_weights, _ = self._event_distribution(law, index + 1)
        if cancel == 1.:
            return NominalEvent(None, None, None, cancel)
        dep, arr = float(dep_times @ dep_weights), float(arr_times @ arr_weights)
        return NominalEvent(dep - self._trips[trip][index].scheduled, dep, arr, cancel)
