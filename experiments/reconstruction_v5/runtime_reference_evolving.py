"""A causal finite-grid model with distinct departure and arrival delays.

Each trip has ordered potential event clocks and a correlated segment-service
chain. Delays evolve through running, dwell and omitted-station gaps. Exact
receipts, issued forecasts and first-hand wait/ride evidence condition that
joint chain. Service can stop and resume within a trip.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np

from src.simulator import JourneyScenario, VehicleEvent
from src.trip_belief import DELAYS, NominalEvent


def logsumexp(values, axis=None, keepdims=False):
    """Stable reduction of real log weights, including impossible (-inf) rows."""
    peak = np.max(values, axis=axis, keepdims=True)
    anchor = np.where(np.isfinite(peak), peak, 0.)
    with np.errstate(divide='ignore'):
        reduced = np.log(np.exp(values - anchor).sum(axis=axis, keepdims=True)) + anchor
    return reduced if keepdims else np.squeeze(reduced, axis=axis)


@dataclass(frozen=True)
class _Event:
    cid: int
    kind: str
    scheduled: int


@dataclass
class _ChainLaw:
    times: list
    forward: list
    transitions: list
    emissions: list
    canceled: list

    @cached_property
    def sampling_weights(self):
        return {}

    @cached_property
    def marginals(self):
        backwards = np.zeros(len(self.times[-1]))
        probabilities = [None] * len(self.times)
        for i in range(len(self.times) - 1, -1, -1):
            weights = self.forward[i] + backwards
            probabilities[i] = np.exp(weights - logsumexp(weights))
            if i:
                backwards = logsumexp(self.transitions[i - 1] + self.emissions[i] + backwards, axis=1)
        return probabilities

    def draw(self, rng):
        chosen = np.empty(len(self.times), dtype=int)
        next_index = None
        for i in range(len(chosen) - 1, -1, -1):
            key = i, next_index
            probabilities = self.sampling_weights.get(key)
            if probabilities is None:
                weights = self.forward[i]
                if next_index is not None:
                    weights = weights + self.transitions[i][:, next_index]
                probabilities = np.exp(weights - logsumexp(weights))
                probabilities.flags.writeable = False
                self.sampling_weights[key] = probabilities
            next_index = rng.choice(len(probabilities), p=probabilities)
            chosen[i] = next_index
        return chosen


class EvolvingTripBelief:
    """Adaptive-grid inference for ordered clocks and segment availability.

    ``public_occurrences=True`` assumes immediate local actual publications.
    Feed absence is informative under that synthetic contract. Delayed or
    incomplete feeds use False, retaining only received facts and ride/wait
    evidence. Off-grid exact observations augment the integration grid, so the
    physical continuous-clock posterior is approximated.
    """
    def __init__(self, graph, *, minute=60, travel_sd=2.0, dwell_sd=1.0,
                 travel_drift=0.0, dwell_drift=0.0, public_occurrences=True,
                 cancel_persistence=.9, gap_sd=None, gap_drift=None):
        self.graph, self.minute = graph, minute
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
        self._nominal_cache = {}
        self._transition_cache = {}
        self._positions = {}
        for c in sorted(graph.connections, key=lambda c: (c.dep_time, c.arr_time, c.id)):
            trip = c.trip_id or f"connection:{c.id}"
            self._trip_by_cid[c.id] = trip
            self._trips.setdefault(trip, []).extend((_Event(c.id, "departure", c.dep_time),
                                                     _Event(c.id, "arrival", c.arr_time)))
        for trip, slots in self._trips.items():
            self._positions.update({(event.cid, event.kind): (trip, i) for i, event in enumerate(slots)})

    def copy(self):
        branch = copy(self)
        for attr in ("_known_status", "_known_times", "_forecasts", "_excluded", "_cannot_cancel", "_revision", "_cache", "_nominal_cache"):
            setattr(branch, attr, getattr(self, attr).copy())
        return branch

    def _touch(self, trip):
        self._revision[trip] = self._revision.get(trip, 0) + 1

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

    def _grid(self, slot, shifts, anchors):
        times = np.unique(np.concatenate((slot.scheduled + shifts, anchors)))
        clocks = np.tile(times, 2)
        canceled = np.repeat([False, True], len(times))
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
        key = self._revision.get(trip, 0), pars["mean"], pars["std"], pars["cancel_rate"]
        if trip in self._cache and self._cache[trip][0] == key:
            return self._cache[trip][1]
        observed = [(s, self._known_times[s.cid, s.kind]) for s in slots
                    if (s.cid, s.kind) in self._known_times]
        shifts = np.unique(np.r_[DELAYS * self.minute, [t - s.scheduled for s, t in observed]]).astype(int)
        anchors = np.asarray([t for _, t in observed], dtype=int)
        times, canceled, emissions = zip(*(self._grid(slot, shifts, anchors) for slot in slots))
        delays = [(t - s.scheduled) / self.minute for t, s in zip(times, slots)]
        q = pars["cancel_rate"]
        initial = -.5 * ((delays[0] - pars["mean"]) / max(pars["std"], .5)) ** 2
        initial -= logsumexp(initial[~canceled[0]])
        with np.errstate(divide="ignore"):
            initial += np.log(np.where(canceled[0], q, 1 - q))
        forward = [initial + emissions[0]]
        transitions = []
        for i in range(1, len(slots)):
            travel = slots[i].kind == "arrival"
            gap = (not travel and self.graph.connections[slots[i - 1].cid].arr_stop
                   != self.graph.connections[slots[i].cid].dep_stop)
            sd = self.travel_sd if travel else self.gap_sd if gap else self.dwell_sd
            drift = self.travel_drift if travel else self.gap_drift if gap else self.dwell_drift
            kernel_key = (slots[i].scheduled - slots[i - 1].scheduled, travel, sd, drift,
                          tuple(delays[i - 1]), tuple(delays[i]),
                          None if travel else (q, self.cancel_persistence))
            # Shared kernels serve fixed, unobserved grids. Observed-clock
            # kernels belong to their law, so discarded hypothetical paths
            # release their anchor-specific matrices.
            log_step = self._transition_cache.get(kernel_key) if not observed else None
            if log_step is None:
                log_step = -.5 * ((delays[i][None, :] - delays[i - 1][:, None] - drift) / sd) ** 2
                ordered = times[i][None, :] >= times[i - 1][:, None]
                log_step = np.where(ordered, log_step, -np.inf)
                # Normalize potential-clock transitions once per availability state.
                norm = logsumexp(log_step[:, ~canceled[i]], axis=1, keepdims=True)
                log_step -= np.where(np.isfinite(norm), norm, 0.0)
                if travel:
                    status_probability = (canceled[i - 1][:, None] == canceled[i][None, :]).astype(float)
                else:
                    next_cancel = self.cancel_persistence * canceled[i - 1][:, None] + (1 - self.cancel_persistence) * q
                    status_probability = np.where(canceled[i][None, :], next_cancel, 1 - next_cancel)
                with np.errstate(divide="ignore"):
                    log_step += np.log(status_probability)
                log_step.flags.writeable = False
                if not observed:
                    self._transition_cache[kernel_key] = log_step
            transitions.append(log_step)
            forward.append(logsumexp(forward[-1][:, None] + log_step, axis=0) + emissions[i])
        if not np.isfinite(logsumexp(forward[-1])):
            raise ValueError(f"Observations leave no ordered trip event path for {trip}")
        law = _ChainLaw(times, forward, transitions, emissions, canceled)
        self._cache[trip] = key, law
        return law

    def event_distribution(self, cid, kind, parameters):
        """Conditional operating clock distribution and cancellation mass."""
        trip, i = self._positions[cid, kind]
        law = self._law(trip, parameters)
        operating = ~law.canceled[i]
        weights = law.marginals[i][operating]
        cancel = float(law.marginals[i][~operating].sum())
        if weights.sum() == 0:
            return np.empty(0, dtype=int), np.empty(0), 1.
        return law.times[i][operating], weights / weights.sum(), cancel

    def sample(self, parameters, rng, n):
        banks = [{} for _ in range(n)]
        for trip, slots in self._trips.items():
            law = self._law(trip, parameters)
            for world in banks:
                indices = law.draw(rng)
                for i in range(0, len(slots), 2):
                    dep, arr = slots[i:i + 2]
                    canceled = bool(law.canceled[i][indices[i]])
                    departure, arrival = int(law.times[i][indices[i]]), int(law.times[i + 1][indices[i + 1]])
                    if canceled and self.public_occurrences:
                        departure, arrival = dep.scheduled, arr.scheduled
                    world[dep.cid] = VehicleEvent(999 if canceled else departure - dep.scheduled,
                                                  canceled, departure, arrival)
        return [JourneyScenario(events, minute=self.minute,
                                publications=None if self.public_occurrences else {}) for events in banks]

    def mean_events(self, parameters):
        nominal = {}
        for trip, slots in self._trips.items():
            self._law(trip, parameters)
            key = self._cache[trip][0]
            cached = self._nominal_cache.get(trip)
            if cached is None or cached[0] != key:
                events = {}
                for i in range(0, len(slots), 2):
                    cid = slots[i].cid
                    dep_times, dep_weights, cancel = self.event_distribution(cid, "departure", parameters)
                    arr_times, arr_weights, _ = self.event_distribution(cid, "arrival", parameters)
                    if cancel == 1.:
                        event = NominalEvent(None, None, None, cancel)
                    else:
                        dep, arr = float(dep_times @ dep_weights), float(arr_times @ arr_weights)
                        event = NominalEvent(dep - slots[i].scheduled, dep, arr, cancel)
                    events[cid] = event
                cached = key, events
                self._nominal_cache[trip] = cached
            nominal.update(cached[1])
        return nominal
