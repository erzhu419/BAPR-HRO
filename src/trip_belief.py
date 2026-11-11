"""Condition an independent-trip plug-in delay model on public journey history.

Unknown trips have 70 integer delays (-5..64) and an explicit cancellation
outcome. Gaussian route parameters are supplied by the caller and held fixed
within a sampling bank. The masks condition on events, polling absence and
committed attempts; this is not a model of shared latent disruption factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import copy as shallow_copy
import math
from typing import Iterable
import numpy as np

from .bocd.regime_detector import DelayObservation
from .simulator import JourneyScenario, VehicleEvent
from .transit_graph import TransitGraph


DELAYS = np.arange(-5, 65, dtype=int)
OUTCOMES = np.concatenate((DELAYS, [999]))
CANCEL_INDEX = len(DELAYS)


@dataclass(frozen=True)
class NominalEvent:
    """Conditional mean operating times, with separate cancellation mass."""
    mean_delay: float | None
    departure: float | None
    arrival: float | None
    cancel_probability: float


class TripBelief:
    def __init__(self, graph: TransitGraph):
        self.graph = graph
        self.known: dict[tuple, tuple[bool, int]] = {}
        self._allowed: dict[tuple, np.ndarray] = {}
        self._history_revision: dict[tuple, int] = {}
        self._nominal_cache: dict[tuple, tuple] = {}
        self._trip_by_cid = {}
        self._connections_by_trip = {}
        for c in sorted(graph.connections, key=lambda c: (c.dep_time, c.id)):
            key = ('trip', c.trip_id) if c.trip_id else ('connection', c.id)
            self._trip_by_cid[c.id] = key
            self._connections_by_trip.setdefault(key, []).append(c)

    def copy(self) -> TripBelief:
        """Fork public history for a rollout, sharing only the static graph/index."""
        branch = shallow_copy(self)
        branch.known = self.known.copy()
        branch._allowed = {key: mask.copy() for key, mask in self._allowed.items()}
        branch._history_revision = self._history_revision.copy()
        branch._nominal_cache = self._nominal_cache.copy()
        return branch

    def _touch(self, key):
        self._history_revision[key] = self._history_revision.get(key, 0) + 1

    def _mask(self, key):
        return self._allowed.setdefault(key, np.ones(len(OUTCOMES), dtype=bool))

    def observe_vehicle_event(self, cid: int, canceled: bool, delay: int):
        """Record an exact public vehicle outcome, shared by its entire trip."""
        key = self._trip_by_cid[cid]
        value = (bool(canceled), 999 if canceled else int(delay))
        if key in self.known and self.known[key] != value:
            raise ValueError(f'Inconsistent public observations of trip {key}')
        index = CANCEL_INDEX if canceled else int(delay) - int(DELAYS[0])
        in_grid = canceled or 0 <= index < CANCEL_INDEX
        if in_grid and not self._mask(key)[index]:
            raise ValueError(f'Public vehicle event contradicts earlier history for {key}')
        if self.known.get(key) != value:
            self.known[key] = value
            self._touch(key)

    def observe_poll(self, stop: int, time: int,
                     messages: Iterable[DelayObservation], seen_before: set[str]):
        """Condition on the simulator's last-10-minute, first-five message poll.

        The full sort key is (actual time, connection ID, event kind). A poll
        returning fewer than five messages rules out all additional unseen
        events in the window. A full poll only rules out additional events
        ordered before or at its fifth message. Returned messages also fix
        their trip's exact outcome; repeating the exact hook is harmless.
        """
        messages = tuple(messages)
        returned_keys = []
        for message in messages:
            cid_text, kind = message.event_id.split(':', 1)
            cid = int(cid_text)
            returned_keys.append((message.actual_time, cid, kind))
            self.observe_vehicle_event(cid, message.is_canceled, message.actual_delay)
        cutoff = max(returned_keys) if len(messages) == 5 else None

        local = {}
        for c in self.graph.get_connections_from(stop) + self.graph.get_connections_to(stop):
            key = self._trip_by_cid[c.id]
            if key not in self.known:
                local.setdefault(key, {})[c.id] = c
        for key, connections in local.items():
            mask = self._mask(key)
            for c in connections.values():
                event_specs = []
                if c.dep_stop == stop:
                    event_specs.append(('departure', c.dep_time))
                if c.arr_stop == stop:
                    event_specs.append(('arrival', c.arr_time))
                for kind, scheduled in event_specs:
                    if f'{c.id}:{kind}' in seen_before:
                        continue
                    event_times = scheduled + DELAYS
                    excluded = (time - 10 <= event_times) & (event_times <= time)
                    if cutoff is not None:
                        cutoff_time, cutoff_cid, cutoff_kind = cutoff
                        tie_before = (c.id, kind) <= (cutoff_cid, cutoff_kind)
                        excluded &= ((event_times < cutoff_time)
                                     | ((event_times == cutoff_time) & tie_before))
                    mask[:CANCEL_INDEX] &= ~excluded
                if c.dep_stop == stop and f'{c.id}:cancel' not in seen_before:
                    cancel_key = (c.dep_time, c.id, 'cancel')
                    if (time - 10 <= c.dep_time <= time
                            and (cutoff is None or cancel_key <= cutoff)):
                        mask[CANCEL_INDEX] = False
            self._touch(key)

    def observe_attempt(self, cid: int, kind: str, time: int):
        """Condition on a public missed departure or completed waiting window.

        A timeout is compatible with a later operating departure, or with a
        canceled connection whose scheduled announcement has not yet occurred.
        Boarded and explicitly canceled outcomes use observe_vehicle_event.
        """
        key = self._trip_by_cid[cid]
        c = self.graph.connections[cid]
        if kind not in ('missed', 'timeout'):
            raise ValueError('Attempt kind must be missed or timeout')
        allowed = np.zeros(len(OUTCOMES), dtype=bool)
        if kind == 'missed':
            allowed[:CANCEL_INDEX] = c.dep_time + DELAYS < time
        else:
            allowed[:CANCEL_INDEX] = c.dep_time + DELAYS > time
            allowed[CANCEL_INDEX] = c.dep_time > time
        if key in self.known:
            canceled, delay = self.known[key]
            consistent = ((not canceled and c.dep_time + delay < time) if kind == 'missed'
                          else ((c.dep_time if canceled else c.dep_time + delay) > time))
            if not consistent:
                raise ValueError(f'Attempt contradicts known trip {key}')
        self._mask(key)[:] &= allowed
        self._touch(key)

    @staticmethod
    def _route_log_weights(parameters):
        mean, std = float(parameters['mean']), max(float(parameters['std']), .5)
        cancel = float(parameters.get('cancel_rate', 0.0))
        if not 0 <= cancel <= 1:
            raise ValueError('Cancellation probability must be in [0,1]')
        log_operating = -.5 * ((DELAYS - mean) / std) ** 2
        peak = float(log_operating.max())
        log_z = peak + math.log(float(np.exp(log_operating - peak).sum()))
        log_weights = np.full(len(OUTCOMES), -np.inf)
        if cancel < 1:
            log_weights[:CANCEL_INDEX] = log_operating - log_z + math.log1p(-cancel)
        if cancel > 0:
            log_weights[CANCEL_INDEX] = math.log(cancel)
        return log_weights

    def posterior(self, cid: int, dists: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return outcome delays (999 means cancel) and conditioned probabilities.

        Exact observed outcomes take precedence over fitted plug-in parameters.
        A known operating delay outside the unknown-trip grid remains exact.
        """
        key = self._trip_by_cid[cid]
        if key in self.known:
            canceled, delay = self.known[key]
            return np.array([999 if canceled else delay], dtype=int), np.array([1.0])
        first = self._connections_by_trip[key][0]
        params = dists.get(first.route, {'mean': 1.0, 'std': 2.0, 'cancel_rate': .01})
        log_weights = self._route_log_weights(params)
        log_weights[~self._mask(key)] = -np.inf
        peak = float(log_weights.max())
        if not math.isfinite(peak):
            raise ValueError(f'Public history has zero support for trip {key}')
        weights = np.exp(log_weights - peak)
        weights /= weights.sum()
        return OUTCOMES.copy(), weights

    def sample(self, dists: dict, rng: np.random.Generator, n: int) -> list[JourneyScenario]:
        """Draw one shared conditional scenario bank for all candidate actions."""
        worlds = [{} for _ in range(n)]
        for connections in self._connections_by_trip.values():
            outcomes, probabilities = self.posterior(connections[0].id, dists)
            draws = rng.choice(outcomes, size=n, p=probabilities)
            for world, delay in zip(worlds, draws):
                canceled = delay == 999
                for c in connections:
                    world[c.id] = VehicleEvent(
                        int(delay), bool(canceled),
                        c.dep_time if canceled else int(c.dep_time + delay),
                        c.arr_time if canceled else int(c.arr_time + delay),
                    )
        return [JourneyScenario(world) for world in worlds]

    def mean_events(self, dists: dict) -> dict[int, NominalEvent]:
        """Conditional operating mean times and cancellation probabilities.

        These are summaries, not a scenario containing the expected cost of a
        nonlinear journey policy. Sampling uses exactly the same posterior.
        """
        events = {}
        for key, connections in self._connections_by_trip.items():
            params = dists.get(connections[0].route)
            parameter_key = ((float(params['mean']), float(params['std']),
                              float(params.get('cancel_rate', 0.0)))
                             if params is not None else (1.0, 2.0, .01))
            revision = self._history_revision.get(key, 0)
            cached = self._nominal_cache.get(key)
            if cached is None or cached[:2] != (parameter_key, revision):
                outcomes, probabilities = self.posterior(connections[0].id, dists)
                operating = outcomes != 999
                p_operating = float(probabilities[operating].sum())
                p_cancel = float(probabilities[~operating].sum())
                delay = (float(np.dot(outcomes[operating], probabilities[operating]) / p_operating)
                         if p_operating else None)
                trip_events = {}
                for c in connections:
                    trip_events[c.id] = NominalEvent(delay,
                        c.dep_time + delay if delay is not None else None,
                        c.arr_time + delay if delay is not None else None, p_cancel)
                cached = (parameter_key, revision, trip_events)
                self._nominal_cache[key] = cached
            events.update(cached[2])
        return events
