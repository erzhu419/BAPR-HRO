"""Analytic committed-action bounds and fixed-itinerary witnesses.

The real-data contract has no public occurrence feed. A failed attempt then
returns to the same stop at a known clock. Trip factors are independent in
EvolvingTripBelief, while all event/service dependence within a trip remains.
The bounds integrate that current finite-grid joint law, not a sampled bank.
"""
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from .committed_transit import OBSERVE, WAIT
from .evolving_trip_belief import EvolvingTripBelief, logsumexp
from .trip_marginal_tilt import TripMarginalTilt


@dataclass(frozen=True)
class CommitmentRisk:
    catch_probability: float
    failure_probability: float
    expected_first_cost: float
    ready: int
    failure_clock: int
    direct_stop_upper: float | None


@dataclass(frozen=True)
class ItineraryValue:
    success_probability: float
    success_gain: float
    upper: float
    arrival_times: np.ndarray
    arrival_mass: np.ndarray


@dataclass(frozen=True)
class RescueValue:
    upper: float
    commitment_lower: float
    fixed_rescue_family_lower: float
    first: CommitmentRisk
    primary: ItineraryValue
    rescue: ItineraryValue


@dataclass(frozen=True)
class TransferGateValue:
    """Full-current-model certificate for an immediate same-vehicle suffix."""
    lower: float
    upper: float
    regret_bound: float
    transfer_gate: int
    success_probability: float
    path: tuple


def _prefix_window_mass(ready, mass, departures, patience, deadline):
    """Integrate arbitrary arrival phases without an incoming-by-clock tensor.

    WAIT advances60seconds from each arrival's phase. The resulting ready(t)
    can jump backwards, so sort *ready clocks*, not incoming arrival clocks.
    """
    valid = ready < deadline
    order = np.argsort(ready[valid], kind='stable')
    clocks = ready[valid][order]
    prefix = np.r_[0., np.cumsum(mass[valid][order])]
    upper = np.searchsorted(clocks, departures, side='right')
    lower = np.searchsorted(clocks, departures - patience, side='left')
    return np.where(departures <= deadline, prefix[upper] - prefix[lower], 0.)


class CommitmentBounds:
    """Reuse posterior-chain messages across current-root analytic queries."""

    def __init__(self, belief, parameters, graph, destination, deadline):
        if not isinstance(belief, (EvolvingTripBelief, TripMarginalTilt)) or belief.public_occurrences:
            raise TypeError('Analytic itinerary witnesses require independent trips and no occurrence feed')
        self.belief, self.graph = belief.copy(), graph
        self.parameters = {route: dict(row) for route, row in parameters.items()}
        self.destination, self.deadline = destination, deadline
        self.minute, self.patience = belief.minute, 12 * belief.minute
        self._laws, self._posterior_edges = {}, {}
        self.stats = {'law_queries': 0, 'root_joint_cells': 0, 'posterior_edge_cells': 0,
                      'path_matrix_cells': 0, 'window_input_cells': 0, 'seconds': 0.}

    def _law(self, trip):
        if trip not in self._laws:
            self._laws[trip] = self.belief._law(trip, self.parameters)
            self.stats['law_queries'] += 1
        return self._laws[trip]

    @staticmethod
    def _backward(law, slot):
        return law._backward_from(max(slot, law._last_effect), slot)

    def _posterior_edge(self, trip, slot):
        key = trip, slot
        if key not in self._posterior_edges:
            law = self._law(trip)
            log_mass = (law._transition(slot) + law.emissions[slot + 1]
                        + self._backward(law, slot + 1))
            norm = logsumexp(log_mass, axis=1, keepdims=True)
            edge = np.exp(log_mass - np.where(np.isfinite(norm), norm, 0.))
            edge.flags.writeable = False
            self._posterior_edges[key] = edge
            self.stats['posterior_edge_cells'] += edge.size
        return self._posterior_edges[key]

    def commit(self, state, cid):
        started = perf_counter()
        c = self.graph.connections[cid]
        transfer = (0 if state.current_trip is None or c.trip_id == state.current_trip
                    else self.graph.stops[state.stop].min_transfer_time)
        ready = min(self.deadline, max(state.time, state.last_arrival + transfer))
        failure_clock = min(self.deadline, ready + self.patience)
        cap = self.deadline - state.time
        if ready >= self.deadline:
            return CommitmentRisk(0., 1., cap, ready, failure_clock,
                                  float(cap) if c.arr_stop == self.destination else None)
        trip, slot = self.belief._positions[cid, 'departure']
        law = self._law(trip)
        # The posterior edge already contains all future evidence. Multiplying
        # it by the boundary-aware marginal avoids forcing a full-trip forward
        # pass after the rider has supplied an exact same-vehicle separator.
        joint = law.marginal(slot)[:, None] * self._posterior_edge(trip, slot)
        joint /= joint.sum()
        caught = (~law.canceled[slot] & (ready <= law.times[slot])
                  & (law.times[slot] <= failure_clock))
        catch_probability = float(joint[caught].sum())
        failure_probability = float(joint[~caught].sum())
        success_cost = float((joint[caught].sum(axis=0)
                              * (np.minimum(law.times[slot + 1], self.deadline) - state.time)).sum())
        lower = success_cost + failure_probability * (failure_clock - state.time)
        upper = success_cost + failure_probability * cap if c.arr_stop == self.destination else None
        self.stats['root_joint_cells'] += joint.size
        self.stats['seconds'] += perf_counter() - started
        return CommitmentRisk(catch_probability, failure_probability, lower, ready, failure_clock, upper)

    def action_lower(self, state, actions):
        result = {}
        for action in actions:
            if action == WAIT:
                result[action] = min(self.deadline, state.time + self.minute) - state.time
            elif action == OBSERVE:
                result[action] = 0.
            else:
                result[action] = self.commit(state, action[1]).expected_first_cost
        return result

    def _ready(self, times, cid, previous_trip, *, initial=None, first_action_now=False):
        c = self.graph.connections[cid]
        transfer = (0 if previous_trip is None or previous_trip == c.trip_id
                    else self.graph.stops[c.dep_stop].min_transfer_time)
        last = np.full(len(times), initial.last_arrival) if initial is not None else times
        platform = last + transfer
        if first_action_now:
            return np.maximum(times, platform).astype(int)
        opening = c.dep_time - self.lookahead
        menu = times + self.minute * np.ceil(np.maximum(0, opening - times) / self.minute)
        ready = np.maximum(menu, platform)
        event = self.belief.mean_events(self.parameters)[cid]
        nominal_departure = c.dep_time if event.departure is None else round(event.departure)
        delayed = nominal_departure > ready + self.patience
        aligned = times + self.minute * np.ceil(np.maximum(0, nominal_departure - self.patience - times) / self.minute)
        return np.maximum(np.where(delayed, aligned, menu), platform).astype(int)

    def itinerary(self, state, path, *, first_action_now=True, lookahead=None):
        """Cost of a fixed legal itinerary, stopping at deadline after any miss.

        Each trip appears once, with consecutive connection event slots inside
        its block. The policy waits in minute increments from the actual arrival
        phase until the fixed nominal boarding window; it never sees private
        future clocks. Root first_action_now fixes the selected root attempt.
        """
        started = perf_counter()
        self.lookahead = 25 * self.minute if lookahead is None else lookahead
        path = tuple(path)
        cap = self.deadline - state.time
        if not path:
            gain = float(cap) if state.stop == self.destination else 0.
            return ItineraryValue(float(state.stop == self.destination), gain, cap - gain,
                                  np.array([state.time]), np.array([float(state.stop == self.destination)]))
        if state.stop == self.destination:
            raise ValueError('An itinerary cannot leave an already reached destination')
        groups, seen_trips, stop = [], set(), state.stop
        for cid in path:
            c = self.graph.connections[cid]
            if c.dep_stop != stop or cid in state.unavailable_ids:
                raise ValueError('Itinerary must be physically contiguous and use available connections')
            trip, slot = self.belief._positions[cid, 'departure']
            if not groups or trip != groups[-1][0]:
                if trip in seen_trips:
                    raise ValueError('Exact independent-trip itinerary evaluation cannot revisit a trip')
                seen_trips.add(trip)
                groups.append((trip, [cid]))
            else:
                if slot != self.belief._positions[groups[-1][1][-1], 'arrival'][1] + 1:
                    raise ValueError('Same-trip blocks must use consecutive modeled connection slots')
                groups[-1][1].append(cid)
            stop = c.arr_stop
            if stop == self.destination and cid != path[-1]:
                raise ValueError('The witness must terminate on its first destination arrival')
        if stop != self.destination:
            raise ValueError('A complete witness itinerary must end at the destination')
        first = self.graph.connections[path[0]]
        if first_action_now and first.dep_time > state.time + self.lookahead:
            raise ValueError('The fixed first attempt is outside the current scheduled lookahead')
        incoming_times, incoming_mass = np.array([state.time]), np.array([1.])
        previous_trip = state.current_trip
        for group_number, (trip, cids) in enumerate(groups):
            entry = self.graph.connections[cids[0]]
            if not entry.pickup_allowed and previous_trip != entry.trip_id:
                raise ValueError('A new trip must permit boarding')
            slot = self.belief._positions[cids[0], 'departure'][1]
            law = self._law(trip)
            initial = state if group_number == 0 else None
            ready = self._ready(incoming_times, cids[0], previous_trip, initial=initial,
                                first_action_now=first_action_now and group_number == 0)
            window = _prefix_window_mass(ready, incoming_mass, law.times[slot], self.patience, self.deadline)
            mass = law.marginal(slot) * window * ~law.canceled[slot]
            self.stats['window_input_cells'] += len(ready) + len(law.times[slot])
            for position, cid in enumerate(cids):
                dep = self.belief._positions[cid, 'departure'][1]
                if position:
                    edge = self._posterior_edge(trip, dep - 1)
                    ready = self._ready(law.times[dep - 1], cid, trip)
                    caught = ((ready[:, None] <= law.times[dep][None, :])
                              & (law.times[dep][None, :] <= ready[:, None] + self.patience)
                              & (ready[:, None] < self.deadline)
                              & (law.times[dep][None, :] <= self.deadline)
                              & ~law.canceled[dep][None, :])
                    mass = mass @ (edge * caught)
                    self.stats['path_matrix_cells'] += edge.size
                edge = self._posterior_edge(trip, dep)
                mass = (mass @ edge) * (law.times[dep + 1] <= self.deadline)
                self.stats['path_matrix_cells'] += edge.size
            final = self.belief._positions[cids[-1], 'arrival'][1]
            incoming_times, incoming_mass = law.times[final], mass
            previous_trip = trip
        gain = float(np.dot(incoming_mass, self.deadline - incoming_times))
        probability = float(incoming_mass.sum())
        self.stats['seconds'] += perf_counter() - started
        return ItineraryValue(probability, gain, float(cap - gain), incoming_times, incoming_mass)

    def with_rescue(self, state, primary, rescue, *, lookahead=None):
        """Exact root-failure rescue witness; later misses stop at deadline."""
        primary = tuple(primary)
        if not primary:
            raise ValueError('A root-failure witness needs an initial connection')
        root_trip = self.belief._trip_by_cid[primary[0]]
        if any(self.belief._trip_by_cid[cid] == root_trip for cid in rescue):
            raise ValueError('Independent rescue must exclude the failed first trip')
        first = self.commit(state, primary[0])
        primary_value = self.itinerary(state, primary, first_action_now=True, lookahead=lookahead)
        failed = replace(state, time=first.failure_clock)
        rescue_value = self.itinerary(failed, rescue, first_action_now=False, lookahead=lookahead)
        upper = self.deadline - state.time - primary_value.success_gain - first.failure_probability * rescue_value.success_gain
        family_lower = first.expected_first_cost + first.failure_probability * rescue_value.upper
        return RescueValue(float(upper), first.expected_first_cost, float(family_lower),
                           first, primary_value, rescue_value)

    def transfer_gate(self, state, path, *, lookahead=None):
        """Certify a fresh onboard suffix when transfer time blocks shortcuts.

        The policy attempts each consecutive same-trip leg immediately, using
        the original12-minute commitment. It does not wait for a nominal clock.
        None means the topology/support checks do not prove the certificate.
        Other trips and all future information are unrestricted in the oracle
        lower bound. No other-trip HMM or network search is required.
        """
        started = perf_counter()
        path = tuple(path)
        lookahead = 25 * self.minute if lookahead is None else lookahead
        if not path or state.time != state.last_arrival or state.current_trip is None:
            return None
        connections = [self.graph.connections[cid] for cid in path]
        trip = self.belief._trip_by_cid[path[0]]
        if connections[0].trip_id != state.current_trip or any(
                self.belief._trip_by_cid[cid] != trip for cid in path):
            return None
        stop, previous_slot = state.stop, None
        departure_stops = []
        for c in connections:
            slot = self.belief._positions[c.id, 'departure'][1]
            if (c.dep_stop != stop or c.id in state.unavailable_ids
                    or (previous_slot is not None and slot != previous_slot + 2)
                    or c.dep_time > state.time + lookahead):
                return None
            departure_stops.append(c.dep_stop)
            stop, previous_slot = c.arr_stop, slot
        if stop != self.destination or len(set(departure_stops)) != len(departure_stops):
            return None
        transfer = min(self.graph.stops[stop].min_transfer_time for stop in departure_stops)
        gate = min(self.deadline, state.time + transfer)
        if gate > min(self.deadline, state.time + self.patience):
            return None
        # A later return of the same trip, or a zero-time historical edge, can
        # invalidate the simple transfer gate. Check all same-trip departures
        # at these stops rather than pretending current_trip implies one edge.
        for c in connections:
            for alternative in self.graph.get_connections_from(c.dep_stop):
                if (alternative.id == c.id or alternative.id in state.unavailable_ids
                        or alternative.trip_id != state.current_trip):
                    continue
                support = self.belief.hard_support(alternative.id, 'departure', self.parameters)
                arrival = self.belief.hard_support(alternative.id, 'arrival', self.parameters)
                if (support is None or support[1] < state.time
                        or alternative.dep_time - lookahead >= gate
                        or (arrival is not None and arrival[0] >= gate)):
                    continue
                return None
        law = self._law(trip)
        entry = self.belief._positions[path[0], 'departure'][1]
        mass = law.marginal(entry) * (~law.canceled[entry]
                & (law.times[entry] >= state.time)
                & (law.times[entry] <= min(self.deadline, state.time + self.patience)))
        for number, cid in enumerate(path):
            dep = self.belief._positions[cid, 'departure'][1]
            if number:
                edge = self._posterior_edge(trip, dep - 1)
                catch = (~law.canceled[dep][None, :]
                         & (law.times[dep][None, :] >= law.times[dep - 1][:, None])
                         & (law.times[dep][None, :] <= law.times[dep - 1][:, None] + self.patience)
                         & (law.times[dep][None, :] <= self.deadline))
                mass = mass @ (edge * catch)
                self.stats['path_matrix_cells'] += edge.size
            edge = self._posterior_edge(trip, dep)
            mass = (mass @ edge) * (law.times[dep + 1] <= self.deadline)
            self.stats['path_matrix_cells'] += edge.size
        arrival = law.times[self.belief._positions[path[-1], 'arrival'][1]]
        success = float(mass.sum())
        failure = 1. - success
        cap = self.deadline - state.time
        upper = float(cap - np.dot(mass, self.deadline - arrival))
        lower = float(np.dot(mass, np.minimum(arrival, gate) - state.time)
                      + failure * (gate - state.time))
        self.stats['seconds'] += perf_counter() - started
        return TransferGateValue(lower, upper, max(0., upper - lower), gate, success, path)


def evaluate_itinerary(belief, parameters, state, path_cids, destination, deadline, **kwargs):
    return CommitmentBounds(belief, parameters, belief.graph, destination, deadline).itinerary(state, path_cids, **kwargs)


def first_failure_rescue_upper(belief, parameters, state, primary, rescue, destination, deadline, **kwargs):
    return CommitmentBounds(belief, parameters, belief.graph, destination, deadline).with_rescue(state, primary, rescue, **kwargs)


def transfer_gate_certificate(belief, parameters, state, path, destination, deadline, **kwargs):
    """Transfer-gate bound also permits a globally shared day mixture.

    The oracle comparison is pathwise, so arbitrary dependence on other trips
    does not invalidate it. Only the current trip's conditional law is needed.
    Component policies use identical immediate-stay instructions, never delta.
    """
    path = tuple(path)
    if isinstance(belief, EvolvingTripBelief):
        return CommitmentBounds(belief, parameters, belief.graph, destination, deadline).transfer_gate(state, path, **kwargs)
    from .joint_day_belief import JointDayBelief
    from .independent_day_mixture_belief import IndependentDayMixtureBelief
    if not isinstance(belief, JointDayBelief) or belief.public_occurrences:
        raise TypeError('Transfer gate supports ordered trips or a shared-day ordered mixture without occurrence feed')
    frozen = belief.copy()
    if isinstance(frozen, IndependentDayMixtureBelief):
        _, weights = frozen.trip_posterior(tuple(path)[0], parameters)
    else:
        _, weights = frozen.day_posterior(parameters)
    values = []
    for component, pars in zip(frozen.components, frozen._shifted_parameters):
        value = CommitmentBounds(component, pars, frozen.graph, destination, deadline).transfer_gate(state, path, **kwargs)
        if value is None:
            return None
        values.append(value)
    lower = float(sum(w * value.lower for w, value in zip(weights, values)))
    upper = float(sum(w * value.upper for w, value in zip(weights, values)))
    probability = float(sum(w * value.success_probability for w, value in zip(weights, values)))
    return TransferGateValue(lower, upper, max(0., upper - lower), values[0].transfer_gate,
                             probability, tuple(path))
