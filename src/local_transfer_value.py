"""Value the same local public fallback used after a failed transfer."""

from dataclasses import replace

import numpy as np

from .public_wait_recourse import _backward_through
from .risk_bounds import CommitmentBounds


class _FirstVisibleThrough:
    """Expected deadline savings and arrival mass of first visible direct service."""

    def __init__(self, bounds, paths):
        rows = []
        for order, (cid, path) in enumerate(paths.items()):
            trip, first = bounds.belief._positions[cid, 'departure']
            slots = [bounds.belief._positions[leg, 'departure'][1] for leg in path]
            law = bounds._law(trip)
            values = _backward_through(bounds, trip, slots, path, {})
            for state, mass in enumerate(law.marginal(first)):
                clock = int(law.times[first][state])
                if mass > 0 and not law.canceled[first][state] and clock < bounds.deadline:
                    rows.append((clock, order, mass, *values[state]))
        rows.sort()
        data = np.asarray(rows).reshape(-1, 5)
        self.clocks, self.order, self.mass = data[:, :3].T
        self.values = data[:, 3:]
        self.prefix = []
        for i in range(len(paths)):
            mass = np.where(self.order == i, self.mass, 0.)
            self.prefix.append(np.r_[0., np.cumsum(mass)])
        self.cache = {}

    def after(self, earliest):
        if earliest not in self.cache:
            start = np.searchsorted(self.clocks, earliest)
            weights = self.mass[start:].copy()
            for i, prefix in enumerate(self.prefix):
                # Sorted by time and then the issued simultaneous priority. A
                # candidate can depart only once, so it does not compete with itself.
                previous = prefix[start:-1] - prefix[start]
                weights *= np.where(self.order[start:] == i, 1., np.maximum(0., 1. - previous))
            self.cache[earliest] = weights @ self.values[start:]
        return self.cache[earliest]


def _failure_at_transfer(bounds, arrived, previous_trip, target, departures, fallback):
    """Return rescue gain, rescued mass, and original failure mass by arrival phase."""
    graph, deadline = bounds.graph, bounds.deadline
    trip, slot = bounds.belief._positions[target, 'departure']
    law = bounds._law(trip)
    dep = law.times[slot][None, :]
    operating = ~law.canceled[slot][None, :]
    physical = arrived + graph.stops[graph.connections[target].dep_stop].min_transfer_time
    ready = bounds._ready(arrived, target, previous_trip, departures=departures)[:, None]
    expiry = np.minimum(ready + bounds.patience, deadline)
    caught = operating & (ready <= dep) & (dep <= expiry) & (dep < deadline)
    # A departure before local arrival is never seen. A departure at arrival
    # or during the transfer is seen, but fallback boarding still obeys MCT.
    observed_early = operating & (arrived[:, None] <= dep) & (dep < ready)
    start = np.where(observed_early, np.maximum(dep, physical[:, None]), expiry)
    fail_mass = (~caught) * law.marginal(slot)
    values = np.zeros((*start.shape, 2))
    for t in np.unique(start[fail_mass > 0]):
        if t < deadline:
            values[start == t] = fallback.after(int(t))
    return np.column_stack((np.einsum('ad,adk->ak', fail_mass, values), fail_mass.sum(axis=1)))


def _from_first_departure(bounds, trip, block, departures, terminal):
    """Contract an arbitrary terminal value through the first vehicle block."""
    law = bounds._law(trip)
    value = terminal
    for i, cid in reversed(tuple(enumerate(block))):
        slot = bounds.belief._positions[cid, 'departure'][1]
        value = bounds._posterior_edge(trip, slot) @ value
        if i == 0:
            return value
        arrival = law.times[slot - 1]
        ready = bounds._ready(arrival, cid, trip, departures=departures)
        dep = law.times[slot]
        legal = ((ready[:, None] <= dep) & (dep <= ready[:, None] + bounds.patience)
                 & (ready[:, None] < bounds.deadline) & (dep <= bounds.deadline)
                 & ~law.canceled[slot])
        value = (bounds._posterior_edge(trip, slot - 1) * legal) @ value


def add_transfer_rescue_scores(scores, belief, parameters, goal, controllers):
    """Update the departure marks and return their local failure components.

    Each controller supplies the first transfer and the ordered direct fallback
    paths for its fixed itinerary. The calculation keeps the frozen chain laws,
    integrates the first missed transfer, and treats fallback trips as independent
    of the attempted trips and of each other. It uses local departures only. The
    result is a model score, not a guarantee for a broader public history.
    """
    bounds = CommitmentBounds(belief, parameters, scores.graph, goal, scores.deadline)
    bounds.lookahead = 25 * bounds.minute
    fallback_cache, diagnostics = {}, {}
    for cid, controller in controllers.items():
        index = controller.transfer_index
        target, fallbacks = controller.target, controller.fallback_paths
        if index is None or cid not in scores.rows:
            continue
        plan = scores.paths[cid]
        block = plan.primary[:index]
        assert index > 0 and plan.primary[index] == target
        trip, first = belief._positions[cid, 'departure']
        fallback_trips = [belief._trip_by_cid[option] for option in fallbacks]
        primary_trips = {belief._trip_by_cid[leg] for leg in plan.primary}
        assert len(set(fallback_trips)) == len(fallback_trips)
        assert not primary_trips.intersection(fallback_trips)
        key = tuple(fallbacks.items())
        if key not in fallback_cache:
            fallback_cache[key] = _FirstVisibleThrough(bounds, fallbacks)
        fallback = fallback_cache[key]
        law = bounds._law(trip)
        last = belief._positions[block[-1], 'arrival'][1]
        departures = dict(plan.departures)
        forecasts, rows = {}, {}
        connection = scores.graph.connections[cid]
        for state in np.flatnonzero(~law.canceled[first]):
            clock = int(law.times[first][state])
            old = scores.rows[cid].get(clock)
            if old is None or clock < scores.ready:
                continue
            if connection.dep_distribution is not None and clock > connection.dep_distribution.support_max:
                continue
            lookahead = max(25 * bounds.minute, connection.dep_time - clock)
            if lookahead not in forecasts:
                bounds.lookahead = lookahead
                arrived = law.times[last]
                terminal = _failure_at_transfer(bounds, arrived, trip, target, departures, fallback)
                terminal[arrived > scores.deadline] = 0.
                forecasts[lookahead] = _from_first_departure(bounds, trip, block, departures, terminal)
                bounds.lookahead = 25 * bounds.minute
            gain, rescued, missed = map(float, forecasts[lookahead][state])
            scores.rows[cid][clock] = replace(
                old, expected_arrival_or_deadline=old.expected_arrival_or_deadline-gain,
                success_probability=min(1., old.success_probability+rescued))
            rows[clock] = dict(failure_probability=missed, rescue_probability=rescued,
                               rescue_gain=gain,
                               failure_terminal_clock=scores.deadline-gain/missed if missed > 0 else None)
        diagnostics[cid] = rows
    return diagnostics
