"""Population bounds from current P0, without sampled or realized endpoints.

Destination relaxation is valid for every admissible policy. Commitment
values, when supplied below, bound the causal optimum only; they do not
replace the value of a specified cheap or finite-lookahead policy.
"""
from collections import defaultdict

import numpy as np

from experiments.reconstruction_v41.physical_posterior import posterior_transition


def destination_endpoints(node, position=None):
    """Reachable absorbing arrival IDs in the remaining declared menu.

    For a Ride, position is the first remaining path index (inclusive).
    Traversal stops at its first destination; later scoring-only exits do
    not create additional physical opportunities. An onboard current index
    includes its current ALIGHT exit as well as later CONTINUE exits.
    """
    endpoints = set()

    def visit(current, start=0):
        if hasattr(current, 'path'):
            for cid in current.path[start:]:
                if cid not in current.exits:
                    continue
                tail = current.exits[cid]
                if tail is None:
                    endpoints.add(cid)
                    break
                visit(tail)
        else:
            for ride in current.rides.values():
                visit(ride)

    visit(node, 0 if position is None else position)
    return tuple(sorted(endpoints))


def destination_lower_bound(graph, root_law, deadline, node, history, clock, position=None):
    """Absolute expected lower bound for V* and every fixed causal policy.

    Relax boarding, transfer feasibility and nonanticipation, allowing any
    remaining destination endpoint. Each cancelled, past or late endpoint
    has value H. For one endpoint per trip, the relaxation is integrated
    exactly. Multiple endpoints on one trip use a union CDF bound; only
    distinct physical trips have their survival probabilities multiplied.

    root_law must contain the query's already prepared P0 laws. No received
    forecast is reselected and no future publication is accessed. `graph`
    is accepted for the common bound interface; schedules are not used.
    """
    start, end = min(float(clock), float(deadline)), float(deadline)
    if start == end:
        return end
    grouped = defaultdict(list)
    for cid in destination_endpoints(node, position):
        trip, slot = root_law.model._positions[cid, 'arrival']
        grouped[trip].append(slot)
    if not grouped:
        return end
    laws, grid = {}, {start, end}
    for trip, slots in grouped.items():
        law = root_law.law(trip, history)
        rows = []
        for slot in slots:
            mass = law.marginal(slot)
            valid = (~law.canceled[slot]) & (law.times[slot] >= start) & (law.times[slot] < end) & (mass > 0)
            times, probabilities = law.times[slot][valid], mass[valid]
            rows.append((times, probabilities))
            grid.update(map(float, times))
        laws[trip] = rows
    times = np.array(sorted(grid))
    survival = np.ones(len(times)-1)
    for rows in laws.values():
        endpoint_mass = np.zeros(len(times))
        for clocks, probabilities in rows:
            np.add.at(endpoint_mass, np.searchsorted(times, clocks), probabilities)
        # Bonferroni/union bound within one trip, not independence of slots.
        survival *= np.maximum(0., 1.-np.cumsum(endpoint_mass)[:-1])
    value = start+float(np.dot(np.diff(times), survival))
    return min(end, max(start, value))


def _commitment_routes(graph, node, position=0):
    """Each leg is (executed cid path, transfer time after this leg)."""
    if not hasattr(node, 'path'):
        for ride in node.rides.values():
            yield from _commitment_routes(graph, ride)
        return
    prefix = []
    for cid in node.path[position:]:
        prefix.append(cid)
        if cid not in node.exits:
            continue
        tail = node.exits[cid]
        if tail is None:
            yield ((tuple(prefix), 0.),)
            break
        transfer = float(graph.stops[tail.stop].min_transfer_time)
        for suffix in _commitment_routes(graph, tail):
            yield ((tuple(prefix), transfer), *suffix)


def commitment_upper_bound(graph, root_law, deadline, node, history, clock, position=None,
                           ready=None, after=float('-inf'), max_routes=32, available=None):
    """Exact feasible-route upper bound on the optimal continuation only.

    At each transfer the route waits for its one predetermined vehicle,
    ignores all other observations, and fails at H if it cannot board.
    Distinct legs must use distinct physical trips. This restriction only
    limits candidate *upper-bound policies*, never the actual action menu.

    Ride position starts with an unexecuted departure. For forced CONTINUE
    pass the index after the current arrival; forced ALIGHT passes its tail
    Platform. At an existing platform pass its original ready time, and
    forced WAIT also passes after=clock so current offers cannot be reused.
    A forced visible BOARD passes its Ride with position=0 and ready=clock.

    All returned values use the complete finite P0 law, not sample banks.
    Exhausting max_routes or available() keeps the best valid upper bound
    already obtained (initially H). It is not a bound on specified cheap or
    finite-lookahead policy values unless that policy is this commitment.
    """
    end, now = float(deadline), float(clock)
    if ready is None:
        ready = now if hasattr(node, 'path') else now+graph.stops[node.stop].min_transfer_time
    threshold = max(now, float(ready))
    result = dict(upper=end, route=None, examined=0, admitted=0,
        repeated_trip_rejections=0, structure_rejections=0, stopped=None,
        scope='Feasible fixed-route policy upper bound for V* only; population P0, not a sample mean')
    if threshold >= end:
        return result
    laws, transitions, suffixes = {}, {}, {}

    def trip_law(trip):
        if trip not in laws:
            laws[trip] = root_law.law(trip, history)
        return laws[trip]

    def marks(route):
        if route in suffixes:
            return suffixes[route]
        path, transfer = route[0]
        positions = [root_law.model._positions[cid,kind]
                     for cid in path for kind in ('departure','arrival')]
        trip = positions[0][0]
        law = trip_law(trip)
        first, last = positions[0][1], positions[-1][1]
        active_slots = {slot for _,slot in positions}
        arrival = law.times[last]
        if len(route) == 1:
            gain = np.maximum(0., end-arrival)
        else:
            later_times, later_mass, later_gain = marks(route[1:])
            weighted = later_mass*later_gain
            order = np.argsort(later_times,kind='stable')
            later_times, weighted = later_times[order], weighted[order]
            sums = np.r_[np.cumsum(weighted[::-1])[::-1], 0.]
            gain = sums[np.searchsorted(later_times,arrival+transfer,side='left')]
        for slot in range(last,first-1,-1):
            if slot != last:
                key = trip, slot
                if key not in transitions:
                    transitions[key] = posterior_transition(law, slot)
                gain = transitions[key] @ gain
            if slot in active_slots:
                gain = np.where(law.canceled[slot] | (law.times[slot] >= end), 0., gain)
        value = law.times[first], law.marginal(first), gain
        suffixes[route] = value
        return value

    for route in _commitment_routes(graph,node,0 if position is None else position):
        if max_routes is not None and result['examined'] >= max_routes:
            result['stopped'] = 'route_limit'
            break
        if available is not None and not available():
            result['stopped'] = 'compute_limit'
            break
        result['examined'] += 1
        trips, invalid = [], False
        for path, _ in route:
            positions = [root_law.model._positions[cid,kind]
                         for cid in path for kind in ('departure','arrival')]
            trip = positions[0][0]
            if any(t != trip for t,_ in positions) or any(
                    b[1] <= a[1] for a,b in zip(positions,positions[1:])):
                invalid = True
                break
            trips.append(trip)
        if invalid:
            result['structure_rejections'] += 1
            continue
        if len(trips) != len(set(trips)):
            result['repeated_trip_rejections'] += 1
            continue
        times, mass, gain = marks(route)
        allowed = (times >= threshold) & (times > after)
        value = end-float(np.dot(mass, np.where(allowed,gain,0.)))
        value = min(end,max(now,value))
        result['admitted'] += 1
        if value < result['upper']:
            result.update(upper=value,route=[dict(path=list(p),transfer_after=m) for p,m in route])
    return result
