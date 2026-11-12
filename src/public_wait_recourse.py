"""Publicly screened through-service controls for conditional rescue."""

import numpy as np

from .conditional_rescue_policy import ConditionalItineraries, DepartureScore
from .public_trip_conditionals import condition_on_departure
from .risk_bounds import CommitmentBounds
from .tail_rescue import TailRescuePlan


def through_itineraries(graph, options):
    """Keep the screened paths that stay on a single identified vehicle trip."""
    return {cid: option for cid, option in options.items()
            if graph.connections[cid].trip_id is not None
            and all(graph.connections[leg].trip_id == graph.connections[cid].trip_id
                    for leg in option.path)}


def _backward_through(bounds, trip, slots, path, departures):
    """Contract the committed fixed-path gain and success for all entry states."""
    law = bounds._law(trip)
    deadline = bounds.deadline
    final = law.times[slots[-1] + 1]
    value = np.column_stack((np.maximum(0, deadline-final), final <= deadline))
    for i in range(len(path)-1, -1, -1):
        slot, cid = slots[i], path[i]
        value = bounds._posterior_edge(trip, slot) @ value
        if i == 0:
            return value
        arrival = law.times[slot-1]
        ready = bounds._ready(arrival, cid, trip, departures=departures)
        departure = law.times[slot]
        valid = ((ready[:, None] <= departure) & (departure <= ready[:, None]+bounds.patience)
                 & (ready[:, None] < deadline) & (departure <= deadline) & ~law.canceled[slot])
        edge = bounds._posterior_edge(trip, slot-1)
        value = ((edge * valid) @ value) * (arrival <= deadline)[:, None]


def compile_through_scores(belief, parameters, graph, root, goal, deadline,
                           original_paths, extras, *, point=False):
    """Use the original integrated A/B wait reference and filter other paths.

    V16's nominal screen bounds the first departure's scheduled lookahead at
    these roots, so its fixed-path scoring uses the same 1500-second lookahead
    for all first-clock conditions. In the exact control the ordered-chain
    backward pass retains service and clock dependence. The point control
    evaluates only conditional operating clock means on those same paths.
    """
    through = through_itineraries(graph, extras)
    scores = ConditionalItineraries(belief, parameters, graph, root, goal, deadline,
                                    dict(original_paths))
    bounds = CommitmentBounds(belief, parameters, graph, goal, deadline)
    bounds.lookahead = 1500
    for cid, option in through.items():
        path = option.path
        trip, slot = belief._positions[cid, 'departure']
        slots = [belief._positions[leg, 'departure'][1] for leg in path]
        first = graph.connections[cid]
        if slots != list(range(slot, slot+2*len(path), 2)) or first.dep_time-root.time > bounds.lookahead:
            scores.excluded[cid] = 'The path is not a consecutive fixed-lookahead through itinerary'
            continue
        law = bounds._law(trip)
        probability = law.marginal(slot)
        plan = TailRescuePlan(path, first_action_now=True)
        departures = dict(plan.departures)
        integrated = None if point else _backward_through(bounds, trip, slots, path, departures)
        support = np.count_nonzero(bounds._posterior_edge(trip, slot)[:, ~law.canceled[slot+1]], axis=1)
        rows = {}
        for i in np.flatnonzero(~law.canceled[slot] & (probability > 0.)):
            clock = int(law.times[slot][i])
            if not root.time <= clock < deadline:
                continue
            if point:
                conditional = condition_on_departure(belief, parameters, cid, clock)
                score = scores._mean_clock_score(conditional, bounds, None, trip, path, departures, clock)
                success = float(score < deadline)
            elif first.dep_distribution is not None and clock > first.dep_distribution.support_max:
                score, success = deadline, 0.
            else:
                score, success = deadline-float(integrated[i, 0]), float(integrated[i, 1])
            rows[clock] = DepartureScore(clock, float(probability[i]), score, success, int(support[i]))
        if rows:
            scores.paths[cid], scores.rows[cid] = plan, rows
    return scores, through
