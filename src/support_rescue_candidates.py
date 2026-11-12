"""Timetable-early rescue candidates supported by the frozen departure law."""

from types import SimpleNamespace

import numpy as np

from .committed_transit import CommittedState
from .public_rescue_candidates import PublicItinerary
from .public_trip_conditionals import condition_on_departure
from .recourse_itinerary_router import (
    _ConditionalNominal, build_trip_blocks, expand_trip_blocks, nominal_itinerary,
)


def screen_support_itineraries(graph, belief, parameters, root, goal, deadline,
                               original, window=720, *, min_mass=.01, max_new=12):
    """Add legal scheduled-early departures by probability of being catchable.

    Freeze the paths from the root's model and timetable before replay. A
    possible delayed departure only admits a candidate to the scoring menu;
    execution still requires it to be publicly visible and physically
    boardable. The timetable menu remains unchanged.
    """
    ready = max(root.time, root.last_arrival + graph.stops[root.stop].min_transfer_time)
    end = min(ready + window, deadline - 1)
    ranked = []
    screened = 0
    for connection in graph.get_connections_from(root.stop):
        cid = connection.id
        if (cid in original or cid in root.used_connections or not connection.pickup_allowed
                or connection.dep_time >= ready):
            continue
        support = belief.hard_support(cid, 'departure', parameters)
        if support is None or support[1] < ready or support[0] > end:
            continue
        screened += 1
        clocks, weights, canceled = belief.event_distribution(cid, 'departure', parameters)
        valid = (clocks >= ready) & (clocks <= end) & (weights > 0.)
        mass = float(weights[valid].sum() * (1. - canceled))
        if not mass or mass < min_mass:
            continue
        eligible = clocks[valid]
        mode = int(eligible[np.argmax(weights[valid])])
        ranked.append((-mass, connection.dep_time, cid, mode, int(eligible.min())))

    audit = dict(ready=ready, window_end=end, min_mass=min_mass, max_new=max_new,
                 scheduled_early_hard_screened=screened, above_mass_threshold=len(ranked), rows=[])
    if not ranked or not max_new:
        return {}, audit

    scheduled = {c.id: SimpleNamespace(departure=c.dep_time, arrival=c.arr_time)
                 for c in graph.connections}
    blocks = build_trip_blocks(graph)
    extras = {}
    for neg_mass, scheduled_departure, cid, mode, earliest in sorted(ranked):
        row = dict(cid=cid, scheduled_departure=scheduled_departure, operating_mass=-neg_mass,
                   mode_departure=mode, earliest_departure=earliest)
        audit['rows'].append(row)
        if len(extras) == max_new:
            row['status'] = 'candidate_cap'
            continue
        connection = graph.connections[cid]
        for departure in dict.fromkeys((mode, earliest)):
            conditioned = condition_on_departure(belief, parameters, cid, departure)
            assert conditioned is not None
            arrival = conditioned.marginal(cid).mean_operating_clock
            assert arrival is not None
            at = max(departure, round(arrival))
            if at >= deadline:
                continue
            used = root.used_connections | {cid}
            after = CommittedState(connection.arr_stop, at, connection.trip_id, at, used)
            trip = belief._trip_by_cid[cid]
            nominal = _ConditionalNominal(scheduled, belief, trip, conditioned.law)
            suffix = nominal_itinerary(graph, after, goal, nominal, scheduled, deadline, 1500,
                                       trip_blocks=blocks)
            if suffix is None:
                continue
            path = expand_trip_blocks(graph, (cid,) + suffix, goal, root.used_connections, blocks)
            if not path:
                continue
            extras[cid] = PublicItinerary(path, graph.connections[path[-1]].arr_time)
            row.update(status='added', representative_departure=departure, representative_arrival=at,
                       path=path)
            break
        else:
            row['status'] = 'no_nominal_path'
    return extras, audit
