"""Nominal path templates for public-event rollout at the waiting platform."""

from types import SimpleNamespace

import numpy as np

from experiments.reconstruction_v18.run_policy_pair import root_menu, support_additions
from src.platform_nominal_index import PlatformNominalIndex
from src.public_trip_conditionals import condition_on_departure
from src.public_wait_recourse import through_itineraries
from src.recourse_itinerary_router import _ConditionalNominal


def candidate_paths(graph, belief, parameters, root, goal, deadline, plan, *, direct):
    original, extras = root_menu(graph, root, goal, deadline, plan)
    extras.update(support_additions(graph, belief, parameters, root, goal, deadline, original, extras))
    if direct:
        extras = through_itineraries(graph, extras)
    paths = {cid: value.primary for cid, value in original.items()}
    paths.update((cid, value.path) for cid, value in extras.items())
    if direct:
        return paths, dict(screened=len(paths), extra_nominal=0)

    ready = max(root.time, root.last_arrival + graph.stops[root.stop].min_transfer_time)
    index = PlatformNominalIndex(graph, goal, root.time, deadline, unavailable=root.used_connections)
    scheduled = {c.id: SimpleNamespace(departure=c.dep_time, arrival=c.arr_time)
                 for c in graph.connections}
    supported, added = set(), 0
    for c in graph.get_connections_from(root.stop):
        cid = c.id
        if cid in root.used_connections or not c.pickup_allowed:
            continue
        support = belief.hard_support(cid, 'departure', parameters)
        if support is None or support[1] < ready or support[0] >= deadline:
            continue
        clocks, mass, _ = belief.event_distribution(cid, 'departure', parameters)
        valid = (ready <= clocks) & (clocks < deadline) & (mass > 0)
        if not valid.any():
            continue
        supported.add(cid)
        if cid in paths:
            continue
        times, weights = clocks[valid], mass[valid]
        for tick in dict.fromkeys((int(times[np.argmax(weights)]), int(times.min()))):
            condition = condition_on_departure(belief, parameters, cid, tick)
            nominal = _ConditionalNominal(scheduled, belief, belief._trip_by_cid[cid], condition.law)
            path = index.conditional_path(cid, nominal, tick)
            if path:
                paths[cid] = path
                added += 1
                break
    return paths, dict(screened=len(paths)-added, extra_nominal=added, physically_supported=len(supported),
                       supported_without_template=sorted(supported-set(paths)))
