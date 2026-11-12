"""Check indexed broad template discovery against the prior nominal searches."""

from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v15.run_attribution_banks import prepare
from experiments.reconstruction_v18.run_policy_pair import root_menu, support_additions
from src.committed_transit import CommittedState
from src.platform_nominal_index import PlatformNominalIndex
from src.public_trip_conditionals import condition_on_departure
from src.recourse_itinerary_router import _ConditionalNominal, build_trip_blocks, expand_trip_blocks, nominal_itinerary


def nominal_finish(graph, path, nominal, tick):
    prior = graph.connections[path[0]]
    at = max(tick, round(nominal[prior.id].arrival))
    legal = True
    for cid in path[1:]:
        after, event = graph.connections[cid], nominal[cid]
        same = prior.trip_id is not None and after.trip_id == prior.trip_id
        dep = round(event.departure)
        ready = at if same else at+graph.stops[after.dep_stop].min_transfer_time
        legal &= after.dep_stop == prior.arr_stop and dep >= ready and (same or after.pickup_allowed)
        at, prior = dep+round(event.arrival-event.departure), after
    return at, legal


def audit(case):
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    original, extras = root_menu(graph, root, goal, deadline, plan)
    extras.update(support_additions(graph, belief, parameters, root, goal, deadline, original, extras))
    ready = max(root.time, root.last_arrival+graph.stops[root.stop].min_transfer_time)
    scheduled = {c.id: SimpleNamespace(departure=c.dep_time, arrival=c.arr_time) for c in graph.connections}
    start = perf_counter()
    blocks = build_trip_blocks(graph)
    index = PlatformNominalIndex(graph, goal, root.time, deadline, unavailable=root.used_connections,
                                 trip_blocks=blocks)
    sweep = perf_counter()-start
    timing = Counter()
    counts = Counter()
    coverage_differences = []
    for c in graph.get_connections_from(root.stop):
        cid = c.id
        if cid in root.used_connections or cid in original or cid in extras or not c.pickup_allowed:
            continue
        support = belief.hard_support(cid, 'departure', parameters)
        if support is None or support[1] < ready or support[0] >= deadline:
            continue
        clocks, mass, _ = belief.event_distribution(cid, 'departure', parameters)
        valid = (ready <= clocks) & (clocks < deadline) & (mass > 0)
        if not valid.any():
            continue
        counts['supported_additions'] += 1
        times, weights = clocks[valid], mass[valid]
        found = {}
        for tick in dict.fromkeys((int(times[np.argmax(weights)]), int(times.min()))):
            condition = condition_on_departure(belief, parameters, cid, tick)
            modes = ('indexed', 'repeated') if cid % 2 else ('repeated', 'indexed')
            for mode in modes:
                if mode in found:
                    continue
                nominal = _ConditionalNominal(scheduled, belief, belief._trip_by_cid[cid], condition.law)
                start = perf_counter()
                if mode == 'indexed':
                    path = index.conditional_path(cid, nominal, tick)
                else:
                    predicted = condition.marginal(cid).mean_operating_clock
                    path = None
                    if predicted is not None and predicted <= deadline:
                        arrival = max(tick, round(predicted))
                        state = CommittedState(c.arr_stop, arrival, c.trip_id, arrival, root.used_connections | {cid})
                        suffix = nominal_itinerary(graph, state, goal, nominal, scheduled, deadline, 1500,
                                                   trip_blocks=blocks)
                        if suffix is not None:
                            path = expand_trip_blocks(graph, (cid,) + suffix, goal, root.used_connections, blocks)
                timing[mode] += perf_counter()-start
                if path:
                    finish, legal = nominal_finish(graph, path, nominal, tick)
                    legal &= (path[0] == cid and graph.connections[path[-1]].arr_stop == goal and finish <= deadline)
                    if mode == 'indexed':
                        assert legal and expand_trip_blocks(graph, path, goal, root.used_connections, blocks) == path
                    else:
                        counts['repeated_nominally_infeasible'] += not legal
                    counts[mode+'_templates'] += 1
                    found[mode] = (path, finish)
            if len(found) == 2:
                break
        if len(found) == 1:
            coverage_differences.append(dict(connection=cid, method=next(iter(found))))
        elif len(found) == 2:
            a, b = found['indexed'], found['repeated']
            counts['identical_path'] += a[0] == b[0]
            if a[1] == b[1]:
                counts['same_finish'] += 1
            elif a[1] < b[1]:
                counts['indexed_earlier_finish'] += 1
            else:
                counts['indexed_later_finish'] += 1
    assert not coverage_differences
    return dict(case=case, counts=counts, sweep_seconds=sweep, query_seconds=timing,
                sweep_and_indexed_queries_seconds=sweep+timing['indexed'], coverage_differences=coverage_differences)


def main():
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    result = dict(scope='Previously frozen roots, nominal template discovery only. Query timing excludes shared support '
                        'screening and departure conditioning. Alternate the order with fresh conditional mapping caches.',
                  roots=[audit(case) for case in CASES])
    output = Path(__file__).with_name('nominal_index_audit.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
