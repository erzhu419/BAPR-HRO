"""Rebuild the supplied uncapped probe and explain remaining finite scope."""
from collections import Counter, defaultdict
from copy import deepcopy

from experiments.reconstruction_v32.screen import downstream_options, retain_options, final_services
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import label_coverage
from src.recourse_itinerary_router import build_trip_blocks, trip_key


def wide_root(graph, root, protocol):
    """Remove only the four-first-vehicles cap at the already selected hubs."""
    wide = deepcopy(root)
    blocks = build_trip_blocks(graph)
    first = graph.connections[root['first_cid']]
    origin_routes = graph.get_routes_at_stop(first.dep_stop)
    origin_trips = {trip_key(c) for c in graph.get_connections_from(first.dep_stop)
                    if c.pickup_allowed and first.dep_time <= c.dep_time < root['deadline']}
    rows = []
    for old, hub in zip(root['hubs'], wide['hubs'], strict=True):
        options = downstream_options(graph, blocks, hub['stop'], hub['scheduled_ready'],
                                     root['deadline'], (), protocol).get(root['goal'], [])
        original = retain_options(graph, options, protocol['max_first_cids_per_hub'])
        if original != [dict(o, first_path=tuple(o['first_path'])) for o in old['options']]:
            # JSON lists versus generator tuples are representation only.
            import json
            if json.loads(json.dumps(original)) != old['options']:
                raise ValueError('Schedule reconstruction changed the original four-car menu')
        keep = {o['first_cid'] for o in old['options']}
        for option in options:
            rows.append(dict(arrival_cid=hub['arrival_cid'], stop=hub['stop'],
                first_cid=option['first_cid'], first_path=list(option['first_path']),
                tail_paths=[list(p) for p in option['tail_paths']],
                original_retained=option['first_cid'] in keep,
                original_exclusion=None if option['first_cid'] in keep else 'four_first_vehicle_cap',
                exclusion_is_safe_dominance=False))
        hub['options'] = options
        variants, itineraries = defaultdict(list), []
        for option in options:
            variants[option['first_cid']].append(option)
            for tail in option['tail_paths'] or [None]:
                itinerary = [option['first_path']]+([] if tail is None else [tail])
                itineraries.append(dict(rides=itinerary,
                    trip_ids=[trip_key(graph.connections[p[0]]) for p in itinerary],
                    routes=[graph.connections[p[0]].route for p in itinerary],
                    scheduled_arrival=graph.connections[itinerary[-1][-1]].arr_time))
        first_trips = {trip_key(graph.connections[o['first_cid']]) for o in options}
        tail_trips = {trip_key(graph.connections[p[0]]) for o in options for p in o['tail_paths']}
        hub.update(variants=dict(variants), tails=itineraries,
            raw_first_cids=len(variants), retained_first_cids=len(variants),
            raw_option_count=len(options), retained_option_count=len(options),
            overlapping_first_tail_trips=sorted(first_trips & tail_trips),
            root_tail_overlap=sorted(first_trips & tail_trips),
            incoming_trip_reused=first.trip_id in first_trips | tail_trips,
            downstream_only_first_cids=[o['first_cid'] for o in options
                if graph.connections[o['first_cid']].route not in origin_routes])
    trip_hubs = defaultdict(set)
    for hub in wide['hubs']:
        for option in hub['options']:
            for path in [option['first_path'], *option['tail_paths']]:
                trip_hubs[trip_key(graph.connections[path[0]])].add(hub['stop'])
    finals = {trip for hub in wide['hubs'] for option in hub['options']
              for trip in final_services(graph, option)}
    wide.update(shared_tail_trips={trip:sorted(stops) for trip,stops in trip_hubs.items() if len(stops)>1},
        origin_replicable_final_trips=sorted(finals & origin_trips),
        downstream_only_final_trips=sorted(finals-origin_trips),
        has_downstream_only_direct_tail=any(not o['tail_paths'] and
            graph.connections[o['first_cid']].route not in origin_routes
            for hub in wide['hubs'] for o in hub['options']),
        has_two_transfer_option=any(o['tail_paths'] for hub in wide['hubs'] for o in hub['options']),
        has_direct_tail_option=any(not o['tail_paths'] for hub in wide['hubs'] for o in hub['options']),
        selection_metadata_scope='Original capped-root and hub selection; selection tags were not recomputed after expansion')
    return wide, rows


def generation_ledger(graph, root, protocol):
    """Record finite candidate filters and every reached generation frontier.

    No outcome/forecast selects the enumeration or its exclusion reasons.
    Non-generated longer paths are represented by their cutoff prefix, not
    invented as a complete all-network route list.
    """
    blocks = build_trip_blocks(graph)
    trips, positions = blocks
    departures, cutoffs, visited = [], [], set()

    def paths(cid, depth):
        connection = graph.connections[cid]
        path, stop = [], connection.dep_stop
        for leg_id in trips[trip_key(connection)][positions[cid]:]:
            leg = graph.connections[leg_id]
            reason = ('data_topology_gap' if leg.dep_stop != stop else
                      'scheduled_arrival_horizon' if leg.arr_time >= root['deadline'] else
                      'eight_segment_limit' if len(path)==protocol['max_tail_legs'] else None)
            if reason:
                cutoffs.append(dict(depth=depth, first_cid=cid, prefix=list(path),
                    next_cid=leg_id, reason=reason, safe_pruning=False))
                break
            path.append(leg_id)
            stop = leg.arr_stop
            yield tuple(path)
        else:
            cutoffs.append(dict(depth=depth, first_cid=cid, prefix=list(path),
                next_cid=None, reason='end_of_local_vehicle_data', safe_pruning=False))

    def station(stop, ready, excluded, depth):
        context = stop, float(ready), tuple(excluded), depth
        if context in visited:
            return []
        visited.add(context)
        counts, allowed = Counter(), []
        for c in sorted(graph.get_connections_from(stop), key=lambda c:(c.dep_time,c.id)):
            reasons = []
            if not c.pickup_allowed: reasons.append('no_pickup')
            if c.trip_id in excluded: reasons.append('excluded_same_trip')
            if c.dep_time < ready: reasons.append('scheduled_departure_before_ready')
            if c.dep_time > ready+protocol['tail_first_wait']: reasons.append('900_second_window')
            if c.dep_time >= root['deadline']: reasons.append('scheduled_departure_horizon')
            if not reasons:
                if counts[c.route] >= protocol['services_per_route']:
                    reasons.append('two_services_per_route')
                else:
                    counts[c.route] += 1
                    allowed.append(c.id)
            departures.append(dict(stop=stop, scheduled_ready=ready, depth=depth,
                excluded_trips=list(excluded), cid=c.id, trip=c.trip_id,
                scheduled_departure=c.dep_time, reasons=reasons,
                safe_pruning=bool(reasons) and all(r=='no_pickup' for r in reasons)))
        return allowed

    for hub in root['hubs']:
        for cid in station(hub['stop'],hub['scheduled_ready'],(),1):
            for path in paths(cid,1):
                last = graph.connections[path[-1]]
                ready = last.arr_time+graph.stops[last.arr_stop].min_transfer_time
                for nxt in station(last.arr_stop,ready,(trip_key(graph.connections[cid]),),2):
                    for tail in paths(nxt,2):
                        cutoffs.append(dict(depth=2,first_cid=nxt,prefix=list(tail),next_cid=None,
                            reason='two_tail_rides_limit',safe_pruning=False))
    return dict(departures=departures, path_frontiers=cutoffs,
        declared_support={'data':'Existing dated local graph only',
            'strict_dominance_pruning':0,'public_physical_impossibility_pruning':0},
        counts=dict(departure_contexts=len(visited), candidate_occurrences=len(departures),
                    reasons=dict(Counter(reason for r in departures for reason in r['reasons'])),
                    frontiers=dict(Counter(r['reason'] for r in cutoffs))))


def missing_labels(graph, metadata, root):
    edges, issues = label_coverage(graph, metadata, root)
    return dict(menu_edges=len(edges), issues=issues,
        usage='Outcome availability audit only; never a root/menu/action selection rule')


def scheduled_early_support(graph, model, history, stop, ready, deadline, action_cids):
    """Current public support of schedule-early departures, without labels."""
    rows = []
    for c in graph.get_connections_from(stop):
        if not c.pickup_allowed or c.dep_time >= ready:
            continue
        trip, slot = model._positions[c.id,'departure']
        law = history.law(trip)
        mass = law.marginal(slot)
        legal = (~law.canceled[slot]) & (law.times[slot]>=ready) & (law.times[slot]<deadline)
        probability = float(mass[legal].sum())
        rows.append(dict(cid=c.id,trip=trip,scheduled_departure=c.dep_time,
            ready=ready,in_action_menu=c.id in action_cids,
            public_legal_mass=probability,public_support_legal=bool((mass[legal]>0).any()),
            legal_support_min=float(law.times[slot][legal & (mass>0)].min()) if probability>0 else None))
    return rows
