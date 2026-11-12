"""Static discovery of onboard choices with distinct downstream opportunities.

Only schedule fields are read. Selection is saved before any replay metadata
is supplied to the controls. The finite search is a mechanism probe, not a
network prevalence estimate. No model parameter or actual loss selects roots.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import random
from time import perf_counter

from src.recourse_itinerary_router import build_trip_blocks, trip_key
from src.transit_graph import Connection, Stop, TransitGraph


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'experiments/reconstruction_v25/city_transfer_endpoints.json.gz'
DAYS = ('2023-07-16', '2023-07-17')
PROTOCOL = dict(seed=20261112, days=DAYS, first_departure_window=[28800, 30600],
    max_seed_first_trips_per_day=48, max_selected_per_day=6,
    horizon=2400, max_first_legs=8, max_tail_legs=8,
    tail_first_wait=900, services_per_route=2, max_tail_rides=2,
    max_hubs=3, max_first_cids_per_hub=4,
    selection='Two roots per layer and date: downstream-only direct tail, cross-hub shared trip, and two-transfer. Distinct first vehicles and route/OD, schedule only.',
    label_access=False)


def static_graphs(source=SOURCE):
    """Keep the dated replay's local IDs without reading endpoint labels."""
    with gzip.open(source, 'rt') as handle:
        archive = json.load(handle)
    stops = {key: row for day in archive['dates'].values()
             for key, row in day['stops'].items()}
    with (ROOT / 'experiments/reconstruction_data_audit/station_transfer_bounds.json').open() as handle:
        transfers = json.load(handle)['stations']
    stop_ids = {key: i for i, key in enumerate(sorted(stops))}
    graphs = {}
    for day in DAYS:
        graph = TransitGraph()
        for key, row in stops.items():
            mct = transfers.get(key, {}).get('conservative_max_stated_seconds')
            graph.add_stop(Stop(stop_ids[key], row['name'], row['latitude'],
                                row['longitude'], 120 if mct is None else mct))
        for row in archive['dates'][day]['connections']:
            if row['dropoff_allowed']:
                graph.add_connection(Connection(len(graph.connections), row['route_id'],
                    row['trip_id'], stop_ids[row['dep_stop']], stop_ids[row['arr_stop']],
                    row['scheduled_departure_seconds'], row['scheduled_arrival_seconds'],
                    pickup_allowed=row['pickup_allowed']))
        graphs[day] = graph
    return graphs


def ride_prefixes(graph, blocks, cid, deadline, max_legs):
    trips, positions = blocks
    at, path = graph.connections[cid].dep_stop, []
    for leg_id in trips[trip_key(graph.connections[cid])][positions[cid]:]:
        leg = graph.connections[leg_id]
        if leg.dep_stop != at or leg.arr_time >= deadline or len(path) == max_legs:
            break
        path.append(leg_id)
        at = leg.arr_stop
        yield tuple(path)


def departures(graph, stop, ready, deadline, excluded=(), protocol=PROTOCOL):
    """A declared static cap per line, identical for every solver."""
    count = Counter()
    result = []
    for c in sorted(graph.get_connections_from(stop), key=lambda c: (c.dep_time, c.id)):
        if (not c.pickup_allowed or c.trip_id in excluded or c.dep_time < ready
                or c.dep_time > ready + protocol['tail_first_wait'] or c.dep_time >= deadline):
            continue
        if count[c.route] == protocol['services_per_route']:
            continue
        count[c.route] += 1
        result.append(c.id)
    return result


def downstream_options(graph, blocks, stop, ready, deadline, excluded=(), protocol=PROTOCOL):
    """All selected direct rides and one-transfer continuations, by goal.

    At a second platform all retained services to the same goal share one
    RouteOption. Multiple alighting points of the first tail ride remain
    separate alternatives; they are not silently optimized using outcomes.
    """
    by_goal = defaultdict(list)
    for cid in departures(graph, stop, ready, deadline, excluded, protocol):
        first_trip = trip_key(graph.connections[cid])
        for path in ride_prefixes(graph, blocks, cid, deadline, protocol['max_tail_legs']):
            last = graph.connections[path[-1]]
            by_goal[last.arr_stop].append(dict(first_cid=cid, first_path=path,
                transfer_stop=None, tail_paths=[], scheduled_arrival=last.arr_time,
                scheduled_transfer_ready=None))
            if protocol['max_tail_rides'] < 2:
                continue
            transfer_ready = last.arr_time + graph.stops[last.arr_stop].min_transfer_time
            tails = defaultdict(list)
            for nxt in departures(graph, last.arr_stop, transfer_ready, deadline,
                                   (*excluded, first_trip), protocol):
                for tail in ride_prefixes(graph, blocks, nxt, deadline, protocol['max_tail_legs']):
                    goal = graph.connections[tail[-1]].arr_stop
                    if goal != stop:
                        tails[goal].append(tail)
            for goal, paths in tails.items():
                by_goal[goal].append(dict(first_cid=cid, first_path=path,
                    transfer_stop=last.arr_stop, tail_paths=sorted(set(paths)),
                    scheduled_arrival=min(graph.connections[p[-1]].arr_time for p in paths),
                    scheduled_transfer_ready=transfer_ready))
    return by_goal


def final_services(graph, option):
    paths = option['tail_paths'] or [option['first_path']]
    return {trip_key(graph.connections[path[-1]]) for path in paths}


def retain_options(graph, options, limit):
    """Choose first vehicles statically, retaining all their alighting variants."""
    best = {}
    for option in options:
        trip = trip_key(graph.connections[option['first_cid']])
        key = (option['scheduled_arrival'], bool(option['tail_paths']), option['first_path'])
        if trip not in best or key < best[trip][0]:
            best[trip] = key, option
    ordered = [o for _, o in sorted(best.values(), key=lambda pair: pair[0])]
    selected, seen = [], set()
    for option in ordered:
        route = graph.connections[option['first_cid']].route
        if route not in seen:
            selected.append(option)
            seen.add(route)
        if len(selected) == limit:
            break
    selected_ids = {o['first_cid'] for o in selected}
    selected.extend(o for o in ordered if o['first_cid'] not in selected_ids)
    keep = {o['first_cid'] for o in selected[:limit]}
    return sorted((o for o in options if o['first_cid'] in keep),
                  key=lambda o: (o['first_cid'], len(o['first_path']), o['first_path']))


def structure_roots(graph, day, cid, protocol=PROTOCOL):
    """Enumerate goals with two different reachable downstream opportunities."""
    blocks = build_trip_blocks(graph)
    first = graph.connections[cid]
    start, deadline = first.dep_time - 30, first.dep_time - 30 + protocol['horizon']
    path = tuple(p[-1] for p in ride_prefixes(graph, blocks, cid, deadline, protocol['max_first_legs']))
    full_same_vehicle_goals = {graph.connections[p[-1]].arr_stop
        for p in ride_prefixes(graph, blocks, cid, deadline, len(graph.connections))}
    origin_routes = graph.get_routes_at_stop(first.dep_stop)
    origin_trips = {trip_key(c) for c in graph.get_connections_from(first.dep_stop)
                    if c.pickup_allowed and first.dep_time <= c.dep_time < deadline}
    by_goal = defaultdict(list)
    for index, leg_id in enumerate(path):
        leg = graph.connections[leg_id]
        ready = leg.arr_time + graph.stops[leg.arr_stop].min_transfer_time
        options = downstream_options(graph, blocks, leg.arr_stop, ready, deadline,
                                     (), protocol)
        for goal, candidates in options.items():
            if goal == first.dep_stop or goal in full_same_vehicle_goals:
                continue
            retained = retain_options(graph, candidates, protocol['max_first_cids_per_hub'])
            unique = [o['first_cid'] for o in retained
                      if graph.connections[o['first_cid']].route not in origin_routes]
            variants = defaultdict(list)
            rides = []
            for option in retained:
                variants[option['first_cid']].append(option)
                for tail in option['tail_paths'] or [None]:
                    itinerary = [option['first_path']] + ([] if tail is None else [tail])
                    rides.append(dict(rides=itinerary,
                        trip_ids=[trip_key(graph.connections[p[0]]) for p in itinerary],
                        routes=[graph.connections[p[0]].route for p in itinerary],
                        scheduled_arrival=graph.connections[itinerary[-1][-1]].arr_time))
            first_trips = {trip_key(graph.connections[o['first_cid']]) for o in retained}
            tail_trips = {trip_key(graph.connections[p[0]]) for o in retained for p in o['tail_paths']}
            by_goal[goal].append(dict(arrival_cid=leg_id, path_index=index,
                stop=leg.arr_stop, prefix=path[:index+1], scheduled_ready=ready,
                options=retained, variants=dict(variants), tails=rides,
                raw_option_count=len(candidates), retained_option_count=len(retained),
                raw_first_cids=len({o['first_cid'] for o in candidates}),
                retained_first_cids=len(variants),
                overlapping_first_tail_trips=sorted(first_trips & tail_trips),
                root_tail_overlap=sorted(first_trips & tail_trips),
                incoming_trip_reused=first.trip_id in first_trips | tail_trips,
                downstream_only_first_cids=unique))
    roots, failures = [], Counter()
    for goal, hubs in sorted(by_goal.items()):
        if len({h['stop'] for h in hubs}) < 2:
            failures['fewer_than_two_hubs'] += 1
            continue
        if not any(h['downstream_only_first_cids'] for h in hubs):
            failures['no_downstream_only_line'] += 1
            continue
        final_sets = [{t for o in h['options'] for t in final_services(graph, o)} for h in hubs]
        if len({frozenset(ts) for ts in final_sets}) < 2 or len(set.union(*final_sets)) < 2:
            failures['equivalent_final_services'] += 1
            continue
        if not set.union(*final_sets).difference(origin_trips):
            failures['all_final_services_available_at_origin'] += 1
            continue
        # Include the first, last, and first unique-opportunity hub. This cap
        # is applied before labels and shared by all competing controls.
        raw_hubs = len(hubs)
        if len(hubs) > protocol['max_hubs']:
            unique_index = next(i for i, h in enumerate(hubs) if h['downstream_only_first_cids'])
            indices = sorted({0, len(hubs)-1, unique_index})
            hubs = [hubs[i] for i in indices]
        trip_hubs = defaultdict(set)
        for h in hubs:
            for option in h['options']:
                for ride in [option['first_path'], *option['tail_paths']]:
                    trip_hubs[trip_key(graph.connections[ride[0]])].add(h['stop'])
        shared = {trip: sorted(stops) for trip, stops in trip_hubs.items() if len(stops) > 1}
        final_sets = [{t for o in h['options'] for t in final_services(graph, o)} for h in hubs]
        downstream_final = set.union(*final_sets).difference(origin_trips)
        if not downstream_final:
            failures['hub_cap_lost_downstream_final_service'] += 1
            continue
        roots.append(dict(root_id=f'onboard_{day}_{cid}_{goal}', day=day,
            root_kind='onboard', start=start, deadline=deadline, stop=first.dep_stop,
            goal=goal, origin_name=graph.stops[first.dep_stop].name,
            goal_name=graph.stops[goal].name, first_cid=cid, first_trip=first.trip_id,
            first_route=first.route, first_path=path, scheduled_board=first.dep_time,
            hubs=hubs, shared_tail_trips=shared,
            raw_hub_count=raw_hubs,
            origin_replicable_final_trips=sorted(set.union(*final_sets) & origin_trips),
            downstream_only_final_trips=sorted(downstream_final),
            has_downstream_only_direct_tail=any(not o['tail_paths'] and
                graph.connections[o['first_cid']].route not in origin_routes
                for h in hubs for o in h['options']),
            has_two_transfer_option=any(o['tail_paths'] for h in hubs for o in h['options']),
            has_direct_tail_option=any(not o['tail_paths'] for h in hubs for o in h['options']),
            legal_action_contract='Onboard until an eligible arrival; alight or continue. MCT at each changed vehicle.',
            selection_uses_actual=False))
    return roots, dict(failures), len(by_goal)


def screen_day(graph, day, protocol=PROTOCOL):
    lower, upper = protocol['first_departure_window']
    eligible = [c for c in graph.connections if c.pickup_allowed and lower <= c.dep_time <= upper]
    # One departure per physical first vehicle avoids nearby windows of the
    # same trip being counted as distinct discoveries. Shuffle before picking
    # its departure point, so this does not systematically start at terminals.
    rng = random.Random(protocol['seed'] + int(day[-2:]))
    rng.shuffle(eligible)
    unique, seen = [], set()
    for c in eligible:
        if c.trip_id not in seen:
            unique.append(c.id)
            seen.add(c.trip_id)
    seeds = unique[:protocol['max_seed_first_trips_per_day']]
    ledger, catalog = [], []
    for draw, cid in enumerate(seeds):
        candidates, excluded, goals = structure_roots(graph, day, cid, protocol)
        ledger.append(dict(draw=draw, first_cid=cid, first_trip=graph.connections[cid].trip_id,
            distinct_goals_considered=goals, eligible_goals=[r['goal'] for r in candidates],
            excluded=excluded))
        catalog.extend(candidates)
    selected, selected_ids, seen_route_od, seen_trips = [], set(), set(), set()
    groups = [('downstream_only_direct', [r for r in catalog if r['has_downstream_only_direct_tail']]),
              ('shared_across_hubs', [r for r in catalog if r['shared_tail_trips']]),
              ('two_transfers', [r for r in catalog if r['has_two_transfer_option']])]
    layer_counts = Counter()
    for _ in range(2):
        for layer, group in groups:
            candidate = next((r for r in group if r['root_id'] not in selected_ids
                and r['first_trip'] not in seen_trips
                and (r['first_route'], r['stop'], r['goal']) not in seen_route_od), None)
            if candidate is None:
                continue
            candidate['selected_layer'] = layer
            selected.append(candidate)
            selected_ids.add(candidate['root_id'])
            seen_route_od.add((candidate['first_route'], candidate['stop'], candidate['goal']))
            seen_trips.add(candidate['first_trip'])
            layer_counts[layer] += 1
            if len(selected) == protocol['max_selected_per_day']:
                break
    return dict(network_stops=len(graph.stops), network_connections=len(graph.connections),
        eligible_departures=len(eligible), eligible_distinct_first_trips=len(unique),
        sampled_first_trips=len(seeds), candidate_root_goal_pairs=len(catalog),
        selected_count=len(selected), selected_layers=dict(layer_counts),
        eligible_layers={layer: len(group) for layer, group in groups},
        ledger=ledger, selected=selected,
        candidate_catalog=[dict(root_id=r['root_id'], first_cid=r['first_cid'], goal=r['goal'],
            hub_stops=[h['stop'] for h in r['hubs']], shared_trip_count=len(r['shared_tail_trips']),
            two_transfer=r['has_two_transfer_option'], direct_tail=r['has_direct_tail_option']) for r in catalog])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE / 'selection.json')
    args = parser.parse_args()
    started = perf_counter()
    graphs = static_graphs()
    by_day = {day: screen_day(graphs[day], day) for day in DAYS}
    result = dict(protocol=PROTOCOL, source=str(SOURCE.relative_to(ROOT)), by_day=by_day,
        selected=[root for day in DAYS for root in by_day[day]['selected']],
        selection_wall=perf_counter()-started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({day: {key: val for key, val in row.items()
        if key not in ('ledger', 'selected', 'candidate_catalog')} for day, row in by_day.items()}, indent=2))


if __name__ == '__main__':
    main()
