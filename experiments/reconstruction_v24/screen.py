"""Schedule-only menus with an actual downstream transfer and a rescue service."""

import gzip
import json
from collections import Counter
from functools import cache
from pathlib import Path

from experiments.reconstruction_v4.run_city_replay import OD_NAMES
from experiments.reconstruction_v6.run_city_new_od import PAIRS
from experiments.reconstruction_v22.run_natural_screen import CAP, PUBLICATION, READY_TIMES
from src.platform_nominal_index import PlatformNominalIndex
from src.recourse_itinerary_router import build_trip_blocks, trip_key
from src.transit_graph import Connection, Stop, TransitGraph


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'experiments/reconstruction_v23/city_transfer_endpoints.json.gz'
DATES = ('2023-07-23', '2023-07-24')


def static_graphs(source=SOURCE):
    """Read only static fields; preserve the existing replay's stop/edge IDs."""
    with gzip.open(source, 'rt') as handle:
        archive = json.load(handle)
    stops = {key: row for day in archive['dates'].values() for key, row in day['stops'].items()}
    with (ROOT / 'experiments/reconstruction_data_audit/station_transfer_bounds.json').open() as handle:
        transfers = json.load(handle)['stations']
    stop_ids = {key: i for i, key in enumerate(sorted(stops))}
    graphs = {}
    for day in DATES:
        graph = TransitGraph()
        for key, row in stops.items():
            mct = transfers.get(key, {}).get('conservative_max_stated_seconds') or 120
            graph.add_stop(Stop(stop_ids[key], row['name'], row['latitude'], row['longitude'], mct))
        for row in archive['dates'][day]['connections']:
            if not row['dropoff_allowed']:
                continue
            graph.add_connection(Connection(len(graph.connections), row['route_id'], row['trip_id'],
                stop_ids[row['dep_stop']], stop_ids[row['arr_stop']],
                row['scheduled_departure_seconds'], row['scheduled_arrival_seconds'],
                pickup_allowed=row['pickup_allowed']))
        graphs[day] = graph
    return graphs


def build_menu(graph, blocks, stop, goal, entry, ready, deadline, incoming_trip=None):
    """One schedule-best alighting choice per first trip; retain every tail ride."""
    trips, positions = blocks
    counts = Counter()

    def same_trip_paths(cid):
        connection = graph.connections[cid]
        at, path = connection.dep_stop, []
        for leg_id in trips[trip_key(connection)][positions[cid]:]:
            leg = graph.connections[leg_id]
            if leg.dep_stop != at or leg.arr_time > deadline:
                break
            path.append(leg_id)
            at = leg.arr_stop
            yield at, tuple(path)

    @cache
    def tails(platform):
        result = []
        for c in graph.get_connections_from(platform):
            if not c.pickup_allowed or not entry <= c.dep_time < deadline:
                continue
            for at, path in same_trip_paths(c.id):
                if at == goal:
                    result.append(path)
                    break
        return tuple(sorted(result, key=lambda p: (graph.connections[p[0]].dep_time, p)))

    by_trip = {}
    for c in graph.get_connections_from(stop):
        trip = trip_key(c)
        if (not c.pickup_allowed or not ready-120 <= c.dep_time <= ready+600
                or c.dep_time >= deadline or trip == incoming_trip):
            continue
        counts['first_connections_considered'] += 1
        candidates = []
        for at, prefix in same_trip_paths(c.id):
            leg = graph.connections[prefix[-1]]
            if at == goal:
                candidates.append(dict(first_cid=c.id, first_path=prefix, transfer_stop=None,
                    tail_paths=(), scheduled_arrival=leg.arr_time, scheduled_transfer_ready=None))
                break
            transfer_ready = leg.arr_time + graph.stops[at].min_transfer_time
            alternatives = tuple(path for path in tails(at)
                                 if trip_key(graph.connections[path[0]]) != trip)
            nominally_catchable = [path for path in alternatives
                                  if graph.connections[path[0]].dep_time >= transfer_ready]
            if len({trip_key(graph.connections[path[0]]) for path in nominally_catchable}) < 2:
                continue
            primary = min(nominally_catchable,
                          key=lambda path: (graph.connections[path[-1]].arr_time, path))
            candidates.append(dict(first_cid=c.id, first_path=prefix, transfer_stop=at,
                tail_paths=alternatives, scheduled_arrival=graph.connections[primary[-1]].arr_time,
                scheduled_transfer_ready=transfer_ready))
        counts['alighting_options_before_selection'] += len(candidates)
        for candidate in candidates:
            key = (candidate['scheduled_arrival'], candidate['transfer_stop'] is not None,
                   candidate['first_path'], candidate['tail_paths'])
            if trip not in by_trip or key < by_trip[trip][0]:
                by_trip[trip] = key, candidate
    menu = [candidate for _, candidate in sorted(by_trip.values(), key=lambda item: item[0])]
    counts['retained_options'] = len(menu)
    counts['transfer_options'] = sum(option['transfer_stop'] is not None for option in menu)
    counts['direct_options'] = len(menu)-counts['transfer_options']
    counts['tail_paths'] = sum(len(option['tail_paths']) for option in menu)
    return menu, dict(counts)


def independent_menu(graph, catalog, incoming_trip=None):
    first_trips = {trip_key(graph.connections[option['first_cid']]) for option in catalog}
    if incoming_trip is not None:
        first_trips.add(incoming_trip)
    independent, counts = [], Counter()
    for option in catalog:
        if option['transfer_stop'] is None:
            independent.append(option)
            continue
        tails = tuple(path for path in option['tail_paths']
                      if trip_key(graph.connections[path[0]]) not in first_trips)
        counts['removed_overlapping_tail_paths'] += len(option['tail_paths'])-len(tails)
        catchable = [path for path in tails if graph.connections[path[0]].dep_time
                     >= option['scheduled_transfer_ready']]
        if len({trip_key(graph.connections[path[0]]) for path in catchable}) < 2:
            counts['excluded_options_without_independent_rescue'] += 1
            continue
        independent.append(dict(option, tail_paths=tails,
            scheduled_arrival=min(graph.connections[path[-1]].arr_time for path in catchable)))
    return independent, counts


def select_pair(graph, independent):
    if len(independent) < 2:
        return [], 'fewer_than_two_independent_options'
    if not any(o['transfer_stop'] is not None for o in independent):
        return [], 'no_downstream_transfer_option'
    first = min(independent, key=lambda o: (graph.connections[o['first_cid']].dep_time,
                                            o['scheduled_arrival'], o['first_cid']))
    others = [o for o in independent if o['first_cid'] != first['first_cid']
              and (first['transfer_stop'] is not None or o['transfer_stop'] is not None)]
    second = min(others, key=lambda o: (o['scheduled_arrival'],
                                      o['transfer_stop'] is not None, o['first_cid']))
    return [first, second], 'earliest_first_departure_and_best_alternative_with_transfer'


def screen_day(day, graph):
    blocks = build_trip_blocks(graph)
    names = {stop.name: stop.id for stop in graph.stops.values()}
    roots, counts = [], Counter()
    for od, (origin_name, goal_name) in enumerate((*OD_NAMES, *PAIRS)):
        origin, goal = names[origin_name], names[goal_name]
        for start in READY_TIMES:
            counts['od_windows'] += 1
            deadline = start+CAP
            nominal = PlatformNominalIndex(graph, goal, start, deadline, trip_blocks=blocks)
            route = nominal.path(origin, start+PUBLICATION)
            boundaries = [('origin', origin, start, start+PUBLICATION, ())]
            if route:
                first_trip = trip_key(graph.connections[route[0]])
                transfer = next((i for i, cid in enumerate(route)
                                 if trip_key(graph.connections[cid]) != first_trip), None)
                if transfer is not None:
                    prefix = route[:transfer]
                    incoming = graph.connections[prefix[-1]]
                    ready = incoming.arr_time+max(graph.stops[incoming.arr_stop].min_transfer_time, PUBLICATION)
                    boundaries.append(('first_transfer', incoming.arr_stop, incoming.arr_time, ready, prefix))
                    counts['scheduled_first_transfer_roots'] += 1
            else:
                counts['no_nominal_path'] += 1
            for kind, stop, entry, ready, prefix in boundaries:
                incoming_trip = trip_key(graph.connections[prefix[-1]]) if prefix else None
                catalog, menu_counts = build_menu(graph, blocks, stop, goal, entry, ready, deadline, incoming_trip)
                counts.update(menu_counts)
                independent, removed = independent_menu(graph, catalog, incoming_trip)
                counts.update(removed)
                options, selection_reason = select_pair(graph, independent)
                best_direct = min((o for o in independent if o['transfer_stop'] is None),
                                  key=lambda o: (o['scheduled_arrival'], o['first_cid']), default=None)
                omitted_direct = (best_direct is not None and options
                                  and best_direct['first_cid'] not in {o['first_cid'] for o in options})
                counts['roots'] += 1
                counts['roots_with_catalog_options'] += bool(catalog)
                counts['roots_with_catalog_transfer_options'] += any(o['transfer_stop'] is not None for o in catalog)
                counts['roots_with_selected_pair'] += bool(options)
                counts['selected_transfer_options'] += sum(o['transfer_stop'] is not None for o in options)
                counts['selected_direct_options'] += sum(o['transfer_stop'] is None for o in options)
                counts['roots_with_direct_transfer_pair'] += bool(options) and any(o['transfer_stop'] is None for o in options)
                counts['selected_pairs_omitting_schedule_best_direct'] += bool(omitted_direct)
                roots.append(dict(root_id=f'{day}_{od}_{start}_{kind}', day=day, od=od,
                    origin_name=origin_name, goal_name=goal_name, goal=goal, start=start, deadline=deadline,
                    root_kind=kind, stop=stop, scheduled_entry=entry, scheduled_ready=ready,
                    prefix=prefix, options=options, catalog_options=catalog,
                    full_options=independent, selection_reason=selection_reason,
                    best_direct_first_cid=None if best_direct is None else best_direct['first_cid'],
                    pair_omits_best_direct=bool(omitted_direct),
                    independent_options=len(independent), menu_counts=menu_counts))
    return dict(counts=dict(counts), roots=roots)


def main():
    results = {day: screen_day(day, graph) for day, graph in static_graphs().items()}
    output = dict(scope='Schedule-only development screen on the two already inspected V23 archive days.',
                  source=str(SOURCE.relative_to(ROOT)), by_day=results)
    (HERE/'selection.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({day: result['counts'] for day, result in results.items()}, indent=2))


if __name__ == '__main__':
    main()
