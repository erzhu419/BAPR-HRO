"""Schedule-first natural roots, followed by explicit source eligibility.

The menu is constructed before checking labels or messages. Eligibility never
computes losses, candidate forecast values or policy advantages.
"""

from collections import Counter
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v32.screen import structure_roots
from experiments.reconstruction_v33.nodes import root_ride


HERE = Path(__file__).resolve().parent


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as handle:
        return json.load(handle)


def write(path, value):
    with (gzip.open(path, 'wt') if str(path).endswith('.gz') else open(path, 'w')) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=None if str(path).endswith('.gz') else 2)


def context(data, day):
    if 'station_transfer_bounds' not in data:
        data['station_transfer_bounds'] = read(
            HERE.parent/'reconstruction_data_audit/station_transfer_bounds.json')['stations']
    return dated_graph(data, day, 'gtfs_station_max', True)


def static_catalog(data, protocol):
    rows, ledger = [], {}
    for day in protocol['main_dates']:
        graph = context(data, day)[0]
        lo, hi = protocol['structure']['first_departure_window']
        first = {}
        for c in sorted(graph.connections, key=lambda c: (c.dep_time, c.id)):
            if c.pickup_allowed and lo <= c.dep_time <= hi:
                first.setdefault(c.trip_id, c.id)
        rejected, considered = Counter(), 0
        before = len(rows)
        for cid in first.values():
            candidates, excluded, goals = structure_roots(graph, day, cid, protocol['structure'])
            rows.extend(candidates)
            rejected.update(excluded)
            considered += goals
        ledger[day] = dict(stops=len(graph.stops), connections=len(graph.connections),
            first_vehicles=len(first), considered_root_goals=considered,
            qualifying_roots=len(rows)-before, topology_exclusions=dict(rejected))
        print(day, 'static roots', len(rows)-before, flush=True)
    return dict(scope='Schedule-only menus; no actual status/clock or forecast is used.',
                by_day=ledger, candidates=rows)


def label_coverage(graph, metadata, root):
    ride = root_ride(graph, root)
    edges = sorted({cid for path in ride.paths() for cid in path})
    bad = []
    for cid in edges:
        row = metadata[cid]
        dep, arr = row['departure'], row['arrival']
        if dep['status'] not in ('REAL', 'CANCELED') or arr['status'] not in ('REAL', 'CANCELED'):
            bad.append(dict(cid=cid, reason='missing_or_ambiguous', departure=dep['status'], arrival=arr['status']))
        elif dep['official_journey'] is None or dep['official_journey'] != arr['official_journey']:
            bad.append(dict(cid=cid, reason='inconsistent_endpoint_identity'))
        elif dep['status'] == arr['status'] == 'REAL' and dep['time_seconds'] > arr['time_seconds']:
            bad.append(dict(cid=cid, reason='inconsistent_endpoint_clock'))
    for path in ride.paths():
        for before, after in zip(path, path[1:]):
            a, b = metadata[before]['arrival'], metadata[after]['departure']
            if a['official_journey'] != b['official_journey']:
                bad.append(dict(cid=after, reason='inconsistent_same_vehicle_identity'))
            elif a['status'] == b['status'] == 'REAL' and a['time_seconds'] > b['time_seconds']:
                bad.append(dict(cid=after, reason='inconsistent_same_vehicle_clock'))
    # The state is an already boarded vehicle, not a hypothetical canceled ride.
    first_exit = root['hubs'][0]['arrival_cid']
    first_prefix = root['first_path'][:root['first_path'].index(first_exit)+1]
    if any(metadata[cid][kind]['status'] != 'REAL'
           for cid in first_prefix for kind in ('departure', 'arrival')):
        bad.append(dict(cid=root['first_cid'], reason='unobserved_initial_onboard_prefix'))
    return edges, bad


def message_eligibility(graph, metadata, source_to_local, root, frames):
    first_exit = root['hubs'][0]['arrival_cid']
    now = metadata[first_exit]['arrival']['time_seconds']
    if now is None:
        return []
    ride = root_ride(graph, root)
    source_trip = graph.connections[root['first_cid']].trip_id
    tail_edges = {cid for path in ride.paths() for cid in path
                  if graph.connections[cid].trip_id != source_trip}
    witnesses = []
    for frame in frames:
        if frame['receipt_seconds'] > now:
            break
        for source_id, kind, status, _prediction in frame['endpoint_updates']:
            cid = source_to_local.get(source_id)
            if cid not in tail_edges or status != 'FORECAST':
                continue
            c = graph.connections[cid]
            planned = c.dep_time if kind == 'departure' else c.arr_time
            if planned > now:
                witnesses.append(dict(cid=cid, kind=kind, receipt=frame['receipt_seconds'],
                    decision_clock=now, scheduled_target=planned))
    return witnesses


def select(data, messages, catalog, protocol):
    output, eligible_ledger, selected = {}, [], []
    for day in protocol['main_dates']:
        graph, _, _, mapping, metadata, _ = context(data, day)
        eligible = []
        for root in (r for r in catalog['candidates'] if r['day'] == day):
            edges, bad = label_coverage(graph, metadata, root)
            signals = message_eligibility(graph, metadata, mapping, root, messages['dates'][day]['frames'])
            eligible_ledger.append(dict(root_id=root['root_id'], day=day, menu_edges=len(edges),
                complete_labels=not bad, incomplete=bad, predecision_numeric_messages=len(signals),
                message_timing_example=signals[-1] if signals else None))
            if not bad and signals:
                eligible.append(root)
        groups = [('downstream_only_direct', [r for r in eligible if r['has_downstream_only_direct_tail']]),
                  ('shared_across_hubs', [r for r in eligible if r['shared_tail_trips']]),
                  ('two_transfers', [r for r in eligible if r['has_two_transfer_option']])]
        trips, route_od, chosen = set(), set(), []
        while len(chosen) < protocol['structure']['max_selected_per_day']:
            prior = len(chosen)
            for layer, group in groups:
                candidate = next((r for r in group if r['first_trip'] not in trips
                    and (r['first_route'],r['stop'],r['goal']) not in route_od), None)
                if candidate is not None:
                    selected_root = dict(candidate, selected_layer=layer,
                        static_menu_uses_actual=False,
                        selection_uses_actual_coverage_and_timing=True)
                    selected_root.pop('selection_uses_actual',None)
                    chosen.append(selected_root)
                    trips.add(candidate['first_trip'])
                    route_od.add((candidate['first_route'],candidate['stop'],candidate['goal']))
                if len(chosen) == protocol['structure']['max_selected_per_day']:
                    break
            if len(chosen) == prior:
                break
        output[day] = dict(**catalog['by_day'][day], complete_with_message=len(eligible), selected=len(chosen))
        selected.extend(chosen)
        print(day, 'complete+message',len(eligible),'selected',len(chosen),flush=True)
    return dict(protocol='protocol.json', model='model_fit.json', selection_uses_loss=False,
                by_day=output, eligibility=eligible_ledger, selected=selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static-only', action='store_true')
    args = parser.parse_args()
    began = perf_counter()
    protocol, data = read(HERE/'protocol.json'), read(HERE/'data.json.gz')
    target = HERE/'static_catalog.json.gz'
    if target.exists():
        catalog = read(target)
    else:
        catalog = static_catalog(data, protocol)
        write(target, catalog)
    if not args.static_only:
        messages = read(HERE/'messages.json.gz')
        result = select(data, messages, catalog, protocol)
        result['wall'] = perf_counter()-began
        write(HERE/'selection.json', result)


if __name__ == '__main__':
    main()
