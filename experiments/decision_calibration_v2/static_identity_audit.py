"""Audit two public GTFS identities within the frozen regional connection graph.

This writes correspondence evidence only; it never rewrites trips, messages,
menus, probabilities, labels or outcomes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from experiments.decision_calibration_v2.prepare_static import dated_connections_from_static

IDENTITY = ('operator_id', 'vehicle_number', 'route_id')
SEGMENT = ('dep_stop', 'arr_stop', 'dep_sequence', 'arr_sequence',
           'scheduled_departure_seconds', 'scheduled_arrival_seconds',
           'pickup_allowed', 'dropoff_allowed')
PLATFORM = ('dep_platform_id', 'arr_platform_id')


def trip_records(connections):
    grouped = defaultdict(list)
    for row in connections:
        grouped[row['trip_id']].append(row)
    records = {}
    for trip, rows in grouped.items():
        identities = {tuple(row[k] for k in IDENTITY) for row in rows}
        if len(identities) != 1:
            raise ValueError(f'Public trip {trip} has conflicting operator/vehicle/route metadata')
        identity = next(iter(identities))
        segments = tuple(sorted(tuple(row[k] for k in SEGMENT) for row in rows))
        platforms = tuple(sorted(tuple(row[k] for k in (*SEGMENT, *PLATFORM)) for row in rows))
        records[trip] = dict(identity=identity, signature=(*identity, segments),
                             platforms=platforms, segments=segments, connection_count=len(rows))
    return records


def public_description(trip, record):
    return dict(trip_id=trip, **dict(zip(IDENTITY, record['identity'])),
                connection_count=record['connection_count'])


def compare_day(day, old_connections, new_connections):
    old, new = trip_records(old_connections), trip_records(new_connections)
    old_index, new_index, new_vehicles = defaultdict(list), defaultdict(list), defaultdict(list)
    for trip, record in old.items():
        old_index[record['signature']].append(trip)
    for trip, record in new.items():
        new_index[record['signature']].append(trip)
        new_vehicles[record['identity'][:2]].append(trip)
    rows, matched_new = [], set()
    for trip, record in sorted(old.items()):
        row = public_description(trip, record)
        matches = new_index.get(record['signature'], [])
        if len(old_index[record['signature']]) == len(matches) == 1:
            target = matches[0]
            same_platforms = record['platforms'] == new[target]['platforms']
            status = 'same_literal_id' if target == trip else 'renamed_id'
            if not same_platforms:
                status += '_with_platform_change'
            row.update(status=status, mapped_trip_id=target, planned_platform_sequence_equal=same_platforms)
            matched_new.add(target)
        elif matches:
            row.update(status='ambiguous_signature', mapped_trip_id=None,
                       old_signature_trips=sorted(old_index[record['signature']]),
                       new_signature_trips=sorted(matches))
        else:
            counterparts = new_vehicles.get(record['identity'][:2], [])
            row.update(status='changed_public_service' if counterparts else 'absent_in_new_graph',
                       mapped_trip_id=None, same_vehicle_new_trips=sorted(counterparts),
                       old_segments=record['segments'],
                       new_counterparts=[dict(**public_description(t, new[t]), segments=new[t]['segments'])
                                         for t in sorted(counterparts)])
        rows.append(row)
    literal = []
    for trip in sorted(old.keys() & new.keys()):
        before, after = old[trip], new[trip]
        if before['signature'] != after['signature']:
            literal.append(dict(trip_id=trip, old_identity=dict(zip(IDENTITY,before['identity'])),
                new_identity=dict(zip(IDENTITY,after['identity'])), old_segments=before['segments'],
                new_segments=after['segments']))
    counts = Counter(row['status'] for row in rows)
    summary = dict(day=day, old_trips=len(old), new_trips=len(new),
        old_connections=len(old_connections), new_connections=len(new_connections),
        statuses=dict(counts), exact_unique_matches=len(matched_new),
        literal_id_overlap=len(old.keys() & new.keys()), literal_id_changed_signature=len(literal),
        planned_platform_changed_trips=sum(row.get('planned_platform_sequence_equal') is False for row in rows),
        changed_public_service_trips=counts['changed_public_service'],
        ambiguous_signature_trips=counts['ambiguous_signature'],
        unmatched_new_trips=len(new.keys()-matched_new),
        all_old_graph_trips_exactly_mapped=len(matched_new)==len(old),
        graph_signature_multiset_equal=Counter(r['signature'] for r in old.values()) ==
                                       Counter(r['signature'] for r in new.values()))
    return dict(summary=summary, rows=rows, reused_literal_id_changes=literal,
                unmatched_new=[public_description(t,new[t]) for t in sorted(new.keys()-matched_new)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-static', type=Path, required=True)
    parser.add_argument('--new-static', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dates = [f'2023-03-{number:02d}' for number in range(16,20)]
    _, old = dated_connections_from_static(dates, args.old_static)
    _, new = dated_connections_from_static(dates, args.new_static)
    days = {day:compare_day(day,old[day],new[day]) for day in dates}
    args.output.mkdir(parents=True, exist_ok=True)
    mapping_path = args.output/'static_identity_mapping.json.gz'
    with gzip.open(mapping_path, 'wt') as handle:
        json.dump(dict(days=days), handle, separators=(',',':'))
    receipt = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        sources=dict(old_static=str(args.old_static), new_static=str(args.new_static)),
        dates=dates, identity_fields=IDENTITY, connection_signature_fields=SEGMENT,
        planned_platform_fields=PLATFORM, matching='Exact equality of every regional connection; unique on both sides; no fuzzy or outcome matching',
        scope='Original 16 stations and07:30–09:35 physical-adjacency graph, including every trip connection retained by dated_connections_from_static; not a nationwide or full-day equivalence claim',
        target_actual_or_messages_or_probabilities_or_losses_read=False,
        graph_menu_mct_or_messages_changed=False,
        mapping_direction='Each rows.trip_id is the old frozen-graph ID; rows.mapped_trip_id is its uniquely matched ID in the new public static version. Reverse only exact unique matches to decode new-version messages.',
        mapping_file=mapping_path.name, mapping_bytes=mapping_path.stat().st_size,
        days=[days[day]['summary'] for day in dates])
    (args.output/'static_identity_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
