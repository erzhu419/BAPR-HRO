"""Recover declared platform pairs without changing the station-level model.

Only static trip, sequence, stop and schedule fields enter the mapping. A
planned platform or published assignment is never labelled an actual platform.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from experiments.decision_gap_v1.panel import read, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def seconds(value):
    h, m, s = map(int, value.split(':'))
    return 3600*h+60*m+s


def static_sources(connections, regional=ROOT/'data/zurich_gtfs', static=ROOT/'data/swiss_gtfs'):
    """Index only the requested trip/stop-sequence endpoints from public CSVs."""
    wanted = {(row['trip_id'], row[side+'_sequence'])
              for row in connections for side in ('dep', 'arr')}
    endpoints = defaultdict(list)
    with (Path(regional)/'stop_times.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            key = row['trip_id'], int(row['stop_sequence'])
            if key in wanted:
                endpoints[key].append(row)
    with (Path(static)/'stops.txt').open(encoding='utf-8-sig') as handle:
        stops = {row['stop_id']:row for row in csv.DictReader(handle)}
    transfers = defaultdict(list)
    with (Path(static)/'transfers.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            transfers[row['from_stop_id'], row['to_stop_id']].append(row)
    return endpoints, stops, transfers


def endpoint(connection, kind, index, stops):
    side = 'arr' if kind == 'arrival' else 'dep'
    key = connection['trip_id'], connection[side+'_sequence']
    rows = index.get(key, [])
    result = dict(trip_id=key[0], stop_sequence=key[1], event_kind=kind,
        station_id=connection[side+'_stop'], candidates=rows, stop_id=None,
        platform_specific=False, actual_platform_verified=False)
    if len(rows) != 1:
        return dict(result, status='missing' if not rows else 'ambiguous')
    row = rows[0]
    if row['stop_id'].split(':')[0] != connection[side+'_stop']:
        return dict(result, status='station_mismatch')
    if seconds(row[kind+'_time']) != connection['scheduled_'+kind+'_seconds']:
        return dict(result, status='schedule_mismatch')
    stop = stops.get(row['stop_id'])
    if stop is None:
        return dict(result, status='missing_stop_metadata')
    result.update(stop_id=row['stop_id'], stop_name=stop.get('stop_name'),
        location_type=stop.get('location_type'), parent_station=stop.get('parent_station'),
        platform_specific=(stop.get('location_type') in ('', '0')
                           and stops.get(stop.get('parent_station'), {}).get('location_type') == '1'))
    return dict(result, status='resolved', source='static stop_times trip_id and stop_sequence')


def transfer_pair(incoming, outgoing, fixed_mct, index, stops, transfers):
    """Verify an exact directed planned-platform rule, preserving unknowns."""
    arrival, departure = endpoint(incoming, 'arrival', index, stops), endpoint(outgoing, 'departure', index, stops)
    result = dict(arrival=arrival, departure=departure, fixed_mct_seconds=fixed_mct,
        exact_records=[], applicable_inherited_records=[], station_alias_records=[],
        exact_direct_pair=False, applicable_inherited_rule=False, planned_pair_source_verified=False,
        actual_platform_pair_verified=False, stated_pair_seconds=None, fixed_value_matches=None)
    if arrival['status'] != 'resolved' or departure['status'] != 'resolved':
        return dict(result, status='endpoint_unresolved')
    a, b = arrival['stop_id'], departure['stop_id']
    exact = transfers.get((a, b), [])
    result['exact_records'] = exact
    result['exact_direct_pair'] = bool(exact)
    # In this Swiss feed the bare BPUIC is itself location_type=0. Only
    # Parent<BPUIC> is a station; string-prefix similarity grants no inheritance.
    parent_a, parent_b = arrival.get('parent_station'), departure.get('parent_station')
    aliases_a = [a]+([parent_a] if stops.get(parent_a, {}).get('location_type') == '1' else [])
    aliases_b = [b]+([parent_b] if stops.get(parent_b, {}).get('location_type') == '1' else [])
    inherited = []
    for pair in ((left,right) for left in aliases_a for right in aliases_b):
        if pair != (a,b):
            inherited.extend(transfers.get(pair, []))
    result.update(applicable_inherited_records=inherited, applicable_inherited_rule=bool(inherited))
    for pair in dict.fromkeys(((arrival['station_id'],b),(a,departure['station_id']))):
        if pair != (a,b) and not (pair[0] in aliases_a and pair[1] in aliases_b):
            result['station_alias_records'].extend(transfers.get(pair, []))
    if not (arrival['platform_specific'] and departure['platform_specific']):
        return dict(result, status='station_only_endpoint')
    applicable = [*exact, *inherited]
    if not applicable:
        return dict(result, status='same_platform_without_stated_rule' if a == b else 'exact_pair_rule_missing')
    kinds = {row['transfer_type'] for row in applicable}
    if kinds == {'3'}:
        return dict(result, status='planned_pair_prohibited')
    if '3' in kinds:
        return dict(result, status='conflicting_applicable_rules')
    stated = [int(row['min_transfer_time']) for row in applicable
              if row['transfer_type'] == '2' and row.get('min_transfer_time')]
    if not stated or len(set(stated)) != 1 or any(row['transfer_type'] != '2' for row in applicable):
        return dict(result, status='no_unique_explicit_minimum')
    value = stated[0]
    result.update(planned_pair_source_verified=True, stated_pair_seconds=value,
                  fixed_value_matches=value == fixed_mct)
    return dict(result, status='planned_pair_verified' if value == fixed_mct else 'fixed_mct_differs_from_planned_pair')


def audit_development():
    """Audit both transfer pairs for all580 frozen events, without loss filtering."""
    from experiments.reconstruction_v34.discover import context
    data = read(ROOT/'experiments/reconstruction_v34/data.json.gz')
    registration = read(ROOT/'experiments/decision_gap_v1/events/registration.json.gz')
    connections = [row for day in data['dates'].values() for row in day['connections']]
    index, stops, transfers = static_sources(connections)
    contexts = {}
    pairs = []
    for root in registration['roots']:
        if root['day'] not in contexts:
            contexts[root['day']] = context(data, root['day'])
        metadata = contexts[root['day']][4]
        for event in root['rows']:
            for role, arrival, departure, mct in (
                ('upstream',event['current_cid'],event['upstream_cid'],event['upstream_mct']),
                ('hub',event['source_arrival_cid'],event['hub_cid'],event['downstream_mct'])):
                pairs.append(dict(root_index=root['root_index'], day=root['day'],
                    event_index=event['event_index'], role=role, arrival_cid=arrival, departure_cid=departure,
                    **transfer_pair(metadata[arrival], metadata[departure], mct, index, stops, transfers)))
    summary = dict(roots=len(registration['roots']), registered_events=registration['summary']['opportunities'],
        pairs=len(pairs), statuses=dict(Counter(row['status'] for row in pairs)),
        planned_pair_source_verified=sum(row['planned_pair_source_verified'] for row in pairs),
        exact_direct_pairs=sum(row['exact_direct_pair'] for row in pairs),
        inherited_rule_pairs=sum(row['applicable_inherited_rule'] for row in pairs),
        actual_platform_pairs_verified=sum(row['actual_platform_pair_verified'] for row in pairs),
        fixed_mct_changed=False,
        static_endpoint_keys=len(index), static_endpoint_ambiguities=sum(len(rows)!=1 for rows in index.values()),
        key_negative_roots=[row for row in pairs if (row['root_index'],row['event_index']) in ((6,0),(18,0),(32,2))],
        scope='All frozen development events; scheduled platforms and published rules only, no actual-platform or independent-date confirmation',
        transfer_semantics='Only a location_type=1 parent_station contributes inherited transfer rules; the bare Swiss BPUIC is not promoted from location_type=0 to station.',
        transfer_reference='https://gtfs.org/documentation/schedule/reference/#transferstxt')
    write(HERE/'platform_pair_audit.json.gz', dict(summary=summary, rows=pairs,
        sources=['data/zurich_gtfs/stop_times.txt','data/swiss_gtfs/stops.txt','data/swiss_gtfs/transfers.txt'],
        dropped_fields='build_dated_replay.py:dated_connections maps full stop_id to its station prefix while retaining trip and sequence'))
    write(HERE/'platform_pair_summary.json', summary)
    print({key:value for key,value in summary.items() if key!='key_negative_roots'})


def audit_rt_schema():
    """Inspect one already-developed public frame; no new-date values opened."""
    import gzip
    import tarfile
    from google.transit import gtfs_realtime_pb2 as pb

    day = '2023-10-25'
    data = read(ROOT/'experiments/reconstruction_v34/data.json.gz')
    messages = read(ROOT/'experiments/reconstruction_v34/messages.json.gz')
    member_name = messages['dates'][day]['frames'][0]['archive_member']
    trips = {row['trip_id'] for row in data['dates'][day]['connections']}
    archive_path = ROOT/f'data/swiss_rt/gtfs-rt/{day}.tar.bz2'
    with tarfile.open(archive_path, 'r|bz2') as archive:
        for member in archive:
            if member.name == member_name:
                feed = pb.FeedMessage.FromString(archive.extractfile(member).read())
                break
        else:
            raise ValueError('Existing public receipt source member was not found')
    counts, rows = Counter(), []
    for entity in feed.entity:
        if not entity.HasField('trip_update') or entity.trip_update.trip.trip_id not in trips:
            continue
        update = entity.trip_update
        counts['trip_updates'] += 1
        for stop in update.stop_time_update:
            counts['stop_updates'] += 1
            counts['stop_id_present'] += bool(stop.stop_id)
            counts['stop_id_has_suffix'] += ':' in stop.stop_id
            counts['assigned_stop_id_present'] += bool(stop.stop_time_properties.assigned_stop_id)
            rows.append(dict(trip_id=update.trip.trip_id, stop_sequence=stop.stop_sequence,
                published_stop_id=stop.stop_id, assigned_stop_id=stop.stop_time_properties.assigned_stop_id or None))
    with gzip.open(ROOT/f'experiments/reconstruction_data_audit/rail_replay_sources/{day}.csv.gz','rt') as handle:
        actual_columns = handle.readline().strip().split(';')
    result = dict(scope='Single existing developed-day public frame; schema/content availability probe, not independent calibration',
        day=day, archive=str(archive_path.relative_to(ROOT)), archive_member=member_name,
        counts=dict(counts), rows=rows, actual_source_columns=actual_columns,
        actual_platform_columns=[], actual_platform_pairs_verified=0,
        extraction_loss='The existing RT extractor tests station prefixes then stores only cid/kind/status/time, dropping the full published stop ID and assignment fields.',
        interpretation='Published stop IDs identify public platform information. They do not independently verify the physically used arrival/departure platform. No actual-platform field exists in this endpoint CSV schema.',
        field_reference='https://gtfs.org/documentation/realtime/reference/#message-stoptimeproperties')
    write(HERE/'rt_platform_schema_audit.json', result)
    print(dict(day=day, **counts))
    return result


if __name__ == '__main__':
    audit_development()
    audit_rt_schema()
