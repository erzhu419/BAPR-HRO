"""Resolve original V35 training queries using public static endpoint identities."""
from collections import Counter, defaultdict

VEHICLE = ('operator_id', 'vehicle_number', 'route_id')
ENDPOINT = (*VEHICLE, 'stop_id', 'stop_sequence', 'kind', 'scheduled_seconds')


def key(row, fields):
    return tuple(row[name] for name in fields)


def index(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[key(row, fields)].append(row)
    return result


def query_endpoints(connections):
    return [dict(source_connection_id=c['id'], trip_id=c['trip_id'],
                 **{name:c[name] for name in VEHICLE}, stop_id=c[side+'_stop'],
                 stop_sequence=c[side+'_sequence'], kind=kind,
                 scheduled_seconds=c['scheduled_'+kind+'_seconds'])
            for c in connections for kind, side in (('departure','dep'),('arrival','arr'))]


def literal_identity(query, catalog):
    """Classify the public meaning of the ID used by the old decoder."""
    trips = [row for row in catalog['trips'] if row['trip_id'] == query['trip_id']]
    if len(trips) != 1:
        return 'unresolved_literal_id', None
    source = trips[0]
    if key(source, VEHICLE[:2]) != key(query, VEHICLE[:2]):
        return 'different_physical_train', source
    endpoints = [row for row in catalog['endpoints']
                 if row['trip_id'] == query['trip_id']
                 and row['stop_sequence'] == query['stop_sequence'] and row['kind'] == query['kind']]
    if len(endpoints) == 1 and key(endpoints[0], ENDPOINT) == key(query, ENDPOINT):
        return 'same_public_endpoint', source
    return 'same_train_changed_public_endpoint', source


def audit_day(day, connections, old_catalog, source_catalog):
    old_endpoints = index(old_catalog['endpoints'], ENDPOINT)
    source_endpoints = index(source_catalog['endpoints'], ENDPOINT)
    old_vehicles = index(old_catalog['trips'], VEHICLE)
    source_vehicles = index(source_catalog['trips'], VEHICLE)
    source_trains = index(source_catalog['trips'], VEHICLE[:2])
    rows = []
    for query in query_endpoints(connections):
        signature, vehicle = key(query, ENDPOINT), key(query, VEHICLE)
        before, after = old_endpoints[signature], source_endpoints[signature]
        old_trips, new_trips = old_vehicles[vehicle], source_vehicles[vehicle]
        if len(old_trips) != 1 or len(new_trips) > 1 or len(before) != 1 or len(after) > 1:
            status = 'ambiguous_public_identity'
        elif not new_trips:
            status = ('same_physical_train_changed_route' if source_trains[vehicle[:2]]
                      else 'physical_service_absent_from_active_regional_catalog')
        elif not after:
            status = 'same_service_endpoint_changed_or_absent'
        elif before[0]['trip_id'] != query['trip_id']:
            raise ValueError('Frozen query does not belong to its original public static endpoint')
        else:
            status = 'same_literal_id' if after[0]['trip_id'] == query['trip_id'] else 'renamed_id'
        literal, source_trip = literal_identity(query, source_catalog)
        matched = status in ('same_literal_id', 'renamed_id')
        rows.append(dict(date=day, **query, status=status,
                         mapped_trip_id=after[0]['trip_id'] if matched else None,
                         original_literal_identity=literal,
                         original_literal_source_trip=source_trip))
    return dict(rows=rows, summary=dict(date=day, queries=len(rows),
        statuses=dict(Counter(r['status'] for r in rows)),
        original_literal_identities=dict(Counter(r['original_literal_identity'] for r in rows)),
        matched_endpoints=sum(r['mapped_trip_id'] is not None for r in rows)))


def decoder_maps(evidence):
    trips, canonical = defaultdict(dict), {}
    for row in evidence['rows']:
        raw = row['mapped_trip_id']
        if raw is None:
            continue
        target = row['source_connection_id'], row['stop_id'], row['scheduled_seconds']
        endpoint = row['stop_sequence'], row['kind']
        if endpoint in trips[raw] and trips[raw][endpoint] != target:
            raise ValueError('Two frozen queries claim one public endpoint')
        if raw in canonical and canonical[raw] != row['trip_id']:
            raise ValueError('A public train has multiple frozen identities')
        trips[raw][endpoint] = target
        canonical[raw] = row['trip_id']
    return dict(trips), canonical


def canonical_frames(frames, canonical):
    """Endpoint IDs already address frozen queries; map all trip statuses too."""
    for frame in frames:
        raw = frame['trip_id']
        frame['raw_trip_id'] = raw
        frame['trip_id'] = canonical[raw]
        frame['trip_statuses'] = [[canonical[t], status] for t, status in frame['trip_statuses']]
    return frames


def compare_queries(old_rows, corrected_rows, evidence):
    identity = lambda r: (r['date'], r['source_connection_id'], r['kind'])
    old = {identity(r):r for r in old_rows}
    fixed = {identity(r):r for r in corrected_rows}
    mapping = {identity(r):r for day in evidence.values() for r in day['rows']}
    if old.keys() != fixed.keys() or old.keys() != mapping.keys():
        raise ValueError('The original training query denominator changed')
    rows = []
    for item, before in old.items():
        after, public = fixed[item], mapping[item]
        for name in ('trip_id','route_id','stop_id','stop_sequence','scheduled',
                     'prediction_clock','actual_status','actual_seconds','official_journey'):
            if before[name] != after[name]:
                raise ValueError(f'Frozen query or label changed: {item}, {name}')
        if before['error_eligible'] and after['error_eligible']:
            status = ('retained_same_numeric_error' if before['error_seconds'] == after['error_seconds']
                      else 'retained_changed_numeric_error')
        elif before['error_eligible']:
            status = 'lost_eligibility'
        elif after['error_eligible']:
            status = 'recovered_eligibility'
        else:
            status = 'still_ineligible'
        rows.append(dict(date=item[0], source_connection_id=item[1], kind=item[2],
            identity_status=public['status'], original_literal_identity=public['original_literal_identity'],
            eligibility_change=status, original_eligible=before['error_eligible'],
            corrected_eligible=after['error_eligible'], original_error_seconds=before['error_seconds'],
            corrected_error_seconds=after['error_seconds'],
            original_forecast_seconds=before['forecast_seconds'],
            corrected_forecast_seconds=after['forecast_seconds']))
    return dict(rows=rows, summary=dict(queries=len(rows),
        eligibility_changes=dict(Counter(r['eligibility_change'] for r in rows)),
        original_eligible_identity=dict(Counter(r['original_literal_identity'] for r in rows if r['original_eligible'])),
        original_eligible=sum(r['original_eligible'] for r in rows),
        corrected_eligible=sum(r['corrected_eligible'] for r in rows)))
