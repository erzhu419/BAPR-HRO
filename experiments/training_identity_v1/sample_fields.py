"""Trace both publications for every originally eligible V35 training query."""
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import json
from pathlib import Path
import tarfile
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2 as pb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def query_key(row):
    return row['date'], row['source_connection_id'], row['kind']


def field_record(row, publication, feed, entities, source_catalog, midnight):
    entity = entities[publication['entity_id']]
    update = entity.trip_update
    trip = update.trip
    raw_trip = publication.get('raw_trip_id', publication['trip_id'])
    if trip.trip_id != raw_trip:
        raise ValueError('Stored publication does not identify the raw trip entity')
    candidates = [stop for stop in update.stop_time_update
        if stop.HasField('stop_sequence') and stop.stop_sequence == row['stop_sequence']
        and (not stop.stop_id or stop.stop_id.split(':')[0] == row['stop_id'])]
    if len(candidates) != 1:
        raise ValueError('Selected publication has no unique raw endpoint update')
    stop = candidates[0]
    value = getattr(stop, row['kind'])
    has_kind = stop.HasField(row['kind'])
    raw_time = int(value.time) if has_kind and value.HasField('time') else None
    raw_delay = int(value.delay) if has_kind and value.HasField('delay') else None
    mode = 'time' if raw_time is not None else 'delay' if raw_delay is not None else 'absent'
    decoded = (raw_time-midnight if mode == 'time' else
               row['scheduled']+raw_delay if mode == 'delay' else None)
    captured = int(datetime.fromisoformat(Path(publication['archive_member']).stem).timestamp())-midnight
    header = int(feed.header.timestamp)-midnight if feed.header.HasField('timestamp') else None
    measured = int(update.timestamp)-midnight if update.HasField('timestamp') else None
    receipt = max(x for x in (captured, header, measured) if x is not None)
    source_trips = source_catalog['trips'].get(raw_trip, [])
    endpoints = source_catalog['endpoints'].get((raw_trip, row['stop_sequence'], row['kind']), [])
    source_trip = source_trips[0] if len(source_trips) == 1 else None
    source_endpoint = endpoints[0] if len(endpoints) == 1 else None
    raw_route = trip.route_id if trip.HasField('route_id') else None
    source_schedule = source_endpoint['scheduled_seconds'] if source_endpoint is not None else None
    return dict(mode=mode, publication_present=True,
        archive_member=publication['archive_member'], entity_id=entity.id,
        raw_trip_id=raw_trip, canonical_trip_id=row['trip_id'],
        raw_route_id=raw_route, raw_start_time=trip.start_time, raw_start_date=trip.start_date,
        raw_stop_id=stop.stop_id, raw_stop_sequence=stop.stop_sequence,
        raw_stop_status=pb.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(stop.schedule_relationship),
        raw_time_epoch=raw_time, raw_time_seconds=None if raw_time is None else raw_time-midnight,
        raw_delay_seconds=raw_delay,
        raw_uncertainty=int(value.uncertainty) if has_kind and value.HasField('uncertainty') else None,
        both_time_and_delay_present=raw_time is not None and raw_delay is not None,
        query_scheduled_seconds=row['scheduled'], stored_forecast_seconds=row['forecast_seconds'],
        decoded_using_frozen_query_schedule=decoded,
        decoded_matches_stored_forecast=decoded == row['forecast_seconds'],
        capture_seconds=captured, header_seconds=header, update_seconds=measured,
        recomputed_receipt_seconds=receipt, stored_receipt_seconds=publication['receipt_seconds'],
        receipt_matches_stored=receipt == publication['receipt_seconds'],
        source_catalog_trip=source_trip, source_catalog_endpoint=source_endpoint,
        raw_route_matches_source_catalog=(None if source_trip is None else raw_route == source_trip['route_id']),
        source_catalog_scheduled_seconds=source_schedule,
        source_schedule_minus_query_schedule=None if source_schedule is None else source_schedule-row['scheduled'],
        decoded_using_source_catalog_schedule=(raw_time-midnight if mode == 'time' else
            source_schedule+raw_delay if mode == 'delay' and source_schedule is not None else None))


def main():
    original = {query_key(r):r for r in read(ROOT/'experiments/reconstruction_v35/events.json.gz')['rows']
                if r['split'] == 'training' and r['error_eligible']}
    corrected = {query_key(r):r for r in read(HERE/'corrected_training_events.json.gz')['rows']}
    comparison = {query_key(r):r for r in read(HERE/'query_comparison.json.gz')['rows']}
    catalog = read(HERE/'active_public_endpoints.json.gz')['dates']
    result = {key:dict(date=key[0], source_connection_id=key[1], kind=key[2],
        original_literal_identity=comparison[key]['original_literal_identity'],
        eligibility_change=comparison[key]['eligibility_change'],
        original_eligible=True, corrected_eligible=corrected[key]['error_eligible']) for key in original}
    requested = defaultdict(lambda:defaultdict(list))
    for key, before in original.items():
        for side, row in [('old',before), ('new',corrected[key])]:
            publication = row['forecast_publication']
            if publication is None:
                result[key][side] = dict(mode='absent', publication_present=False,
                    absence_scope='No selected valid numeric publication in the saved query; '
                        'this does not assert absence of all raw messages or StopTimeEvents.',
                    stored_forecast_seconds=row['forecast_seconds'])
            else:
                requested[key[0]][publication['archive_member']].append((key,side,row,publication))
    archives = []
    for day in sorted(requested):
        source_catalog = dict(trips=defaultdict(list), endpoints=defaultdict(list))
        for trip in catalog[day]['trips']:
            source_catalog['trips'][trip['trip_id']].append(trip)
        for endpoint in catalog[day]['endpoints']:
            source_catalog['endpoints'][endpoint['trip_id'],endpoint['stop_sequence'],endpoint['kind']].append(endpoint)
        midnight = int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('Europe/Zurich')).timestamp())
        archive = ROOT/f'data/swiss_rt/gtfs-rt/{day}.tar.bz2'
        decoded_members = []
        with tarfile.open(archive, 'r|bz2') as stream:
            for member in stream:
                if member.name not in requested[day]:
                    continue
                feed = pb.FeedMessage.FromString(stream.extractfile(member).read())
                entities = {entity.id:entity for entity in feed.entity if entity.HasField('trip_update')}
                for key, side, row, publication in requested[day][member.name]:
                    result[key][side] = field_record(row, publication, feed, entities, source_catalog, midnight)
                decoded_members.append(member.name)
        if set(decoded_members) != set(requested[day]):
            raise ValueError('At least one referenced archive capture was not found')
        archives.append(dict(date=day, path=str(archive), bytes=archive.stat().st_size,
            referenced_capture_members=len(decoded_members)))
    rows = list(result.values())
    publications = [r[side] for r in rows for side in ('old','new') if r[side]['publication_present']]
    wrong_same = [r for r in rows if r['original_literal_identity']=='different_physical_train'
                  and r['eligibility_change']=='retained_same_numeric_error']
    def modes(group):
        return [dict(old_mode=k[0], new_mode=k[1], endpoints=n)
                for k,n in sorted(Counter((r['old']['mode'],r['new']['mode']) for r in group).items())]
    groups = []
    for category in sorted({(r['original_literal_identity'],r['eligibility_change']) for r in rows}):
        subset = [r for r in rows if (r['original_literal_identity'],r['eligibility_change'])==category]
        groups.append(dict(original_literal_identity=category[0],eligibility_change=category[1],
            endpoints=len(subset), value_modes=modes(subset)))
    evidence = dict(endpoints=len(wrong_same), value_modes=modes(wrong_same),
        same_value_delay_distribution_seconds=dict(sorted(Counter(
            r['old']['raw_delay_seconds'] for r in wrong_same).items())),
        source_schedule_minus_query_distribution_seconds=dict(sorted(Counter(
            r['old']['source_schedule_minus_query_schedule'] for r in wrong_same).items())),
        both_delay_zero=sum(r['old']['mode']==r['new']['mode']=='delay' and
                           r['old']['raw_delay_seconds']==r['new']['raw_delay_seconds']==0 for r in wrong_same),
        both_equal_nonzero_delay=sum(r['old']['mode']==r['new']['mode']=='delay' and
            r['old']['raw_delay_seconds']==r['new']['raw_delay_seconds'] and r['old']['raw_delay_seconds']!=0 for r in wrong_same),
        old_public_source_schedule_differs_from_frozen_query=sum(
            r['old']['source_schedule_minus_query_schedule'] not in (None,0) for r in wrong_same),
        old_decoded_public_source_time_differs_from_stored_forecast=sum(
            r['old']['decoded_using_source_catalog_schedule'] != r['old']['stored_forecast_seconds'] for r in wrong_same))
    with gzip.open(HERE/'sample_field_rows.json.gz','wt') as handle:
        json.dump(dict(rows=rows),handle,separators=(',',':'))
    audit = dict(scope='All originally eligible334 endpoint queries; original and corrected selected publications traced to raw archive fields. No mapping or fitted parameter changed.',
        endpoints=len(rows), archives=archives, rows_file='sample_field_rows.json.gz',
        payloads_saved=False, mapping_or_fit_changed=False, groups=groups,
        selected_publication_references=len(publications),
        all_selected_forecasts_reproduced=all(p['decoded_matches_stored_forecast'] for p in publications),
        all_selected_receipts_reproduced=all(p['receipt_matches_stored'] for p in publications),
        raw_route_vs_source_catalog=dict(Counter(str(p['raw_route_matches_source_catalog']) for p in publications)),
        wrong_train_same_numeric=evidence)
    (HERE/'sample_field_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
