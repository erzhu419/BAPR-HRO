"""Match a bounded Ist-Daten rail sample to Zurich graph events.

Only rows tagged REAL verify an event time. The archive prefixes cover SBB
journeys and cannot supply the missing competing services for a city replay.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime
from io import BytesIO, StringIO
import json
from pathlib import Path
import pickle
import struct
import sys
import tarfile
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile
from zoneinfo import ZoneInfo
import zlib

from google.transit import gtfs_realtime_pb2 as pb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.swiss_full.build_route_priors_temporal import build_temporal_priors
from src.gtfs_parser import load_stops, parse_time
from src.gtfs_rt_parser import scheduled_services_by_date

DATES = ('2023-10-12', '2023-10-25')
ARCHIVE = 'https://archive.opentransportdata.swiss/istdaten/2023/ist-daten-2023-10.zip'
DOCUMENTATION = 'https://opentransportdata.swiss/en/cookbook/historic-and-statistics-cookbook/actual-data/'
PREFIX_BYTES = 1024 ** 2
AGENCY = '11'
OPERATOR = '85:11'
FORECAST_CUTOFF_MINUTES = 5
FIELDS = {
    'departure': ('ABFAHRTSZEIT', 'AB_PROGNOSE', 'AB_PROGNOSE_STATUS'),
    'arrival': ('ANKUNFTSZEIT', 'AN_PROGNOSE', 'AN_PROGNOSE_STATUS'),
}


def get_range(byte_range, count):
    request = Request(ARCHIVE, headers={'Range': f'bytes={byte_range}'})
    with urlopen(request, timeout=30) as response:
        if response.status != 206 or int(response.headers['Content-Length']) != count:
            raise ValueError('Ist-Daten server did not honor the bounded byte range')
        content_range = response.headers['Content-Range']
        return response.read(count), int(content_range.rsplit('/', 1)[1])


def archive_prefixes():
    tail, size = get_range('-65557', 65557)
    with ZipFile(BytesIO(tail)) as archive:
        for service_date in DATES:
            member = archive.getinfo(f'ist-daten-2023-10/{service_date}_istdaten.csv')
            if member.compress_type != ZIP_DEFLATED or member.compress_size < PREFIX_BYTES:
                raise ValueError(f'Unexpected compression for {member.filename}')
            start = member.header_offset + size - len(tail)
            block, _ = get_range(f'{start}-{start + PREFIX_BYTES - 1}', PREFIX_BYTES)
            header = struct.unpack_from('<4s5H3L2H', block)
            if header[0] != b'PK\x03\x04':
                raise ValueError(f'Unexpected ZIP header for {member.filename}')
            content = zlib.decompressobj(-15).decompress(block[30 + sum(header[-2:]):])
            lines = content[:content.rfind(b'\n') + 1].decode('utf-8-sig')
            rows = list(csv.DictReader(StringIO(lines), delimiter=';'))
            # The last train-number group can be truncated by the byte cap.
            tail_group = rows[-1]['BETREIBER_ID'], rows[-1]['LINIEN_ID']
            yield service_date, (start, start + PREFIX_BYTES - 1), [
                row for row in rows if (row['BETREIBER_ID'], row['LINIEN_ID']) != tail_group
            ]


def static_graph_events(graph, identity):
    zurich = ROOT / 'data/zurich_gtfs'
    ids = {c.trip_id for c in graph.connections if identity[c.trip_id]['agency_id'] == AGENCY}
    with (zurich / 'trips.txt').open(encoding='utf-8-sig') as handle:
        trips = {row['trip_id']: row for row in csv.DictReader(handle) if row['trip_id'] in ids}
    stops_by_coordinate = defaultdict(set)
    for sid, stop in load_stops(str(zurich)).items():
        stops_by_coordinate[stop.name, stop.lat, stop.lon].add(sid.split(':')[0])

    stop_times = defaultdict(set)
    with (zurich / 'stop_times.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['trip_id'] in trips:
                for kind in FIELDS:
                    uic = row['stop_id'].split(':')[0]
                    scheduled_time = row[f'{kind}_time']
                    minute = parse_time(scheduled_time)
                    key = row['trip_id'], kind, minute, uic
                    seconds = 60 * minute + int(scheduled_time.split(':')[-1])
                    stop_times[key].add((int(row['stop_sequence']), seconds))

    events = {}
    for connection in graph.connections:
        trip_id = connection.trip_id
        if trip_id not in trips:
            continue
        for kind, stop_id, minute in (
            ('departure', connection.dep_stop, connection.dep_time),
            ('arrival', connection.arr_stop, connection.arr_time),
        ):
            stop = graph.stops[stop_id]
            uics = stops_by_coordinate[stop.name, stop.lat, stop.lon]
            candidates = {(uic, sequence, seconds) for uic in uics
                          for sequence, seconds in stop_times[trip_id, kind, minute, uic]}
            if len(candidates) == 1:
                uic, sequence, seconds = candidates.pop()
                events[connection.id, kind] = {
                    'trip_id': trip_id, 'train_number': trips[trip_id]['trip_short_name'],
                    'service_id': trips[trip_id]['service_id'], 'route_id': identity[trip_id]['route_id'],
                    'uic': uic, 'stop_sequence': sequence, 'scheduled_minute': minute,
                    'gtfs_scheduled_seconds': seconds,
                }
    return events


def seconds_of_service(value, service_date):
    parsed = datetime.strptime(value, '%d.%m.%Y %H:%M:%S' if value.count(':') == 2 else '%d.%m.%Y %H:%M')
    return int((parsed - datetime.fromisoformat(service_date)).total_seconds())


def match_realized(service_date, rows, events, active):
    active_events = {key: event for key, event in events.items() if event['service_id'] in active}
    train_numbers = {event['train_number'] for event in active_events.values()}
    records = defaultdict(set)
    status_counts = Counter()
    cancellations = set()
    for row in rows:
        if row['BETREIBER_ID'] != OPERATOR or row['LINIEN_ID'] not in train_numbers:
            continue
        if datetime.strptime(row['BETRIEBSTAG'], '%d.%m.%Y').date().isoformat() != service_date:
            continue
        for kind, (planned, time, status) in FIELDS.items():
            if not row[planned]:
                continue
            minute = seconds_of_service(row[planned], service_date) // 60
            key = row['LINIEN_ID'], row['BPUIC'], kind, minute
            if row['FAELLT_AUS_TF'] == 'true':
                cancellations.add(key)
                continue
            if row['ZUSATZFAHRT_TF'] == 'true' or row['DURCHFAHRT_TF'] == 'true':
                continue
            seconds = seconds_of_service(row[time], service_date) if row[time] else None
            records[key].add((row['FAHRT_BEZEICHNER'], row[status], seconds))

    matched = {}
    for (connection_id, kind), event in active_events.items():
        key = event['train_number'], event['uic'], kind, event['scheduled_minute']
        options = records[key]
        if key in cancellations:
            status_counts['canceled_flag_at_matching_stop'] += 1
            continue
        if len(options) != 1:
            status_counts['ambiguous_key' if options else 'absent_key'] += 1
            continue
        journey, status, seconds = next(iter(options))
        status_counts[status or 'empty'] += 1
        if status == 'REAL' and seconds is not None:
            matched[connection_id, kind] = dict(event, official_journey=journey,
                                                actual_seconds=seconds,
                                                delay_seconds=seconds - 60 * event['scheduled_minute'])
    return matched, dict(status_counts), len(active_events)


def observed_transfers(graph, legs):
    departures = defaultdict(list)
    for c in legs:
        departures[graph.connections[c].dep_stop].append(c)
    transfers = []
    for incoming, leg in legs.items():
        arrival = graph.connections[incoming]
        buffer = graph.stops[arrival.arr_stop].min_transfer_time
        for outgoing in departures[arrival.arr_stop]:
            departure = graph.connections[outgoing]
            after = legs[outgoing]
            if leg['arrival']['official_journey'] == after['departure']['official_journey']:
                continue
            slack = departure.dep_time - arrival.arr_time - buffer
            if 0 <= slack <= 15:
                actual_slack = (after['departure']['actual_seconds'] - leg['arrival']['actual_seconds']
                                - 60 * buffer)
                transfers.append({'arrival_connection': incoming, 'departure_connection': outgoing,
                                  'scheduled_slack_min': slack, 'actual_slack_seconds': actual_slack})
    return transfers


def latest_direct_forecasts(service_date, events, windows):
    """Read the latest explicit forecast for each requested event and time window."""
    local_midnight = int(datetime.fromisoformat(service_date).replace(tzinfo=ZoneInfo('Europe/Zurich')).timestamp())
    endpoints = defaultdict(list)
    for (checkpoint, key), (start, end) in windows.items():
        event = events[key]
        endpoints[event['trip_id']].append((checkpoint, key, event, start, end))
    lower = min(start for start, _ in windows.values())
    upper = max(end for _, end in windows.values())
    compact_date = service_date.replace('-', '')
    forecasts = {}
    path = ROOT / f'data/swiss_rt/gtfs-rt/{service_date}.tar.bz2'
    with tarfile.open(path, 'r|bz2') as archive:
        for member in archive:
            if not member.name.endswith('.gtfsrt'):
                continue
            capture = int(datetime.fromisoformat(Path(member.name).stem).timestamp()) - local_midnight
            if not lower <= capture <= upper:
                continue
            feed = pb.FeedMessage.FromString(archive.extractfile(member).read())
            receipt = max(capture, feed.header.timestamp - local_midnight) if feed.header.HasField('timestamp') else capture
            for entity in feed.entity:
                if not entity.HasField('trip_update'):
                    continue
                update = entity.trip_update
                if (update.trip.trip_id not in endpoints or update.trip.start_date != compact_date
                        or update.trip.schedule_relationship != pb.TripDescriptor.SCHEDULED):
                    continue
                by_sequence = {stop.stop_sequence: stop for stop in update.stop_time_update
                               if stop.HasField('stop_sequence')}
                for checkpoint, key, event, start, end in endpoints[update.trip.trip_id]:
                    _, kind = key
                    planned = event['gtfs_scheduled_seconds']
                    if not start <= receipt <= end:
                        continue
                    forecast_key = checkpoint, key
                    if forecast_key in forecasts and receipt <= forecasts[forecast_key]['receipt_seconds']:
                        continue
                    stop = by_sequence.get(event['stop_sequence'])
                    if (stop is None or stop.schedule_relationship != pb.TripUpdate.StopTimeUpdate.SCHEDULED
                            or (stop.stop_id and stop.stop_id.split(':')[0] != event['uic']) or not stop.HasField(kind)):
                        continue
                    estimate = getattr(stop, kind)
                    if estimate.HasField('time'):
                        forecast = estimate.time - local_midnight
                    elif estimate.HasField('delay'):
                        forecast = planned + estimate.delay
                    else:
                        continue
                    forecasts[forecast_key] = {'receipt_seconds': receipt, 'forecast_seconds': forecast}
    return forecasts


def score_precutoff_forecasts(service_date, realized, priors):
    """Latest explicit local GTFS-RT forecast received 5–60 minutes before schedule."""
    windows = {('precutoff', key): (event['gtfs_scheduled_seconds'] - 3600,
                                  event['gtfs_scheduled_seconds'] - 60 * FORECAST_CUTOFF_MINUTES)
               for key, event in realized.items()}
    forecasts = latest_direct_forecasts(service_date, realized, windows)
    rows = []
    for (_, (connection_id, kind)), forecast in sorted(forecasts.items()):
        event = realized[connection_id, kind]
        actual = event['actual_seconds']
        prior = priors.get(event['route_id'])
        if prior is None:
            continue
        rows.append(dict(forecast, connection_id=connection_id, kind=kind,
                         scheduled_seconds=event['gtfs_scheduled_seconds'], actual_seconds=actual,
                         forecast_error_min=(forecast['forecast_seconds'] - actual) / 60,
                         schedule_error_min=-event['delay_seconds'] / 60,
                         prior_error_min=prior['mean'] - event['delay_seconds'] / 60))
    return {'cutoff_min_before_scheduled': FORECAST_CUTOFF_MINUTES, 'lookback_min_before_scheduled': 60,
            'n_matched_forecasts_with_past_prior': len(rows),
            'n_received_before_realized_event': sum(row['receipt_seconds'] < row['actual_seconds'] for row in rows),
            'paired_mae_min': {name: sum(abs(row[f'{name}_error_min']) for row in rows) / len(rows) if rows else None
                               for name in ('forecast', 'schedule', 'prior')},
            'rows': rows}


def main():
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as handle:
        graph = pickle.load(handle)
    with (ROOT / 'data/per_day_distributions_corrected.pkl').open('rb') as handle:
        priors = build_temporal_priors(pickle.load(handle))
    identity = json.loads((HERE / 'graph_identity.json').read_text())['by_trip']
    events = static_graph_events(graph, identity)
    calendars = scheduled_services_by_date(ROOT / 'data/swiss_gtfs', DATES,
                                           {event['service_id'] for event in events.values()})
    results = {}
    for service_date, byte_range, rows in archive_prefixes():
        realized, counts, active_events = match_realized(service_date, rows, events, calendars[service_date])
        legs = {}
        for connection_id, kind in realized:
            if kind != 'departure' or (connection_id, 'arrival') not in realized:
                continue
            departure = realized[connection_id, 'departure']
            arrival = realized[connection_id, 'arrival']
            if (departure['official_journey'] == arrival['official_journey']
                    and departure['actual_seconds'] <= arrival['actual_seconds']):
                legs[connection_id] = {'departure': departure, 'arrival': arrival}
        transfers = observed_transfers(graph, legs)
        training = priors[service_date]
        errors = defaultdict(list)
        for event in realized.values():
            if event['route_id'] in training['priors']:
                delay = event['delay_seconds'] / 60
                forecast = training['priors'][event['route_id']]['mean']
                errors[event['route_id']].append((delay, forecast - delay))
        route_errors = {route: {'n_events': len(rows), 'schedule_mae_min': sum(abs(a) for a, _ in rows) / len(rows),
                                'prior_mae_min': sum(abs(e) for _, e in rows) / len(rows),
                                'prior_bias_min': sum(e for _, e in rows) / len(rows)}
                        for route, rows in sorted(errors.items())}
        flat = [pair for route in errors.values() for pair in route]
        results[service_date] = {
            'archive_byte_range': byte_range, 'prefix_rows': len(rows), 'static_graph_endpoint_matches': len(events),
            'active_graph_endpoints': active_events, 'join_status_counts': counts,
            'realized_events': len(realized), 'realized_legs': len(legs),
            'scheduled_transfer_opportunities': len(transfers),
            'realized_missed_transfers': sum(row['actual_slack_seconds'] < 0 for row in transfers),
            'past_training_dates': training['train_dates'], 'n_scored_events': len(flat),
            'schedule_mae_min': sum(abs(a) for a, _ in flat) / len(flat) if flat else None,
            'prior_mae_min': sum(abs(e) for _, e in flat) / len(flat) if flat else None,
            'prior_bias_min': sum(e for _, e in flat) / len(flat) if flat else None,
            'by_route': route_errors, 'legs': legs, 'transfers': transfers,
            'precutoff_forecasts': score_precutoff_forecasts(service_date, realized, training['priors']),
        }
    output = {'scope': 'Held-out measured SBB rail endpoints and transfer opportunities in bounded archive prefixes. '
              'Conditional on unambiguous REAL records; sampled pre-cutoff forecasts from local GTFS-RT. '
              'Missing competing non-rail trips precludes citywide journey replay.',
              'archive': ARCHIVE, 'documentation': DOCUMENTATION, 'compressed_bytes_requested': 65557 + len(DATES) * PREFIX_BYTES,
              'operator': OPERATOR, 'dates': results}
    (HERE / 'realized_subset.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    for service_date, values in results.items():
        print(service_date, {key: values[key] for key in ('active_graph_endpoints', 'realized_events', 'realized_legs',
                                                          'scheduled_transfer_opportunities', 'realized_missed_transfers',
                                                          'schedule_mae_min', 'prior_mae_min')})


if __name__ == '__main__':
    main()
