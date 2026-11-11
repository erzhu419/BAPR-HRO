"""Extract identified GTFS-RT predictions and explicit service statuses.

These feeds contain predictions, not a vehicle-event replay.  Delay severity,
trip cancellation, skipped stops, and absent prediction data remain distinct.
"""
from __future__ import annotations

import csv
from datetime import date, datetime
import os
import pickle
import tarfile
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
from google.transit import gtfs_realtime_pb2 as pb


def parse_single_snapshot(data: bytes, route_ids: Optional[set[str]] = None,
                          service_date: Optional[str] = None) -> list[dict]:
    """Keep trip identity, publication time, status and forecast separately.

    A trip-status record is emitted even for a canceled trip with no stop updates.
    One arrival prediction (or departure fallback) is kept per stop update.
    ``event_time`` is a forecast timestamp; it is never labeled a realized event.
    """
    feed = pb.FeedMessage()
    feed.ParseFromString(data)
    records = []
    for entity in feed.entity:
        if not entity.HasField('trip_update'):
            continue
        tu = entity.trip_update
        if route_ids is not None and tu.trip.route_id not in route_ids:
            continue
        if service_date is not None and tu.trip.start_date != service_date:
            continue
        trip_status = pb.TripDescriptor.ScheduleRelationship.Name(tu.trip.schedule_relationship)
        published_at = int(feed.header.timestamp) if feed.header.HasField('timestamp') else None
        measured_at = int(tu.timestamp) if tu.HasField('timestamp') else None
        base = dict(route_id=tu.trip.route_id, trip_id=tu.trip.trip_id,
                    service_date=tu.trip.start_date, start_time=tu.trip.start_time,
                    published_at=published_at, measured_at=measured_at, trip_relationship=trip_status,
                    is_canceled=trip_status == 'CANCELED', source='prediction')
        records.append(dict(base, source='declared_service_status', record_type='trip_status', stop_id='',
                            stop_sequence=None, stop_relationship=None,
                            event_kind=None, event_time=None, delay_seconds=None, hour=-1))
        for stu in tu.stop_time_update:
            relationship = pb.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(stu.schedule_relationship)
            kind = None
            for field in ('delay', 'time'):
                for candidate in ('arrival', 'departure'):
                    if stu.HasField(candidate) and getattr(stu, candidate).HasField(field):
                        kind = candidate
                        break
                if kind:
                    break
            event = getattr(stu, kind) if kind else None
            event_time = int(event.time) if event is not None and event.HasField('time') else None
            delay = int(event.delay) if event is not None and event.HasField('delay') else None
            records.append(dict(base, record_type='stop_prediction', stop_id=stu.stop_id,
                                stop_sequence=int(stu.stop_sequence) if stu.HasField('stop_sequence') else None,
                                stop_relationship=relationship, event_kind=kind,
                                event_time=event_time, delay_seconds=delay,
                                hour=datetime.fromtimestamp(event_time, ZoneInfo('Europe/Zurich')).hour
                                if event_time is not None else -1))
    return records


def event_key(record: dict) -> tuple:
    """Vehicle-service identity; a later snapshot is a revision, not a new trip."""
    return (record['route_id'], record['service_date'], record['trip_id'],
            record['start_time'], record['record_type'],
            record['stop_sequence'] if record['stop_sequence'] is not None else record['stop_id'])


def latest_predictions(records) -> list[dict]:
    """Use one latest prediction/status per identified event for calibration."""
    latest = {}
    for record in records:
        key = event_key(record)
        previous = latest.get(key)
        if previous is None or (record['published_at'] or 0) >= (previous['published_at'] or 0):
            latest[key] = record
    return list(latest.values())


def process_day(tar_path: str, sample_interval: int = 10,
                max_snapshots: Optional[int] = None, route_ids: Optional[set[str]] = None,
                service_date: Optional[str] = None) -> list[dict]:
    """Stream selected snapshots; retain latest event revisions, not repetitions.

    ``max_snapshots`` limits small field audits.  No archive is unpacked to disk.
    Archive parse errors propagate rather than masquerading as an empty day.
    """
    def records():
        sampled = 0
        index = 0
        with tarfile.open(tar_path, 'r|bz2') as tar:
            for member in tar:
                if not member.name.endswith('.gtfsrt'):
                    continue
                selected = index % sample_interval == 0
                index += 1
                if not selected:
                    continue
                handle = tar.extractfile(member)
                yield from parse_single_snapshot(handle.read(), route_ids, service_date)
                sampled += 1
                if max_snapshots is not None and sampled >= max_snapshots:
                    break
    return latest_predictions(records())


def load_route_metadata(gtfs_dir: str) -> dict[str, dict]:
    """Keep agency and route IDs; equal public line names need not be one route."""
    with open(os.path.join(gtfs_dir, 'routes.txt'), encoding='utf-8-sig') as handle:
        return {row['route_id']: {'agency_id': row['agency_id'],
                                 'name': row['route_short_name'] or row['route_long_name'] or row['route_id']}
                for row in csv.DictReader(handle)}


def recover_graph_route_identity(graph, gtfs_dir: str) -> dict:
    """Recover exact graph identities by trip ID, without public-name matching.

    Only the trips already present in the small graph are retained while the
    static trips file is read once. The graph and historical pickles are not
    changed by this lookup.
    """
    wanted = {connection.trip_id for connection in graph.connections if connection.trip_id}
    routes = {trip: set() for trip in wanted}
    services = {trip: set() for trip in wanted}
    with open(os.path.join(gtfs_dir, 'trips.txt'), encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['trip_id'] in wanted:
                routes[row['trip_id']].add(row['route_id'])
                services[row['trip_id']].add(row['service_id'])
    metadata = load_route_metadata(gtfs_dir)
    by_trip = {}
    for trip, matches in routes.items():
        if len(matches) == 1:
            rid = next(iter(matches))
            if rid in metadata:
                service = next(iter(services[trip])) if len(services[trip]) == 1 else None
                by_trip[trip] = dict(route_id=rid, service_id=service, **metadata[rid])
    return {
        'by_trip': by_trip,
        'missing_trip_ids': sorted(trip for trip, matches in routes.items() if not matches),
        'ambiguous_trip_ids': {trip: sorted(matches) for trip, matches in routes.items() if len(matches) > 1},
        'missing_route_metadata': sorted({rid for matches in routes.values() for rid in matches if rid not in metadata}),
        'matched_connections': sum(connection.trip_id in by_trip for connection in graph.connections),
        'total_connections': len(graph.connections),
    }


def scheduled_services_by_date(gtfs_dir, dates, service_ids) -> dict[str, set[str]]:
    """Resolve selected GTFS calendars, including service-date exceptions."""
    weekdays = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')
    date_keys = {day.replace('-', ''): (day, weekdays[date.fromisoformat(day).weekday()]) for day in dates}
    active = {day: set() for day, _ in date_keys.values()}
    with open(os.path.join(gtfs_dir, 'calendar.txt'), encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            service = row['service_id']
            if service not in service_ids:
                continue
            for compact, (day, weekday) in date_keys.items():
                if row['start_date'] <= compact <= row['end_date'] and row[weekday] == '1':
                    active[day].add(service)
    with open(os.path.join(gtfs_dir, 'calendar_dates.txt'), encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            service, compact = row['service_id'], row['date']
            if service not in service_ids or compact not in date_keys:
                continue
            day, _ = date_keys[compact]
            if row['exception_type'] == '1':
                active[day].add(service)
            elif row['exception_type'] == '2':
                active[day].discard(service)
    return active


def summarize_predictions(records, route_metadata: Optional[dict] = None) -> dict:
    """Summarize unique forecast events and cancellation among observed trips.

    Cancellation uses scheduled/canceled trip statuses as its denominator, not
    stop forecasts. Missing trip updates supply no outcome and no denominator.
    """
    records = latest_predictions(records)
    trips = {(r['route_id'], r['service_date'], r['trip_id'], r['start_time']): r
             for r in records if r['record_type'] == 'trip_status'}
    grouped = {}
    for r in records:
        rid = r['route_id']
        out = grouped.setdefault(rid, {'delays': [], 'by_hour': {}, 'n_canceled': 0,
                                      'n_status_trips': 0, 'n_skipped': 0, 'n_no_data': 0})
        if r['record_type'] == 'trip_status':
            if r['trip_relationship'] in ('SCHEDULED', 'CANCELED'):
                out['n_status_trips'] += 1
                out['n_canceled'] += int(r['is_canceled'])
            continue
        out['n_skipped'] += int(r['stop_relationship'] == 'SKIPPED')
        out['n_no_data'] += int(r['stop_relationship'] == 'NO_DATA')
        trip = trips.get((rid, r['service_date'], r['trip_id'], r['start_time']), r)
        if trip['is_canceled'] or r['stop_relationship'] in ('SKIPPED', 'NO_DATA'):
            continue
        if r['delay_seconds'] is None:
            continue
        delay = r['delay_seconds'] / 60.0
        if abs(delay) > 120:
            continue
        out['delays'].append(delay)
        if r['hour'] >= 0:
            out['by_hour'].setdefault(r['hour'], []).append(delay)
    distributions = {}
    for rid, values in grouped.items():
        arr = np.asarray(values.pop('delays'), dtype=float)
        hours = values.pop('by_hour')
        n = len(arr)
        count = values['n_status_trips']
        distributions[rid] = {
            **values, **((route_metadata or {}).get(rid, {'name': rid, 'agency_id': ''})),
            'route_id': rid, 'delays_min': arr,
            'mean': float(arr.mean()) if n else None,
            'std': float(arr.std()) if n else None,
            'median': float(np.median(arr)) if n else None,
            'p95': float(np.percentile(arr, 95)) if n else None,
            'n_obs': n, 'n_severe_delay': int((np.abs(arr) > 30).sum()),
            'cancel_rate': values['n_canceled'] / count if count else None,
            'cancel_source': 'explicit_status_among_observed_scheduled_trips',
            'data_kind': 'latest_unique_predictions_not_realized_events',
            'by_hour': {hour: {'mean': float(np.mean(ds)), 'std': float(np.std(ds)), 'n': len(ds)}
                        for hour, ds in hours.items()},
        }
    return distributions


def build_delay_distributions(rt_dir: str, route_metadata: Optional[dict] = None,
                              n_days: int = 7, sample_interval: int = 20,
                              verbose: bool = True) -> dict:
    """Build forecast-calibrated distributions, keyed by original route ID."""
    files = sorted(f for f in os.listdir(rt_dir) if f.endswith('.tar.bz2'))[:n_days]
    records = []
    for filename in files:
        day = process_day(os.path.join(rt_dir, filename), sample_interval)
        records.extend(day)
        if verbose:
            print(f'{filename}: {len(day)} unique prediction/status records', flush=True)
    return summarize_predictions(records, route_metadata)


def save_distributions(distributions: dict, path: str):
    with open(path, 'wb') as handle:
        pickle.dump({rid: dict(d, delays_min=d['delays_min'].tolist())
                     for rid, d in distributions.items()}, handle)


def load_distributions(path: str) -> dict:
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    for value in data.values():
        value['delays_min'] = np.array(value['delays_min'])
    return data


if __name__ == '__main__':
    distributions = build_delay_distributions('data/swiss_rt/gtfs-rt',
                                             load_route_metadata('data/swiss_gtfs'))
    save_distributions(distributions, 'data/delay_distributions_corrected.pkl')
