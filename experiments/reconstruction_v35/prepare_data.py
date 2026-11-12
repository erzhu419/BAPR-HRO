"""Prepare past-date endpoint/forecast errors from existing local archives."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.reconstruction_v34 import messages as extraction
from src.calibration_model import PublicForecasts


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT = ROOT/'experiments/reconstruction_data_audit'
V5 = ROOT/'experiments/reconstruction_v5'
TRAIN = ('2023-10-01', '2023-10-02')
VALIDATION = tuple(f'2023-10-{d:02}' for d in range(3, 12))
DATES = TRAIN+VALIDATION
ACTUALS = AUDIT/'rail_calibration_dates_v2.json.gz'
LEGACY = V5/'calibration_public_frames.json.gz'


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else path.open()) as handle:
        return json.load(handle)


def write(path, value):
    with (gzip.open(path, 'wt') if str(path).endswith('.gz') else path.open('w')) as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False,
                  separators=(',', ':') if str(path).endswith('.gz') else None,
                  indent=None if str(path).endswith('.gz') else 2)


def file_info(path):
    stat = path.stat()
    return dict(path=str(path.relative_to(ROOT)), bytes=stat.st_size,
                modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())


def extract_day(day, connections):
    # Each worker reuses the exact V34 parser, with a V35 output destination.
    extraction.HERE = HERE
    return extraction.extract(day, connections)


def endpoint_rows(day, connections, frames, legacy_frames):
    """Reconstruct latest valid numeric messages at each fixed public cutoff."""
    by_id = {c['id']: c for c in connections}
    by_trip = defaultdict(list)
    for c in connections:
        by_trip[c['trip_id']].append(c['id'])
    queries = sorted((c[f'scheduled_{kind}_seconds']-300, c['id'], kind)
                     for c in connections for kind in ('departure', 'arrival'))
    forecasts, trip_statuses, endpoint_statuses = {}, {}, {}
    legacy = PublicForecasts(connections, legacy_frames)
    frame_index, rows = 0, []
    identities = set()
    for cutoff, cid, kind in queries:
        while frame_index < len(frames) and frames[frame_index]['receipt_seconds'] <= cutoff:
            frame = frames[frame_index]
            receipt = frame['receipt_seconds']
            for trip, status in frame['trip_statuses']:
                trip_statuses[trip] = status, receipt
                if status in ('CANCELED', 'DELETED'):
                    for target in by_trip.get(trip, ()):
                        forecasts.pop((target, 'departure'), None)
                        forecasts.pop((target, 'arrival'), None)
            for target, event_kind, status, predicted in frame['endpoint_updates']:
                key = target, event_kind
                endpoint_statuses[key] = status, receipt
                if status == 'FORECAST' and predicted is not None:
                    trip = by_id[target]['trip_id']
                    notice = trip_statuses.get(trip)
                    if not (notice and notice[0] in ('CANCELED', 'DELETED') and notice[1] >= receipt):
                        forecasts[key] = predicted, frame_index
                elif status in ('NO_DATA', 'SKIPPED') or (status == 'FORECAST' and predicted is None):
                    forecasts.pop(key, None)
            frame_index += 1
        c = by_id[cid]
        side = 'dep' if kind == 'departure' else 'arr'
        endpoint = c[kind]
        actual = endpoint['time_seconds'] if endpoint['status'] == 'REAL' else None
        future = actual is not None and cutoff < actual
        issued = forecasts.get((cid, kind))
        publication = None
        if issued is not None:
            prediction, source_frame_index = issued
            frame = frames[source_frame_index]
            publication = dict(frame_index=source_frame_index, **{key: frame[key] for key in (
                'archive_member', 'entity_id', 'trip_id', 'capture_seconds', 'header_seconds',
                'update_seconds', 'receipt_seconds')})
            receipt = frame['receipt_seconds']
        else:
            prediction, receipt = None, None
        old = legacy.at(cutoff).values.get((cid, kind))
        identity = day, c['trip_id'], c[f'{side}_sequence'], kind
        if identity in identities:
            raise ValueError(f'Duplicate physical endpoint: {identity}')
        identities.add(identity)
        eligible = future and prediction is not None
        rows.append(dict(date=day, split='training' if day in TRAIN else 'validation',
            source_connection_id=cid, trip_id=c['trip_id'], route_id=c['route_id'], route_name=c['route_name'],
            stop_id=c[f'{side}_stop'], stop_sequence=c[f'{side}_sequence'], kind=kind,
            scheduled=c[f'scheduled_{kind}_seconds'], prediction_clock=cutoff,
            actual_status=endpoint['status'], actual_seconds=actual,
            official_journey=endpoint['official_journey'], forecast_seconds=prediction,
            forecast_receipt=receipt, forecast_publication=publication,
            forecast_age=None if receipt is None else cutoff-receipt,
            cutoff_before_actual=None if actual is None else future,
            receipt_before_actual=None if actual is None or receipt is None else receipt < actual,
            actual_lead_seconds=None if actual is None else actual-cutoff,
            message_trip_status=trip_statuses.get(c['trip_id']),
            message_endpoint_status=endpoint_statuses.get((cid, kind)),
            error_eligible=eligible, error_seconds=actual-prediction if eligible else None,
            legacy_forecast_seconds=None if old is None else old[0],
            legacy_forecast_receipt=None if old is None else old[1],
            legacy_prediction_changed=(None if old is None else old[0]) != prediction))
    return rows


def describe(rows):
    real = [r for r in rows if r['actual_status'] == 'REAL']
    errors = [r['error_seconds'] for r in rows if r['error_eligible']]
    return dict(endpoints=len(rows), distinct_trips=len({(r['date'], r['trip_id']) for r in rows}),
        status_counts=dict(Counter(r['actual_status'] for r in rows)), real_endpoints=len(real),
        cutoff_before_actual=sum(r['cutoff_before_actual'] for r in real),
        cutoff_at_or_after_actual=sum(not r['cutoff_before_actual'] for r in real),
        latest_numeric_forecast=sum(r['forecast_seconds'] is not None for r in rows),
        real_without_numeric_forecast=sum(r['forecast_seconds'] is None for r in real),
        error_samples=len(errors), legacy_prediction_changes=sum(r['legacy_prediction_changed'] for r in rows),
        error_mean_seconds=float(np.mean(errors)) if errors else None,
        error_mae_seconds=float(np.mean(np.abs(errors))) if errors else None,
        error_sd_seconds=float(np.std(errors, ddof=1)) if len(errors) > 1 else None)


def main():
    HERE.mkdir(exist_ok=True)
    actuals, legacy = read(ACTUALS), read(LEGACY)
    results = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(extract_day, day, actuals['dates'][day]['connections']) for day in DATES]
        for future in as_completed(jobs):
            day, result, reused = future.result()
            results[day] = result
            print(day, 'reused' if reused else 'extracted', result['summary'], flush=True)
    messages = dict(dates={day: results[day] for day in DATES}, sample_seconds=extraction.SAMPLE_SECONDS,
        capture_window_seconds=extraction.WINDOW, timezone='Europe/Zurich',
        availability='max(archive capture, feed header, individual TripUpdate timestamp)',
        scope='Existing local 10/01–11 archives; all days previously used for development. '
              'Numeric fields are forecasts; service notices are not actual endpoint labels.')
    write(HERE/'messages.json.gz', messages)
    rows = [row for day in DATES for row in endpoint_rows(day, actuals['dates'][day]['connections'],
            results[day]['frames'], legacy['dates'][day]['frames'])]
    events = dict(train_dates=TRAIN, validation_dates=VALIDATION, rows=rows,
        provenance=dict(actual_source=str(ACTUALS.relative_to(ROOT)), message_source='messages.json.gz',
            source_ids='Original source connection IDs; no dated_graph local-ID remapping.',
            cutoff='scheduled event clock minus 300 seconds',
            error_eligibility='REAL, cutoff strictly before actual, and a latest valid numeric forecast at cutoff.',
            error_orientation='actual minus forecast', dates_previously_developed=True,
            static_version='20231024', static_scope='Retrospective historically valid snapshot used for model development.',
            new_server_transfer_bytes=0, model_or_policy_executed=False))
    write(HERE/'events.json.gz', events)
    route_kind = []
    for split in ('training', 'validation'):
        groups = defaultdict(list)
        for row in rows:
            if row['split'] == split:
                groups[row['route_id'], row['kind']].append(row)
        for (route, kind), grouped in sorted(groups.items()):
            route_kind.append(dict(split=split, route_id=route, route_name=grouped[0]['route_name'],
                                   kind=kind, **describe(grouped)))
    inventory = dict(stage='V35 past-date message-model validation inputs',
        train_dates=TRAIN, validation_dates=VALIDATION, dates_previously_developed=True,
        sources=[file_info(ACTUALS), file_info(LEGACY), file_info(V5/'calibration_protocol.json')],
        raw_archives=[file_info(ROOT/f'data/swiss_rt/gtfs-rt/{day}.tar.bz2') for day in DATES],
        original_endpoint_fields=list(actuals['dates'][TRAIN[0]]['connections'][0]),
        original_actual_fields=list(actuals['dates'][TRAIN[0]]['connections'][0]['departure']),
        event_fields=list(rows[0]), source_id_contract=events['provenance']['source_ids'],
        source_join='Date, operator/train number, BPUIC, event kind and scheduled minute; ambiguous identities remain unresolved.',
        static_feed=(ROOT/'data/swiss_gtfs/feed_info.txt').read_text().strip(),
        message_contract=messages['availability'], sample_seconds=extraction.SAMPLE_SECONDS,
        capture_window_seconds=extraction.WINDOW,
        by_day={day: dict(events=describe([r for r in rows if r['date'] == day]),
            messages=results[day]['summary'], actual_source=actuals['dates'][day]['filtered_source']) for day in DATES},
        by_split={split: describe([r for r in rows if r['split'] == split]) for split in ('training', 'validation')},
        route_kind=route_kind, outputs=[file_info(HERE/'messages.json.gz'), file_info(HERE/'events.json.gz')],
        new_server_transfer_bytes=0, local_raw_archives_copied=False,
        limitations=['Previously developed dates validate the message model away from V34 decision evaluation; they are not pristine holdouts.',
            'Actual sources are 16-station, 07:30–09:35 subsets from compressed daily prefixes; missing remains unresolved.',
            'The static snapshot version is Oct24; earlier dates are retrospective model-development data.',
            'Error counts are physical endpoints, correlated within trips and days; they are not independent observations.'])
    write(HERE/'data_inventory.json', inventory)
    print('events', len(rows), 'split summaries', inventory['by_split'], flush=True)


if __name__ == '__main__':
    main()
