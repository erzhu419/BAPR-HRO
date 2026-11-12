"""Inventory local public-message/actual joins without computing policy loss."""
from collections import Counter
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUDIT = ROOT / 'experiments/reconstruction_data_audit'
V5 = ROOT / 'experiments/reconstruction_v5'
V6 = ROOT / 'experiments/reconstruction_v6'
ACTUAL_SOURCES = [
    AUDIT / 'rail_calibration_dates_v2.json.gz',
    AUDIT / 'dated_replay_v2.json.gz',
    V5 / 'calibration_confirmation_actuals.json.gz',
    V5 / 'calibration_relative_confirmation_actuals.json.gz',
    V6 / 'calibration_confirmation_actuals.json.gz',
    V6 / 'calibration_selection_actuals.json.gz',
    V6 / 'decision_risk_confirmation_actuals.json.gz',
    V6 / 'decision_risk_completion_actuals.json.gz',
    V6 / 'calibration_selection_full_source_actuals.json.gz',
]
FRAME_SOURCES = [
    V5 / 'calibration_public_frames.json.gz',
    V5 / 'calibration_confirmation_public_frames.json.gz',
    V5 / 'calibration_relative_confirmation_public_frames.json.gz',
]
IDENTITY = ('id', 'trip_id', 'route_id', 'dep_stop', 'arr_stop', 'dep_sequence',
            'arr_sequence', 'scheduled_departure_seconds', 'scheduled_arrival_seconds')


def read(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def main():
    actual, actual_paths, versions = {}, {}, []
    identity_agreements = []
    sources = ACTUAL_SOURCES + ([HERE / 'data.json.gz'] if (HERE / 'data.json.gz').exists() else [])
    for path in sources:
        data = read(path)
        for day, row in data['dates'].items():
            connections = row['connections']
            versions.append(dict(day=day, path=str(path.relative_to(ROOT)),
                connections=len(connections),
                endpoint_statuses=dict(Counter(c[k]['status'] for c in connections for k in ('departure', 'arrival'))),
                outcomes=dict(Counter(c['outcome'] for c in connections))))
            if day in actual:
                old = [[c[k] for k in IDENTITY] for c in actual[day]['connections']]
                new = [[c[k] for k in IDENTITY] for c in connections]
                identity_agreements.append(dict(day=day, earlier=actual_paths[day],
                    later=str(path.relative_to(ROOT)), same_static_ids=old == new))
            actual[day], actual_paths[day] = row, str(path.relative_to(ROOT))
    frames, frame_paths = {}, {}
    for path in FRAME_SOURCES:
        data = read(path)
        for day, row in data['dates'].items():
            frames[day], frame_paths[day] = row['frames'], str(path.relative_to(ROOT))
    rows = []
    for day, actual_day in sorted(actual.items()):
        conns = {c['id']: c for c in actual_day['connections']}
        updates = [(f, u) for f in frames.get(day, ()) for u in f['endpoint_updates']]
        forecasts = [(f, u) for f, u in updates if u[2] == 'FORECAST']
        unmapped = sorted({u[0] for _, u in updates if u[0] not in conns})
        real = [(f, u) for f, u in forecasts if u[0] in conns and conns[u[0]][u[1]]['status'] == 'REAL']
        before = [(f, u) for f, u in real if f['receipt_seconds'] < conns[u[0]][u[1]]['time_seconds']]
        rows.append(dict(day=day, actual_path=actual_paths[day], forecast_path=frame_paths.get(day),
            raw_rt_available=(ROOT / f'data/swiss_rt/gtfs-rt/{day}.tar.bz2').exists(),
            scheduled_connections=len(conns), outcome_counts=dict(Counter(c['outcome'] for c in conns.values())),
            forecast_frames=len(frames.get(day, ())), forecast_revisions=len(forecasts),
            distinct_forecast_endpoints=len({(u[0], u[1]) for _, u in forecasts}),
            numeric_revisions_with_real_endpoint=len(real),
            numeric_revisions_before_own_real_endpoint_using_legacy_clock=len(before),
            distinct_real_endpoints_with_prior_forecast=len({(u[0], u[1]) for _, u in before}),
            unmapped_source_ids=unmapped))
    output = dict(scope='Local inventory only: no controller losses, candidate utility, or new model fitting.',
        summary=dict(actual_dates=len(actual), existing_forecast_dates=len(frames),
            intersection_dates=len(set(actual) & set(frames)),
            distinct_real_endpoints_with_prior_forecast=sum(r['distinct_real_endpoints_with_prior_forecast'] for r in rows),
            raw_rt_date_range=['2023-10-01', '2023-11-04']),
        static_feed=(ROOT / 'data/swiss_gtfs/feed_info.txt').read_text().strip(),
        days=rows, actual_source_versions=versions, repeated_date_static_id_checks=identity_agreements,
        schemas=dict(
            actual_identity=['service date', 'operator_id', 'vehicle_number', 'BPUIC', 'event kind', 'scheduled minute'],
            actual_endpoint=['status', 'time_seconds', 'official_journey', 'reported_time_seconds', 'canceled_flag', 'reported_status'],
            gtfs_rt_identity=['trip_id', 'start_date', 'stop_sequence', 'stop_id root UIC'],
            legacy_frame=['archive_member', 'capture_seconds', 'header_seconds', 'receipt_seconds', 'trip_statuses', 'endpoint_updates']),
        causal_scope='Legacy receipt=max(capture,header), missing TripUpdate timestamp. Inventory pre-event '
            'counts do not establish availability before any particular alighting decision. V34 messages.py '
            're-extracts individual update clocks; execution must consume only receipt<=decision clock.',
        reusable_interfaces={
            'static_graph': 'build_dated_replay.dated_connections dates; no actuals are read',
            'actual_join': 'build_dated_replay.attach_outcomes preserves REAL/CANCELED/MISSING/AMBIGUOUS and official journey consistency',
            'graph_world': 'run_dated_replay.dated_graph(data,day,gtfs_station_max,True) returns graph,world,stop_ids,source_to_local,metadata,omitted',
            'soft_prediction': 'EvolvingTripBelief.observe_forecast(cid,kind,predicted_time,sd_seconds); latest Gaussian emission, not actual evidence',
            'recursive_controller': 'V33 nodes/planner/journey retain all downstream exits; V33 ConditionalEngine needs forecast records and cache invalidation before reuse'},
        candidate_design='Primary dates 10/25,26,27,28,30,31 follow static feed 20231024. Freeze static roots/menus '
            'and model before losses; require at least two legal downstream exits plus distinct continuation '
            'sets from schedule only. Keep missing endpoints/cancellation in the menu and report censored '
            'real executions. Diagnose message timing relative to actual irreversible decisions after the '
            'selection freeze. Preserve both signs; compare strong rolling distributional/first-offer controls.')
    (HERE / 'local_inventory.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output['summary'], indent=2))
    print('date connections measured revisions real endpoints with prior forecast')
    for r in rows:
        print(r['day'], r['scheduled_connections'], r['outcome_counts'].get('MEASURED', 0),
              r['forecast_revisions'], r['distinct_real_endpoints_with_prior_forecast'])


if __name__ == '__main__':
    main()
