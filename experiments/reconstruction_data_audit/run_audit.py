"""Bounded field audit: two streamed Swiss snapshots and a small saved graph.

No realized-event calibration or full-day population rate is inferred here.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import sys
import tarfile
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.gtfs_rt_parser import parse_single_snapshot, summarize_predictions, load_route_metadata

ROOT = Path(__file__).resolve().parents[2]


def main():
    archive = ROOT / 'data/swiss_rt/gtfs-rt/2023-10-01.tar.bz2'
    metadata = load_route_metadata(str(ROOT / 'data/swiss_gtfs'))
    snapshots = []
    records = []
    with tarfile.open(archive, 'r|bz2') as tar:
        for member in tar:
            if not member.name.endswith('.gtfsrt'):
                continue
            batch = parse_single_snapshot(tar.extractfile(member).read())
            snapshots.append({'member': member.name, 'published_at': batch[0]['published_at'],
                              'trip_statuses': dict(Counter(r['trip_relationship'] for r in batch
                                                           if r['record_type'] == 'trip_status')),
                              'stop_statuses': dict(Counter(r['stop_relationship'] for r in batch
                                                           if r['record_type'] == 'stop_prediction')),
                              'n_stop_predictions': sum(r['record_type'] == 'stop_prediction' for r in batch),
                              'n_numeric_delay': sum(r['delay_seconds'] is not None for r in batch),
                              'n_absolute_forecast_time': sum(r['event_time'] is not None for r in batch),
                              'n_measurement_timestamp': sum(r['measured_at'] is not None for r in batch)})
            records.extend(batch)
            if len(snapshots) == 2:
                break
    summaries = summarize_predictions(records, metadata)
    by_name = defaultdict(list)
    for rid, item in metadata.items():
        by_name[item['name']].append({'route_id': rid, 'agency_id': item['agency_id']})
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as handle:
        graph = pickle.load(handle)
    unique_mapping = {}
    ambiguous = {}
    missing = []
    for name in sorted({connection.route for connection in graph.connections}):
        matches = by_name[name]
        if len(matches) == 1:
            unique_mapping[name] = dict(matches[0], present_in_bounded_feed=matches[0]['route_id'] in summaries)
        elif matches:
            ambiguous[name] = matches
        else:
            missing.append(name)
    output = {
        'scope': 'Two archive-order snapshots, predictions and declared statuses; not realized replay or population cancellation estimation.',
        'archive': str(archive.relative_to(ROOT)), 'archive_bytes': archive.stat().st_size,
        'snapshots': snapshots,
        'raw_records': len(records),
        'unique_trip_statuses': sum(s['n_status_trips'] for s in summaries.values()),
        'unique_delay_predictions': sum(s['n_obs'] for s in summaries.values()),
        'explicit_canceled_trips': sum(s['n_canceled'] for s in summaries.values()),
        'severe_delay_predictions': sum(s['n_severe_delay'] for s in summaries.values()),
        'route_id_count': len(summaries),
        'graph_short_names': len(unique_mapping) + len(ambiguous) + len(missing),
        'unambiguous_route_id_mapping': unique_mapping,
        'ambiguous_names_skipped': ambiguous, 'unmatched_names_skipped': missing,
        'gtfs_reference': 'https://gtfs.org/documentation/realtime/reference/',
        'information_semantics': 'feed header timestamp is creation/publication proxy; TripUpdate timestamp is vehicle progress measurement; StopTimeEvent time remains a forecast.',
    }
    destination = Path(__file__).with_name('field_audit.json')
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in output.items()
                      if key not in ('unambiguous_route_id_mapping', 'ambiguous_names_skipped')}, indent=2))
    print(f'unambiguous={len(unique_mapping)}, ambiguous_skipped={len(ambiguous)}, unmatched_skipped={len(missing)}')


if __name__ == '__main__':
    main()
