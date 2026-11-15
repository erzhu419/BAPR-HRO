"""Decode the proven public GTFS renaming without changing the frozen graph.

Every source ID is interpreted in the new version, including literal IDs that
were reused for a different train. No literal-ID fallback is permitted.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path

from experiments.decision_calibration_v2.fetch_messages import extract_messages


def decoding_connections(connections, evidence):
    rows = evidence['rows']
    if any(row['status'] not in ('renamed_id', 'same_literal_id') or
           not row['planned_platform_sequence_equal'] for row in rows):
        raise ValueError('Only exact unique public identities with unchanged platforms can be decoded')
    forward = {row['trip_id']:row['mapped_trip_id'] for row in rows}
    reverse = {new:old for old, new in forward.items()}
    if set(forward) != {row['trip_id'] for row in connections} or len(reverse) != len(forward):
        raise ValueError('Identity correspondence must cover the frozen graph bijectively')
    return [dict(row, trip_id=forward[row['trip_id']]) for row in connections], reverse


def canonical_frames(frames, reverse):
    for frame in frames:
        source_trip = frame['trip_id']
        frame['raw_trip_id'] = source_trip
        frame['trip_id'] = reverse[source_trip]
        frame['trip_statuses'] = [[reverse[trip], status] for trip, status in frame['trip_statuses']]
    return frames


def correct_day(day, connections, evidence, archive, directory):
    temporary_connections, reverse = decoding_connections(connections, evidence)
    raw_output = directory/'raw_decoding'/f'{day}.json.gz'
    result = extract_messages(day, temporary_connections, archive, raw_output)
    canonical_frames(result['frames'], reverse)
    result['identity_correction'] = dict(
        source='static_identity_mapping.json.gz',
        interpretation='New-version raw IDs decoded through the exact public bijection; '
            'all output trip IDs restored to the frozen canonical graph. No literal fallback.',
        graph_connections_changed=False, menu_changed=False, raw_forecast_values_modified=False,
        raw_availability_clocks_modified=False, canonical_consumed_histories_corrected=True,
        posterior_predictions_may_change=True,
        mapped_trip_count=len(reverse), raw_decoding_output=str(raw_output),
        reused_literal_ids_with_changed_signature=evidence['summary']['literal_id_changed_signature'])
    target = directory/f'{day}.json.gz'
    with gzip.open(target, 'wt') as handle:
        json.dump(result, handle, separators=(',', ':'))
    audit = dict(day=day, output_complete=True, filtered_bytes=target.stat().st_size,
        identity_correction=result['identity_correction'], summary=result['summary'])
    target.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return dict(day=day, captures=result['summary']['sampled_snapshots'],
        matched_messages=result['summary']['matched_trip_messages'],
        distinct_trips=result['summary']['distinct_trips'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.directory/'scheduled.json.gz', 'rt') as handle:
        scheduled = json.load(handle)
    with gzip.open(args.directory/'static_identity_mapping.json.gz', 'rt') as handle:
        identities = json.load(handle)['days']
    manifest = json.loads((args.directory/'date_manifest.json').read_text())
    sources = {row['day']:row['rt'] for row in manifest['days']}
    directory = args.directory/'messages_corrected'
    directory.mkdir(exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(correct_day, day, scheduled['dates'][day]['connections'],
            evidence, args.cache/sources[day]['filename'], directory)
            for day, evidence in identities.items()]
        for future in as_completed(jobs):
            row = future.result()
            results.append(row)
            print(json.dumps(row), flush=True)
    (directory/'summary.json').write_text(json.dumps(dict(
        original_messages_overwritten=False, days=sorted(results, key=lambda row:row['day'])), indent=2)+'\n')


if __name__ == '__main__':
    main()
