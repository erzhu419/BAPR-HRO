"""Match received platform announcements to registered pairs, without control changes."""
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from experiments.decision_gap_v1.panel import read, write

HERE = Path(__file__).resolve().parent


def index_announcements(frames):
    records = defaultdict(list)
    for frame in frames:
        for update in frame.get('platform_stop_updates', []):
            records[frame['trip_id'], update['stop_sequence']].append(dict(
                receipt_seconds=frame['receipt_seconds'], capture_seconds=frame['capture_seconds'],
                archive_member=frame['archive_member'], entity_id=frame['entity_id'],
                published_stop_id=update['published_stop_id'], stop_status=update['stop_status']))
    for rows in records.values():
        rows.sort(key=lambda row:row['receipt_seconds'])
    return records


def received(endpoint, clock, index):
    if clock is None:
        return dict(status='initial_clock_unknown', announcement=None)
    rows = index.get((endpoint['trip_id'], endpoint['stop_sequence']), [])
    position = bisect_right([row['receipt_seconds'] for row in rows], clock)-1
    if position < 0:
        return dict(status='no_received_platform_record', announcement=None)
    latest = rows[position]
    stated = latest['published_stop_id']
    return dict(status='received' if stated else 'latest_record_has_no_stop_id',
        announcement=latest, matches_scheduled_platform=(stated == endpoint['stop_id']) if stated else None,
        actual_platform_verified=False)


def run(directory=HERE):
    manifest = read(directory/'date_manifest.json')
    pairs, daily = [], []
    for day in manifest['selected_dates']:
        result = read(directory/'results'/f'{day}.json.gz')
        index = index_announcements(read(directory/'messages'/f'{day}.json.gz')['frames'])
        clocks = {root['root_index']:root['initial_state']['clock'] for root in result['roots']}
        rows = []
        for pair in result['platform_pairs']:
            clock = clocks[pair['root_index']]
            rows.append(dict(day=day, root_index=pair['root_index'], event_index=pair['event_index'],
                role=pair['role'], decision_clock=clock,
                arrival=received(pair['arrival'], clock, index),
                departure=received(pair['departure'], clock, index),
                planned_rule_status=pair['status'], fixed_mct_seconds=pair['fixed_mct_seconds']))
        endpoints = [row[side] for row in rows for side in ('arrival', 'departure')]
        daily.append(dict(day=day, pairs=len(rows), endpoint_statuses=dict(Counter(r['status'] for r in endpoints)),
            announced_endpoints_different_from_static=sum(r.get('matches_scheduled_platform') is False for r in endpoints)))
        pairs.extend(rows)
    write(directory/'platform_announcements.json.gz', dict(days=daily, pairs=pairs,
        scope='Last received stop-sequence platform record at or before the actual first decision; repeated endpoints retain the full registered pair denominator. This is announced information, not observed platform truth. No model, MCT, probability or gate was changed.'))
    write(directory/'platform_announcements_summary.json', dict(days=daily,
        pairs=len(pairs), actual_platform_pairs_verified=0,
        endpoint_statuses=dict(sum((Counter(row['endpoint_statuses']) for row in daily), Counter())),
        announced_endpoints_different_from_static=sum(row['announced_endpoints_different_from_static'] for row in daily)))


if __name__ == '__main__':
    run()
