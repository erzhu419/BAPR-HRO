"""Join saved platform announcements to every registered first-decision pair."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.decision_calibration_v2.platform_announcements import index_announcements, received
from experiments.decision_gap_v1.panel import read, write


SOURCE_SEMANTICS = dict(
    evidence='Published StopTimeUpdate.stop_id at the registered stop sequence, received by the first decision.',
    interpretation='Announcement evidence; neither the platform physically used nor measured walking time.',
    realized_platform_guarantee_found=False,
    documentation_scope='Current official documentation explains field semantics; it is not a historical 2023 platform observation.',
    sources=[dict(
        url='https://opentransportdata.swiss/en/cookbook/realtime-prediction-cookbook/gtfs-rt/',
        supporting_statement='Trip updates publish delays and service changes; Swiss vehicle positions are not provided by this interface.'),
        dict(url='https://gtfs.org/documentation/realtime/reference/#message-stoptimeupdate',
        supporting_statement='StopTimeUpdate.stop_id identifies a stop in the corresponding static feed; real-time assignments are described separately under StopTimeProperties.assigned_stop_id.')])


def endpoint_receipt(endpoint, clock, index):
    value = received(endpoint, clock, index)
    value['scheduled_endpoint'] = endpoint
    value['actual_platform_verified'] = False
    if value['status'] != 'received':
        comparison = 'announcement_unavailable'
    elif endpoint.get('stop_id') is None:
        comparison = 'static_endpoint_unresolved'
        value['matches_scheduled_platform'] = None
    else:
        comparison = 'same' if value['matches_scheduled_platform'] else 'different'
    value['static_comparison'] = comparison
    return value


def run(directory):
    dates = read(directory/'date_manifest.json')['selected_dates']
    pairs, daily = [], []
    for day in dates:
        result = read(directory/'events/results'/f'{day}.json.gz')
        index = index_announcements(read(directory/'messages'/f'{day}.json.gz')['frames'])
        clocks = {root['root_index']:root['initial_state']['clock'] for root in result['roots']}
        rows = []
        for pair in result['platform_pairs']:
            clock = clocks[pair['root_index']]
            rows.append(dict(day=day, root_index=pair['root_index'], event_index=pair['event_index'],
                role=pair['role'], decision_clock=clock,
                arrival=endpoint_receipt(pair['arrival'], clock, index),
                departure=endpoint_receipt(pair['departure'], clock, index),
                planned_rule_status=pair['status'], fixed_mct_seconds=pair['fixed_mct_seconds'],
                actual_platform_pair_verified=False))
        endpoints = [row[side] for row in rows for side in ('arrival', 'departure')]
        comparison = Counter(row['static_comparison'] for row in endpoints)
        daily.append(dict(day=day, roots=len(result['roots']), pairs=len(rows), endpoints=len(endpoints),
            endpoint_statuses=dict(Counter(row['status'] for row in endpoints)),
            static_comparison=dict(comparison),
            announced_endpoints_different_from_static=comparison['different'],
            pairs_with_both_announcements=sum(all(row[side]['status']=='received'
                for side in ('arrival', 'departure')) for row in rows)))
        pairs.extend(rows)
    summary = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(), selected_dates=dates,
        days=daily, roots=sum(row['roots'] for row in daily), pairs=len(pairs), endpoints=2*len(pairs),
        actual_platform_pairs_verified=0,
        endpoint_statuses=dict(sum((Counter(row['endpoint_statuses']) for row in daily), Counter())),
        static_comparison=dict(sum((Counter(row['static_comparison']) for row in daily), Counter())),
        announced_endpoints_different_from_static=sum(row['announced_endpoints_different_from_static'] for row in daily),
        pairs_with_both_announcements=sum(row['pairs_with_both_announcements'] for row in daily),
        source_semantics=SOURCE_SEMANTICS,
        full_receipts='platform_announcements.json.gz',
        scope='All registered pairs, including repeated endpoints and missing announcements. Last received record at or before each root first decision. Model, MCT, controls and budgets unchanged; raw archives were not reread.')
    write(directory/'platform_announcements.json.gz', dict(summary=summary, pairs=pairs))
    write(directory/'platform_announcements_summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    run(parser.parse_args().directory)
