"""Compare published platform rules for the frozen four-day registered pairs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from experiments.decision_calibration_v1.platform_pairs import static_sources, transfer_pair
from experiments.reconstruction_v34.discover import context


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else path.open()) as handle:
        return json.load(handle)


def applicable_view(result):
    """Only applicable directed rules enter; bare-station aliases do not."""
    rules = [*result['exact_records'], *result['applicable_inherited_records']]
    return dict(arrival_stop_id=result['arrival']['stop_id'],
        departure_stop_id=result['departure']['stop_id'],
        arrival_status=result['arrival']['status'], departure_status=result['departure']['status'],
        rules=sorted(rules, key=lambda row:tuple(sorted(row.items()))),
        status=result['status'], stated_pair_seconds=result['stated_pair_seconds'],
        fixed_value_matches=result['fixed_value_matches'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--old-static', type=Path, required=True)
    parser.add_argument('--new-static', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    selection = read(directory/'selection.json')
    scheduled = read(directory/'scheduled.json.gz')
    identities = read(directory/'static_identity_mapping.json.gz')['days']
    dates = sorted(identities)
    old_connections = [row for day in dates for row in scheduled['dates'][day]['connections']]
    maps = {day:{row['trip_id']:row['mapped_trip_id'] for row in identities[day]['rows']}
            for day in dates}
    translated = {day:{row['id']:dict(row,trip_id=maps[day][row['trip_id']])
                       for row in scheduled['dates'][day]['connections']} for day in dates}
    new_connections = [row for day in dates for row in translated[day].values()]
    old_index, old_stops, old_transfers = static_sources(old_connections,
        args.old_static/'data/zurich_gtfs', args.old_static/'data/swiss_gtfs')
    new_index, new_stops, new_transfers = static_sources(new_connections,
        args.new_static/'data/zurich_gtfs', args.new_static/'data/swiss_gtfs')
    metadata = {day:context(scheduled,day)[4] for day in dates}
    rows = []
    for root_index, selected in enumerate(selection['selected']):
        day = selected['day']
        if day not in identities:
            continue
        menu = read(directory/'menus'/f'{root_index:03}.json.gz')
        for event in menu['registration']['rows']:
            for role, incoming_cid, outgoing_cid, mct in (
                ('upstream',event['current_cid'],event['upstream_cid'],event['upstream_mct']),
                ('downstream',event['source_arrival_cid'],event['hub_cid'],event['downstream_mct'])):
                incoming, outgoing = metadata[day][incoming_cid], metadata[day][outgoing_cid]
                new_incoming, new_outgoing = translated[day][incoming['id']],translated[day][outgoing['id']]
                old = applicable_view(transfer_pair(incoming,outgoing,mct,old_index,old_stops,old_transfers))
                new = applicable_view(transfer_pair(new_incoming,new_outgoing,mct,new_index,new_stops,new_transfers))
                rows.append(dict(day=day,root_index=root_index,event_index=event['event_index'],role=role,
                    incoming_cid=incoming_cid,outgoing_cid=outgoing_cid,frozen_mct_seconds=mct,
                    old_public_rule=old,new_public_rule=new,applicable_rules_and_result_equal=old==new))
    counts = Counter(row['day'] for row in rows)
    changed = [row for row in rows if not row['applicable_rules_and_result_equal']]
    receipt = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_frozen_at_utc=selection['frozen_at_utc'],
        sources=dict(old_static=str(args.old_static),new_static=str(args.new_static),
                     registration='selection.json and menus/*.json.gz',
                     graph='scheduled.json.gz',identity='static_identity_mapping.json.gz'),
        published_transfer_rows=dict(old=sum(map(len,old_transfers.values())),new=sum(map(len,new_transfers.values()))),
        dates=dates,registered_pairs=len(rows),pairs_by_day=dict(counts),
        distinct_directed_platform_mct_pairs=len({(row['old_public_rule']['arrival_stop_id'],
            row['old_public_rule']['departure_stop_id'],row['frozen_mct_seconds']) for row in rows}),
        changed_applicable_rule_or_result_pairs=len(changed),all_registered_pairs_equal=not changed,
        old_status_counts=dict(Counter(row['old_public_rule']['status'] for row in rows)),
        new_status_counts=dict(Counter(row['new_public_rule']['status'] for row in rows)),
        changed_pairs=changed,actual_or_messages_or_probabilities_or_losses_read=False,
        frozen_graph_menu_mct_changed=False,actual_platform_identity_inferred=False,
        interpretation='This compares applicable planned directed platform rules for registered pairs only. It does not infer an actual platform or a pedestrian travel-time observation.')
    with gzip.open(directory/'static_transfer_version_pairs.json.gz','wt') as handle:
        json.dump(rows,handle,separators=(',',':'))
    (directory/'static_transfer_version_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
