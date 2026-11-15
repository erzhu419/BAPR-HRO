"""Readable exports of the frozen event ledger and saved-bank decomposition."""
import csv
import json
from pathlib import Path

from experiments.decision_gap_v1.panel import read

HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:json.dumps(value, ensure_ascii=False) if isinstance(value,(dict,list)) else value
                          for key,value in row.items()} for row in rows)


def run():
    registration = read(HERE/'events/registration.json.gz')
    groups = {root['root_index']: {row['event_index']:row for row in root['rows']}
              for root in read(HERE/'events/group_registration.json.gz')['roots']}
    ledger, contributions = [], []
    for root in registration['roots']:
        index = root['root_index']
        evaluated = {row['event_index']:row for row in read(HERE/f'events/roots/{index:03}.json.gz')['rows']}
        bank = read(HERE/f'events/banks/{index:03}.json.gz')
        bank_rows = {row['event_index']:row for row in bank['rows']}
        for event in root['rows']:
            event_index = event['event_index']
            scored, features = evaluated[event_index], groups[index][event_index]
            row = dict(root_index=index, day=root['day'], event_index=event_index,
                incoming_trip=event['incoming_trip'], outgoing_trip=event['outgoing_trip'],
                current_cid=event['current_cid'], upstream_cid=event['upstream_cid'],
                hub_cid=event['hub_cid'], source_arrival_cid=event['source_arrival_cid'],
                hub_in_action_menu=event['hub_in_action_menu'],
                model_status=scored['model']['status'], actual_status=scored['actual']['status'],
                model_partitions=scored['model'].get('partitions'),
                actual_partition=scored['actual']['actual_partition'],
                actual_partition_bounds=scored['actual']['partition_bounds'],
                model_detail=scored['model'], public_features=features)
            ledger.append(row)
            part = bank_rows.get(event_index, {})
            shared = dict(root_index=index, day=root['day'], event_index=event_index,
                          bank_status=bank['status'], orientation='ALIGHT minus CONTINUE')
            if 'partitions' in part:
                contributions.extend(dict(**shared, partition=key, **value,
                    overall_difference_bounds=part['overall_difference_bounds'],
                    unresolved_pairs=part['unresolved_pairs']) for key,value in part['partitions'].items())
            else:
                contributions.append(dict(**shared, partition='unresolved', reason=part.get('reason', bank.get('reason'))))
    write_csv(HERE/'transfer_event_ledger.csv', ledger)
    write_csv(HERE/'event_contributions.csv', contributions)
    summary = read(HERE/'events/summary.json')
    days = [dict(unit=unit, day=day, **values) for unit,rows in summary['descriptive_groups'].items()
            for day,values in rows['by_date'].items()]
    write_csv(HERE/'calibration_by_day.csv', days)
    print(dict(events=len(ledger), contribution_rows=len(contributions), day_rows=len(days),
               scope='Development diagnostics; independent confirmation remains pending'))


if __name__ == '__main__':
    run()
