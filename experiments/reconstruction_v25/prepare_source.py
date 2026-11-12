"""Prepare the frozen July 16/17 archive; filter the full source only remotely."""

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import gzip
import json
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen


DATES = ('2023-07-16', '2023-07-17')
SOURCE = ('https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd_2023/download/'
          'Fahrzeiten_SOLL_IST_20230716_20230722.csv')
START, END = 27000, 34500


def filter_remote(args):
    """Download on the server, then retain the declared small operating peak."""
    if not args.source_cache.exists():
        temporary = args.source_cache.with_suffix('.csv.partial')
        with urlopen(SOURCE, timeout=90) as response, temporary.open('wb') as output:
            copyfileobj(response, output, 1024*1024)
        temporary.replace(args.source_cache)
    date_keys = {date.fromisoformat(day).strftime('%d.%m.%y'): day for day in DATES}
    counts, scanned = Counter(), 0
    with args.source_cache.open(encoding='utf-8-sig', newline='') as handle, gzip.open(
            args.output, 'wt', encoding='utf-8', newline='') as output:
        reader = csv.DictReader(handle)
        writer = csv.DictWriter(output, reader.fieldnames)
        writer.writeheader()
        for row in reader:
            scanned += 1
            day = date_keys.get(row['betriebsdatum'])
            if day is None or not row['soll_ab_von']:
                continue
            offset = (datetime.strptime(row['datum_von'], '%d.%m.%y').date()
                      - date.fromisoformat(day)).days*86400
            if START <= int(row['soll_ab_von'])+offset < END:
                writer.writerow(row)
                counts[day] += 1
    audit = dict(source=SOURCE, remote_source_cache=str(args.source_cache),
                 remote_source_bytes=args.source_cache.stat().st_size,
                 scanned_weekly_rows=scanned, local_subset_bytes=args.output.stat().st_size,
                 window_seconds=[START, END], rows_by_date={day: counts[day] for day in DATES})
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


def join_source(args):
    """Join only after the parent experiment has frozen policy and menus."""
    from experiments.reconstruction_data_audit import build_vbz_replay_sample as join
    from experiments.reconstruction_data_audit.build_vbz_city_replay import STOP_NAME_ALIASES
    from experiments.reconstruction_v23.prepare_city_transfer_source import match_day

    root = Path(__file__).resolve().parents[2]
    audit_dir = root/'experiments/reconstruction_data_audit'
    selection = json.loads((args.schedule_dir/'selection.json').read_text())
    extraction = json.loads(args.rows.with_suffix('.audit.json').read_text())
    if extraction['source'] != SOURCE or set(extraction['rows_by_date']) != set(DATES):
        raise ValueError('Filtered source does not match the frozen dates and municipal week')
    stop_file = audit_dir/'vbz_stop_mapping.csv'
    crosswalk = join.stop_crosswalk(root/'data/swiss_gtfs', stop_file.read_bytes())
    with stop_file.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['halt_lang'] in STOP_NAME_ALIASES:
                crosswalk[row['halt_id']] = STOP_NAME_ALIASES[row['halt_lang']]
    join.DATES = DATES
    date_audit_path = args.schedule_dir/'date_audit.json'
    date_audit = json.loads(date_audit_path.read_text())
    first_read = datetime.now(timezone.utc).isoformat()
    date_audit.update(pre_acquisition_actual_endpoints_opened=False, new_actual_endpoints_opened=True,
                      menu_frozen_at_utc=selection['frozen_at_utc'],
                      first_local_actual_read_at_utc=first_read)
    date_audit_path.write_text(json.dumps(date_audit, ensure_ascii=False, indent=2)+'\n')
    with gzip.open(args.rows, 'rb') as handle:
        measured, exclusions = join.measured_rows(handle.read(), crosswalk, normal_only=False)
    dates, coverage = {}, {}
    for day in DATES:
        with gzip.open(args.schedule_dir/f'city_schedule_{day}.json.gz', 'rt') as handle:
            schedule = json.load(handle)
        if schedule['date'] != day or schedule['time_window_seconds'] != [START, END]:
            raise ValueError('Static schedule differs from the frozen date/window')
        coverage[day] = match_day(day, schedule, measured[day], exclusions)
        dates[day] = {**schedule, 'coverage': coverage[day]}
    dataset = dict(scope='Frozen two-date retrospective evaluation; policy fitting dates are later.',
                   train_dates=[], holdout_dates=list(DATES), dates=dates, source=SOURCE,
                   matching_table=join.STOPS, planned_time_join_precision_seconds=60)
    with gzip.open(args.output, 'wt', encoding='utf-8') as handle:
        json.dump(dataset, handle, ensure_ascii=False, separators=(',', ':'))
    audit = dict(source_extraction=extraction, coverage=coverage,
                 menu_frozen_at_utc=selection['frozen_at_utc'],
                 joined_at_utc=datetime.now(timezone.utc).isoformat(),
                 matching_rule='Unchanged V23 static aliases, calendar and minute planned-clock matching.')
    args.output.with_name(args.output.name.removesuffix('.json.gz')+'_audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    date_audit.update(joined_at_utc=audit['joined_at_utc'],
                      endpoint_archive=str(args.output),
                      subset_bytes=extraction['local_subset_bytes'])
    date_audit_path.write_text(json.dumps(date_audit, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({day: record['connection_outcome_counts'] for day, record in coverage.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    remote = commands.add_parser('filter-remote')
    remote.add_argument('--source-cache', type=Path, required=True)
    remote.add_argument('--output', type=Path, required=True)
    remote.set_defaults(run=filter_remote)
    local = commands.add_parser('join')
    local.add_argument('--rows', type=Path, required=True)
    local.add_argument('--schedule-dir', type=Path, required=True)
    local.add_argument('--output', type=Path, required=True)
    local.set_defaults(run=join_source)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
