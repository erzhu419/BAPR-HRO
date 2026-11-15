"""Cache compressed members first; decode only after dates and menus are frozen."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
from io import StringIO
import json
from pathlib import Path
import subprocess
import struct
import sys
import zlib

from experiments.decision_calibration_v2.filter_actuals_remote import fetch_range, in_window
from experiments.reconstruction_data_audit.build_dated_replay import STATIONS


def cache_day(directory, day):
    """No deflate decoding, CSV schema or target value is accessed here."""
    directory = Path(directory)
    metadata = json.loads((directory / f'actual_metadata_{day[:7]}.json').read_text())
    member, = [row for row in metadata['members']
               if Path(row['name']).name == f'{day}_istdaten.csv']
    target = directory / 'actual_cache' / f'{day}.deflate'
    receipt = target.with_suffix('.cache.json')
    target.parent.mkdir(exist_ok=True)
    if receipt.exists() and target.exists():
        value = json.loads(receipt.read_text())
        if target.stat().st_size != member['compressed_bytes']:
            raise ValueError('Cached member length differs from official directory')
        return dict(day=day, exit_code=0, reused_complete_member=True,
                    compressed_bytes=value['compressed_bytes'])
    header = fetch_range(metadata['url'], member['absolute_local_header_offset'], 30)
    fields = struct.unpack('<4s5H3L2H', header)
    offset = member['absolute_local_header_offset'] + 30 + sum(fields[-2:])
    size = member['compressed_bytes']
    boundaries = [size * i // 8 for i in range(9)]
    parts = directory / 'actual_cache_parts' / day
    parts.mkdir(parents=True, exist_ok=True)

    def fetch_part(index):
        start, stop = boundaries[index:index+2]
        path = parts / f'{index:02}.deflate'
        if path.exists():
            if path.stat().st_size != stop-start:
                raise ValueError('Previously completed range has a different length')
            return dict(part=index, start=offset+start, bytes=stop-start, reused=True)
        raw = fetch_range(metadata['url'], offset+start, stop-start)
        temporary = path.with_suffix('.partial')
        temporary.write_bytes(raw)
        temporary.replace(path)  # Final part names denote completely received ranges.
        return dict(part=index, start=offset+start, bytes=len(raw), reused=False)

    with ThreadPoolExecutor(max_workers=8) as pool:
        completed = list(pool.map(fetch_part, range(8)))
    temporary = target.with_suffix('.partial')
    with temporary.open('wb') as output:
        for index in range(8):
            output.write((parts / f'{index:02}.deflate').read_bytes())
    temporary.replace(target)
    value = dict(day=day, source_url=metadata['url'], member=member['name'],
        compressed_bytes=size, uncompressed_bytes=member['uncompressed_bytes'],
        member_data_byte_range=[offset, offset+size-1], completed_ranges=completed,
        cached_at_utc=datetime.now(timezone.utc).isoformat(),
        cache_scope='Compressed bytes only; no decompression or CSV schema read',
        target_values_read=False)
    receipt.write_text(json.dumps(value, indent=2)+'\n')
    return dict(day=day, exit_code=0, reused_complete_member=False,
                compressed_bytes=size, target_values_read=False)


def cache(directory, days=None):
    directory = Path(directory)
    candidate = json.loads((directory / 'actual_sources.json').read_text())['candidate_dates']
    days = candidate if days is None else days
    if set(days)-set(candidate):
        raise ValueError('Date outside the metadata candidate list')
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(cache_day, directory, day): day for day in days}
        for job in as_completed(jobs):
            try:
                row = job.result()
            except Exception as error:
                row = dict(day=jobs[job], exit_code=1, error=f'{type(error).__name__}: {error}')
            rows.append(row)
            print(json.dumps(row), flush=True)
    value = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        rows=sorted(rows, key=lambda row: row['day']), max_connections=32,
        target_values_read=False, csv_schema_read=False, decompression_performed=False,
        unfrozen_candidates='Unused compressed bytes remain cached if a date is excluded by clocks; no outcome-based selection')
    attempt = len(list(directory.glob('actual_cache_attempt_*.json'))) + 1
    (directory / f'actual_cache_attempt_{attempt:02d}.json').write_text(json.dumps(value, indent=2)+'\n')
    (directory / 'actual_cache.json').write_text(json.dumps(value, indent=2)+'\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


def filter_day(directory, day):
    """A subprocess of acquire; consume the retained member without networking."""
    directory = Path(directory)
    compressed = directory / 'actual_cache' / f'{day}.deflate'
    source = json.loads(compressed.with_suffix('.cache.json').read_text())
    raw = zlib.decompress(compressed.read_bytes(), -15)
    if len(raw) != source['uncompressed_bytes']:
        raise ValueError('Incomplete decompressed daily CSV')
    reader = csv.DictReader(StringIO(raw.decode('utf-8-sig')), delimiter=';')
    output = directory / 'actual' / f'{day}.csv.gz'
    counts, statuses = Counter(), Counter()
    scanned = selected = 0
    with gzip.open(output, 'wt', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, reader.fieldnames, delimiter=';')
        writer.writeheader()
        for row in reader:
            scanned += 1
            if row['BPUIC'] not in STATIONS or not in_window(row, 27000, 34500):
                continue
            writer.writerow(row)
            selected += 1
            counts[row['BETREIBER_ID'], row['BETREIBER_ABK']] += 1
            statuses[row['AN_PROGNOSE_STATUS'], row['AB_PROGNOSE_STATUS'], row['FAELLT_AUS_TF']] += 1
    audit = dict(archive=source['source_url'], date=day, member=source['member'],
        member_data_byte_range=source['member_data_byte_range'], complete_daily_member=True,
        remote_bytes_requested=0, reused_compressed_member=str(compressed),
        scanned_daily_rows=scanned, selected_rows=selected, local_subset_bytes=output.stat().st_size,
        fields=reader.fieldnames,
        operator_counts=[dict(operator_id=k[0], operator_name=k[1], rows=n) for k,n in counts.most_common()],
        status_counts=[dict(arrival=k[0], departure=k[1], canceled=k[2], rows=n) for k,n in statuses.most_common()])
    output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('fields','operator_counts','status_counts')}), flush=True)


def acquire(directory, days=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'date_manifest.json').read_text())
    selection = json.loads((directory / 'selection.json').read_text())
    if not manifest['dates_frozen'] or not selection['frozen_at_utc']:
        raise ValueError('Date and public-menu freezes must precede actual values')
    if not isinstance(selection['selected'], list):
        raise ValueError('Frozen public selection is missing')
    frozen = manifest['selected_dates']
    days = frozen if days is None else days
    if set(days) - set(frozen):
        raise ValueError('Requested actual date is outside the frozen cohort')
    target = directory / 'actual'
    target.mkdir(exist_ok=True)
    filter_file = directory / 'actual_filter.json'
    filter_file.write_text(json.dumps(dict(stops=sorted(STATIONS),
        time_window_seconds=[27000, 34500], operator_status_filter_applied=False,
        dates_frozen_at_utc=manifest['frozen_at_utc'],
        menus_frozen_at_utc=selection['frozen_at_utc']), indent=2) + '\n')
    started = datetime.now(timezone.utc).isoformat()

    def run_day(day):
        output = target / f'{day}.csv.gz'
        audit_file = output.with_suffix('.audit.json')
        if audit_file.exists():
            audit = json.loads(audit_file.read_text())
            if audit['date'] == day and audit['complete_daily_member'] and output.exists():
                return dict(day=day, exit_code=0, existing_complete_extraction=True)
        attempt = len(list(target.glob(f'{day}.attempt_*.log'))) + 1
        command = [sys.executable, '-m', 'experiments.natural_gap_v2.source_actuals',
                   'filter-day', '--directory', str(directory), '--days', day]
        run = subprocess.run(command, text=True, capture_output=True)
        log = target / f'{day}.attempt_{attempt:02d}.log'
        log.write_text(run.stdout + run.stderr)
        row = dict(day=day, exit_code=run.returncode, log=str(log),
                   completed_at_utc=datetime.now(timezone.utc).isoformat())
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(run_day, day) for day in days]
        rows = [job.result() for job in as_completed(jobs)]
    receipt = dict(started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        dates_frozen_at_utc=manifest['frozen_at_utc'],
        public_menus_frozen_at_utc=selection['frozen_at_utc'],
        rows=sorted(rows, key=lambda row: row['day']),
        simultaneous_days=4, ranges_per_day=0, network_bytes_during_filter=0,
        monthly_archive_downloaded=False, source_rows_transferred_locally=False)
    attempt = len(list(directory.glob('actual_acquisition_attempt_*.json'))) + 1
    (directory / f'actual_acquisition_attempt_{attempt:02d}.json').write_text(
        json.dumps(receipt, indent=2) + '\n')
    (directory / 'actual_acquisition.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


def summarize(directory):
    """Return small receipts only; daily CSV payloads remain on the server."""
    directory = Path(directory)
    manifest = json.loads((directory/'date_manifest.json').read_text())
    acquisition = json.loads((directory/'actual_acquisition.json').read_text())
    rows, schemas = [], []
    for day in manifest['selected_dates']:
        path = directory/'actual'/f'{day}.csv.audit.json'
        row = json.loads(path.read_text())
        if row['fields'] not in schemas:
            schemas.append(row['fields'])
        rows.append({key:row[key] for key in ('date','complete_daily_member',
            'scanned_daily_rows','selected_rows','local_subset_bytes','remote_bytes_requested',
            'operator_counts','status_counts')})
    value = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        dates=manifest['selected_dates'], rows=rows, field_schemas=schemas,
        all_complete=all(row['complete_daily_member'] for row in rows),
        selected_rows=sum(row['selected_rows'] for row in rows),
        filtered_csv_gzip_bytes=sum(row['local_subset_bytes'] for row in rows),
        network_bytes_during_filter=sum(row['remote_bytes_requested'] for row in rows),
        dates_frozen_at_utc=acquisition['dates_frozen_at_utc'],
        menus_frozen_at_utc=acquisition['public_menus_frozen_at_utc'],
        actual_filter_started_at_utc=acquisition['started_at_utc'],
        actual_filter_completed_at_utc=acquisition['completed_at_utc'],
        rows_or_archives_transferred_to_local=False)
    (directory/'actual_filter_summary.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({key:value for key,value in value.items() if key not in ('rows','field_schemas')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('cache', 'filter', 'filter-day', 'summary'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--days', nargs='+')
    args = parser.parse_args()
    if args.stage == 'filter-day':
        # The child must enforce the same freezes even when invoked directly.
        manifest = json.loads((args.directory/'date_manifest.json').read_text())
        selection = json.loads((args.directory/'selection.json').read_text())
        if not manifest['dates_frozen'] or not selection['frozen_at_utc'] or args.days[0] not in manifest['selected_dates']:
            raise ValueError('Date and public-menu freezes required before decompression')
        filter_day(args.directory, args.days[0])
    elif args.stage == 'cache':
        cache(args.directory, args.days)
    elif args.stage == 'summary':
        summarize(args.directory)
    else:
        acquire(args.directory, args.days)
