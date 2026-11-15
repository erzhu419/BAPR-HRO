"""Cache candidate RT archives and inspect capture filenames before date freeze.

No protobuf parser is imported. The archive stream is traversed only for tar
headers; prediction payloads are neither interpreted nor saved separately.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile
from time import perf_counter
from zoneinfo import ZoneInfo

from experiments.decision_calibration_v2.cache_rt import download_range
from experiments.natural_gap_v2.rt_clock import civil_service_seconds


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def capture_metadata(source, archive, rule, report_path):
    day = source['date']
    midnight = int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('Europe/Zurich')).timestamp())
    captures, civil_captures, civil_offsets, matched, unexpected = [], [], set(), [], []
    members = 0
    with tarfile.open(archive, 'r|bz2') as stream:
        for member in stream:
            members += 1
            if not member.name.endswith('.gtfsrt'):
                if member.isfile():
                    unexpected.append(member.name)
                continue
            clock = datetime.fromisoformat(Path(member.name).stem)
            elapsed = int(clock.timestamp())-midnight
            civil = civil_service_seconds(clock.timestamp(),day)
            captures.append(elapsed)
            civil_captures.append(civil)
            civil_offsets.add(civil-elapsed)
            if 25200 <= civil < 34860 and civil % 600 < 60:
                matched.append(dict(member=member.name, capture_seconds=civil,
                                    legacy_elapsed_seconds=elapsed, bin_start=civil//600*600))
    bins = {row['bin_start'] for row in matched}
    missing = sorted(set(rule['required_bin_starts'])-bins)
    result = dict(date=day, archive=str(archive), archive_bytes=archive.stat().st_size,
        scanned_at_utc=datetime.now(timezone.utc).isoformat(), tar_members=members,
        gtfsrt_capture_members=len(captures), legacy_elapsed_capture_range=[min(captures),max(captures)] if captures else None,
        capture_range=[min(civil_captures),max(civil_captures)] if civil_captures else None,
        original_window_capture_members=sum(25200 <= c < 34860 for c in civil_captures),
        filename_civil_minus_elapsed_seconds=sorted(civil_offsets),
        sampled_members=matched, sampled_capture_count=len(matched),
        matching_bins=sorted(bins), missing_bins=missing, capture_coverage_complete=not missing,
        unexpected_regular_members=unexpected, protobuf_values_opened=False,
        header_timestamps_opened=False, actual_endpoint_values_opened=False)
    write(report_path,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--connections', type=int, default=32)
    args = parser.parse_args()
    directory = args.directory
    metadata = json.loads((directory/'rt_source_metadata.json').read_text())
    rule = json.loads((directory/'rt_metadata_rule.json').read_text())
    cache = directory/'cache/rt'
    cache.mkdir(parents=True, exist_ok=True)
    reports = directory/'rt_clocks'
    reports.mkdir(exist_ok=True)
    audit_path = directory/'rt_acquisition.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else dict(
        started_at_utc=datetime.now(timezone.utc).isoformat(), phase='metadata prescreen before date freeze',
        rule_frozen_at_utc=rule['frozen_at_utc'], protobuf_values_opened=False, days={}, errors=[])
    audit.setdefault('transport_attempts',[]).append(dict(
        started_at_utc=datetime.now(timezone.utc).isoformat(),connections=args.connections,
        retained_range_bytes=sum(path.stat().st_size for path in cache.glob('*.ranges/*.part'))))
    write(audit_path,audit)
    started = perf_counter()
    clock_jobs = {}
    with ThreadPoolExecutor(max_workers=2) as clocks, ThreadPoolExecutor(max_workers=args.connections) as pool:
        jobs, plans, pending_ranges = {}, {}, deque()
        for source in metadata['candidate_days']:
            day = source['date']
            target = cache/source['filename']
            report = reports/f'{day}.json'
            if target.exists() and target.stat().st_size == source['size_bytes']:
                if report.exists():
                    audit['days'][day] = json.loads(report.read_text())
                else:
                    clock_jobs[clocks.submit(capture_metadata,source,target,rule,report)] = day
                continue
            folder = cache/(day+'.ranges')
            folder.mkdir(exist_ok=True)
            parts = []
            for start in range(0,source['size_bytes'],8*1024*1024):
                end = min(start+8*1024*1024,source['size_bytes'])-1
                path = folder/f'{start:012d}.part'
                item = (source,start,end,path)
                parts.append(path)
                pending_ranges.append((day,item))
            plans[day] = dict(source=source,target=target,folder=folder,parts=parts,
                              pending=len(parts),failed=False)
        def replenish():
            while pending_ranges and len(jobs) < args.connections:
                day,item = pending_ranges.popleft()
                jobs[pool.submit(download_range,*item)] = (day,item,1)
        replenish()
        # Replenish failed partial ranges promptly; completed parts are reused.
        while jobs:
            future = next(as_completed(jobs))
            day,item,attempt = jobs.pop(future)
            plan = plans[day]
            try:
                future.result()
            except Exception as error:
                failure = dict(date=day,start=item[1],end=item[2],attempt=attempt,error=repr(error),
                    at_utc=datetime.now(timezone.utc).isoformat())
                audit['errors'].append(failure)
                write(audit_path,audit)
                print(json.dumps(failure),flush=True)
                if attempt < 4:
                    jobs[pool.submit(download_range,*item)] = (day,item,attempt+1)
                    continue
                plan['failed'] = True
            plan['pending'] -= 1
            replenish()
            if plan['pending'] or plan['failed']:
                continue
            temporary = plan['target'].with_suffix('.assembling')
            with temporary.open('wb') as handle:
                for part in plan['parts']:
                    with part.open('rb') as fragment:
                        shutil.copyfileobj(fragment,handle)
            if temporary.stat().st_size != plan['source']['size_bytes']:
                raise ValueError('Assembled archive length differs from public directory metadata')
            temporary.replace(plan['target'])
            for part in plan['parts']:
                part.unlink()
            plan['folder'].rmdir()
            print(json.dumps(dict(date=day,status='cached',bytes=plan['target'].stat().st_size)),flush=True)
            clock_jobs[clocks.submit(capture_metadata,plan['source'],plan['target'],rule,reports/f'{day}.json')] = day
        for future in as_completed(clock_jobs):
            day = clock_jobs[future]
            try:
                report = future.result()
                write(reports/f'{day}.json',report)
                audit['days'][day] = report
                print(json.dumps(dict(date=day,status='clock_metadata',
                    sampled=report['sampled_capture_count'],missing_bins=report['missing_bins'])),flush=True)
            except Exception as error:
                audit['errors'].append(dict(date=day,phase='tar_clock_metadata',error=repr(error)))
            audit['elapsed_this_run_seconds'] = perf_counter()-started
            write(audit_path,audit)
    audit['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    audit['unfinished_dates'] = [r['date'] for r in metadata['candidate_days'] if r['date'] not in audit['days']]
    write(audit_path,audit)
    if audit['unfinished_dates']:
        raise RuntimeError('Incomplete source dates retained with range errors for resumption')


if __name__ == '__main__':
    main()
