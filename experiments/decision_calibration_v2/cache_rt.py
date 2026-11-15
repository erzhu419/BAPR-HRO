"""Cache only the preregistered RT archives on the server, without decoding."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter
from urllib.request import Request, urlopen


def download_range(source, start, end, path):
    """HTTP serves the same public object faster on the research servers."""
    expected = end-start+1
    previous = path.stat().st_size if path.exists() else 0
    if previous == expected:
        return 0
    url = source['url'].replace('https://mirror.traines.eu/', 'http://mirror.traines.eu/')
    request = Request(url, headers={'Range':f'bytes={start+previous}-{end}'})
    with urlopen(request, timeout=120) as response, path.open('ab') as handle:
        if response.status != 206 or response.headers['Content-Range'] != (
                f'bytes {start+previous}-{end}/{source["size_bytes"]}'):
            raise ValueError('Mirror did not return the requested archive range')
        shutil.copyfileobj(response, handle, length=262144)
    if path.stat().st_size != expected:
        raise ValueError(f'Incomplete RT archive range {start}-{end}: '
                         f'retained {path.stat().st_size} of {expected} bytes')
    return expected-previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--connections', type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not manifest['dates_frozen']:
        raise ValueError('Date manifest must be frozen before acquisition')
    started = perf_counter()
    audit = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        date_freeze_utc=manifest['frozen_at_utc'], cache=str(args.cache),
        archive_values_decoded=False, connections=args.connections,
        transport='HTTP range against the same mirror host/path; full archive bytes retained', days=[], errors=[])
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    plans, jobs = {}, {}

    def record(day, target, downloaded):
        row = dict(date=day, source=str(target),
            archive_bytes=target.stat().st_size, downloaded_bytes=downloaded,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_wall_seconds=perf_counter()-started)
        audit['days'].append(row)
        audit['days'].sort(key=lambda r:r['date'])
        audit['total_downloaded_bytes'] = sum(r['downloaded_bytes'] for r in audit['days'])
        audit['wall_seconds'] = perf_counter()-started
        args.audit.write_text(json.dumps(audit, indent=2)+'\n')
        print(json.dumps(row), flush=True)

    with ThreadPoolExecutor(max_workers=args.connections) as pool:
        for row in manifest['days']:
            source, day = row['rt'], row['day']
            target = args.cache/source['filename']
            if target.exists() and target.stat().st_size == source['size_bytes']:
                record(day, target, 0)
                continue
            prefix = target.with_suffix(target.suffix+'.part')
            prefix_bytes = prefix.stat().st_size if prefix.exists() else 0
            folder = args.cache/(day+'.ranges')
            folder.mkdir(exist_ok=True)
            parts = []
            for start in range(prefix_bytes, source['size_bytes'], 8*1024*1024):
                end = min(start+8*1024*1024, source['size_bytes'])-1
                part = folder/f'{start:012d}.part'
                parts.append(part)
                jobs[pool.submit(download_range, source, start, end, part)] = day
            plans[day] = dict(target=target, prefix=prefix, parts=parts,
                folder=folder, pending=len(parts), downloaded=0, expected=source['size_bytes'], failed=False)
        for future in as_completed(jobs):
            day = jobs[future]
            plan = plans[day]
            try:
                plan['downloaded'] += future.result()
            except Exception as error:
                plan['failed'] = True
                failure = dict(date=day, error=repr(error))
                audit['errors'].append(failure)
                args.audit.write_text(json.dumps(audit, indent=2)+'\n')
                print(json.dumps(failure), flush=True)
            plan['pending'] -= 1
            if plan['pending'] or plan['failed']:
                continue
            with plan['prefix'].open('ab') as handle:
                for part in plan['parts']:
                    with part.open('rb') as fragment:
                        shutil.copyfileobj(fragment, handle)
            if plan['prefix'].stat().st_size != plan['expected']:
                raise ValueError('Assembled RT archive length differs from frozen metadata')
            plan['prefix'].replace(plan['target'])
            for part in plan['parts']:
                part.unlink()
            plan['folder'].rmdir()
            record(day, plan['target'], plan['downloaded'])
    if audit['errors']:
        raise RuntimeError('Incomplete archive ranges are retained for resuming')


if __name__ == '__main__':
    main()
