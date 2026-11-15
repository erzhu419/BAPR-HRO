"""Server entry for the eight-range public-static acquisition used in V2.

Only the official ZIPs and filtered static inputs are stored on the server.
Use the frozen metadata/date manifest beside this module; no target outcome or
message value is read. Existing complete ZIPs are reused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import os
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

from experiments.decision_gap_v1.panel import read, write
from experiments.decision_calibration_v2.prepare_static import (
    dated_connections_from_static, extract_static,
)

HERE = Path(__file__).resolve().parent
BLOCK_BYTES, WORKERS = 1024**2, 8


def acquire(runtime_root):
    runtime_root = Path(runtime_root).resolve()
    manifest = read(HERE/'date_manifest.json')
    if not manifest['dates_frozen']:
        raise ValueError('Freeze the metadata date panel first')
    cache = runtime_root/'cache'
    cache.mkdir(parents=True, exist_ok=True)
    receipts = []
    for version in sorted({row['static_version'] for row in manifest['days']}):
        metadata = read(HERE/f'static_metadata_{version}.json')
        url, size = metadata['url'], metadata['archive_bytes']
        path = cache/url.rsplit('/', 1)[1]
        tick, downloaded, reused = perf_counter(), 0, path.exists()
        if not reused:
            temporary = path.with_suffix('.range-download')
            fd = os.open(temporary, os.O_CREAT|os.O_TRUNC|os.O_WRONLY, 0o644)
            os.ftruncate(fd, size)

            def fetch(start):
                end = min(start+BLOCK_BYTES, size)-1
                with urlopen(Request(url, headers={'Range': f'bytes={start}-{end}'}), timeout=90) as response:
                    if response.status != 206 or response.headers['Content-Range'] != f'bytes {start}-{end}/{size}':
                        raise ValueError('Official server did not honor the requested range')
                    raw = response.read()
                if len(raw) != end-start+1:
                    raise ValueError('Incomplete static archive range')
                os.pwrite(fd, raw, start)
                return len(raw)

            try:
                with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                    for future in as_completed([pool.submit(fetch, start) for start in range(0, size, BLOCK_BYTES)]):
                        downloaded += future.result()
                        if downloaded//(8*BLOCK_BYTES) != (downloaded-BLOCK_BYTES)//(8*BLOCK_BYTES):
                            print(json.dumps(dict(stage='download', version=version, bytes=downloaded,
                                                  total=size, seconds=perf_counter()-tick)), flush=True)
            finally:
                os.close(fd)
            temporary.rename(path)
        output = runtime_root/'inputs'/f'static_{version}'
        audit = extract_static(path, output)
        dates = [row['day'] for row in manifest['days'] if row['static_version'] == version]
        stops, connections = dated_connections_from_static(dates, output)
        with gzip.open(output/'scheduled.json.gz', 'wt') as handle:
            json.dump(dict(stops=stops, dates={day:dict(connections=rows) for day,rows in connections.items()},
                           scope='Public schedules only; no outcomes attached'), handle)
        link = HERE/'static'/version
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(output, target_is_directory=True)
        row = dict(version=version, url=url, archive_bytes=size, downloaded_bytes=downloaded,
            reused_complete_zip=reused, workers=WORKERS, range_bytes=BLOCK_BYTES,
            seconds=perf_counter()-tick, runtime_root=str(output),
            scheduled_counts={day:len(rows) for day,rows in connections.items()},
            static_preparation=audit, target_actual_or_rt_read=False)
        receipts.append(row)
        write(HERE/'static_acquisition.json', dict(versions=receipts, target_actual_or_rt_read=False))
        print(json.dumps(dict(version=version, scheduled_counts=row['scheduled_counts'], output=str(output))), flush=True)
    return receipts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-root', type=Path, required=True)
    acquire(parser.parse_args().runtime_root)
