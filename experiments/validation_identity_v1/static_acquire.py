"""Acquire the metadata-frozen validation releases and export public catalogs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

from experiments.decision_calibration_v2.prepare_static import extract_static
from experiments.training_identity_v1.public_catalog import build_public_catalog

BLOCK_BYTES, WORKERS = 1024**2, 32


def fetch_range(url, start, length, size):
    expected = f'bytes {start}-{start+length-1}/{size}'
    with urlopen(Request(url,headers={'Range':f'bytes={start}-{start+length-1}'}),timeout=90) as response:
        if response.status != 206 or response.headers['Content-Range'] != expected:
            raise ValueError('Official archive did not honor the requested range')
        raw = response.read()
    if len(raw) != length:
        raise ValueError('Incomplete static archive range')
    return raw


def download_archive(url, size, path):
    """Record each fully received and written block; reuse only those receipts."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.range-download')
    journal=path.with_suffix('.blocks.jsonl')
    if path.exists():
        if path.stat().st_size != size:
            raise ValueError('Cached archive differs from frozen source size')
        return dict(reused_complete_zip=True,downloaded_bytes=0,reused_recorded_bytes=size,
                    block_receipts=str(journal),errors=[])
    completed={}
    if temporary.exists():
        if not journal.exists():
            raise ValueError('Cannot resume an incomplete archive without block receipts')
        for line in journal.read_text().splitlines():
            row=json.loads(line)
            if row['status']=='complete':
                completed[row['start']]=row['bytes']
        if temporary.stat().st_size != size:
            raise ValueError('Interrupted archive differs from frozen source size')
        fd=os.open(temporary,os.O_WRONLY)
    else:
        fd=os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
        os.ftruncate(fd,size)
    prior=sum(completed.values())
    tick,downloaded=perf_counter(),0
    errors=[]

    def fetch(start):
        length=min(BLOCK_BYTES,size-start)
        raw=fetch_range(url,start,length,size)
        if os.pwrite(fd,raw,start) != length:
            raise OSError('Incomplete block write; no completion receipt recorded')
        return length

    try:
        with journal.open('a') as receipts, ThreadPoolExecutor(max_workers=WORKERS) as pool:
            jobs={pool.submit(fetch,start):start for start in range(0,size,BLOCK_BYTES) if start not in completed}
            for job in as_completed(jobs):
                start=jobs[job]
                try:
                    length=job.result()
                    completed[start]=length
                    downloaded+=length
                    row=dict(status='complete',start=start,bytes=length)
                except Exception as error:
                    row=dict(status='failed',start=start,error=f'{type(error).__name__}: {error}')
                    errors.append(row)
                row.update(recorded_at_utc=datetime.now(timezone.utc).isoformat(),source_url=url,archive_bytes=size)
                receipts.write(json.dumps(row)+'\n')
                receipts.flush()
                if row['status']=='complete' and downloaded//(16*BLOCK_BYTES)!=(downloaded-length)//(16*BLOCK_BYTES):
                    print(json.dumps(dict(stage='download',network_bytes=downloaded,reused_recorded_bytes=prior,
                                          total=size,seconds=perf_counter()-tick)),flush=True)
    finally:
        os.close(fd)
    if errors:
        raise RuntimeError(f'{len(errors)} archive ranges failed; complete and failed receipts retained in {journal}')
    if sum(completed.values()) != size:
        raise ValueError('Recorded ranges do not cover the complete archive')
    temporary.rename(path)
    return dict(reused_complete_zip=False,downloaded_bytes=downloaded,reused_recorded_bytes=prior,
                block_receipts=str(journal),complete_blocks=len(completed),errors=[],seconds=perf_counter()-tick)


def save_gzip(path,value):
    with gzip.open(path,'wt') as handle:
        json.dump(value,handle,separators=(',',':'))


def acquire(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'static_source_manifest.json').read_text())
    versions=manifest['source_version_by_date']
    catalog_dir=directory/'source_catalogs'
    catalog_dir.mkdir(parents=True,exist_ok=True)
    merged=dict(dates={},source_version_by_date=versions,
        source_metadata_frozen_at_utc=manifest['frozen_at_utc'],
        scope='Full-day active SBB named trips and planned endpoints at the original16 stations; metadata-selected public releases',
        actual_rt_or_validation_scores_read=False)
    reports=[]
    for version in sorted(set(versions.values())):
        meta=manifest['versions'][version]
        dates=[day for day in manifest['dates'] if versions[day]==version]
        existing=manifest['existing_runtime_by_version'].get(version)
        preparation=None
        if existing:
            runtime=Path(existing)
            transport=dict(reused_filtered_static=True,downloaded_bytes=0,
                           original_acquisition_receipt=str(runtime.parents[1]/'static_acquisition.json'))
        else:
            archive=directory/'cache'/meta['url'].rsplit('/',1)[1]
            transport=download_archive(meta['url'],meta['archive_bytes'],archive)
            runtime=directory/'inputs'/f'static_{version}'
            preparation=extract_static(archive,runtime)
            if preparation['feed_info'] != meta['feed_info']:
                raise ValueError('Prepared feed metadata differs from the frozen public source')
        catalog=build_public_catalog(runtime,dates)
        catalog['static_release']=version
        catalog['feed_info']=meta['feed_info']
        path=catalog_dir/f'{version}.json.gz'
        save_gzip(path,catalog)
        merged['dates'].update(catalog['dates'])
        report=dict(static_release=version,dates=dates,runtime=str(runtime),
            source_url=meta['url'],feed_info=meta['feed_info'],transport=transport,
            preparation=preparation,catalog_file=str(path),catalog_bytes=path.stat().st_size,
            counts={day:dict(trips=len(row['trips']),endpoints=len(row['endpoints'])) for day,row in catalog['dates'].items()})
        reports.append(report)
        print(json.dumps({key:value for key,value in report.items() if key!='preparation'}),flush=True)
    target=directory/'active_public_endpoints.json.gz'
    save_gzip(target,merged)
    receipt=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_metadata_frozen_at_utc=manifest['frozen_at_utc'],versions=reports,
        merged_catalog=str(target),merged_catalog_bytes=target.stat().st_size,
        all_dates=sorted(merged['dates']),old_data_models_or_previous_stage_changed=False,
        actual_rt_or_validation_scores_read=False)
    (directory/'static_acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(stage='complete',dates=receipt['all_dates'],merged_catalog_bytes=receipt['merged_catalog_bytes'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    acquire(parser.parse_args().directory)
