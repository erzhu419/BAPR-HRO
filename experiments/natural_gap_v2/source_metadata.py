"""Collect public archive directories and publication identities before outcomes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

from experiments.decision_calibration_v2 import actual_metadata

HERE = Path(__file__).resolve().parent


def static_metadata(version):
    url = ('https://archive.opentransportdata.swiss/timetable_gtfs/'
           f'timetable-2023-gtfs2020/GTFS_FP2023_{version}_04-15.zip')
    code = '''from urllib.request import Request,urlopen
from zipfile import ZipFile
from io import BytesIO
import csv,json,struct,zlib
url=URL
downloaded=0
def fetch(start,length):
 global downloaded
 value=f'bytes={start}-{start+length-1}' if start>=0 else f'bytes=-{length}'
 with urlopen(Request(url,headers={'Range':value}),timeout=60) as response:
  if response.status!=206:raise ValueError('Range response required')
  raw=response.read(length); downloaded+=len(raw)
  return raw,int(response.headers['Content-Range'].rsplit('/',1)[1])
tail,size=fetch(-1,65557)
with ZipFile(BytesIO(tail)) as archive:
 members=[dict(name=m.filename,compressed_bytes=m.compress_size,uncompressed_bytes=m.file_size,
  zip_metadata_timestamp=list(m.date_time),absolute_local_header_offset=m.header_offset+size-len(tail))
  for m in archive.infolist()]
 member=archive.getinfo('feed_info.txt')
 header,_=fetch(member.header_offset+size-len(tail),30)
 fields=struct.unpack('<4s5H3L2H',header)
 raw,_=fetch(member.header_offset+size-len(tail)+30+sum(fields[-2:]),member.compress_size)
 text=zlib.decompress(raw,-15).decode('utf-8-sig')
print(json.dumps(dict(url=url,archive_bytes=size,members=members,
 feed_info=list(csv.DictReader(text.splitlines())),server_downloaded_bytes=downloaded,
 actual_or_realtime_values_opened=False)))
'''.replace('URL', repr(url))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'jtl110gpu2', 'python3 -'],
        input=code, text=True, capture_output=True, check=True)
    value = json.loads(result.stdout)
    value.update(checked_at_utc=datetime.now(timezone.utc).isoformat(), host='jtl110gpu2',
                 static_release=version, local_metadata_response_bytes=len(result.stdout.encode()))
    (HERE / f'static_metadata_{version}.json').write_text(json.dumps(value, indent=2) + '\n')
    return value


def mirror_metadata(day):
    url = f'http://mirror.traines.eu/swiss-gtfs/{day}/'
    with urlopen(url, timeout=60) as response:
        html = response.read().decode()
    match = re.search(r'<a href="(gtfs_fp2023_([0-9-]+)_04-15.zip)">.*?</a>\s+'
                      r'([0-9]+-[A-Za-z]+-[0-9]+ [0-9:]+)\s+(\d+)', html)
    return dict(index_url=url, filename=match[1], static_release=match[2],
                modified=match[3], archive_bytes=int(match[4]), html=html)


def main():
    HERE.mkdir(exist_ok=True)
    actual_metadata.HERE = HERE
    # The March directory was already read without values; reuse that receipt.
    march = HERE.parent / 'decision_calibration_v2/actual_metadata_2023-03.json'
    (HERE / march.name).write_bytes(march.read_bytes())
    with ThreadPoolExecutor(max_workers=4) as pool:
        static_jobs = [pool.submit(static_metadata, version)
                       for version in ('2023-03-22', '2023-03-29')]
        april_job = pool.submit(actual_metadata.month_metadata, '2023-04')
        mirror = list(pool.map(mirror_metadata, ('2023-03-27', '2023-04-03', '2023-04-10')))
        statics = [job.result() for job in static_jobs]
        april = april_job.result()
    value = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        march_metadata_reused_from=str(march), april_metadata='actual_metadata_2023-04.json',
        static_versions=[row['static_release'] for row in statics], mirror_indexes=mirror,
        collector_source='https://github.com/traines-source/public-transport-statistics/blob/master/ingest/extract-gtfsrt.js',
        version_rule='Next ZIP filesystem mtime calendar date at14:00; use the preceding release on switch-day mornings.',
        expected_morning_versions={'2023-03-23..2023-03-29': '2023-03-22',
                                   '2023-03-30..2023-04-05': '2023-03-29'},
        dates_frozen=False, actual_or_rt_values_read=False)
    (HERE / 'static_source_metadata.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(statics=[dict(release=r['static_release'], bytes=r['archive_bytes'],
        feed_info=r['feed_info']) for r in statics], april_archive_bytes=april['archive_bytes'],
        mirror_versions=[{k:v for k,v in r.items() if k!='html'} for r in mirror])))


if __name__ == '__main__':
    main()
