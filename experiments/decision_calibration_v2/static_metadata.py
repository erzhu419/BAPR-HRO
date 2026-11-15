"""Inspect official pre-decision GTFS publication and feed metadata on the server."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('publication_date', nargs='?', default='2023-03-01')
    args = parser.parse_args()
    url = ('https://archive.opentransportdata.swiss/timetable_gtfs/'
           f'timetable-2023-gtfs2020/GTFS_FP2023_{args.publication_date}_04-15.zip')
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
  return raw,int(response.headers['Content-Range'].rsplit('/',1)[1]),dict(response.headers)
tail,size,headers=fetch(-1,65557)
with ZipFile(BytesIO(tail)) as archive:
 members=[dict(name=m.filename,compressed_bytes=m.compress_size,uncompressed_bytes=m.file_size,
    zip_metadata_timestamp=list(m.date_time),absolute_local_header_offset=m.header_offset+size-len(tail))
    for m in archive.infolist()]
 member=archive.getinfo('feed_info.txt')
 offset=member.header_offset+size-len(tail)
 header,_,_=fetch(offset,30)
 fields=struct.unpack('<4s5H3L2H',header)
 raw,_,_=fetch(offset+30+sum(fields[-2:]),member.compress_size)
 text=zlib.decompress(raw,-15).decode('utf-8-sig')
print(json.dumps(dict(url=url,archive_bytes=size,members=members,feed_info=list(csv.DictReader(text.splitlines())),
 response_headers={k:headers.get(k) for k in ('Content-Type','Content-Range','Last-Modified')},
 server_downloaded_bytes=downloaded,actual_or_realtime_values_opened=False)))
'''.replace('URL', repr(url))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'jtl110gpu2', 'python3 -'], input=code, text=True,
                            capture_output=True, check=True)
    value = json.loads(result.stdout)
    value.update(checked_at_utc=datetime.now(timezone.utc).isoformat(), host='jtl110gpu2',
                 official_index_url='https://archive.opentransportdata.swiss/timetable_gtfs.php',
                 publication_version_from_archive_filename=f'{args.publication_date}_04-15',
                 publication_timezone='not stated by archive index',
                 publication_precedes_first_target_date=args.publication_date < '2023-03-09',
                 local_metadata_response_bytes=len(result.stdout.encode()))
    (HERE/f'static_metadata_{args.publication_date}.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k != 'members'}, indent=2))


if __name__ == '__main__':
    main()
