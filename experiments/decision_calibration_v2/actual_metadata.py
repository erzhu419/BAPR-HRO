"""Read official monthly ZIP directory metadata on the server, never CSV values."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def month_metadata(month, host='jtl110gpu2'):
    url = f'https://archive.opentransportdata.swiss/istdaten/{month[:4]}/ist-daten-{month}.zip'
    code = '''from urllib.request import Request,urlopen
from zipfile import ZipFile
from io import BytesIO
import json
url=URL
with urlopen(Request(url,headers={'Range':'bytes=-65557'}),timeout=60) as response:
 if response.status!=206:raise ValueError('Range response required; no full archive download')
 raw=response.read(65557)
 size=int(response.headers['Content-Range'].rsplit('/',1)[1])
 headers={k:response.headers.get(k) for k in ('Content-Type','Content-Range','Last-Modified')}
with ZipFile(BytesIO(raw)) as archive:
 members=[dict(name=m.filename,compressed_bytes=m.compress_size,uncompressed_bytes=m.file_size,
   zip_metadata_timestamp=list(m.date_time),absolute_local_header_offset=m.header_offset+size-len(raw))
   for m in archive.infolist()]
print(json.dumps(dict(url=url,status=206,archive_bytes=size,metadata_bytes_downloaded=len(raw),
 response_headers=headers,members=members,csv_values_opened=False)))
'''.replace('URL', repr(url))
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,'python3 -'],
        input=code,text=True,capture_output=True,check=True)
    value = json.loads(result.stdout)
    value.update(host=host, local_metadata_response_bytes=len(result.stdout.encode()),
                 checked_at_utc=datetime.now(timezone.utc).isoformat())
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE/f'actual_metadata_{month}.json').write_text(json.dumps(value,indent=2)+'\n')
    print(dict(month=month, members=len(value['members']), archive_bytes=value['archive_bytes'],
               server_metadata_bytes=value['metadata_bytes_downloaded'],
               local_metadata_response_bytes=value['local_metadata_response_bytes']))
    return value


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('months', nargs='+')
    args=parser.parse_args()
    for month in args.months:
        month_metadata(month)
