"""Read only server archive names, sizes and existing source schemas."""
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOTS = {
    'jtl110gpu2': [
        '/home/erzhu419/BAPR-HRO/data',
        '/home/erzhu419/BAPR-HRO/data/swiss_gtfs',
        '/home/erzhu419/BAPR-HRO/data/zurich_gtfs',
        '/home/erzhu419/BAPR-HRO/data/swiss_rt',
        '/home/erzhu419/BAPR-HRO/data/swiss_rt/gtfs-rt',
        '/home/erzhu419/BAPR-HRO/experiments/reconstruction_data_audit/remote_actual_20231025',
        '/home/erzhu419/BAPR-HRO/experiments/reconstruction_data_audit/remote_actual_20231025/actual_member_parts',
        '/home/erzhu419/mine_code/BAPR/data',
    ],
    'jtl110gpu': [
        '/home/erzhu419/BAPR-HRO/data', '/home/erzhu419/mine_code/BAPR-HRO/data',
        '/home/erzhu419/mine_code/BAPR/data', '/home/erzhu419/bapr_server/data',
        '/home/erzhu419/bapr_server.old/data',
    ],
    'jtl311linux': [
        '/home/erzhu419/BAPR-HRO/data', '/home/erzhu419/mine_code/BAPR-HRO/data',
        '/home/erzhu419/mine_code/BAPR/data',
    ],
}


def run():
    hosts = []
    total_bytes = 0
    for host, roots in ROOTS.items():
        code = '''import json,gzip,re
from pathlib import Path
roots=ROOTS
out={'directories':{},'static_feed_metadata':{},'schemas':{}}
for name in roots:
 p=Path(name)
 if not p.exists():
  out['directories'][name]={'exists':False}
  continue
 out['directories'][name]={'exists':True,'entries':[
  {'name':f.name,'type':'directory' if f.is_dir() else 'file',
   'bytes':f.stat().st_size if f.is_file() else None,
   'modified_ns':f.stat().st_mtime_ns}
  for f in sorted(p.iterdir()) if f.is_dir() or
  (f.name.endswith(('.csv','.csv.gz','.txt','.zip','.tar.bz2','.pkl')))]}
 feed=p/'feed_info.txt'
 if feed.exists():out['static_feed_metadata'][str(feed)]=feed.read_text()
 sample=p/'rail_calibration_2023-10-01.csv.gz'
 if sample.exists():
  with gzip.open(sample,'rt') as h:out['schemas'][str(sample)]=h.readline().rstrip()
print(json.dumps(out))
'''.replace('ROOTS', repr(roots))
        result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,'python3 -'],
            input=code,text=True,capture_output=True,check=True)
        transferred = len(result.stdout.encode())+len(result.stderr.encode())
        total_bytes += transferred
        hosts.append(dict(host=host, response_bytes=transferred, **json.loads(result.stdout)))
    exposure = json.loads((HERE.parent/'decision_gap_v1/calibration_protocol.json').read_text())['known_exposed_source_dates']
    import re
    actual_dates = sorted({date for host in hosts for row in host['directories'].values()
        for item in row.get('entries', []) if item['name'].endswith('.csv.gz')
        for date in re.findall(r'20\d\d-\d\d-\d\d',item['name'])})
    rt_dates = sorted({date for host in hosts for name,row in host['directories'].items()
        if 'swiss_rt' in name or 'gtfs-rt' in name
        for item in row.get('entries', []) if item['name'].endswith(('.tar.bz2','.zip','.tar.gz'))
        for date in re.findall(r'20\d\d-\d\d-\d\d',item['name'])})
    eligible = sorted(set(actual_dates) & set(rt_dates) - set(exposure))
    output = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Existing user-owned project source locations; names/stat metadata and historical exposed-source header only; no target outcome read',
        hosts=hosts, response_bytes=total_bytes, payload_csv_or_checkpoint_bytes_transferred=0,
        candidate_actual_dates_from_cache_names=actual_dates,
        known_exposed_dates=exposure,
        actual_dates_outside_known_exposure=sorted(set(actual_dates)-set(exposure)),
        remote_gtfs_rt_archive_dates=rt_dates,
        eligible_common_dates=eligible,
        transfer_accounting='response_bytes covers this reproducible inventory call only; earlier shallow-directory reconnaissance also returned metadata only and was not byte-accounted',
        conclusion='No unexposed date with both named archive types found' if not eligible else 'Candidate dates require exposure and static-feed review before outcome access')
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE/'server_inventory.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(dict(hosts=len(hosts), metadata_response_bytes=total_bytes,
               actual_cache_dates=actual_dates, eligible_common_dates=eligible))


if __name__ == '__main__':
    run()
