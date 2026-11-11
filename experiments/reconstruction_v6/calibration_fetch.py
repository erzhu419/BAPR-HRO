"""Pull only predeclared small actual subsets after the calibration freeze."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = '/home/erzhu419/BAPR-HRO/experiments/reconstruction_data_audit/remote_actual_20231025'


def fetch(day):
    remote = f'{REMOTE}/rail_v6_confirmation_{day}.csv.gz'
    command = ['python3', '-u', f'{REMOTE}/filter_actuals_remote.py', '--date', day,
               '--filter', f'{REMOTE}/bapr-rail-stops.json', '--prefix-bytes', '1048576', '--output', remote]
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'jtl110gpu2',
                    shlex.join(command)], check=True, capture_output=True, text=True)
    for path in (remote, remote.removesuffix('.gz') + '.audit.json'):
        subprocess.run(['scp', '-q', f'jtl110gpu2:{path}', str(HERE / 'calibration_confirmation_sources')], check=True)
    return day, (HERE / 'calibration_confirmation_sources' / Path(remote).name).stat().st_size


if __name__ == '__main__':
    receipt = json.loads((HERE / 'calibration_freeze_receipt.json').read_text())
    if receipt['new_actuals_opened']:
        raise ValueError('This entry fetches the first frozen confirmation cohort, not another selected date search')
    (HERE / 'calibration_confirmation_sources').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for day, size in pool.map(fetch, receipt['protocol']['actual_label_confirmation_dates']):
            print(day, size, 'filtered local gzip bytes', flush=True)
