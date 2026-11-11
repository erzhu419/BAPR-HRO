"""Prepare past-date public receipts for the frozen calibration comparison."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_data_audit.extract_dated_public_forecasts import scan
from experiments.reconstruction_data_audit.build_dated_replay import dated_connections


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def main():
    audit = ROOT / 'experiments/reconstruction_data_audit'
    earlier = load(audit / 'rail_calibration_dates_v2.json.gz')
    heldout = load(audit / 'dated_public_forecasts.json.gz')
    dates = dict(heldout['dates'])
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(scan, (day, value)): day for day, value in earlier['dates'].items()}
        for future in as_completed(jobs):
            day, value = future.result()
            dates[day] = value
            print(day, len(value['frames']), value['endpoint_revisions'], flush=True)
    output = {**heldout, 'scope': 'Public GTFS-RT frames from existing local archives. '
              'Earlier and held-out dates use the same ten-minute capture sampling and receipt clock.', 'dates': dates}
    with gzip.open(HERE / 'calibration_public_frames.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(output, handle, separators=(',', ':'))


def prepare_confirmation():
    protocol = json.loads((HERE / 'calibration_protocol.json').read_text())
    stops, connections = dated_connections(protocol['confirmation_dates'])
    schedules = {'scope': 'Static calendar and public forecasts only; actual confirmation outcomes remain unread.',
                 'stops': stops, 'dates': {day: {'connections': rows} for day, rows in connections.items()}}
    dates = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(scan, (day, rows)) for day, rows in schedules['dates'].items()]
        for future in as_completed(futures):
            day, row = future.result()
            dates[day] = row
            print('confirmation_public', day, len(row['frames']), flush=True)
    with gzip.open(HERE / 'calibration_confirmation_schedules.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(schedules, handle, separators=(',', ':'))
    with gzip.open(HERE / 'calibration_confirmation_public_frames.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump({'dates': dates}, handle, separators=(',', ':'))


if __name__ == '__main__':
    if '--confirmation' in sys.argv:
        prepare_confirmation()
    else:
        main()
