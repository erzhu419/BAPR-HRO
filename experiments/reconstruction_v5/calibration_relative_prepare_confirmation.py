"""Prepare predeclared new-date schedules, public receipts and tiny actual joins."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_data_audit.build_dated_replay import attach_outcomes, dated_connections
from experiments.reconstruction_data_audit.extract_dated_public_forecasts import scan


def main():
    protocol = json.loads((HERE / 'calibration_relative_confirmation_protocol.json').read_text())
    dates = protocol['confirmation_dates']
    stops, schedules = dated_connections(dates)
    public = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(scan, (day, {'connections': schedules[day]})) for day in dates]
        for task in as_completed(tasks):
            day, value = task.result()
            public[day] = value
            print('PUBLIC', day, len(value['frames']), value['endpoint_revisions'], flush=True)
    with gzip.open(HERE / 'calibration_relative_confirmation_public_frames.json.gz', 'wt') as handle:
        json.dump({'dates': public}, handle, separators=(',', ':'))
    output = {'scope': 'Predeclared independent dates; remotely filtered partial SBB source; '
              'every active scheduled candidate is retained, including unknown outcomes.',
              'confirmation_dates': dates, 'stops': stops, 'dates': {}}
    for day in dates:
        path = HERE / 'calibration_relative_confirmation_sources' / f'rail_relative_confirmation_{day}.csv.gz'
        with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle, delimiter=';'))
        coverage = attach_outcomes(day, schedules[day], rows)
        audit = json.loads(path.with_suffix('.audit.json').read_text())
        output['dates'][day] = {'connections': schedules[day], 'coverage': coverage,
                                'filtered_source': str(path.relative_to(ROOT)), 'source_audit': audit}
        print('ACTUAL JOIN', day, coverage, flush=True)
    with gzip.open(HERE / 'calibration_relative_confirmation_actuals.json.gz', 'wt') as handle:
        json.dump(output, handle, separators=(',', ':'), allow_nan=False)


if __name__ == '__main__':
    main()
