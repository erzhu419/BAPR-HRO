"""Prepare October 1--11 rail calibration days from small remote-filtered files."""
import csv
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_data_audit.build_dated_replay import attach_outcomes, dated_connections


def main():
    dates = [f'2023-10-{day:02}' for day in range(1, 12)]
    stops, scheduled = dated_connections(dates)
    results = {}
    for day in dates:
        path = HERE / 'rail_calibration_sources' / f'rail_calibration_{day}.csv.gz'
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle, delimiter=';'))
        coverage = attach_outcomes(day, scheduled[day], rows)
        results[day] = {'coverage': coverage, 'connections': scheduled[day],
                        'filtered_source': str(path.relative_to(ROOT))}
        print(day, coverage, flush=True)
    output = {'scope': 'Earlier actual rail calibration dates for the same fixed 16-station cohort. '
              'The server filtered bounded SBB archive prefixes; every calendar-active candidate and unresolved endpoint remains visible.',
              'timezone': 'Europe/Zurich', 'stops': stops, 'dates': results,
              'heldout_dates_excluded': ['2023-10-12', '2023-10-25', '2023-10-27']}
    with gzip.open(HERE / 'rail_calibration_dates_v2.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(output, handle, ensure_ascii=False, separators=(',', ':'))


if __name__ == '__main__':
    main()
