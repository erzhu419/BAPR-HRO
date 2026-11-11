"""Join predeclared confirmation dates using remotely filtered small sources."""
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
    protocol = json.loads((HERE / 'calibration_protocol.json').read_text())
    dates = protocol['confirmation_dates']
    stops, schedules = dated_connections(dates)
    output = {'scope': 'Predeclared new-date rail confirmation with all scheduled candidates and unresolved outcomes.',
              'stops': stops, 'dates': {}, 'confirmation_dates': dates}
    for day in dates:
        path = HERE / 'calibration_confirmation_sources' / f'rail_confirmation_{day}.csv.gz'
        with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
            records = list(csv.DictReader(handle, delimiter=';'))
        connections = schedules[day]
        coverage = attach_outcomes(day, connections, records)
        audit = json.loads(path.with_suffix('.audit.json').read_text())
        output['dates'][day] = {'connections': connections, 'coverage': coverage,
                                'source_audit': audit, 'source': str(path.relative_to(ROOT))}
    with gzip.open(HERE / 'calibration_confirmation_actuals.json.gz', 'wt') as handle:
        json.dump(output, handle, separators=(',', ':'))
    print('Prepared three predeclared dates, retaining unresolved outcomes.')


if __name__ == '__main__':
    main()
