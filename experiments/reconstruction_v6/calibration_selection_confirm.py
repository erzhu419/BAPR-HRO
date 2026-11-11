"""New actual-label confirmation of the frozen selection coverage targets."""
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import pickle
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V5 = ROOT / 'experiments/reconstruction_v5'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V5))
from experiments.reconstruction_v6.calibration_fetch import fetch
from experiments.reconstruction_v6.calibration_run import read, write
from experiments.reconstruction_v6.calibration_confirm import evaluate
from experiments.reconstruction_v6.calibration_selection import summary
from experiments.reconstruction_data_audit.build_dated_replay import dated_connections, attach_outcomes
from calibration_run import data_inputs


def main():
    frozen = read(HERE / 'calibration_selection_frozen.json')
    protocol = frozen['protocol']
    dates = protocol['confirmation_dates']
    with open(HERE / 'calibration_frozen_model.pkl', 'rb') as handle:
        student = pickle.load(handle)
    with open(V5 / 'calibration_relative_variants_frozen.pkl', 'rb') as handle:
        shrunk = pickle.load(handle)['shrunk_relative']
    fit = read(V5 / 'calibration_frozen_fit.json')['evolving_fit']
    with ThreadPoolExecutor(max_workers=3) as pool:
        for day, count in pool.map(fetch, dates):
            print('SMALL SOURCE', day, count, flush=True)
    stops, schedules = dated_connections(dates)
    data = data_inputs()
    actuals = {'stops': stops, 'dates': {}, 'protocol': protocol}
    frozen['new_actuals_opened'] = True
    write(HERE / 'calibration_selection_open_receipt.json', frozen)
    for day in dates:
        path = HERE / 'calibration_confirmation_sources' / f'rail_v6_confirmation_{day}.csv.gz'
        with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle, delimiter=';'))
        coverage = attach_outcomes(day, schedules[day], rows)
        actuals['dates'][day] = {'connections': schedules[day], 'coverage': coverage,
                                'source_audit': read(path.with_suffix('.audit.json'))}
        print('JOIN', day, coverage, flush=True)
    write(HERE / 'calibration_selection_actuals.json.gz', actuals)
    data['dates'].update(actuals['dates'])
    outcomes = {}
    for day in dates:
        outcomes[day] = evaluate(data, day, fit, {'student': student, 'shrunk_relative': shrunk})
        print('SCORED', day, flush=True)
    result = {'protocol': protocol, 'thresholds': frozen['thresholds'], 'rows_by_date': outcomes,
              'confirmation': summary(outcomes, frozen['thresholds']),
              'source_coverage': {day: actuals['dates'][day]['coverage'] for day in dates}}
    write(HERE / 'calibration_selection_confirmation_results.json.gz', result)
    write(HERE / 'calibration_selection_confirmation_summary.json', {k: v for k, v in result.items() if k != 'rows_by_date'})


if __name__ == '__main__':
    main()
