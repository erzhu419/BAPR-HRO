"""Correct the Oct28 truncated source; keep frozen predictions and thresholds."""
import argparse
import csv
import gzip
from pathlib import Path
import pickle
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V5 = ROOT / 'experiments/reconstruction_v5'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V5))
from experiments.reconstruction_v6.calibration_run import read, write
from experiments.reconstruction_v6.calibration_confirm import evaluate
from experiments.reconstruction_v6.calibration_selection import summary
from experiments.reconstruction_data_audit.build_dated_replay import (
    dated_connections, attach_outcomes, actual_endpoints, endpoint_key,
    seconds_of_service, FIELDS,
)
from calibration_run import data_inputs


def supplemental_exact_rows(day, connections, rows):
    """Expose extra-run records only where the strict exact event key is absent.

    This is a labeled source-identity sensitivity, not a claim that the extra
    operational journey belongs to the original planned service. Multiple
    matching extra records remain ambiguous in the original outcome parser.
    """
    regular = actual_endpoints(day, connections, rows)
    expected = {endpoint_key(row, kind) for row in connections for kind in FIELDS}
    additions, used = [], []
    for row in rows:
        if row['ZUSATZFAHRT_TF'] != 'true':
            continue
        candidate = dict(row, ZUSATZFAHRT_TF='false')
        selected = []
        for kind, (planned, _, _) in FIELDS.items():
            key = None if not row[planned] else (
                row['BETREIBER_ID'], row['LINIEN_ID'], row['BPUIC'], kind,
                seconds_of_service(row[planned], day) // 60)
            if key in expected and not regular[key]:
                selected.append(key)
            else:
                candidate[planned] = ''
        if selected:
            additions.append(candidate)
            used.append({'journey': row['FAHRT_BEZEICHNER'], 'keys': selected,
                         'canceled_flag': row['FAELLT_AUS_TF']})
    return rows + additions, used


def main(supplemental=False):
    day = '2023-10-28'
    old = read(HERE / 'calibration_selection_actuals.json.gz')['dates'][day]
    stops, schedules = dated_connections([day])
    path = HERE / 'calibration_confirmation_sources' / f'rail_v6_confirmation_chunked_{day}.csv.gz'
    audit = read(path.with_suffix('.audit.json'))
    if not audit['complete_daily_member']:
        raise ValueError('Source correction requires a complete daily member')
    with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle, delimiter=';'))
    if supplemental:
        rows, extra = supplemental_exact_rows(day, schedules[day], rows)
    else:
        extra = []
    coverage = attach_outcomes(day, schedules[day], rows)
    new = {'connections': schedules[day], 'coverage': coverage, 'source_audit': audit}
    before = {row['id']: row for row in old['connections']}
    changed = []
    unchanged_known = 0
    for row in new['connections']:
        prior = before[row['id']]
        for event in ('arrival', 'departure'):
            a, b = prior[event], row[event]
            if a != b:
                changed.append({'cid': row['id'], 'route': row['route_id'], 'event': event,
                                'before': a, 'after': b})
            elif a['status'] != 'MISSING':
                unchanged_known += 1
    frozen = read(HERE / 'calibration_selection_frozen.json')
    with open(HERE / 'calibration_frozen_model.pkl', 'rb') as handle:
        student = pickle.load(handle)
    with open(V5 / 'calibration_relative_variants_frozen.pkl', 'rb') as handle:
        shrunk = pickle.load(handle)['shrunk_relative']
    fit = read(V5 / 'calibration_frozen_fit.json')['evolving_fit']
    data = data_inputs()
    data['dates'][day] = new
    outcomes = {day: evaluate(data, day, fit, {'student': student, 'shrunk_relative': shrunk})}
    result = {'source_correction': {
        'reason': 'The original1MiB source prefix omitted later rows in the same daily member.',
        'models_and_thresholds_unchanged': True, 'original_results_preserved': True,
        'supplemental_exact_sensitivity': supplemental, 'supplemental_matches': extra,
        'prefix_coverage': old['coverage'], 'complete_coverage': coverage,
        'unchanged_known_endpoints': unchanged_known, 'changed_endpoints': changed,
        'scope': 'Post-opening source diagnostic on the same locked schedule. '
                 'Supplemental exact-key matching changes source eligibility; its scores are a '
                 'sensitivity, not a new independently confirmed calibration guarantee.'},
        'protocol': frozen['protocol'], 'thresholds': frozen['thresholds'],
        'rows_by_date': outcomes, 'confirmation': summary(outcomes, frozen['thresholds'])}
    stem = 'calibration_selection_supplemental' if supplemental else 'calibration_selection_full_source'
    write(HERE / f'{stem}_actuals.json.gz',
          {'stops': stops, 'dates': {day: new}})
    write(HERE / f'{stem}_results.json.gz', result)
    write(HERE / f'{stem}_summary.json',
          {key: value for key, value in result.items() if key != 'rows_by_date'})
    print('SOURCE', {key: result['source_correction'][key] for key in (
        'prefix_coverage', 'complete_coverage', 'unchanged_known_endpoints')}, flush=True)
    print('CHANGED', len(changed), 'SUPPLEMENTAL', len(extra), flush=True)
    for method in ('student', 'shrunk_relative', 'evolving_incoming'):
        for mode in ('fixed_selector', 'menu_max'):
            value = result['confirmation'][day]['fixed_schedule'][method][mode]
            print(method, mode, {key: value[key] for key in (
                'n_incoming_states', 'coverage_lower', 'coverage_upper',
                'unresolved_targets', 'canceled_targets', 'state_mean_clock_width_seconds')},
                value['observable_groups'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--supplemental-exact', action='store_true')
    main(parser.parse_args().supplemental_exact)
