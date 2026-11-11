"""Freeze one conditional risk candidate before reading new actual labels."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V5 = ROOT / 'experiments/reconstruction_v5'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V5))
from calibration_relative_run import training_pairs, select_menu
from calibration_run import data_inputs, law_summary
from src.conditional_risk import ConditionalStudentModel, observable_group, simultaneous_group_envelope

METHODS = ('evolving_incoming', 'shrunk_relative', 'student')
MENUS = ('fixed_schedule', 'actual_current')


def read(path):
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as handle:
        return json.load(handle)


def write(path, value):
    with gzip.open(path, 'wt') if str(path).endswith('.gz') else open(path, 'w') as handle:
        json.dump(value, handle, indent=None if str(path).endswith('.gz') else 2, allow_nan=False)


def add_candidate(data, outcomes, model):
    for day, output in outcomes.items():
        source = {c['id']: c for c in data['dates'][day]['connections']}
        cached = {}
        for menu in MENUS:
            compact = []
            for original in output[menu]['rows']:
                key = original['incoming'], original['outgoing']
                if key not in cached:
                    row = {k: v for k, v in original.items() if k not in ('methods', 'probabilities')}
                    row['methods'] = {name: deepcopy(original['methods'][name]) for name in METHODS[:-1]}
                    row['probabilities'] = {name: original['probabilities'][name] for name in METHODS[:-1]}
                    law = model.predict(source[row['incoming']], source[row['outgoing']], row['clock'])
                    row['methods']['student'] = law_summary(law, row['actual'])
                    row['probabilities']['student'] = law.board_probability(row['ready'])
                    cached[key] = row
                compact.append(cached[key])
            output[menu]['rows'] = compact
        print('CANDIDATE', day, len(cached), flush=True)


def metrics(rows, method, factor=0., offset=0.):
    operating = [row for row in rows if row['actual'] is not None]
    covered, width, score, error = [], [], [], []
    for row in operating:
        prediction = row['methods'][method]
        half = (prediction['upper'] - prediction['lower']) / 2
        low = prediction['lower'] - factor * half - offset
        high = prediction['upper'] + factor * half + offset
        covered.append(low <= row['actual'] <= high)
        width.append(high - low)
        score.append(high - low + 20 * max(low - row['actual'], row['actual'] - high, 0))
        error.append(abs(prediction['mean'] - row['actual']))
    known = [row for row in rows if row['caught'] is not None]
    probability = [row['probabilities'][method] for row in known]
    y = [float(row['caught']) for row in known]
    avg = lambda values: float(np.mean(values)) if len(values) else None
    return {'n': len(rows), 'observed_operating': len(operating), 'known_catch': len(known),
            'coverage': avg(covered), 'covered': int(sum(covered)), 'width_seconds': avg(width),
            'interval_score': avg(score), 'mae_seconds': avg(error),
            'brier': avg((np.asarray(probability) - y) ** 2),
            'signed_catch_error': avg(np.asarray(probability) - y)}


def summarize(outcomes, factors):
    summary = {}
    old_offsets = read(V5 / 'calibration_relative_variants_frozen_offsets.json')['offset_seconds']
    for day, output in outcomes.items():
        summary[day] = {}
        for menu in MENUS:
            rows = output[menu]['rows']
            group = {}
            for method in METHODS:
                group[method] = {'raw': metrics(rows, method),
                    'group_envelope': metrics(rows, method, factors[menu][method]),
                    'observable_groups': {name: metrics([r for r in rows if observable_group(r) == name],
                                                         method, factors[menu][method])
                                          for name in ('ordinary', 'late')},
                    'selected': metrics(select_menu(rows, method), method, factors[menu][method])}
                if method in old_offsets[menu]:
                    group[method]['v5_original_envelope'] = metrics(rows, method, offset=old_offsets[menu][method])
                group[method]['all_present_groups_meet_90'] = all(
                    value['coverage'] is None or value['coverage'] >= .9
                    for value in group[method]['observable_groups'].values())
            summary[day][menu] = group
    return summary


def freeze():
    protocol = read(HERE / 'calibration_protocol.json')
    frozen = read(V5 / 'calibration_frozen_fit.json')
    data = data_inputs()
    with open(V5 / 'calibration_frozen_model.pkl', 'rb') as handle:
        marginal = pickle.load(handle)
    pairs = training_pairs(data, protocol['training_dates'], frozen['evolving_fit'])
    model = ConditionalStudentModel(marginal, pairs)
    with open(HERE / 'calibration_frozen_model.pkl', 'wb') as handle:
        pickle.dump(model, handle)
    write(HERE / 'calibration_frozen_fit.json', {'protocol': protocol,
        'beta': model.beta, 'base_scale_seconds': model.base_scale, 'scale_slope': model.scale_slope,
        'training_counts': model.training_counts, 'optimizer': model.fit_diagnostics,
        'source_text': (ROOT / 'src/conditional_risk.py').read_text()})
    print('FIT FROZEN', model.beta, model.base_scale, model.scale_slope, flush=True)
    original = read(V5 / 'calibration_relative_variants_results.json.gz')
    outcomes = original['rows_by_date']
    add_candidate(data, outcomes, model)
    factors, details = {}, {}
    for menu in MENUS:
        factors[menu], details[menu] = {}, {}
        for method in METHODS:
            factors[menu][method], details[menu][method] = simultaneous_group_envelope(
                {day: outcomes[day][menu]['rows'] for day in protocol['calibration_dates']}, method)
    write(HERE / 'calibration_frozen_envelopes.json', {'factors': factors, 'details': details})
    # This receipt is completed before any new actual source is fetched/opened.
    write(HERE / 'calibration_freeze_receipt.json', {'protocol': protocol,
        'fit': read(HERE / 'calibration_frozen_fit.json'), 'envelopes': factors,
        'new_actuals_opened': False, 'frozen_sources': {
            'model': (ROOT / 'src/conditional_risk.py').read_text(),
            'evaluator': Path(__file__).read_text()}})
    development = {day: outcomes[day] for day in protocol['development_dates']}
    result = {'protocol': protocol, 'factors': factors, 'development': summarize(development, factors),
              'rows_by_date': development}
    write(HERE / 'calibration_development_results.json.gz', result)
    write(HERE / 'calibration_development_summary.json', {k: v for k, v in result.items() if k != 'rows_by_date'})
    print('ENVELOPES FROZEN', factors, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
