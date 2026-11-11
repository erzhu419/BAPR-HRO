"""Calibrate selection-aware service/clock regions with censored score bounds."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import pickle
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v6.calibration_run import read, write, add_candidate, metrics, V5, METHODS, MENUS
from src.conditional_risk import observable_group

MODES = ('fixed_selector', 'menu_max')


def states(rows):
    result = defaultdict(list)
    for row in rows:
        result[row['incoming']].append(row)
    return list(result.values())


def select(menu, method):
    return min(menu, key=lambda row: (-row['probabilities'][method], row['scheduled'], row['outgoing']))


def score_bounds(row, method):
    # The prediction region includes CANCELED as an explicit service category.
    # This gives clock/service coverage; it does not identify cancel probability.
    if row['status'] == 'CANCELED':
        return -math.inf, -math.inf
    if row['actual'] is None:
        return -math.inf, math.inf
    estimate = row['methods'][method]
    half = (estimate['upper'] - estimate['lower']) / 2
    value = max(estimate['lower'] - row['actual'], row['actual'] - estimate['upper']) / half
    return value, value


def state_score(menu, method, mode):
    considered = [select(menu, method)] if mode == 'fixed_selector' else menu
    scores = [score_bounds(row, method) for row in considered]
    return max(value[0] for value in scores), max(value[1] for value in scores)


def thresholds(outcomes, dates):
    result = {}
    for name in MENUS:
        result[name] = {}
        for method in METHODS:
            result[name][method] = {}
            for mode in MODES:
                daily = {}
                for day in dates:
                    records = states(outcomes[day][name]['rows'])
                    lower, upper, groups = [], [], {}
                    for group in ('ordinary', 'late'):
                        selected = [menu for menu in records if observable_group(menu[0]) == group]
                        values = [state_score(menu, method, mode) for menu in selected]
                        if values:
                            index = math.ceil(.9 * len(values)) - 1
                            lo = sorted(v[0] for v in values)[index]
                            hi = sorted(v[1] for v in values)[index]
                            lower.append(lo); upper.append(hi)
                            groups[group] = {'n_incoming_states': len(values),
                                'score_lower': lo if math.isfinite(lo) else None,
                                'score_upper': hi if math.isfinite(hi) else None,
                                'upper_unbounded': hi == math.inf}
                        else:
                            groups[group] = {'n_incoming_states': 0}
                    daily[day] = {'lower': max(lower), 'upper': max(upper), 'groups': groups}
                low = max(0., max(value['lower'] for value in daily.values()))
                high = max(0., max(value['upper'] for value in daily.values()))
                result[name][method][mode] = {'factor_lower': low if math.isfinite(low) else None,
                    'factor_upper': high if math.isfinite(high) else None, 'vacuous': high == math.inf,
                    'date_scores': {day: {'lower': v['lower'] if math.isfinite(v['lower']) else None,
                                         'upper': v['upper'] if math.isfinite(v['upper']) else None,
                                         'upper_unbounded': v['upper'] == math.inf,
                                         'groups': v['groups']} for day, v in daily.items()}}
    return result


def evaluate_menu(rows, method, mode, factor):
    records = states(rows)
    lower, upper, widths = [], [], []
    observed, unknown, canceled = 0, 0, 0
    for menu in records:
        lo, hi = state_score(menu, method, mode)
        lower.append(hi <= factor)
        upper.append(lo <= factor)
        considered = [select(menu, method)] if mode == 'fixed_selector' else menu
        observed += sum(row['actual'] is not None for row in considered)
        unknown += sum(row['status'] != 'CANCELED' and row['actual'] is None for row in considered)
        canceled += sum(row['status'] == 'CANCELED' for row in considered)
        widths.append(np.mean([(row['methods'][method]['upper'] - row['methods'][method]['lower'])
                              * (1 + factor) for row in considered]))
    selected_rows = [select(menu, method) for menu in records]
    return {'n_incoming_states': len(records), 'observed_operating_targets': observed,
            'unresolved_targets': unknown, 'canceled_targets': canceled,
            'coverage_lower': float(np.mean(lower)) if records else None,
            'coverage_upper': float(np.mean(upper)) if records else None,
            'state_mean_clock_width_seconds': float(np.mean(widths)) if math.isfinite(factor) and widths else None,
            'vacuous': not math.isfinite(factor),
            'selected_clock_metrics': metrics(selected_rows, method, factor)
                                      if math.isfinite(factor) else None}


def summary(outcomes, bounds):
    result = {}
    for day, output in outcomes.items():
        result[day] = {}
        for name in MENUS:
            result[day][name] = {}
            for method in METHODS:
                result[day][name][method] = {}
                for mode in MODES:
                    bound = bounds[name][method][mode]
                    factor = math.inf if bound['vacuous'] else bound['factor_upper']
                    rows = output[name]['rows']
                    values = evaluate_menu(rows, method, mode, factor)
                    values['observable_groups'] = {group: evaluate_menu(
                        [row for row in rows if observable_group(row) == group], method, mode, factor)
                        for group in ('ordinary', 'late')}
                    values['all_present_groups_lower_at_least90'] = all(
                        g['coverage_lower'] is None or g['coverage_lower'] >= .9
                        for g in values['observable_groups'].values())
                    result[day][name][method][mode] = values
    return result


def freeze():
    sys.path.insert(0, str(V5))
    from calibration_run import data_inputs
    protocol = read(HERE / 'calibration_protocol.json')
    with open(HERE / 'calibration_frozen_model.pkl', 'rb') as handle:
        model = pickle.load(handle)
    original = read(V5 / 'calibration_relative_variants_results.json.gz')['rows_by_date']
    add_candidate(data_inputs(), original, model)
    bound = thresholds(original, protocol['calibration_dates'])
    selection_protocol = {'training_dates': protocol['training_dates'],
        'calibration_dates': protocol['calibration_dates'],
        'confirmation_dates': ['2023-10-14', '2023-10-17', '2023-10-28'],
        'actuals_status_before_freeze': 'No actual-label source/score in V5 or first V6 calibration cohort; only old RT archives/date lists existed.',
        'prediction_methods': METHODS, 'groups': protocol['observable_groups'],
        'modes': {'fixed_selector': 'Max raw frozen catch probability; tie scheduled time then outgoing ID. Only this selector protected.',
                  'menu_max': 'Maximum nonconformity across all outgoing candidates for each incoming state. Protects any first selector within the frozen menu.'},
        'service_region': 'Union of CANCELED category and operating clock interval. No cancellation probability calibration is claimed.',
        'missingness': 'Unknown clock/service gives score interval[-infinity,+infinity]; calibration factor has identified lower/upper. Use upper for finite coverage; retain vacuous if infinite.',
        'rank_target': 'Each group/date contributes ceil(.9*n_incoming)-th state score; date is maximum over present groups; largest of9 calibration dates. Under exchangeable date blocks, >=90% probability that every present group has >=90% incoming states covered next date.',
        'scope': 'Counterfactual retained input states, not the adaptively visited journey-state distribution. Both menus scored separately. Input-only selector never reads outgoing truth.',
        'no_further_selection': 'Student/shrunk parameters unchanged; only this predetermined selection score is calibrated. No models, groups, dates or factors are chosen using the new three date labels.'}
    write(HERE / 'calibration_selection_protocol.json', selection_protocol)
    write(HERE / 'calibration_selection_frozen.json', {'protocol': selection_protocol,
        'thresholds': bound, 'new_actuals_opened': False,
        'source_text': Path(__file__).read_text(), 'point_model_fit': read(HERE / 'calibration_frozen_fit.json')})
    previous = {day: original[day] for day in protocol['development_dates']}
    previous.update(read(HERE / 'calibration_confirmation_results.json.gz')['rows_by_date'])
    write(HERE / 'calibration_selection_development_summary.json', summary(previous, bound))
    print('SELECTION FROZEN', {name: {method: {mode: value['factor_upper'] for mode, value in modes.items()}
        for method, modes in methods.items()} for name, methods in bound.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
