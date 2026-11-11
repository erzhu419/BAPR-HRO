"""Frozen shrinkage and heavy-tail conditional alternatives to coefficient one."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_relative_run as relative_run
from calibration_run import data_inputs, law_summary, load, write
from src.calibration_relative import ShrunkRelativeDelayModel
from src.calibration_day_effect import HeavyTailDayModel

NEW_METHODS = ('shrunk_relative', 'heavy_tail_day')


def numerical_reference(marginal, pairs, coarse):
    real_pairs = [(a, b) for rows in pairs.values() for a, b in rows
                  if a['arrival']['status'] == b['departure']['status'] == 'REAL']
    ordered = sorted(real_pairs, key=lambda pair: pair[0]['arrival']['time_seconds'] - marginal.prior(pair[0], 'arrival').mean)
    fine = HeavyTailDayModel(marginal, pairs, delta_step=coarse.delta_step / 2)
    records = []
    for index in (0, len(ordered) // 2, len(ordered) - 1):
        incoming, outgoing = ordered[index]
        now = incoming['arrival']['time_seconds']
        a, b = coarse.predict(incoming, outgoing, now), fine.predict(incoming, outgoing, now)
        points = np.linspace(a.quantile(.01), a.quantile(.99), 11)
        records.append({'training_incoming': incoming['id'], 'training_outgoing': outgoing['id'],
                        'observed_incoming_residual': now - marginal.prior(incoming, 'arrival').mean,
                        'lower_difference_seconds': a.quantile(.05) - b.quantile(.05),
                        'upper_difference_seconds': a.quantile(.95) - b.quantile(.95),
                        'max_cdf_difference': max(abs(a.cdf(t) - b.cdf(t)) for t in points)})
    return {'coarse_step_seconds': coarse.delta_step, 'fine_step_seconds': fine.delta_step, 'training_states': records}


def add_predictions(data, outcomes, models):
    for day, day_output in outcomes.items():
        source = {c['id']: c for c in data['dates'][day]['connections']}
        shared = {}
        for menu in ('fixed_schedule', 'actual_current'):
            for row in day_output[menu]['rows']:
                key = row['incoming'], row['outgoing']
                if key not in shared:
                    incoming, outgoing = source[row['incoming']], source[row['outgoing']]
                    values, probabilities = {}, {}
                    for name, model in models.items():
                        law = model.predict(incoming, outgoing, row['clock'])
                        values[name] = law_summary(law, row['actual'])
                        probabilities[name] = law.board_probability(row['ready'])
                    shared[key] = values, probabilities
                values, probabilities = shared[key]
                row['methods'].update(values)
                row['probabilities'].update(probabilities)
        print('VARIANTS', day, len(shared), flush=True)


def main():
    frozen = json.loads((HERE / 'calibration_frozen_fit.json').read_text())
    fit, original = frozen['evolving_fit'], frozen['protocol']
    data = data_inputs()
    pairs = relative_run.training_pairs(data, original['parameter_training_dates'], fit)
    with (HERE / 'calibration_frozen_model.pkl').open('rb') as handle:
        marginal = pickle.load(handle)
    models = {'shrunk_relative': ShrunkRelativeDelayModel(marginal, pairs),
              'heavy_tail_day': HeavyTailDayModel(marginal, pairs, delta_step=7.5)}
    convergence = numerical_reference(marginal, pairs, models['heavy_tail_day'])
    write(HERE / 'calibration_relative_numerics.json', convergence)
    for row in convergence['training_states']:
        if max(abs(row['lower_difference_seconds']), abs(row['upper_difference_seconds'])) > 1 or row['max_cdf_difference'] > .001:
            raise ValueError('Training-state integration reference exceeds the predeclared numerical tolerance')
    models['heavy_tail_day'].clear_caches()
    models['shrunk_relative']._cache.clear()
    models['shrunk_relative']._quantiles.clear()
    protocol = {'training_dates': original['parameter_training_dates'], 'calibration_dates': original['calibration_dates'],
                'development_dates': original['heldout_dates'], 'future_confirmation_dates': ['2023-10-16', '2023-10-20', '2023-10-31'],
                'new_methods': NEW_METHODS, 'shrunk_coefficient': models['shrunk_relative'].coefficient,
                'cauchy_scale_seconds': models['heavy_tail_day'].scale, 'delta_step_seconds': 7.5,
                'density_interpolation_seconds': 1., 'kernel_seconds': marginal.sigma, 'route_pseudocount': 20,
                'definition': 'Shrinkage uses zero-intercept constrained training-pair regression of centered outgoing '
                'on incoming delay and the corresponding empirical residual law. Heavy-tail uses independent '
                'route-centered incoming/outgoing role-noise KDEs plus a common zero-location Cauchy day effect; '
                'the incoming residual updates the day-effect posterior. Cauchy scale is the unscaled MAD of '
                'training-pair incoming residuals about their training median. No development value selects a parameter.',
                'numerics': 'Base delta support [-300,3840] seconds, extended using the observed incoming residual and '
                'the fixed training-noise extrema plus nine kernel standard deviations. Exact anchor is included. '
                'Prior cell masses are Cauchy CDF differences. Likelihood and outgoing CDF use one-second tables. '
                'Outside-support unnormalized likelihood mass is bounded using the Gaussian nine-sigma density. '
                'Grid evidence is numerical, so its ratio is a diagnostic rather than a certified posterior bound.',
                'local_joint_assumption': 'Given delta and route roles, centered incoming/outgoing noises are '
                'independent with the earlier KDE laws. The predictor conditions only on the rider-known incoming '
                'arrival. Arbitrary route-specific day loadings, serial day drift, and selection on unobserved '
                'outcomes are outside this local joint law. Canceled service remains a separate earlier route marginal.'}
    write(HERE / 'calibration_relative_variants_frozen_fit.json', protocol)
    with (HERE / 'calibration_relative_variants_frozen.pkl').open('wb') as handle:
        pickle.dump(models, handle)
    print('VARIANT FITS FROZEN', protocol['shrunk_coefficient'], protocol['cauchy_scale_seconds'], flush=True)
    # Reuse the exact stored public states and menus of the coefficient-one run.
    original_results = load(HERE / 'calibration_relative_results.json.gz')
    outcomes = original_results['rows_by_date']
    add_predictions(data, outcomes, models)
    methods = (*relative_run.METHODS, *NEW_METHODS)
    offsets, scores = relative_run.offsets_by_menu(outcomes, original['calibration_dates'], .1, methods)
    write(HERE / 'calibration_relative_variants_frozen_offsets.json', {'offset_seconds': offsets, 'date_scores': scores})
    summaries = {day: relative_run.summarize(outcomes[day], offsets, methods) for day in original['heldout_dates']}
    diagnostics = list(models['heavy_tail_day'].diagnostics.values())
    result = {'protocol': {**original_results['protocol'], **protocol}, 'offset_seconds': offsets,
              'date_scores': scores, 'development': summaries, 'rows_by_date': outcomes,
              'integration_diagnostics': diagnostics}
    with gzip.open(HERE / 'calibration_relative_variants_results.json.gz', 'wt') as handle:
        json.dump(result, handle, separators=(',', ':'), allow_nan=False)
    write(HERE / 'calibration_relative_variants_summary.json', {key: result[key] for key in ('protocol', 'offset_seconds', 'development')})
    for day, value in summaries.items():
        for menu in ('fixed_schedule', 'actual_current'):
            print('RESULT', day, menu, {name: {'coverage': round(row['date_calibrated']['interval']['coverage'], 4),
                'width': round(row['date_calibrated']['interval']['mean_width_seconds'], 1),
                'score': round(row['date_calibrated']['interval']['mean_interval_score'], 1),
                'brier': round(row['date_calibrated']['probability']['brier'], 5)}
                for name, row in value[menu]['methods'].items()}, flush=True)


if __name__ == '__main__':
    main()
