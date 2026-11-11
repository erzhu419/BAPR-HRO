"""Freeze one past-only grid-duration candidate; no held-out fitting."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import fit_training
from src.ordered_transition_fit import continuous_mean, duration_probabilities, fit_ordered_duration

SOURCE = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
OUTPUT = Path(__file__).with_name('ordered_duration_training_fit.json')


def training_durations(data):
    groups = defaultdict(list)
    for day in data['train_dates']:
        trips = defaultdict(list)
        for c in data['dates'][day]['connections']:
            if not c['dropoff_allowed']:
                continue
            trips[c['trip_id']].append(c)
            if c['outcome'] == 'MEASURED':
                groups['travel'].append((c['scheduled_arrival_seconds'] - c['scheduled_departure_seconds'],
                                         c['arrival']['time_seconds'] - c['departure']['time_seconds']))
        for rows in trips.values():
            rows.sort(key=lambda c: c['dep_sequence'])
            for before, after in zip(rows, rows[1:]):
                if before['arrival']['status'] != 'REAL' or after['departure']['status'] != 'REAL':
                    continue
                kind = ('gap' if before['arr_stop'] != after['dep_stop'] else
                        'dwell' if before['arr_sequence'] == after['dep_sequence'] else None)
                if kind:
                    groups[kind].append((after['scheduled_departure_seconds'] - before['scheduled_arrival_seconds'],
                                         after['departure']['time_seconds'] - before['arrival']['time_seconds']))
    return {kind: np.asarray(rows, dtype=float) for kind, rows in groups.items()}


def main():
    data, _ = load_city(SOURCE)
    baseline = fit_training(data, retain_no_pickup=True)
    candidate, continuous_candidate, fits, bins = deepcopy(baseline), deepcopy(baseline), {}, []
    for kind, rows in training_durations(data).items():
        scheduled, elapsed = rows.T
        continuous = fit_ordered_duration(scheduled, elapsed, objective='continuous')
        grid = fit_ordered_duration(scheduled, elapsed, objective='grid')
        fits[kind] = {'continuous': continuous, 'grid': grid,
                      'negative_observed_duration_count': int(np.sum(elapsed < 0))}
        candidate['increments'][kind + '_drift'] = grid['location_seconds'] / 60
        candidate['increments'][kind + '_sd'] = grid['scale_seconds'] / 60
        continuous_candidate['increments'][kind + '_drift'] = continuous['location_seconds'] / 60
        continuous_candidate['increments'][kind + '_sd'] = continuous['scale_seconds'] / 60
        for duration in sorted(set(scheduled)):
            values = elapsed[scheduled == duration]
            cmu, csd = continuous['location_seconds'], continuous['scale_seconds']
            gmu, gsd = grid['location_seconds'], grid['scale_seconds']
            clocks, cp = duration_probabilities(duration, cmu, csd)
            grid_clocks, gp = duration_probabilities(duration, gmu, gsd)
            bins.append({'kind': kind, 'scheduled_seconds': duration, 'n': len(values),
                         'actual_mean': float(np.mean(values)), 'actual_sd': float(np.std(values)),
                         'actual_zero_count': int(np.sum(values == 0)),
                         'continuous_fit_continuous_mean': float(continuous_mean(duration, cmu, csd)),
                         'continuous_fit_grid_mean': float(clocks @ cp),
                         'grid_fit_grid_mean': float(grid_clocks @ gp),
                         'grid_fit_grid_sd': float(np.sqrt(gp @ (grid_clocks - grid_clocks @ gp) ** 2))})
        means = [r for r in bins if r['kind'] == kind]
        fits[kind]['grid_fitted_mean_residual_score_seconds'] = float(
            sum(r['n'] * (r['grid_fit_grid_mean'] - r['actual_mean']) for r in means) / len(elapsed))
    candidate['increment_source'] = ('July25 only, expected categorical log likelihood after mean-preserving '
                                     'randomized coarsening onto nonnegative 60-second duration grid.')
    continuous_candidate['increment_source'] = ('July25 only continuous nonnegative truncated-normal likelihood. '
        'Deployed only with the experiment CDF-cell duration kernel and fixed 20-second central grid.')
    zero_dwell = next(r for r in bins if r['kind'] == 'dwell' and r['scheduled_seconds'] == 0)
    m = zero_dwell['actual_mean']
    result = {'input': str(SOURCE.relative_to(ROOT)), 'train_dates': data['train_dates'],
              'candidate_selection': 'One grid fit chosen by declared training objective; no held-out loss used.',
              'continuous_zero_atom': '1498 exact-zero dwells exist. All three continuous optimizations converge '
                                      'to the same finite interior fit; the continuous density is a misspecified '
                                      'reference, not a zero-atom generative model.',
              'grid_observation_likelihood': 'For y between a=60*floor(y/60) and b=a+60, maximize '
                  '(1-f)*log p(a|d)+f*log p(b|d), f=(y-a)/60. p(k|d) is the row-normalized Gaussian '
                  'potential on k=0,60,120,... . This is expected randomized-coarsening log likelihood.',
              'integration': 'Upper integration limit includes observed maximum and extends 12 sigma beyond '
                             'the Gaussian location. Only location and scale are fitted; no held-out tuning.',
              'deployment_limit': 'Not exact adaptive-grid HMM maximum likelihood: finite delay boundaries, '
                                  'known absolute anchors and full chain conditioning change transition rows. '
                                  'Cold 60-second lattice cannot match the observed zero-dwell mean and variance simultaneously.',
              'zero_dwell_grid_minimum_sd_at_observed_mean': float(np.sqrt(60 * m - m * m)),
              'zero_dwell_actual_sd': zero_dwell['actual_sd'],
              'fine_grid_recipe': 'Largest divisor of 60 no greater than observed zero-dwell '
                                  'mean+variance/mean: 20 seconds; central delay [-300,300] seconds, '
                                  'original minute tails and exact observation anchors retained. '
                                  'This necessary variance condition is not a guarantee of model fit.',
              'baseline_training': baseline, 'candidate_training': candidate,
              'continuous_training': continuous_candidate, 'fits': fits, 'duration_bins': bins}
    if OUTPUT.exists():
        if json.loads(OUTPUT.read_text()) != result:
            raise ValueError('Frozen candidate differs: choose a new versioned output')
    else:
        OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'increments': candidate['increments'],
                      'major_bins': [r for r in bins if r['n'] >= 1000]}, indent=2))


if __name__ == '__main__':
    main()
