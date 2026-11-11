"""Read out continuous and common finite-grid conditional forecasts separately."""
from collections import defaultdict
from copy import deepcopy
import pickle
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V5 = ROOT / 'experiments/reconstruction_v5'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V5))
from experiments.reconstruction_v6.calibration_run import read, write, metrics, MENUS
from calibration_run import data_inputs
from calibration_relative_run import build_day
from src.trip_marginal_tilt import TripMarginalTilt


def main():
    with open(HERE / 'calibration_frozen_model.pkl', 'rb') as handle:
        model = pickle.load(handle)
    fit = read(V5 / 'calibration_frozen_fit.json')['evolving_fit']
    factor = read(HERE / 'calibration_frozen_envelopes.json')['factors']
    data = data_inputs()
    data['dates'].update(read(HERE / 'calibration_confirmation_actuals.json.gz')['dates'])
    outcomes = read(HERE / 'calibration_development_results.json.gz')['rows_by_date']
    outcomes.update(read(HERE / 'calibration_confirmation_results.json.gz')['rows_by_date'])
    reports = {}
    for day, output in outcomes.items():
        evaluator, _ = build_day(data, day, fit, [])
        graph = evaluator.graph
        by_source = {row['id']: row for row in data['dates'][day]['connections']}
        grouped = defaultdict(dict)
        for menu in MENUS:
            for row in output[menu]['rows']:
                grouped[row['incoming']][row['outgoing']] = row
        differences, support_losses, duplicate_trips, records = [], [], 0, {}
        for incoming_id, candidates in grouped.items():
            before = by_source[incoming_id]
            clock = next(iter(candidates.values()))['clock']
            base = evaluator.base.copy()
            local_incoming = evaluator.mapping[incoming_id]
            base.observe_connection_event(local_incoming, 'arrival', clock - graph.connections[local_incoming].arr_time)
            anchors = {}
            for outgoing in sorted(candidates, key=lambda cid: (by_source[cid]['scheduled_departure_seconds'], cid)):
                local = evaluator.mapping[outgoing]
                trip = graph.connections[local].trip_id
                if trip in anchors:
                    duplicate_trips += 1
                else:
                    anchors[trip] = outgoing
            targets = {evaluator.mapping[cid]: model.predict(before, by_source[cid], clock) for cid in anchors.values()}
            common = TripMarginalTilt(base, evaluator.priors, targets)
            for outgoing, row in candidates.items():
                local = evaluator.mapping[outgoing]
                clocks, weights, cancel = common.event_distribution(local, 'departure', evaluator.priors)
                cumulative = weights.cumsum()
                quantile = lambda p: float(clocks[min(np.searchsorted(cumulative, p), len(clocks) - 1)])
                estimate = {'lower': quantile(.05), 'upper': quantile(.95), 'mean': float(clocks @ weights),
                            'cancel_probability': cancel,
                            'pit': float(weights[clocks <= row['actual']].sum()) if row['actual'] is not None else None}
                caught = (clocks >= row['ready']) & (clocks <= row['ready'] + 720)
                probability = (1 - cancel) * float(weights[caught].sum())
                continuous = row['probabilities']['student']
                differences.append(probability - continuous)
                records[incoming_id, outgoing] = estimate, probability
        for menu in MENUS:
            for row in output[menu]['rows']:
                estimate, probability = records[row['incoming'], row['outgoing']]
                row['methods']['student_common_projection'] = estimate
                row['probabilities']['student_common_projection'] = probability
        reports[day] = {'n_pair_contexts': len(records), 'n_incoming_contexts': len(grouped),
            'nonanchor_same_trip_candidates': duplicate_trips,
            'mean_abs_catch_difference': float(np.mean(np.abs(differences))),
            'maximum_abs_catch_difference': float(np.max(np.abs(differences))),
            'mean_signed_catch_difference': float(np.mean(differences)),
            'menus': {menu: {'continuous_raw': metrics(output[menu]['rows'], 'student'),
                'projected_raw': metrics(output[menu]['rows'], 'student_common_projection'),
                'continuous_group_envelope': metrics(output[menu]['rows'], 'student', factor[menu]['student']),
                'projected_reused_factor_diagnostic': metrics(output[menu]['rows'], 'student_common_projection', factor[menu]['student'])}
                for menu in MENUS}}
        print('PROJECTED', day, reports[day]['mean_abs_catch_difference'], reports[day]['maximum_abs_catch_difference'], flush=True)
    result = {'scope': 'Posthoc representation diagnostic on dates already exposed in the earlier risk readout; '
        'not a new independent confirmation of the projected model. One common posterior per incoming context, '
        'union of two declared menus, earliest scheduled anchor per trip. All root actions share that posterior. '
        'Cancellation and previous facts retain the original model. Reused continuous envelope factor on discrete '
        'quantiles is an empirical diagnostic, not a transferred coverage theorem.',
        'reports': reports, 'rows_by_date': outcomes,
        'source_text': {name: (ROOT / name).read_text() for name in ('src/conditional_risk.py', 'src/trip_marginal_tilt.py',
                                                                 'experiments/reconstruction_v6/calibration_projection.py')}}
    write(HERE / 'calibration_projection_results.json.gz', result)
    write(HERE / 'calibration_projection_summary.json', {k: v for k, v in result.items() if k not in ('rows_by_date', 'source_text')})


if __name__ == '__main__':
    main()
