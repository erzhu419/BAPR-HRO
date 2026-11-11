"""Pair every completed model contrast on the same six development journeys."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def paired(rows, action, reference):
    indexed = {}
    for row in rows:
        indexed.setdefault((row['date'], row['query']), {})[row['method']] = row
    result = []
    for key, values in sorted(indexed.items()):
        if action in values and reference in values:
            a, b = values[action], values[reference]
            assert a['status'] == b['status'] == 'arrived'
            result.append({'date': key[0], 'query': key[1], 'action_loss': a['loss_seconds'],
                           'reference_loss': b['loss_seconds'], 'difference_seconds': a['loss_seconds'] - b['loss_seconds']})
    assert len(result) == 6
    return {'n_common': 6, 'mean_difference_seconds': sum(row['difference_seconds'] for row in result) / 6,
            'cases': result}


def main():
    main = json.loads((HERE / 'joint_day_v2_results.json').read_text())
    control = json.loads((HERE / 'joint_day_trip_control_results.json').read_text())
    charged = json.loads((HERE / 'joint_day_coupling_charged_results.json').read_text())
    assert len(main['rows']) == 24 and len(control['rows']) == 12 and len(charged['rows']) == 18
    logical = main['rows'] + control['rows']
    result = {'scope': 'Frozen original three development dates, two ODs each. '
              'Differences are realized replay policy contrasts, with dependent pairs and no independent-confirmation claim.',
              'logic': {f'{a}_minus_{b}': paired(logical, a, b) for a, b in (
                  ('joint_rollout', 'trip_mixture_rollout'), ('joint_rollout', 'independent_rollout'),
                  ('joint_nominal', 'trip_mixture_nominal'), ('joint_nominal', 'independent_nominal'))},
              'charged': {f'{a}_minus_{b}': paired(charged['rows'], a, b) for a, b in (
                  ('joint_rollout', 'trip_mixture_rollout'), ('joint_rollout', 'independent_rollout'))},
              'logic_summary': main['summary'] | control['summary'], 'charged_summary': charged['summary']}
    result['logical_coupling_by_valuation_interaction_seconds'] = (
        result['logic']['joint_rollout_minus_trip_mixture_rollout']['mean_difference_seconds'] -
        result['logic']['joint_nominal_minus_trip_mixture_nominal']['mean_difference_seconds'])
    (HERE / 'joint_day_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for timing in ('logic', 'charged'):
        print(timing, {name: value['mean_difference_seconds'] for name, value in result[timing].items()})


if __name__ == '__main__':
    main()
