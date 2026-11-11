"""Compare fixed-policy estimates to their analytic full-model values."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    source = json.loads((HERE / 'itinerary_mc_results.json').read_text())
    groups = []
    for ci, case in enumerate(source['contexts']):
        truth = {item['label']: item['value'] for item in case['analytic']}
        for n in (4, 8):
            for method in ('iid', 'stratified'):
                rows = [r for r in source['rows'] if (r['context'], r['n'], r['method']) == (ci, n, method)]
                if len(rows) != 8:
                    raise ValueError('The frozen eight-seed comparison is incomplete')
                values = {}
                for label, exact in truth.items():
                    estimates = np.array([next(e['estimate'] for e in r['estimates'] if e['label'] == label) for r in rows])
                    values[label] = {'analytic': exact, 'mc_mean': float(estimates.mean()),
                        'bias_readout': float(estimates.mean() - exact),
                        'variance': float(estimates.var(ddof=1)),
                        'rmse': float(np.sqrt(np.mean((estimates - exact) ** 2)))}
                groups.append({'context': ci, 'date': case['date'], 'number': case['number'],
                    'n': n, 'method': method, 'values': values,
                    'ranking_errors': sum(r['ranking_error'] for r in rows),
                    'mean_family_regret': float(np.mean([r['family_regret'] for r in rows])),
                    'maximum_family_regret': max(r['family_regret'] for r in rows),
                    'mean_fresh_wall_seconds': float(np.mean([r['wall_seconds'] for r in rows])),
                    'mean_fresh_cpu_seconds': float(np.mean([r['cpu_seconds'] for r in rows])),
                    'max_pair_transitions': max(sum(e['transitions'] for e in r['estimates']) for r in rows)})
    result = {'source': 'itinerary_mc_results.json', 'timing': 'itinerary_mc_timing_results.json',
        'scope': 'Three archived public contexts, two fixed-policy witnesses each, eight MC seeds and4/8 samples. '
                 'The two initial contexts have identical posterior inputs; model-family regret is not full-city or physical-day regret.',
        'n_trials': len(source['rows']), 'n_policy_values': 2 * len(source['rows']),
        'groups': groups,
        'contexts': [{key: value for key, value in case.items() if key in (
            'date', 'number', 'policies', 'analytic', 'common_context_path_construction_seconds',
            'posthoc_physical_first_attempt')} for case in source['contexts']]}
    (HERE / 'itinerary_mc_summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
