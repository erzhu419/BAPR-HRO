"""Summarize duration prediction and buffer dependence, without refitting."""
import gzip
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
result = {}
for cohort in ('development', 'confirmation', 'completion'):
    with gzip.open(HERE / f'decision_risk_{cohort}_results.json.gz', 'rt') as handle:
        dates = json.load(handle)['dates']
    result[cohort] = {}
    for day, value in dates.items():
        groups = {}
        for group in ('ordinary', 'late'):
            rows = [r for r in value['rows'] if r['selected'] and r['group'] == group]
            if not rows:
                groups[group] = {'n_selected': 0}
                continue
            n = len(rows)
            success = [r for r in rows if r['outcome'] == 'succeeded']
            unknown = [r for r in rows if r['outcome'] == 'unresolved']
            durations = [r['observed_goal_arrival'] - r['time']
                         if r['outcome'] == 'succeeded' else 3600.
                         if r['outcome'].startswith('failed') else 0. for r in rows]
            groups[group] = {
                'n_selected': n,
                'model_mean_macro_seconds': sum(r['model_upper'] for r in rows) / n,
                'observed_mean_macro_lower_seconds': sum(durations) / n,
                'observed_mean_macro_upper_seconds': (sum(durations) + 3600 * len(unknown)) / n,
                'model_mean_failure_probability': sum(max(0., 1-r['model_success_probability']) for r in rows) / n,
                'observed_failure_probability_lower': (n-len(success)-len(unknown)) / n,
                'observed_failure_probability_upper': (n-len(success)) / n,
                'mean_model_R_seconds': sum(r['model_regret_bound'] for r in rows) / n,
                'mean_R_lower_seconds': sum(r['risk_lower'] for r in rows) / n,
                'mean_R_upper_seconds': sum(r['risk_upper'] for r in rows) / n,
                'R_pointwise_above_one_second_count': sum(r['risk_lower'] > 1 for r in rows),
                'zero_buffer_mean_R_lower_seconds': sum(r['zero_buffer_risk_lower'] for r in rows) / n,
                'zero_buffer_mean_R_upper_seconds': sum(r['zero_buffer_risk_upper'] for r in rows) / n,
                'transfer_buffer_seconds': sorted({r['transfer_seconds'] for r in rows}),
                'target_time_censor_count': len(unknown),
            }
        result[cohort][day] = groups
sizes = {path.name: path.stat().st_size for path in (HERE / 'decision_risk_sources').glob('*')}
result['local_source_bytes'] = {'files': sizes, 'total': sum(sizes.values())}
(HERE / 'decision_risk_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ('confirmation', 'local_source_bytes')}, indent=2))
