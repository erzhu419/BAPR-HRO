"""Summarize the completed bounded discovery; no model rerun or reselection."""
import collections
import hashlib
import json
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
x=json.loads((HERE/'broad_probe.json').read_text())
rows=x['rows'];remaining=x['screened'];funnel={'sampled_new_od':len(remaining)}
for key in ('nonempty','K_2_to_6','two_first_routes','two_transfer_options','no_direct'):
    remaining=[r for r in remaining if r['checks'][key]]
    funnel[key]=len(remaining)
known=[r for r in rows if all(r['actual'][m]['loss'] is not None
    for m in ('first','one','single','combined','pair','dp'))]
def signature(outcome):
    return outcome['action'],outcome['tail'],outcome['loss'],outcome['status'],tuple(
        (r['clock'],r['visible'],r['seen'],r['choice']) for r in outcome['trace'])
summary=dict(scope=x['scope'],protocol=x['protocol'],network=x['network'],
    status=x['status'],runtime_seconds=x['runtime_seconds'],
    selection_seconds=x['selection_seconds'],sequential_funnel=funnel,
    selected=len(x['selected']),completed=len(rows),
    initial_state_model_gaps={m:dict(mean=statistics.mean(r['root_gaps'][m] for r in rows),
        maximum=max(r['root_gaps'][m] for r in rows),
        strictly_positive_above_1e_6=sum(r['root_gaps'][m]>1e-6 for r in rows))
        for m in ('first','one','single','combined','pair')},
    actual=dict(complete_pairs=len(known),missing=len(rows)-len(known),
        statuses=dict(collections.Counter(r['actual']['dp']['status'] for r in rows)),
        mean_loss={m:statistics.mean(r['actual'][m]['loss'] for r in known)
            for m in ('first','one','single','combined','pair','dp')},
        identical_trace_and_outcome_to_dp={m:sum(signature(r['actual'][m])==signature(r['actual']['dp'])
            for r in known) for m in ('one','single','combined','pair')},
        first_to_combined_gain=[dict(root_id=r['root_id'],gain=r['actual']['first']['loss']-
            r['actual']['combined']['loss']) for r in known]),
    checks=dict(max_dp_integration_error=max(r['exact']['dp_integration_error'] for r in rows),
        max_first_threshold_error=max(r['exact']['checked_first_threshold_error'] for r in rows)),
    source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for name in ('broad_probe.py','broad_probe_selection.json','broad_probe.json')},
    interpretation='No extra exact-DP initial-state value over combined in these ten new schedule-selected roots. Six complete actual pairs agree; four unresolved labels are retained. This does not establish universal optimality, independent confirmation, or city-wide prevalence.')
(HERE/'broad_probe_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
