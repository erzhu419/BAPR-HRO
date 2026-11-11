"""Posthoc service-risk refusal audit; no new fit or independent confirmation."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.reconstruction_v5.calibration_run import data_inputs
from experiments.reconstruction_v6.decision_risk_run import evaluate_day
from src.decision_risk import realized_macro_risk, binomial_failure_upper, service_guard, required_zero_failure_blocks

data = data_inputs()
fit = json.loads((HERE / 'decision_risk_frozen.json').read_text())['protocol']['configuration']['training']
routes = defaultdict(lambda: {'by_day': {}, 'n_contexts': 0, 'failure_or_unresolved_contexts': 0})
for day in ('2023-10-01', '2023-10-02'):
    evaluated = evaluate_day(data, day, fit)
    source = {row['id']: row for row in data['dates'][day]['connections']}
    for row in evaluated['rows']:
        before, after = source[row['incoming']], source[row['outgoing']]
        dep, arr = after['departure'], after['arrival']
        actual_trip = before['arrival']['official_journey']
        canceled = any(e['status'] == 'CANCELED' and e['official_journey'] == actual_trip for e in (dep, arr))
        same = actual_trip == dep['official_journey'] == arr['official_journey']
        risk = realized_macro_risk(row['time'], row['time'] + row['transfer_seconds'], row['time'] + 3600,
            dep['time_seconds'] if dep['status'] == 'REAL' else None,
            arr['time_seconds'] if arr['status'] == 'REAL' else None,
            canceled=canceled, same_actual_trip=same)
        failed_or_unresolved = risk.outcome != 'succeeded'
        route = routes[after['route_id']]
        route['n_contexts'] += 1
        route['failure_or_unresolved_contexts'] += int(failed_or_unresolved)
        route['by_day'][day] = route['by_day'].get(day, False) or failed_or_unresolved
for route in routes.values():
    n, k = len(route['by_day']), sum(route['by_day'].values())
    route.update(n_independent_date_units=n, dates_with_any_failure_or_unresolved=k,
        primary_date_block_q_upper=binomial_failure_upper(k, n),
        context_iid_q_upper_sensitivity_only=binomial_failure_upper(
            route['failure_or_unresolved_contexts'], route['n_contexts']))

summaries = {}
for label in ('development', 'confirmation', 'completion'):
    with gzip.open(HERE / f'decision_risk_{label}_results.json.gz', 'rt') as handle:
        result = json.load(handle)['dates']
    summaries[label] = {}
    for day, value in result.items():
        if day not in data['dates']:
            with gzip.open(HERE / f'decision_risk_{label}_actuals.json.gz', 'rt') as handle:
                sources = json.load(handle)['dates'][day]['connections']
        else:
            sources = data['dates'][day]['connections']
        source = {row['id']: row for row in sources}
        groups = defaultdict(list)
        for row in value['rows']:
            if not row['selected']:
                continue
            route = source[row['outgoing']]['route_id']
            statistics = routes.get(route)
            q = statistics['primary_date_block_q_upper'] if statistics else 1.
            success = row['model_success_probability']
            cap = row['time'] + 3600 - row['gate']
            clock = max(0., (row['model_regret_bound'] - max(0., 1-success)*cap) / success) if success else cap
            guarded = service_guard(clock, q, cap)
            groups[row['group']].append(dict(route=route, q_upper=q, **guarded))
        summaries[label][day] = {group: {
            'n_model_selected': len(rows), 'n_authorized': sum(row['authorized'] for row in rows),
            'guard_upper_min_seconds': min(row['risk_upper'] for row in rows),
            'guard_upper_max_seconds': max(row['risk_upper'] for row in rows),
            'routes': dict(Counter(row['route'] for row in rows))}
            for group, rows in groups.items()}
result = dict(posthoc_created_at_utc=datetime.now(timezone.utc).isoformat(),
    assumptions='Primary Bernoulli units are iid same-source route/date blocks, Y=any failure/unresolved next suffix in the observable incoming cohort. This upper bound dominates a same-day chosen suffix marginally, but is not conditional on arbitrary H/gate invocation. Context-iid calculation is a sensitivity, not extra independent dates. A total-risk bound additionally requires the supplied conditional-success clock-tail bound to hold on the target cohort. No shift-day promise.',
    action='All1second real-risk authorizations are refused. q_upper is not a lower bound on actual risk. Existing independent dates are only a posthoc sensitivity for this new guard.',
    training_dates=['2023-10-01','2023-10-02'], training_routes=dict(routes),
    absent_route_q_upper=1., routes_with_zero_model_cancel_prior=[route for route in routes if fit['route_priors'].get(route, fit['route_priors']['pooled'])['cancel_rate'] == 0],
    sample_requirement=dict(cap_seconds=3180, epsilon_seconds=1, alpha=.1,
        formula='ceil(log(alpha)/log(1-epsilon/cap))',
        required_zero_failure_independent_date_blocks=required_zero_failure_blocks(3180.)),
    sensitivity=summaries)
(HERE / 'decision_risk_service_guard.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'sample_requirement':result['sample_requirement'], 'training_routes':dict(routes),
                  'confirmation':summaries['confirmation']}, indent=2))
