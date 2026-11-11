"""Service-support and fixed-law diagnostics on already exposed V1--V6 days."""
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v5.calibration_run import data_inputs
from src.decision_risk import required_zero_failure_blocks
from src.service_state_model import (
    ServiceStateModel, ServiceTrip, completion_label, fit_completion_parameters,
)


def read(path):
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as handle:
        return json.load(handle)


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summarize(rows):
    known = [row for row in rows if row['actual_completion'] is not None]
    return dict(n_selected_zero_q=len(rows), n_known_completion=len(known),
        n_unresolved=len(rows) - len(known), failures=sum(not row['actual_completion'] for row in known),
        old_service_only_brier=float(np.mean([(1. - row['actual_completion']) ** 2 for row in known])) if known else None,
        bayes_service_only_brier=float(np.mean([(row['completion_probability'] - row['actual_completion']) ** 2 for row in known])) if known else None,
        mean_model_service_cost=float(np.mean([row['model_service_cost'] for row in rows])) if rows else None,
        mean_actual_service_cost_lower=float(np.mean([row['actual_service_cost_lower'] for row in rows])) if rows else None,
        mean_actual_service_cost_upper=float(np.mean([row['actual_service_cost_upper'] for row in rows])) if rows else None)


def main():
    HERE.mkdir(exist_ok=True)
    protocol = dict(scope='Posthoc support/model diagnostic; all scored dates were already exposed by V1--V6.',
        training_dates=['2023-10-01', '2023-10-02'],
        prior_shapes=[.5, 1.],
        target='Segment traversal can be completed by the same actual service, independent of catch/deadline timing.',
        baseline='Only original q=0 routes: conditional service completion probability1; not the original catch+clock prediction.',
        population='Original frozen V6 model-selected contexts, then restrict to zero-q route and known completed incoming. Unknown future completion retained.',
        joint_law='Once-per-world route latent Beta initial-failure/stop/restore variables; exact conditioning filters the original service worlds.',
        physical_scope='No arrival-only cancellation is turned into a predeparture nonboarding event. Failure location/time and cancellation receipt remain unidentified.',
        guarantee='Positive support and coherent model conditioning only; correlated route records do not establish frequentist calibration or actual safe authorization.',
        source='Existing local dated-replay/decision-risk records only; no download.',
        source_text={name: (ROOT / name).read_text() for name in
                     ('src/service_state_model.py', 'experiments/reconstruction_v7/service_run.py')})
    write('service_protocol.json', protocol)
    data = data_inputs()
    actuals = read(ROOT / 'experiments/reconstruction_v6/decision_risk_confirmation_actuals.json.gz')
    data['dates'].update(actuals['dates'])
    old = read(ROOT / 'experiments/reconstruction_v6/decision_risk_development_results.json.gz')['dates']
    old.update(read(ROOT / 'experiments/reconstruction_v6/decision_risk_confirmation_results.json.gz')['dates'])
    previous_fit = read(ROOT / 'experiments/reconstruction_v5/calibration_frozen_fit.json')['evolving_fit']
    training = {day: data['dates'][day]['connections'] for day in protocol['training_dates']}
    failure = read(ROOT / 'experiments/reconstruction_v6/decision_risk_failure_diagnosis.json')
    fits, outcomes = {}, {}
    for prior in protocol['prior_shapes']:
        parameters, audit = fit_completion_parameters(training, prior)
        fits[str(prior)] = dict(parameters={route: asdict(value) for route, value in parameters.items()}, audit=audit)
        dates = {}
        for day, result in old.items():
            source = {row['id']: row for row in data['dates'][day]['connections']}
            sequences = defaultdict(list)
            for item in source.values():
                sequences[item['trip_id']].append(item)
            local_laws = {}
            rows = []
            for context in result['rows']:
                if not context['selected']:
                    continue
                incoming, following = source[context['incoming']], source[context['outgoing']]
                route = incoming['route_id']
                if previous_fit['route_priors'][route]['cancel_rate'] != 0:
                    continue
                before, after = completion_label(incoming), completion_label(following)
                if before.can_complete is not True:
                    raise ValueError('The original selected context lacks the completed incoming fact')
                actual = after.can_complete if after.journey == before.journey else None
                trip = incoming['trip_id']
                if trip not in local_laws:
                    ordered = sorted(sequences[trip], key=lambda item: (item['dep_sequence'], item['arr_sequence'], item['id']))
                    local_laws[trip] = ServiceStateModel(
                        (ServiceTrip(route, tuple(item['id'] for item in ordered)),), {route: parameters[route]}).exact()
                posterior = local_laws[trip].condition({incoming['id']: True})
                q = 1. - posterior.completion_probability(following['id'])
                cap = 3600 - context['transfer_seconds']
                rows.append(dict(date=day, group=context['group'], route=route,
                    incoming=incoming['id'], outgoing=following['id'], completion_probability=1. - q,
                    actual_completion=actual, model_service_cost=cap * q,
                    actual_service_cost_lower=cap if actual is False else 0.,
                    actual_service_cost_upper=0. if actual is True else cap,
                    label_reason=after.reason, cancellation_receipt=None))
            dates[day] = dict(summary=summarize(rows), rows=rows)
        outcomes[str(prior)] = dates
    params, _ = fit_completion_parameters(training, .5)
    incoming, outgoing = failure['failure']['incoming'], failure['failure']['outgoing']
    route = failure['incoming_source']['route_id']
    source_trip = [item for item in data['dates']['2023-10-27']['connections']
                   if item['trip_id'] == failure['incoming_source']['trip_id']]
    source_trip.sort(key=lambda item: (item['dep_sequence'], item['arr_sequence'], item['id']))
    local = ServiceStateModel((ServiceTrip(route, tuple(item['id'] for item in source_trip)),), {route: params[route]})
    root = local.exact().condition({incoming: True})
    support = dict(old_failure_probability=0., new_failure_probability=1. - root.completion_probability(outgoing),
        source_incoming=failure['incoming_source'], source_future=failure['future_source'],
        fixed_worlds=[dict(can_complete=dict(zip(local.connections, states)), probability=float(weight))
                      for states, weight in zip(root.worlds, root.weights) if weight > 0],
        publication_information=failure['publication_information'],
        after_offline_failure_label_probability=root.condition({outgoing: False}).completion_probability(outgoing),
        timeline_scope='Offline completion label is scored, not supplied to the time29071 public history.')
    labels = {}
    for day in sorted(old):
        counts = Counter()
        for row in data['dates'][day]['connections']:
            label = completion_label(row)
            counts[label.reason] += 1
            if label.can_complete is False:
                counts['arrival_only_cancel'] += row['arrival']['status'] == 'CANCELED' and row['departure']['status'] != 'CANCELED'
        labels[day] = dict(counts)
    sample_sensitivity = {str(seconds): required_zero_failure_blocks(3180, seconds)
                          for seconds in (1, 30, 60, 120)}
    write('service_results.json', dict(protocol=protocol, training=fits, selected_contexts=outcomes,
        source_label_audit=labels, s2_support=support,
        zero_failure_binomial_day_counts_sensitivity=sample_sensitivity,
        statistical_scope='These day counts belong to the chosen iid-day zero-event Clopper-Pearson construction; not a universal information lower bound.'))
    summary = dict(training={prior: value['audit'] for prior, value in fits.items()},
        panels={prior: {day: value['summary'] for day, value in dates.items()} for prior, dates in outcomes.items()},
        s2_stop_rates={prior: value['parameters'][route]['stop'] for prior, value in fits.items()},
        s2_new_failure_probability=support['new_failure_probability'], source_label_audit=labels,
        zero_failure_binomial_day_counts_sensitivity=sample_sensitivity)
    write('service_summary.json', summary)
    print(json.dumps(dict(s2=support['new_failure_probability'], required_days=sample_sensitivity,
        new_development_dates={date: outcomes['0.5'][date]['summary'] for date in ('2023-10-15','2023-10-19','2023-10-22','2023-10-27')}), indent=2))


if __name__ == '__main__':
    main()
