"""Locate the observed suffix cancellation without exposing it to selection."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.reconstruction_v5.calibration_run import data_inputs
from experiments.reconstruction_v6.decision_risk_run import evaluate_day

data = data_inputs()
fit = json.loads((HERE / 'decision_risk_frozen.json').read_text())['protocol']['configuration']['training']
with gzip.open(HERE / 'decision_risk_development_results.json.gz', 'rt') as handle:
    prior = json.load(handle)['dates']
counts = {}
for i in range(1, 12):
    day = f'2023-10-{i:02d}'
    result = prior[day] if day in prior else evaluate_day(data, day, fit)
    sources = {c['id']: c for c in data['dates'][day]['connections']}
    tally = Counter()
    for row in result['rows']:
        incoming, future = sources[row['incoming']], sources[row['outgoing']]
        actual = incoming['arrival']['official_journey']
        canceled = any(e['status'] == 'CANCELED' and e['official_journey'] == actual
                       for e in (future['departure'], future['arrival']))
        tally['observable_incoming_contexts'] += 1
        if canceled:
            tally['known_canceled_next_contexts'] += 1
            tally['selected_known_canceled_next_contexts'] += int(row['selected'])
            tally['known_canceled_next_S2_contexts'] += int(future['route_id'] == '91-2-B-j23-1')
    counts[day] = dict(tally)
day = '2023-10-27'
failure = next(row for row in prior[day]['rows'] if row['selected'] and row['outcome'] == 'failed_canceled')
source = {c['id']: c for c in data['dates'][day]['connections']}
route = source[failure['outgoing']]['route_id']
result = dict(training_calibration_contexts=counts, failure=failure,
    incoming_source=source[failure['incoming']], future_source=source[failure['outgoing']],
    frozen_route_prior=fit['route_priors'][route],
    cancel_parameter_effect='q=0 omits cancellation states from the Evolving event grids; this suffix has model success_probability1. No timing bias is needed for this failure.',
    publication_information='The official retrospective status supplies no cancellation publication timestamp. The frozen H contains only completed incoming dep/arr and no public notifications. The future CANCELED label cannot be exposed at time29071.',
    calibration_interpretation='B is a rank-based new-date tolerance under exchangeability, allowing up to10% date miscoverage; it is not a hard limit for each calendar date. Oct27 remains a development failure and is never added to calibration.')
(HERE / 'decision_risk_failure_diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'training_calibration_contexts': counts, 'frozen_route_prior': result['frozen_route_prior']}, indent=2))
