"""Score frozen conditional risk methods on predeclared new actual dates."""
import csv
import gzip
import json
from pathlib import Path
import pickle
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V5 = ROOT / 'experiments/reconstruction_v5'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V5))
from experiments.reconstruction_v6.calibration_run import read, write, summarize, METHODS, MENUS
from experiments.reconstruction_data_audit.build_dated_replay import attach_outcomes, dated_connections
from calibration_run import data_inputs, law_summary
from calibration_relative_run import candidate_pairs, build_day


def evaluate(data, day, fit, models):
    evaluator, _ = build_day(data, day, fit, [])
    source, graph = evaluator.source, evaluator.graph
    cached, output = {}, {}
    for menu in MENUS:
        pairs = candidate_pairs(graph, None if menu == 'fixed_schedule' else source)
        rows = []
        counts = {'planned_pairs': len(pairs), 'unobserved_incoming': 0}
        for incoming, outgoing in pairs:
            before, after = source[incoming.id], source[outgoing.id]
            if before['outcome'] != 'MEASURED':
                counts['unobserved_incoming'] += 1
                continue
            key = incoming.id, outgoing.id
            if key not in cached:
                clock = before['arrival']['time_seconds']
                ready = clock + graph.stops[incoming.arr_stop].min_transfer_time
                status = after['departure']['status']
                actual = after['departure']['time_seconds'] if status == 'REAL' else None
                caught = bool(actual is not None and ready <= actual <= ready + 720) if status in ('REAL', 'CANCELED') else None
                # Frozen independent-trip route parameters: observing this
                # different incoming trip leaves the outgoing prior unchanged.
                evolving = evaluator.evolving(after['id'], 'departure', clock, False)
                laws = {'evolving_incoming': evolving,
                        'shrunk_relative': models['shrunk_relative'].predict(before, after, clock),
                        'student': models['student'].predict(before, after, clock)}
                cached[key] = {'incoming': before['id'], 'outgoing': after['id'],
                    'incoming_route': before['route_id'], 'outgoing_route': after['route_id'],
                    'clock': clock, 'ready': ready, 'scheduled': outgoing.dep_time,
                    'incoming_delay': clock - incoming.arr_time, 'actual': actual,
                    'status': status, 'caught': caught,
                    'methods': {name: law_summary(law, actual) for name, law in laws.items()},
                    'probabilities': {name: law.board_probability(ready) for name, law in laws.items()}}
            row = cached[key]
            rows.append(row)
            counts[row['status']] = counts.get(row['status'], 0) + 1
        output[menu] = {'coverage_counts': counts, 'rows': rows}
    return output


def main():
    receipt = read(HERE / 'calibration_freeze_receipt.json')
    protocol = receipt['protocol']
    with open(HERE / 'calibration_frozen_model.pkl', 'rb') as handle:
        student = pickle.load(handle)
    with open(V5 / 'calibration_relative_variants_frozen.pkl', 'rb') as handle:
        shrunk = pickle.load(handle)['shrunk_relative']
    fit = read(V5 / 'calibration_frozen_fit.json')['evolving_fit']
    factors = read(HERE / 'calibration_frozen_envelopes.json')['factors']
    dates = protocol['actual_label_confirmation_dates']
    stops, schedules = dated_connections(dates)
    receipt['new_actuals_opened'] = True
    write(HERE / 'calibration_confirmation_open_receipt.json', receipt)
    actuals = {'dates': {}, 'stops': stops, 'protocol': protocol}
    data = data_inputs()
    for day in dates:
        path = HERE / 'calibration_confirmation_sources' / f'rail_v6_confirmation_{day}.csv.gz'
        with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle, delimiter=';'))
        coverage = attach_outcomes(day, schedules[day], rows)
        actuals['dates'][day] = {'connections': schedules[day], 'coverage': coverage,
            'source_audit': read(path.with_suffix('.audit.json'))}
        print('JOIN', day, coverage, flush=True)
    write(HERE / 'calibration_confirmation_actuals.json.gz', actuals)
    data['dates'].update(actuals['dates'])
    outcomes = {}
    for day in dates:
        outcomes[day] = evaluate(data, day, fit, {'student': student, 'shrunk_relative': shrunk})
        print('SCORED', day, {name: outcomes[day][name]['coverage_counts'] for name in MENUS}, flush=True)
    result = {'protocol': protocol, 'factors': factors, 'rows_by_date': outcomes,
        'confirmation': summarize(outcomes, factors),
        'source_coverage': {day: actuals['dates'][day]['coverage'] for day in dates},
        'interpretation': 'Three independent actual-label date blocks after frozen model/group/envelopes. '
        'Historical public RT forecasts from these dates existed but did not fit this candidate. '
        'CDF probabilities are raw laws; date-group envelopes modify intervals only.'}
    write(HERE / 'calibration_confirmation_results.json.gz', result)
    write(HERE / 'calibration_confirmation_summary.json', {k: v for k, v in result.items() if k != 'rows_by_date'})


if __name__ == '__main__':
    main()
