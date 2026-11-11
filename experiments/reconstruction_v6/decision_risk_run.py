"""Freeze and evaluate a decision-loss target for selected same-car fragments."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shlex
import subprocess
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v5.calibration_run import data_inputs
from experiments.reconstruction_data_audit.build_dated_replay import dated_connections, attach_outcomes
from src.committed_transit import CommittedState
from src.decision_risk import realized_macro_risk, summarize_day, calibrate_daily_groups, assess_day
from src.evolving_trip_belief import EvolvingTripBelief
from src.risk_bounds import CommitmentBounds

CALIBRATION = [f'2023-10-{i:02d}' for i in range(3, 12)]
DEVELOPMENT = ['2023-10-12', '2023-10-25', '2023-10-27']
CONFIRMATION = ['2023-10-15', '2023-10-19', '2023-10-22']
REMOTE = '/home/erzhu419/BAPR-HRO/experiments/reconstruction_data_audit/remote_actual_20231025'
PROTOCOL = HERE / 'decision_risk_protocol.json'


def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt') as handle:
            return json.load(handle)
    return json.loads(Path(path).read_text())


def write(path, value):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'wt') as handle:
            json.dump(value, handle, allow_nan=False, separators=(',', ':'))
    else:
        Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def evaluate_day(data, day, fit):
    graph, _, _, _, source, _ = dated_graph(data, day, 'gtfs_station_max', True)
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    base = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    rows, exclusions = [], Counter()
    for trip, slots in base._trips.items():
        for previous, following in zip(slots[::2], slots[2::2]):
            before, after = graph.connections[previous.cid], graph.connections[following.cid]
            incoming, future = source[before.id], source[after.id]
            if before.arr_stop != after.dep_stop or after.arr_stop == after.dep_stop:
                exclusions['static_noncontiguous_or_goal_already_here'] += 1
                continue
            if incoming['outcome'] != 'MEASURED':
                exclusions['incoming_not_observable_completed_ride'] += 1
                continue
            # This stronger topology restriction makes the measured surrogate
            # independent of whether model hard supports contain actual times.
            same_departures = [c.id for c in graph.get_connections_from(before.arr_stop)
                               if c.trip_id == before.trip_id and c.id != before.id]
            if same_departures != [after.id]:
                exclusions['multiple_same_trip_departures'] += 1
                continue
            time = incoming['arrival']['time_seconds']
            branch = base.copy()
            branch.observe_connection_event(before.id, 'departure',
                incoming['departure']['time_seconds'] - before.dep_time)
            branch.observe_connection_event(before.id, 'arrival', time - before.arr_time)
            state = CommittedState(before.arr_stop, time, before.trip_id, time,
                                   unavailable_ids=frozenset({before.id}), initialized=True)
            index = CommitmentBounds(branch, parameters, graph, after.arr_stop, time + 3600)
            certificate = index.transfer_gate(state, (after.id,))
            key = incoming['arr_stop']
            stated = data['station_transfer_bounds'].get(key, {})
            transfer_source = ('max_stated_station_buffer'
                if stated.get('conservative_max_stated_seconds') is not None else 'assumed120')
            item = dict(date=day, incoming=incoming['id'], outgoing=future['id'], trip_id=trip,
                group='late' if time - before.arr_time > 180 else 'ordinary', time=time,
                incoming_delay_seconds=time - before.arr_time, selected=False,
                transfer_source=transfer_source,
                transfer_seconds=graph.stops[before.arr_stop].min_transfer_time,
                minimum_stated_seconds=stated.get('minimum_stated_seconds'))
            if certificate is None:
                item['gate_state'] = 'structure_or_support_not_certified'
                rows.append(item)
                continue
            item.update(model_regret_bound=certificate.regret_bound,
                model_upper=certificate.upper, model_lower=certificate.lower,
                selected=certificate.regret_bound <= 1., gate=certificate.transfer_gate,
                model_success_probability=certificate.success_probability,
                gate_state='selected' if certificate.regret_bound <= 1. else 'model_gap_above_one_second')
            if item['selected']:
                dep, arr = future['departure'], future['arrival']
                departure = dep['time_seconds'] if dep['status'] == 'REAL' else None
                arrival = arr['time_seconds'] if arr['status'] == 'REAL' else None
                actual_trip = incoming['arrival']['official_journey']
                canceled = any(endpoint['status'] == 'CANCELED'
                    and endpoint['official_journey'] == actual_trip for endpoint in (dep, arr))
                same = actual_trip == dep['official_journey'] == arr['official_journey']
                risk = realized_macro_risk(time, certificate.transfer_gate, time + 3600,
                    departure, arrival, canceled=canceled, same_actual_trip=same)
                zero = realized_macro_risk(time, time, time + 3600, departure, arrival,
                    canceled=canceled, same_actual_trip=same)
                item.update(risk_lower=risk.lower, risk_upper=risk.upper, outcome=risk.outcome,
                    zero_buffer_risk_lower=zero.lower, zero_buffer_risk_upper=zero.upper,
                    future_status={'departure': dep['status'], 'arrival': arr['status']},
                    observed_goal_arrival=arrival, observed_departure=departure,
                    same_actual_trip=same)
            rows.append(item)
    return {'rows': rows, 'summary': summarize_day(rows), 'exclusions': dict(exclusions),
            'source_coverage': data['dates'][day]['coverage']}


def freeze():
    fit = read(ROOT / 'experiments/reconstruction_v5/calibration_frozen_fit.json')['evolving_fit']
    configuration = dict(training_dates=['2023-10-01', '2023-10-02'], calibration_dates=CALIBRATION,
        development_dates=DEVELOPMENT, confirmation_dates=CONFIRMATION, training=fit,
        clock='integer seconds', source_information='Completed incoming own ride dep/arr only; no public occurrence or numeric forecasts.',
        context='Next physically contiguous graph leg of the same trip, goal=its next stop; exactly one available same-trip outgoing leg at current stop. No future target status enters context/selector.',
        selector='Fixed current Evolving posterior transfer_gate regret_bound<=1second; buffer/source/topology conditions unchanged.',
        group='incoming actual arrival delay <=180seconds versus >180seconds',
        deadline='incoming actual arrival +3600seconds',
        target='Maximum across present groups of selected finite-context mean R, not each-call expected risk. R=(B-T)+ on macro success, D-T on failure.',
        missing='Future unmeasured/provider-mismatched endpoints retain R in[0,D-T]; explicit canceled target is a failure.',
        calibration='Largest of nine calibration date upper scores; rank9/10 under exchangeable latent date blocks. Empty group has no restriction; no claim conditional on being called/selected.',
        transfer_contract='GTFS max-stated station buffer when present, otherwise assumed120. This is the project execution contract, not identified physical minimum per platform pair.',
        runtime_scope='Nonblocking audit of a previously issued stay fragment/logical counterfactual; no assertion that a fresh post-arrival computation catches a same-second vehicle.',
        sensitivity='For the same selected contexts report zero-buffer R; do not reselect/refit or treat this as the primary certificate.',
        source='Complete official daily member filtered remotely to the existing16stations/window; only small gzip and audit copied locally.')
    if PROTOCOL.exists():
        protocol = read(PROTOCOL)
        if protocol['configuration'] != configuration:
            raise ValueError('Protocol changed; retain this experiment and use a separately named version')
    else:
        protocol = {'locked_at_utc': datetime.now(timezone.utc).isoformat(), 'configuration': configuration,
            'source_text': {path: (ROOT / path).read_text() for path in (
                'src/decision_risk.py', 'src/risk_bounds.py', 'src/evolving_trip_belief.py',
                'experiments/reconstruction_v6/decision_risk_run.py',
                'experiments/reconstruction_v4/run_dated_replay.py')}}
        write(PROTOCOL, protocol)
    data = data_inputs()
    outcomes = {}
    start = perf_counter()
    for day in CALIBRATION + DEVELOPMENT:
        outcomes[day] = evaluate_day(data, day, fit)
        print(day, outcomes[day]['summary'], flush=True)
    calibration = calibrate_daily_groups({day: outcomes[day]['summary'] for day in CALIBRATION})
    result = dict(protocol=protocol, calibration=calibration, elapsed_seconds=perf_counter() - start,
                  dates=outcomes)
    write(HERE / 'decision_risk_development_results.json.gz', result)
    write(HERE / 'decision_risk_frozen.json', {'protocol': protocol, 'calibration': calibration})
    write(HERE / 'decision_risk_development_summary.json', dict(calibration=calibration,
        days={day: {k: v for k, v in outcome.items() if k != 'rows'} for day, outcome in outcomes.items()}))


def fetch(day, source='prefix'):
    folder = HERE / 'decision_risk_sources'
    folder.mkdir(exist_ok=True)
    stem = 'rail_decision_prefix' if source == 'prefix' else 'rail_decision_risk'
    remote = f'{REMOTE}/{stem}_{day}.csv.gz'
    command = ['python3', '-u', f'{REMOTE}/decision_risk_prefix_remote.py' if source == 'prefix'
               else f'{REMOTE}/calibration_complete_remote.py', '--date', day,
               '--filter', f'{REMOTE}/bapr-rail-stops.json', '--output', remote]
    command += ['--prefix-bytes', '1048576'] if source == 'prefix' else ['--workers', '4']
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'jtl110gpu2',
                    shlex.join(command)], check=True, capture_output=True, text=True)
    for path in (remote, remote.removesuffix('.gz') + '.audit.json'):
        subprocess.run(['scp', '-q', f'jtl110gpu2:{path}', str(folder)], check=True)
    return day, (folder / Path(remote).name).stat().st_size


def confirm():
    frozen = read(HERE / 'decision_risk_frozen.json')
    protocol = frozen['protocol']
    dates = protocol['configuration']['confirmation_dates']
    fit = protocol['configuration']['training']
    write(HERE / 'decision_risk_confirmation_primary_open_receipt.json', dict(
        opened_after_source_correction_at_utc=datetime.now(timezone.utc).isoformat(), frozen=frozen,
        source_correction=read(HERE / 'decision_risk_source_cohort_amendment.json')))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for day, size in pool.map(fetch, dates):
            print('PRIMARY PREFIX / SMALL LOCAL', day, size, flush=True)
    stops, schedules = dated_connections(dates)
    actuals = {'stops': stops, 'dates': {}, 'protocol': protocol}
    data = data_inputs()
    for day in dates:
        path = HERE / 'decision_risk_sources' / f'rail_decision_prefix_{day}.csv.gz'
        with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
            raw = list(csv.DictReader(handle, delimiter=';'))
        coverage = attach_outcomes(day, schedules[day], raw)
        actuals['dates'][day] = {'connections': schedules[day], 'coverage': coverage,
                                'source_audit': read(path.with_suffix('.audit.json'))}
    actuals['source_scope'] = 'primary: same 1MiB-prefix pipeline as calibration'
    write(HERE / 'decision_risk_confirmation_actuals.json.gz', actuals)
    data['dates'].update(actuals['dates'])
    outcomes = {day: evaluate_day(data, day, fit) for day in dates}
    bound = frozen['calibration']['risk_upper_seconds']
    summary = {day: {**{k: v for k, v in outcome.items() if k != 'rows'},
                    'assessment': assess_day(outcome['summary'], bound)} for day, outcome in outcomes.items()}
    write(HERE / 'decision_risk_confirmation_results.json.gz', dict(frozen=frozen, dates=outcomes))
    write(HERE / 'decision_risk_confirmation_summary.json', dict(calibration=frozen['calibration'], days=summary))
    print(json.dumps(summary, indent=2), flush=True)
    # Keep the same dates and selector; changed observable incoming cohorts
    # from source completion are a sensitivity, not rank-calibrated evidence.
    with ThreadPoolExecutor(max_workers=3) as pool:
        for day, size in pool.map(lambda day: fetch(day, 'full'), dates):
            print('COMPLETION SENSITIVITY / SMALL LOCAL', day, size, flush=True)
    stops, schedules = dated_connections(dates)
    complete = {'stops': stops, 'dates': {}, 'source_scope':
        'full-member completion sensitivity; no source-cohort exchangeability claim with prefix calibration'}
    for day in dates:
        path = HERE / 'decision_risk_sources' / f'rail_decision_risk_{day}.csv.gz'
        with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
            raw = list(csv.DictReader(handle, delimiter=';'))
        coverage = attach_outcomes(day, schedules[day], raw)
        complete['dates'][day] = {'connections': schedules[day], 'coverage': coverage,
                                'source_audit': read(path.with_suffix('.audit.json'))}
    write(HERE / 'decision_risk_completion_actuals.json.gz', complete)
    data['dates'].update(complete['dates'])
    sensitivity = {day: evaluate_day(data, day, fit) for day in dates}
    write(HERE / 'decision_risk_completion_results.json.gz', dict(frozen=frozen, dates=sensitivity,
        source_scope=complete['source_scope']))
    write(HERE / 'decision_risk_completion_summary.json', dict(source_scope=complete['source_scope'],
        days={day: {k: v for k, v in result.items() if k != 'rows'} for day, result in sensitivity.items()}))


def score_existing():
    """Apply a documented scorer correction without fetching source again."""
    freeze()
    frozen = read(HERE / 'decision_risk_frozen.json')
    fit, bound = frozen['protocol']['configuration']['training'], frozen['calibration']['risk_upper_seconds']
    for label in ('confirmation', 'completion'):
        actuals = read(HERE / f'decision_risk_{label}_actuals.json.gz')
        data = data_inputs()
        data['dates'].update(actuals['dates'])
        outcomes = {day: evaluate_day(data, day, fit) for day in CONFIRMATION}
        write(HERE / f'decision_risk_{label}_results.json.gz', dict(frozen=frozen, dates=outcomes,
            source_scope=actuals['source_scope']))
        write(HERE / f'decision_risk_{label}_summary.json', dict(calibration=frozen['calibration'],
            source_scope=actuals['source_scope'], days={day: {
                **{k: v for k, v in result.items() if k != 'rows'},
                **({'assessment': assess_day(result['summary'], bound)} if label == 'confirmation' else {})}
                for day, result in outcomes.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'confirm', 'score'))
    arguments = parser.parse_args()
    {'freeze': freeze, 'confirm': confirm, 'score': score_existing}[arguments.stage]()
