"""Frozen single-message comparison and exact two-event posterior scores.

Old marginal predictions are read from V35. Only their previously missing
joint-event scores are computed here; no routing control or sample is run.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.forecast import MessageHistory
from experiments.reconstruction_v35.forecast import FittedMessageHistory
from experiments.reconstruction_v35.validate import metric_means, predict, query_history, write
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from src.evolving_trip_belief import EvolvingTripBelief, logsumexp

HERE = Path(__file__).resolve().parent
V35 = HERE.parent/'reconstruction_v35'
MODELS = ('own_only', 'legacy_messages', 'fitted_messages', 'single_message')
AGE_BINS = ('no_target_forecast', '<=600', '(600,1800]', '>1800')
COUNT_BINS = ('0', '1', '2-4', '5+')


def age_bin(age):
    return ('no_target_forecast' if age is None else '<=600' if age <= 600 else
            '(600,1800]' if age <= 1800 else '>1800')


def count_bin(count):
    return '0' if count == 0 else '1' if count == 1 else '2-4' if count <= 4 else '5+'


def joint_event_scores(law, departure_slot, *, scheduled_departure, scheduled_arrival,
                       actual_departure, actual_arrival):
    """Exact adjacent-slot law, conditioned on both physical events operating."""
    arrival_slot = departure_slot+1
    backward = law._backward_from(len(law.times)-1, arrival_slot)
    log_mass = (law._full_forward(departure_slot)[:, None] + law._transition(departure_slot)
                + law.emissions[arrival_slot][None, :] + backward[None, :])
    normalizer = logsumexp(log_mass)
    if not np.isfinite(normalizer):
        raise FloatingPointError('Nonfinite joint normalization needs diagnosis before scoring')
    mass = np.exp(log_mass-normalizer)
    operating = (~law.canceled[departure_slot])[:, None] & (~law.canceled[arrival_slot])[None, :]
    operating_mass = float(mass[operating].sum())
    if operating_mass <= 0:
        raise UnsupportedHistory('No probability that both scored endpoints operate')
    joint = np.where(operating, mass/operating_mass, 0.)
    departure = law.times[departure_slot][:, None]
    arrival = law.times[arrival_slot][None, :]
    product = joint.sum(axis=1)[:, None]*joint.sum(axis=0)[None, :]
    dep_limit, arr_limit = scheduled_departure+300, scheduled_arrival+300
    runtime_limit = scheduled_arrival-scheduled_departure+60
    masks = {'J1': (departure<=dep_limit) & (arrival<=arr_limit),
             'J2': arrival-departure<=runtime_limit}
    truths = {'J1': actual_departure<=dep_limit and actual_arrival<=arr_limit,
              'J2': actual_arrival-actual_departure<=runtime_limit}
    events = {}
    for name, mask in masks.items():
        probability, independent = float(joint[mask].sum()), float(product[mask].sum())
        truth = float(truths[name])
        events[name] = dict(probability=probability, product_probability=independent,
            joint_minus_product=probability-independent,
            absolute_joint_minus_product=abs(probability-independent),
            actual=truth, Brier=(probability-truth)**2, product_Brier=(independent-truth)**2)
    return dict(status='ok', operating_pair_probability=operating_mass,
        departure_threshold=dep_limit, arrival_threshold=arr_limit, runtime_threshold=runtime_limit,
        conditional_departure_probability=float(joint[departure[:,0]<=dep_limit].sum()),
        conditional_arrival_probability=float(joint[:,arrival[0]<=arr_limit].sum()),
        product_impossible_order_probability=float(product[arrival<departure].sum()), events=events)


def predict_joint(history, row, metadata):
    if row['cid'] is None or any(o['cid'] is None for o in row['own_observations']):
        return dict(status='unsupported', reason='Endpoint omitted from fixed physical graph')
    cid = row['cid']
    trip, slot = history.belief._positions[cid, 'departure']
    arrival_trip, arrival_slot = history.belief._positions[cid, 'arrival']
    if arrival_trip != trip or arrival_slot != slot+1:
        raise ValueError('This panel requires adjacent departure/arrival slots of one connection')
    source = metadata[cid]
    try:
        return joint_event_scores(history.law(trip), slot,
            scheduled_departure=source['scheduled_departure_seconds'],
            scheduled_arrival=source['scheduled_arrival_seconds'],
            actual_departure=source['departure']['time_seconds'], actual_arrival=row['actual'])
    except UnsupportedHistory as error:
        return dict(status='unsupported', reason=str(error))


def selection_receipt(history, trip):
    raw = history.raw_forecast_records(trip)
    completed = max((history.belief._positions[cid, kind][1]
        for cid, kind, mode, value in history._records.get(trip, ()) if mode=='observed'), default=-1)
    eligible = [r for r in raw if history.belief._positions[r[0], r[1]][1]>completed]
    selected = history.selected_forecast(trip)
    return dict(raw_active_count=len(raw), eligible_future_count=len(eligible),
        last_observed_slot=completed, selected_forecast=selected,
        selected_age_seconds=None if selected is None else history.message_clock-selected[4])


def summarize(rows, panel):
    common = [r for r in rows if all(r['predictions'][m]['status']=='ok' for m in MODELS)]
    joint_common = ([r for r in rows if all(r['joint_predictions'][m]['status']=='ok' for m in MODELS)]
                    if panel=='onboard' else [])
    result = dict(queries=len(rows), common_supported=len(common),
        target_forecast_queries=sum(r['target_has_forecast'] for r in rows), models={})
    if panel=='onboard':
        result['joint_common_supported'] = len(joint_common)
        result['joint_event_outcomes'] = {event:dict(n=len(joint_common),
            true=sum(r['joint_predictions']['own_only']['events'][event]['actual'] for r in joint_common))
            for event in ('J1','J2')}
    for model in MODELS:
        result['models'][model] = dict(
            supported=sum(r['predictions'][model]['status']=='ok' for r in rows),
            unsupported=sum(r['predictions'][model]['status']!='ok' for r in rows),
            common_metrics=metric_means(common, model))
        if panel=='onboard':
            result['models'][model].update(
                joint_supported=sum(r['joint_predictions'][model]['status']=='ok' for r in rows),
                joint_unsupported=sum(r['joint_predictions'][model]['status']!='ok' for r in rows),
                joint_metrics={event:{metric:float(np.mean([
                    r['joint_predictions'][model]['events'][event][metric] for r in joint_common]))
                    for metric in ('probability','product_probability','joint_minus_product',
                        'absolute_joint_minus_product','Brier','product_Brier')} if joint_common else {}
                    for event in ('J1','J2')})
    return result


def run_day(day):
    began = perf_counter()
    saved = read(V35/'validation_days'/f'{day}.json.gz')
    data = read(HERE.parent/'reconstruction_data_audit/rail_calibration_dates_v2.json.gz')
    graph, _, _, mapping, metadata, _ = context(data, day)
    frames = read(V35/'message_days'/f'{day}.json.gz')['frames']
    prior = read(HERE.parent/'reconstruction_v5/calibration_frozen_fit.json')['evolving_fit']
    fit = read(V35/'fitted_emission.json')
    model = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **prior['increments'])
    parameters = {c.route:prior['route_priors'].get(c.route,prior['route_priors']['pooled'])
                  for c in graph.connections}
    summaries = {}
    max_marginal_discrepancy = 0.
    for panel, rows in saved['panels'].items():
        single = SingleMessageHistory(model, parameters, fit=fit, frames=frames, source_to_local=mapping)
        old = (dict(own_only=MessageHistory(model, parameters),
            legacy_messages=MessageHistory(model, parameters, frames=frames, source_to_local=mapping),
            fitted_messages=FittedMessageHistory(model, parameters, fit=fit, frames=frames, source_to_local=mapping))
            if panel=='onboard' else {})
        for row in rows:
            history = query_history(single, row)
            row['predictions']['single_message'] = predict(history, row)
            selected = selection_receipt(history, row['trip_id'])
            if selected['raw_active_count'] != row['trip_active_forecast_count'] or history.frame_index!=row['public_frame_count']:
                raise AssertionError('New model must receive the same raw public history as V35')
            row['single_message_selection'] = selected
            row['strata'] = dict(target_message_age=age_bin(None if row['target_forecast'] is None else
                    row['target_forecast']['age_seconds']),
                eligible_future_forecast_count=count_bin(selected['eligible_future_count']))
            if panel=='onboard':
                row['joint_predictions'] = {'single_message':predict_joint(history,row,metadata)}
                for name, master in old.items():
                    previous = query_history(master,row)
                    score = row['joint_predictions'][name] = predict_joint(previous,row,metadata)
                    if score['status']=='ok' and row['predictions'][name]['status']=='ok':
                        difference = abs(score['conditional_arrival_probability']-
                            row['predictions'][name]['conditional_operating_probability'])
                        max_marginal_discrepancy = max(max_marginal_discrepancy,difference)
                        if difference>1e-9:
                            raise AssertionError('Reconstructed pair law disagrees with saved V35 marginal')
        summaries[panel] = summarize(rows,panel)
        print(day,panel,len(rows),'common',summaries[panel]['common_supported'],flush=True)
    output = dict(day=day, panels=saved['panels'], summary=summaries,
        source_eligibility=saved['source_eligibility'],
        old_marginals_reused_from=f'../reconstruction_v35/validation_days/{day}.json.gz',
        max_joint_vs_saved_arrival_CDF_difference=max_marginal_discrepancy, wall=perf_counter()-began)
    write(HERE/'validation_days'/f'{day}.json.gz',output)
    print(day,'completed',round(output['wall'],3),flush=True)
    return day


def macro_summary(per_day, panel):
    summaries = {day:summarize(rows,panel) for day,rows in per_day.items()}
    total = summarize([r for rows in per_day.values() for r in rows],panel)
    macro = {}
    for model in MODELS:
        values = [d['models'][model]['common_metrics'] for d in summaries.values()]
        metrics = sorted({key for value in values for key in value})
        macro[model] = dict(common_metrics={key:float(np.mean([v[key] for v in values if key in v]))
            for key in metrics}, contributing_dates=sum(bool(v) for v in values))
        if panel=='onboard':
            macro[model]['joint_metrics'] = {}
            for event in ('J1','J2'):
                values = [d['models'][model]['joint_metrics'][event] for d in summaries.values()]
                keys = sorted({key for value in values for key in value})
                macro[model]['joint_metrics'][event] = {key:float(np.mean([v[key] for v in values if key in v]))
                    for key in keys}
    total['equal_date_macro'] = macro
    if panel=='onboard':
        total['equal_date_event_rates'] = {event:float(np.mean([
            d['joint_event_outcomes'][event]['true']/d['joint_event_outcomes'][event]['n']
            for d in summaries.values() if d['joint_event_outcomes'][event]['n']]))
            for event in ('J1','J2') if any(d['joint_event_outcomes'][event]['n'] for d in summaries.values())}
    total['by_day'] = summaries
    return total


def day_directions(summary):
    fields = {'CRPS_seconds':('common_metrics','CRPS_seconds')}
    if 'joint_common_supported' in summary:
        fields.update(J1_Brier=('joint_metrics','J1','Brier'), J2_Brier=('joint_metrics','J2','Brier'))
    directions = {}
    for field, path in fields.items():
        gains = {}
        for day, report in summary['by_day'].items():
            old, new = report['models']['fitted_messages'],report['models']['single_message']
            for key in path:
                old, new = old.get(key),new.get(key)
                if old is None or new is None:
                    break
            if old is not None and new is not None:
                gains[day] = old-new
        directions[field] = dict(gain_by_day=gains, positive=sum(v>0 for v in gains.values()),
            negative=sum(v<0 for v in gains.values()), tied=sum(v==0 for v in gains.values()))
    return directions


def aggregate(days):
    saved = {day:read(HERE/'validation_days'/f'{day}.json.gz') for day in days}
    panels, strata = {}, {}
    for panel in ('endpoint','onboard'):
        rows_by_day = {day:d['panels'][panel] for day,d in saved.items()}
        report = panels[panel] = macro_summary(rows_by_day,panel)
        report['single_vs_fitted_day_directions'] = day_directions(report)
        report['target_forecast_subset'] = macro_summary(
            {day:[r for r in rows if r['target_has_forecast']] for day,rows in rows_by_day.items()},panel)
        strata[panel] = {}
        for field, bins in (('target_message_age',AGE_BINS),('eligible_future_forecast_count',COUNT_BINS)):
            strata[panel][field] = {label:macro_summary(
                {day:[r for r in rows if r['strata'][field]==label] for day,rows in rows_by_day.items()},panel)
                for label in bins}
    result = dict(scope='Frozen V36 model-structure diagnostic; old marginal receipts reused unchanged.',
        validation_dates=days, models=MODELS, panels=panels, strata=strata,
        joint_event_scope='Exact next-leg departure/arrival pair posterior conditional on both operating. '
            'Product probabilities use marginals from that same conditional pair law. J2 product may assign '
            'mass to impossible negative running times; this mass is retained in the row diagnostics.',
        v35_gate_unchanged=read(V35/'validation_summary.json')['gate'],
        diagnostic_only=True,
        limitations='Two fixed joint events and conditional marginals do not validate the complete joint '
            'distribution or identify forecast-error correlation. The compression also omits messages on '
            'the publicly completed prefix. Dates are retrospective development data. No new scalar gate.',
        max_joint_vs_saved_arrival_CDF_difference=max(d['max_joint_vs_saved_arrival_CDF_difference'] for d in saved.values()),
        row_files=[f'validation_days/{day}.json.gz' for day in days])
    write(HERE/'validation_summary.json',result)
    print('validation summary complete; diagnostic only',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    protocol = read(HERE/'protocol.json')
    days = protocol['validation']['dates']
    if tuple(protocol['validation']['models'])!=MODELS:
        raise ValueError('Validator model labels must match the frozen protocol')
    (HERE/'validation_days').mkdir(exist_ok=True)
    pending = [day for day in days if not (HERE/'validation_days'/f'{day}.json.gz').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_day,day) for day in pending]):
            future.result()
    aggregate(days)


if __name__=='__main__':
    main()
