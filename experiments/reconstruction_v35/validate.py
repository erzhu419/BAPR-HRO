"""Score the frozen message likelihood on disjoint-date endpoint/ride queries.

This evaluates physical posterior laws, without running a routing control.
Clock scores and the Brier event are conditional on the target operating;
cancellation probability is retained separately.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.forecast import MessageHistory
from experiments.reconstruction_v35.forecast import FittedMessageHistory
from src.evolving_trip_belief import EvolvingTripBelief

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent/'reconstruction_data_audit'
MODELS = ('own_only', 'legacy_messages', 'fitted_messages')
METRICS = ('CRPS_seconds', 'MAE_seconds', 'central_90_coverage',
           'central_90_width_seconds', 'conditional_operating_Brier', 'cancel_probability')


def write(path, value):
    with (gzip.open(path, 'wt') if str(path).endswith('.gz') else open(path, 'w')) as handle:
        json.dump(value, handle, indent=None if str(path).endswith('.gz') else 2, allow_nan=False)


def distribution_scores(clocks, probabilities, actual, *, cancel_probability=0., threshold=None):
    clocks, probabilities = np.asarray(clocks, float), np.asarray(probabilities, float)
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(clocks)):
        raise FloatingPointError('Nonfinite posterior must be diagnosed before evaluating the panel')
    positive = probabilities > 0
    clocks, probabilities = clocks[positive], probabilities[positive]
    if not len(clocks) or probabilities.sum() <= 0:
        raise UnsupportedHistory('No operating posterior mass for the scored REAL target')
    order = np.argsort(clocks)
    clocks, probabilities = clocks[order], probabilities[order]
    probabilities = probabilities/probabilities.sum()
    cumulative = probabilities.cumsum()
    left_mass = np.r_[0., cumulative[:-1]]
    left_moment = np.r_[0., np.cumsum(probabilities*clocks)[:-1]]
    half_pair_distance = float(np.sum(probabilities*(clocks*left_mass-left_moment)))
    crps = float(np.sum(probabilities*np.abs(clocks-actual))-half_pair_distance)
    quantile = lambda q: float(clocks[min(int(np.searchsorted(cumulative, q)), len(clocks)-1)])
    lower, upper = quantile(.05), quantile(.95)
    mean = float(probabilities@clocks)
    result = dict(status='ok', clocks_seconds=clocks.tolist(),
        conditional_operating_probabilities=probabilities.tolist(), predicted_mean_seconds=mean,
        lower_05_seconds=lower, upper_95_seconds=upper, cancel_probability=float(cancel_probability),
        CRPS_seconds=crps, MAE_seconds=abs(mean-actual),
        central_90_coverage=float(lower <= actual <= upper), central_90_width_seconds=upper-lower)
    if threshold is not None:
        probability = float(probabilities[clocks <= threshold].sum())
        truth = float(actual <= threshold)
        result.update(threshold_seconds=threshold, conditional_operating_probability=probability,
                      event_actual=truth, conditional_operating_Brier=(probability-truth)**2)
    return result


def query_history(master, query):
    master.advance(query['clock'])
    history = master.clone() if query['own_observations'] else master
    for observation in query['own_observations']:
        if observation['clock'] > query['clock']:
            raise ValueError('A future own endpoint cannot condition this query')
        history.observe(observation['cid'], observation['kind'], observation['clock'])
    return history


def predict(history, query):
    cid, kind = query['cid'], query['kind']
    if cid is None or any(o['cid'] is None for o in query['own_observations']):
        return dict(status='unsupported', reason='Endpoint omitted from the fixed physical model graph')
    try:
        trip, slot = history.belief._positions[cid, kind]
        law = history.law(trip)
        marginal = law.marginal(slot)
        operating = ~law.canceled[slot]
        return distribution_scores(law.times[slot][operating], marginal[operating], query['actual'],
            cancel_probability=marginal[~operating].sum(), threshold=query.get('threshold'))
    except UnsupportedHistory as error:
        return dict(status='unsupported', reason=str(error))


def make_queries(day, connections, mapping, endpoint_events):
    panels = {'endpoint': [], 'onboard': []}
    counts = {'endpoint': Counter(), 'onboard': Counter()}
    source = {c['id']: c for c in connections}
    for event in endpoint_events:
        counts['endpoint']['source_endpoints'] += 1
        if event['actual_status'] != 'REAL':
            counts['endpoint']['non_REAL_endpoint'] += 1
            continue
        if event['actual_seconds'] <= event['prediction_clock']:
            counts['endpoint']['actual_not_after_query'] += 1
            continue
        cid, kind = event['source_connection_id'], event['kind']
        panels['endpoint'].append(dict(query_id=f'{day}:endpoint:{cid}:{kind}', day=day,
            panel='endpoint', source_cid=cid, cid=mapping.get(cid), kind=kind,
            trip_id=event['trip_id'], route_id=event['route_id'], clock=event['prediction_clock'],
            scheduled=event['scheduled'], actual=event['actual_seconds'], own_observations=[]))
    trips = defaultdict(list)
    for c in connections:
        trips[c['trip_id']].append(c)
    for rows in trips.values():
        rows.sort(key=lambda c: (c['dep_sequence'], c['id']))
        for first, following in zip(rows, rows[1:]):
            counts['onboard']['neighboring_source_rows'] += 1
            if (first['arr_stop'] != following['dep_stop'] or
                    first['arr_sequence'] != following['dep_sequence']):
                counts['onboard']['not_physically_contiguous'] += 1
                continue
            endpoints = [first['departure'], first['arrival'], following['departure'], following['arrival']]
            if any(e['status'] != 'REAL' for e in endpoints):
                counts['onboard']['non_REAL_prefix_or_target'] += 1
                continue
            journeys = {e['official_journey'] for e in endpoints}
            if None in journeys or len(journeys) != 1:
                counts['onboard']['inconsistent_vehicle_identity'] += 1
                continue
            clocks = [e['time_seconds'] for e in endpoints]
            if clocks != sorted(clocks):
                counts['onboard']['inconsistent_chronology'] += 1
                continue
            panels['onboard'].append(dict(query_id=f'{day}:onboard:{first["id"]}:{following["id"]}',
                day=day, panel='onboard', source_cid=following['id'], cid=mapping.get(following['id']),
                incoming_source_cid=first['id'], kind='arrival', trip_id=following['trip_id'],
                route_id=following['route_id'], clock=clocks[1], actual=clocks[3],
                scheduled=following['scheduled_arrival_seconds'],
                threshold=following['scheduled_arrival_seconds']+300,
                own_observations=[dict(cid=mapping.get(first['id']), kind='departure', clock=clocks[0]),
                                  dict(cid=mapping.get(first['id']), kind='arrival', clock=clocks[1])]))
    for panel, rows in panels.items():
        rows.sort(key=lambda r: (r['clock'], r['query_id']))
        counts[panel]['eligible_queries'] = len(rows)
        counts[panel]['unmapped_queries'] = sum(r['cid'] is None or
            any(o['cid'] is None for o in r['own_observations']) for r in rows)
    return panels, {panel: dict(value) for panel, value in counts.items()}


def metric_means(rows, model):
    return {metric: float(np.mean([r['predictions'][model][metric] for r in rows
                                  if metric in r['predictions'][model]]))
            for metric in METRICS if any(metric in r['predictions'][model] for r in rows)}


def summarize_rows(rows):
    common = [r for r in rows if all(r['predictions'][m]['status']=='ok' for m in MODELS)]
    forecast_common = [r for r in common if r['target_has_forecast']]
    result = dict(eligible_queries=len(rows), common_supported=len(common),
        target_forecast_queries=sum(r['target_has_forecast'] for r in rows),
        common_supported_target_forecast=len(forecast_common), models={})
    for model in MODELS:
        supported = [r for r in rows if r['predictions'][model]['status']=='ok']
        result['models'][model] = dict(supported=len(supported), unsupported=len(rows)-len(supported),
            common_metrics=metric_means(common, model), own_supported_metrics=metric_means(supported, model),
            common_target_forecast_metrics=metric_means(forecast_common, model))
    result['fitted_new_unsupported_vs_legacy'] = sum(r['predictions']['fitted_messages']['status']!='ok'
        and r['predictions']['legacy_messages']['status']=='ok' for r in rows)
    result['fitted_new_unsupported_vs_own'] = sum(r['predictions']['fitted_messages']['status']!='ok'
        and r['predictions']['own_only']['status']=='ok' for r in rows)
    return result


def describe_onboard_messages(saved):
    """Read saved predictions only; these strata do not change the frozen gate."""
    subsets = {day:[r for r in d['panels']['onboard'] if r['target_has_forecast'] and
        all(r['predictions'][m]['status']=='ok' for m in MODELS)] for day,d in saved.items()}
    rows = [r for subset in subsets.values() for r in subset]
    ages = [r['target_forecast']['age_seconds'] for r in rows]
    counts = [r['trip_active_forecast_count'] for r in rows]
    quantiles = lambda xs: dict(zip(('min','p10','p25','median','p75','p90','max'),
        map(float,np.quantile(xs,[0,.1,.25,.5,.75,.9,1])))) if xs else {}
    training = [r for r in read(HERE/'events.json.gz')['rows']
                if r['split']=='training' and r['error_eligible']]
    by_day = {}
    for day, subset in subsets.items():
        metrics = {m:metric_means(subset,m) for m in MODELS}
        if not subset:
            by_day[day] = dict(common_target_forecast_queries=0)
            continue
        f,l = metrics['fitted_messages'],metrics['legacy_messages']
        by_day[day] = dict(common_target_forecast_queries=len(subset),models=metrics,
            fitted_vs_legacy=dict(CRPS_gain_seconds=l['CRPS_seconds']-f['CRPS_seconds'],
                Brier_gain=l['conditional_operating_Brier']-f['conditional_operating_Brier'],
                coverage_increase=f['central_90_coverage']-l['central_90_coverage']))
    signs = {}
    for metric in ('CRPS_gain_seconds','Brier_gain','coverage_increase'):
        values = [d['fitted_vs_legacy'][metric] for d in by_day.values() if 'fitted_vs_legacy' in d]
        signs[metric] = dict(positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),
                            tied=sum(v==0 for v in values))
    return dict(scope='Descriptive target-has-forecast stratum of saved onboard rows; no new model or gate.',
        common_target_forecast_queries=len(rows),
        training_error_eligible_queries=len(training),
        training_error_eligible_forecast_age_seconds=quantiles([r['forecast_age'] for r in training]),
        target_forecast_age_seconds=quantiles(ages),
        active_same_trip_forecast_count=quantiles(counts),
        active_same_trip_forecast_count_histogram=dict(sorted(Counter(counts).items())),
        by_day=by_day, fitted_vs_legacy_day_signs=signs,
        interpretation='The overall macro coverage can move farther above 90% while the forecast-present '
            'subset moves upward toward 90% from below. The frozen overall gate remains unchanged; '
            'neither a formal gate failure nor this descriptive subset proves uniform deterioration '
            'or successful joint calibration. Query exposures are clustered within trips and dates.')


def run_day(day):
    began = perf_counter()
    data = read(AUDIT/'rail_calibration_dates_v2.json.gz')
    events = [r for r in read(HERE/'events.json.gz')['rows'] if r['date']==day]
    frames = read(HERE/'message_days'/f'{day}.json.gz')['frames']
    physical_fit = read(HERE.parent/'reconstruction_v5/calibration_frozen_fit.json')['evolving_fit']
    fitted = read(HERE/'fitted_emission.json')
    graph, _, _, mapping, metadata, _ = context(data, day)
    model = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **physical_fit['increments'])
    parameters = {c.route:physical_fit['route_priors'].get(c.route,physical_fit['route_priors']['pooled'])
                  for c in graph.connections}
    panels, counts = make_queries(day, data['dates'][day]['connections'], mapping, events)
    summaries = {}
    for panel, rows in panels.items():
        histories = dict(own_only=MessageHistory(model, parameters),
            legacy_messages=MessageHistory(model, parameters, frames=frames, source_to_local=mapping),
            fitted_messages=FittedMessageHistory(model, parameters, fit=fitted, frames=frames, source_to_local=mapping))
        for row in rows:
            row['predictions'] = {}
            for name, master in histories.items():
                # Unmapped physical slots remain explicit failures in every model.
                if row['cid'] is None or any(o['cid'] is None for o in row['own_observations']):
                    master.advance(row['clock'])
                    history = master
                else:
                    history = query_history(master, row)
                row['predictions'][name] = predict(history, row)
                if name=='legacy_messages':
                    current = history._forecasts.get((row['cid'], row['kind']))
                    row['target_has_forecast'] = current is not None
                    row['target_forecast'] = (dict(predicted=current[0], receipt=current[2],
                        age_seconds=row['clock']-current[2]) if current is not None else None)
                    row['trip_active_forecast_count'] = len(history.forecast_records(row['trip_id']))
                    row['public_frame_count'] = history.frame_index
        summaries[panel] = summarize_rows(rows)
        print(day, panel, json.dumps({k:v for k,v in summaries[panel].items() if k!='models'}), flush=True)
    output = dict(day=day, source_eligibility=counts, panels=panels, summary=summaries, wall=perf_counter()-began)
    write(HERE/'validation_days'/f'{day}.json.gz', output)
    print(day, 'completed', round(output['wall'], 3), flush=True)
    return day


def aggregate(days):
    saved = {day:read(HERE/'validation_days'/f'{day}.json.gz') for day in days}
    panels = {}
    for panel in ('endpoint', 'onboard'):
        total = summarize_rows([r for d in saved.values() for r in d['panels'][panel]])
        macro = {}
        for model in MODELS:
            macro[model] = {}
            for label in ('common_metrics', 'common_target_forecast_metrics'):
                per_day = [d['summary'][panel]['models'][model][label] for d in saved.values()]
                macro[model][label] = {metric:float(np.mean([v[metric] for v in per_day if metric in v]))
                    for metric in METRICS if any(metric in v for v in per_day)}
                macro[model][label+'_contributing_dates'] = sum(bool(v) for v in per_day)
        total['equal_date_macro'] = macro
        panels[panel] = total
    b = panels['onboard']['equal_date_macro']
    f, l, own = (b[m]['common_metrics'] for m in ('fitted_messages','legacy_messages','own_only'))
    criteria = dict(no_increased_unsupported_in_each_panel=all(
        p['models']['fitted_messages']['unsupported']<=p['models']['legacy_messages']['unsupported']
        for p in panels.values()),
        onboard_CRPS_below_legacy=f['CRPS_seconds']<l['CRPS_seconds'],
        onboard_CRPS_no_worse_than_own=f['CRPS_seconds']<=own['CRPS_seconds'],
        onboard_Brier_no_worse_than_legacy=f['conditional_operating_Brier']<=l['conditional_operating_Brier'],
        onboard_coverage_deviation_no_worse_than_legacy=
            abs(f['central_90_coverage']-.9)<=abs(l['central_90_coverage']-.9))
    summary = dict(scope='Frozen 3–11 October development validation; no controls or V34 loss used.',
        training_dates=read(HERE/'fitted_emission.json')['training_dates'], validation_dates=days,
        observation_scope='Onboard queries condition only the first leg departure/arrival and receipts '
            'available at that arrival. The target realization is used only for scoring.',
        score_scope='CRPS, intervals and Brier use the operating-clock law. Cancellation mass is separate. '
            'Brier is P(next arrival<=scheduled arrival+300 | operating,current public history).',
        panels=panels, by_day={day:dict(summary=d['summary'], source_eligibility=d['source_eligibility'],
            wall=d['wall']) for day,d in saved.items()},
        gate=dict(passed=all(criteria.values()), criteria=criteria,
            scope='Development model-retention rule only; not G2 or independent confirmation.'),
        descriptive=describe_onboard_messages(saved),
        row_files=[f'validation_days/{day}.json.gz' for day in days])
    write(HERE/'validation_summary.json', summary)
    print('GATE',json.dumps(summary['gate']),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    protocol, fitted = read(HERE/'protocol.json'), read(HERE/'fitted_emission.json')
    if fitted['training_dates'] != protocol['training_dates']:
        raise ValueError('The frozen emission fit must use only the declared training dates')
    days = protocol['validation_dates']
    (HERE/'validation_days').mkdir(exist_ok=True)
    pending = [day for day in days if not (HERE/'validation_days'/f'{day}.json.gz').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_day, day) for day in pending]):
            future.result()
    aggregate(days)


if __name__ == '__main__':
    main()
