"""Training-only descriptive residual dependence and causal selector coverage."""

from collections import Counter, defaultdict
import gzip
from itertools import combinations
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V35 = HERE.parent/'reconstruction_v35'
TRAIN = ('2023-10-01', '2023-10-02')


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else path.open()) as handle:
        return json.load(handle)


def correlation(pairs, means, weights=None):
    if len(pairs) < 2:
        return None
    values = np.array([[r['error_seconds']-means.get((r['route_id'], r['kind']), 0.)
                        for r in pair] for pair in pairs], dtype=float)
    w = np.ones(len(values)) if weights is None else np.asarray(weights)
    w /= w.sum()
    centered = values-np.sum(values*w[:, None], axis=0)
    variance = np.sum(centered**2*w[:, None], axis=0)
    return (float(np.sum(centered[:, 0]*centered[:, 1]*w)/np.sqrt(np.prod(variance)))
            if np.all(variance > 0) else None)


def pair_summary(pairs, means):
    group_counts = Counter((a['date'], a['trip_id'], a['forecast_receipt']) for a, _ in pairs)
    trip_counts = Counter((a['date'], a['trip_id']) for a, _ in pairs)
    return dict(endpoint_pairs=len(pairs), trip_receipt_groups=len(group_counts),
        trip_days=len(trip_counts), dates=sorted({a['date'] for a, _ in pairs}),
        raw_error_correlation=correlation(pairs, {}),
        route_kind_demeaned_correlation=correlation(pairs, means),
        equal_receipt_group_demeaned_correlation=correlation(pairs, means,
            [1/group_counts[a['date'], a['trip_id'], a['forecast_receipt']] for a, _ in pairs]),
        equal_trip_day_demeaned_correlation=correlation(pairs, means,
            [1/trip_counts[a['date'], a['trip_id']] for a, _ in pairs]))


def residuals(rows):
    means, groups = defaultdict(list), defaultdict(list)
    for row in rows:
        means[row['route_id'], row['kind']].append(row['error_seconds'])
        groups[row['date'], row['trip_id'], row['forecast_receipt']].append(row)
    means = {key: float(np.mean(values)) for key, values in means.items()}
    ordered = [sorted(group, key=lambda r: (r['stop_sequence'], r['kind'] == 'departure'))
               for group in groups.values()]
    pairs = [pair for group in ordered for pair in combinations(group, 2)]
    adjacent = [pair for group in ordered for pair in zip(group, group[1:])]
    ages = []
    for lower, upper in ((0, 300), (300, 600), (600, 1800), (1800, None)):
        subset = [r for r in rows if lower <= r['forecast_age'] and (upper is None or r['forecast_age'] < upper)]
        errors = np.array([r['error_seconds'] for r in subset], dtype=float)
        ages.append(dict(age_seconds=[lower, upper], endpoints=len(subset),
            trip_days=len({(r['date'], r['trip_id']) for r in subset}),
            by_day=dict(Counter(r['date'] for r in subset)),
            mean_error=float(errors.mean()) if len(errors) else None,
            mae=float(np.abs(errors).mean()) if len(errors) else None,
            sd=float(errors.std(ddof=1)) if len(errors) > 1 else None))
    return dict(endpoints=len(rows), trip_days=len({(r['date'], r['trip_id']) for r in rows}),
        receipt_groups=len(groups), receipt_groups_with_pairs=sum(len(g) >= 2 for g in ordered),
        group_size_counts=dict(Counter(len(g) for g in ordered)), all_pairs=pair_summary(pairs, means),
        adjacent_eligible_endpoint_pairs=pair_summary(adjacent, means),
        different_stop_pairs=pair_summary([(a, b) for a, b in pairs if a['stop_sequence'] != b['stop_sequence']], means),
        same_kind_pairs=pair_summary([(a, b) for a, b in pairs if a['kind'] == b['kind']], means),
        by_day={day: dict(endpoints=sum(r['date'] == day for r in rows),
            **pair_summary([(a, b) for a, b in pairs if a['date'] == day], means)) for day in TRAIN},
        age_groups=ages, pair_sources=[dict(date=a['date'], trip=a['trip_id'], receipt=a['forecast_receipt'],
            left=[a['source_connection_id'], a['kind']], right=[b['source_connection_id'], b['kind']],
            left_error=a['error_seconds'], right_error=b['error_seconds']) for a, b in pairs],
        centering='Subtract each route/kind training-sample mean for description only. No predictive parameter is changed.',
        interpretation='These are unconditional actual-minus-forecast residual correlations. Shared trip/day shocks, '
            'overlapping states and forecast construction can induce them. They do not directly test or refute '
            'conditional independence of forecast messages given the entire latent physical trajectory. '
            'Pairs share endpoints/trips; counts are support descriptions, not independent sample sizes.')


def selected_probe(history, trip, clock, model, metadata, target):
    chosen = history.selected_forecast(trip)
    raw = history.raw_forecast_records(trip)
    last = max((model._positions[r[0], r[1]][1] for r in history._records.get(trip, ())
                if r[2] == 'observed'), default=-1)
    candidates = [r for r in raw if model._positions[r[0], r[1]][1] > last]
    expected = min(candidates, key=lambda r: (-r[4], model._positions[r[0], r[1]][1]), default=None)
    result = dict(target=target, clock=clock, trip=trip, raw_forecasts=len(raw),
                  remaining_candidates=len(candidates), latest_observed_slot=last,
                  selector_matches_frozen_rule=chosen == expected, selected=chosen)
    if chosen is None:
        return result
    cid, kind, prediction, _, receipt = chosen
    scheduled = metadata[cid][f'scheduled_{kind}_seconds']
    result.update(source_connection_id=metadata[cid]['id'], kind=kind,
        physical_slot=model._positions[cid, kind][1], scheduled=scheduled, predicted=prediction,
        receipt=receipt, forecast_age=clock-receipt, scheduled_at_or_before_clock=scheduled <= clock,
        scheduled_strictly_before_clock=scheduled < clock,
        predicted_at_or_before_clock=prediction <= clock,
        uses_only_available_receipt=receipt <= clock,
        follows_observed_prefix=model._positions[cid, kind][1] > last)
    return result


def probe_summary(rows):
    active = [r for r in rows if r['selected'] is not None]
    count = sum(r['scheduled_at_or_before_clock'] for r in active)
    strictly_past = sum(r['scheduled_strictly_before_clock'] for r in active)
    return dict(queries=len(rows), queries_with_selection=len(active),
        distinct_trip_days=len({(r['date'], r['trip']) for r in active}),
        scheduled_at_or_before_clock=count, proportion_among_selected=count/len(active) if active else None,
        scheduled_strictly_before_clock=strictly_past,
        strictly_past_proportion=strictly_past/len(active) if active else None,
        predicted_at_or_before_clock=sum(r['predicted_at_or_before_clock'] for r in active),
        all_receipts_available=all(r['uses_only_available_receipt'] for r in active),
        all_follow_observed_prefix=all(r['follows_observed_prefix'] for r in active),
        all_match_frozen_selector=all(r['selector_matches_frozen_rule'] for r in rows))


def panel_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['date'], row['trip']].append(row)
    return dict(summary=probe_summary(rows),
        by_day={day: probe_summary([r for r in rows if r['date'] == day]) for day in TRAIN},
        by_trip_day=[dict(date=day, trip=trip, **probe_summary(group))
                     for (day, trip), group in sorted(groups.items())],
        first_past_scheduled_examples=[r for r in rows if r.get('scheduled_strictly_before_clock')][:8],
        source='All queries are reproducible from the existing V35 event/message tables; examples use encounter order.')


def selector_audit(events, messages):
    from experiments.reconstruction_v4.run_dated_replay import dated_graph
    from experiments.reconstruction_v34.run import setup
    from experiments.reconstruction_v36.forecast import SingleMessageHistory

    data = read(ROOT/'experiments/reconstruction_data_audit/rail_calibration_dates_v2.json.gz')
    data['station_transfer_bounds'] = read(ROOT/'experiments/reconstruction_data_audit/station_transfer_bounds.json')['stations']
    physical_fit = read(HERE.parent/'reconstruction_v34/model_fit.json')
    fitted = read(V35/'fitted_emission.json')
    endpoint_panel, onboard_panel, unmapped = [], [], []
    for day in TRAIN:
        graph, _, _, mapping, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        model, parameters = setup(graph, physical_fit)
        frames = messages['dates'][day]['frames']
        history = SingleMessageHistory(model, parameters, fit=fitted, frames=frames, source_to_local=mapping)
        queries = sorted((r for r in events if r['date'] == day and r['actual_status'] == 'REAL'
                          and r['cutoff_before_actual']), key=lambda r: r['prediction_clock'])
        for event in queries:
            cid = mapping.get(event['source_connection_id'])
            if cid is None:
                unmapped.append([day, event['source_connection_id'], event['kind']])
                continue
            history.advance(event['prediction_clock'])
            trip = model._positions[cid, event['kind']][0]
            endpoint_panel.append(dict(date=day, **selected_probe(history, trip, event['prediction_clock'],
                model, metadata, [event['source_connection_id'], event['kind']])))
        by_trip = defaultdict(list)
        for cid, c in metadata.items():
            by_trip[c['trip_id']].append((cid, c))
        incoming = []
        for trip, connections in by_trip.items():
            connections.sort(key=lambda item: item[1]['dep_sequence'])
            for (cid, before), (nid, after) in zip(connections, connections[1:]):
                if before['arr_sequence'] != after['dep_sequence']:
                    continue
                endpoints = [c[kind] for c in (before, after) for kind in ('departure', 'arrival')]
                if any(e['status'] != 'REAL' for e in endpoints):
                    continue
                clocks = [e['time_seconds'] for e in endpoints]
                if clocks != sorted(clocks) or len({e['official_journey'] for e in endpoints}) != 1:
                    continue
                incoming.append((clocks[1], cid, nid, trip))
        base = SingleMessageHistory(model, parameters, fit=fitted, frames=frames, source_to_local=mapping)
        for clock, cid, nid, trip in sorted(incoming):
            base.advance(clock)
            history = base.clone()
            history.observe(cid, 'departure', metadata[cid]['departure']['time_seconds'])
            history.observe(cid, 'arrival', clock)
            onboard_panel.append(dict(date=day, incoming_source=metadata[cid]['id'],
                **selected_probe(history, trip, clock, model, metadata, [metadata[nid]['id'], 'arrival'])))
    return dict(rule='Per physical trip, remove all forecast slots at or before its latest hard-observed slot. '
            'Among remaining slots choose latest received time, then earliest physical slot. Without hard '
            'observations all slots remain eligible. No scheduled/predicted-time filter is applied.',
        endpoint_panel=panel_summary(endpoint_panel), onboard_panel=panel_summary(onboard_panel),
        unmapped_endpoint_queries=unmapped,
        assessment='The selector uses received messages, known static slot order and hard public observations only, '
            'so it is implementable at the decision time. It is a one-message likelihood approximation with '
            'the same full public record, not a conditional-independence model or an optimal selector. '
            'A selected scheduled time in the past can be a delayed '
            'unobserved service; that condition is not itself a causality error.')


def main():
    events = read(V35/'events.json.gz')['rows']
    training = [r for r in events if r['date'] in TRAIN and r['error_eligible']]
    result = dict(training_dates=TRAIN, previously_developed=True, policy_results_read=False,
        newly_fitted_parameters=False, server_reads=0,
        residual_dependence=residuals(training), selector=selector_audit(events, read(V35/'messages.json.gz')))
    (HERE/'correlation_diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print('residuals', result['residual_dependence']['all_pairs'])
    print('endpoint selector', result['selector']['endpoint_panel']['summary'])
    print('onboard selector', result['selector']['onboard_panel']['summary'])


if __name__ == '__main__':
    main()
