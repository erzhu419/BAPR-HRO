"""Read-only duration attribution for all V30 actions and retained V29 pairs.

The same-stop training means are descriptive references, not a fitted or
evaluated replacement model. They never affect forecasts or action selection.
"""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean

from experiments.reconstruction_v23.run_transfer_screen import TRAINING
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'


def training_station_groups(data):
    """Observed duration lists with the original identity/inclusion rules."""
    groups = defaultdict(list)
    for day in data['train_dates']:
        trips = defaultdict(list)
        for row in data['dates'][day]['connections']:
            if not row['dropoff_allowed']:
                continue
            trips[row['trip_id']].append(row)
            dep, arr = row['departure'], row['arrival']
            if (row['outcome'] == 'MEASURED' and dep.get('official_journey')
                    and dep['official_journey'] == arr.get('official_journey')):
                elapsed = arr['time_seconds']-dep['time_seconds']
                if elapsed >= 0:
                    key = (row['route_id'], 'travel',
                           row['scheduled_arrival_seconds']-row['scheduled_departure_seconds'],
                           row['dep_stop'], row['arr_stop'])
                    groups[key].append(elapsed)
        for rows in trips.values():
            rows.sort(key=lambda row: row['dep_sequence'])
            for before, after in zip(rows, rows[1:]):
                arr, dep = before['arrival'], after['departure']
                if (arr['status'] != 'REAL' or dep['status'] != 'REAL'
                        or not arr.get('official_journey')
                        or arr['official_journey'] != dep.get('official_journey')
                        or before['arr_stop'] != after['dep_stop']
                        or before['arr_sequence'] != after['dep_sequence']):
                    continue
                elapsed = dep['time_seconds']-arr['time_seconds']
                if elapsed >= 0:
                    key = (after['route_id'], 'dwell',
                           after['scheduled_departure_seconds']-before['scheduled_arrival_seconds'],
                           after['dep_stop'])
                    groups[key].append(elapsed)
    return groups


def kernel_record(fit, route, kind, scheduled):
    source = 'route'
    row = fit['route_groups'].get(route, {}).get(kind, {}).get(str(scheduled))
    if row is None:
        source, row = 'pooled', fit['groups'].get(kind, {}).get(str(scheduled))
    if row is None:
        source, row = 'residual', fit['residuals'][kind]
        values = [max(0, scheduled+value) for value in row['values']]
    else:
        values = row['values']
    return dict(source=source, n=row['n'],
                mean=sum(value*count for value, count in zip(values, row['counts']))/row['n'])


def matched_summary(rows):
    actual = [row for row in rows if row['duration_status'] == 'complete_actual_path']
    common = [row for row in actual if row['same_stop_training_mean_duration'] is not None]
    return dict(action_occurrences=len(rows), unique_roots=len({row['root_id'] for row in rows}),
        unique_trips=len({(row['day'], row['trip']) for row in rows}),
        actual_coverage=dict(Counter(row['duration_status'] for row in rows)),
        complete_actual_paths=len(actual), common_paths=len(common),
        common_unique_trips=len({(row['day'], row['trip']) for row in common}),
        missing_training_paths=len(actual)-len(common),
        minimum_station_group_n_counts=dict(Counter(
            '0' if row['same_stop_training_min_n'] == 0 else
            '1-4' if row['same_stop_training_min_n'] < 5 else
            '5-19' if row['same_stop_training_min_n'] < 20 else '20+'
            for row in actual)),
        common_scores={label: dict(bias=mean(row[key]-row['actual_duration'] for row in common),
            mae=mean(abs(row[key]-row['actual_duration']) for row in common))
            for label, key in (('route_kernel', 'kernel_mean_duration'),
                               ('same_stop_training_reference', 'same_stop_training_mean_duration'))}
            if common else {},
        common_actual_mean=mean(row['actual_duration'] for row in common) if common else None)


def main():
    with gzip.open(HERE/'results.json.gz', 'rt') as handle:
        result = json.load(handle)
    fit = json.loads((SOURCE/'model_fit.json').read_text())['duration_fit']
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    training, _ = load_city(TRAINING)
    groups = training_station_groups(training)
    histories = {row['history_id']: row for row in result['histories']}
    selected = [row for row in result['forecasts'] if row['law'] == 'route_empirical20']
    rows = []
    for day in sorted({row['day'] for row in selected}):
        graph, _, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        for forecast in (row for row in selected if row['day'] == day):
            history = histories[forecast['history_id']]
            root, route = history['root'], forecast['route']
            path = next(option['first_path'] for option in root['full_options']
                        if option['first_cid'] == forecast['cid'])
            actual = forecast['actual']
            common = dict(root_id=root['root_id'], start=root['start'], day=day,
                role=forecast['role'], tags=forecast['tags'], cid=forecast['cid'], route=route,
                trip=forecast['trip'], first_path=path,
                physical_stop_sequence=[metadata[path[0]]['dep_stop']]+[metadata[cid]['arr_stop'] for cid in path],
                public_clock=history['clock'], public_departure_delays={
                    str(cid): history['clock']-graph.connections[cid].dep_time for cid in history['visible']})
            if actual['root_departure'] is None or actual['first_arrival'] is None:
                rows.append(dict(**common, duration_status=actual['status'],
                                 actual_duration=None, missing_reason=actual['missing_reason']))
                continue
            segments = []

            def add(kind, scheduled, actual, stops):
                record = kernel_record(fit, route, kind, scheduled)
                observed = groups[(route, kind, scheduled, *stops)]
                segments.append(dict(kind=kind, scheduled=scheduled, stops=stops,
                    actual=actual, kernel=record, actual_minus_kernel=actual-record['mean'],
                    same_stop_training=dict(n=len(observed), mean=mean(observed) if observed else None)))

            for index, cid in enumerate(path):
                row = metadata[cid]
                add('travel', row['scheduled_arrival_seconds']-row['scheduled_departure_seconds'],
                    row['arrival']['time_seconds']-row['departure']['time_seconds'],
                    [row['dep_stop'], row['arr_stop']])
                if index:
                    previous = metadata[path[index-1]]
                    add('dwell', row['scheduled_departure_seconds']-previous['scheduled_arrival_seconds'],
                        row['departure']['time_seconds']-previous['arrival']['time_seconds'], [row['dep_stop']])
            station_means = [segment['same_stop_training']['mean'] for segment in segments]
            rows.append(dict(**common, duration_status='complete_actual_path',
                realized_departure_delay=actual['root_departure']-graph.connections[forecast['cid']].dep_time,
                actual_duration=actual['first_arrival']-actual['root_departure'],
                kernel_mean_duration=sum(segment['kernel']['mean'] for segment in segments),
                same_stop_training_mean_duration=(sum(station_means) if all(value is not None for value in station_means) else None),
                same_stop_training_min_n=min(segment['same_stop_training']['n'] for segment in segments),
                first_arrival_bias=(forecast['scores']['first_arrival']['bias']
                                    if forecast['scores']['first_arrival'] else None),
                remaining_wait_bias=(forecast['scores']['remaining_wait']['bias']
                                     if forecast['scores']['remaining_wait'] else None),
                segments=segments))
    summary = {}
    for role in ('board', 'target'):
        cases = [row for row in rows if row['role'] == role and 'v29_disagreement' in row['tags']]
        segments = [segment for row in cases for segment in row['segments']]
        summary[role] = dict(cases=len(cases), same_physical_stop_sequence=
            len({tuple(row['physical_stop_sequence']) for row in cases}) == 1,
            actual_duration=mean(row['actual_duration'] for row in cases),
            kernel_mean_duration=mean(row['kernel_mean_duration'] for row in cases),
            actual_minus_kernel=mean(row['actual_duration']-row['kernel_mean_duration'] for row in cases),
            first_arrival_bias=mean(row['first_arrival_bias'] for row in cases),
            sources=dict(Counter(segment['kernel']['source'] for segment in segments)),
            min_group_n=min(segment['kernel']['n'] for segment in segments),
            same_stop_training_mean_duration=mean(row['same_stop_training_mean_duration'] for row in cases),
            same_stop_training_min_n=min(segment['same_stop_training']['n'] for segment in segments),
            actual_minus_kernel_by_kind={kind: mean(sum(segment['actual_minus_kernel']
                for segment in row['segments'] if segment['kind'] == kind) for row in cases)
                for kind in ('travel', 'dwell')})
    first_offer = [row for row in rows if 'first_offer' in row['tags']]
    first_summary = {day: {role: matched_summary([row for row in first_offer
                        if row['day'] == day and row['role'] == role]) for role in ('board', 'target')}
                     for day in sorted({row['day'] for row in first_offer})}
    output = dict(scope='Descriptive duration attribution of every V30 first-offer action and all retained V29 pairs; no refit or policy change.',
        sources=dict(forecasts='experiments/reconstruction_v30/results.json.gz',
                     fit='experiments/reconstruction_v25/model_fit.json',
                     training=str(TRAINING.relative_to(HERE.parents[1])), training_dates=training['train_dates']),
        cases=rows, v29_summary=summary, first_offer_summary=first_summary,
        first_offer_total=matched_summary(first_offer),
        first_offer_by_route={route: matched_summary([row for row in first_offer if row['route'] == route])
                              for route in sorted({row['route'] for row in first_offer})},
        interpretation='In the retained V29 board/target paths, all first-path groups are route-specific '
            'with at least 116/182 observations. '
            'The broad duration grouping understates these particular paths. Same-stop training means '
            'are closer to these realized durations, making segment heterogeneity a concrete candidate; '
            'the matched groups contain as few as 3/7 records. On all first-offer paths, compare only '
            'common actual durations and completely matched training paths; retain the unmatched paths '
            'and per-path minimum group counts. This elapsed-duration comparison does not condition '
            'a new probability model, rescore full action costs or measure policy benefit. These '
            'development comparisons do not identify recovery versus segment versus time effects, validate a new conditional model, '
            'or establish a controller improvement. Target realized departure delays are outcome labels, '
            'not publicly available selection inputs.')
    (HERE/'mechanism.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(v29=summary, first_offer=first_summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
