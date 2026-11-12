"""Reproduce the training-only elapsed/previous-delay dependence description."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.reconstruction_v23.run_transfer_screen import TRAINING
from experiments.reconstruction_v4.run_city_replay import load_city


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MIN_GROUP = 20


def elapsed_records(data):
    """Use the exact duration-fit eligibility, retaining prior endpoint delay."""
    records, excluded = [], Counter()

    def add(row, before, after, kind, scheduled_before, scheduled_after):
        left, right = before.get('official_journey'), after.get('official_journey')
        if not left or not right or left != right:
            excluded[kind+'_unmatched_vehicle_identity'] += 1
            return
        elapsed = after['time_seconds']-before['time_seconds']
        if elapsed < 0:
            excluded[kind+'_negative_elapsed'] += 1
            return
        records.append(dict(day=day, route=row['route_id'], route_name=row['route_name'],
            trip=row['trip_id'], official_journey=left, kind=kind,
            scheduled=scheduled_after-scheduled_before, elapsed=elapsed,
            prior_delay=before['time_seconds']-scheduled_before))

    for day in data['train_dates']:
        trips = defaultdict(list)
        for row in data['dates'][day]['connections']:
            if not row['dropoff_allowed']:
                continue
            trips[row['trip_id']].append(row)
            if row['outcome'] == 'MEASURED':
                add(row, row['departure'], row['arrival'], 'travel',
                    row['scheduled_departure_seconds'], row['scheduled_arrival_seconds'])
        for rows in trips.values():
            rows.sort(key=lambda row: row['dep_sequence'])
            for before, after in zip(rows, rows[1:]):
                if before['arrival']['status'] != 'REAL' or after['departure']['status'] != 'REAL':
                    continue
                if before['arr_stop'] != after['dep_stop']:
                    kind = 'gap'
                elif before['arr_sequence'] == after['dep_sequence']:
                    kind = 'dwell'
                else:
                    excluded['nonadjacent_same_stop'] += 1
                    continue
                add(after, before['arrival'], after['departure'], kind,
                    before['scheduled_arrival_seconds'], after['scheduled_departure_seconds'])
    return records, dict(excluded)


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x-x.mean(), y-y.mean()
    denominator = float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(x@y/denominator) if denominator else None


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['route'], row['kind'], row['scheduled']].append(row)
    included = [rows for rows in groups.values() if len(rows) >= MIN_GROUP]
    rows = [r for group in included for r in group]
    if not rows:
        return dict(eligible_rows=len(records), included_rows=0, included_groups=0,
                    within_group_correlation=None)
    x, y = [], []
    for group in included:
        delays = np.asarray([r['prior_delay'] for r in group], dtype=float)
        elapsed = np.asarray([r['elapsed'] for r in group], dtype=float)
        x.extend(delays-delays.mean())
        y.extend(elapsed-elapsed.mean())
    return dict(eligible_rows=len(records), included_rows=len(rows),
        included_groups=len(included), excluded_small_group_rows=len(records)-len(rows),
        trips=len({(r['day'], r['trip']) for r in rows}),
        official_journeys=len({(r['day'], r['official_journey']) for r in rows}),
        within_group_correlation=correlation(x, y),
        raw_correlation=correlation([r['prior_delay'] for r in rows], [r['elapsed'] for r in rows]),
        group_sizes=[dict(route=group[0]['route'], kind=group[0]['kind'],
                         scheduled=group[0]['scheduled'], n=len(group)) for group in included])


def main():
    data, _ = load_city(TRAINING)
    frozen = json.loads((HERE.parent/'reconstruction_v25/model_fit.json').read_text())['duration_fit']
    records, excluded = elapsed_records(data)
    # A different source/eligibility would invalidate attribution to the frozen model.
    counts = defaultdict(Counter)
    for row in records:
        counts[row['route'], row['kind'], row['scheduled']][row['elapsed']] += 1
    expected = {(route, kind, int(scheduled)): Counter(dict(zip(group['values'], group['counts'])))
                for route, kinds in frozen['route_groups'].items()
                for kind, schedules in kinds.items() for scheduled, group in schedules.items()}
    if dict(counts) != expected or data['train_dates'] != frozen['train_dates']:
        raise AssertionError('Probe records do not reproduce the frozen duration-fit atoms')
    by_route = {name: {kind: summarize([r for r in records if r['route_name'] == name and r['kind'] == kind])
                       for kind in ('travel', 'dwell', 'gap')}
                for name in ('2', '9', '32')}
    result = dict(source=str(TRAINING.relative_to(ROOT)), train_dates=data['train_dates'],
        grouping='route_id, event kind, scheduled duration; both variables centered within each group',
        prior_delay='actual previous endpoint minus GTFS scheduled previous endpoint, in seconds',
        elapsed='actual next endpoint minus actual previous endpoint, in seconds',
        minimum_group_rows=MIN_GROUP, frozen_fit_atoms_reproduced=True,
        total_rows=len(records), total_groups=len(counts), excluded=excluded,
        by_route_name=by_route, report_route9_values=dict(travel=-.246, dwell=-.254),
        route9_rounds_to_report={kind: round(by_route['9'][kind]['within_group_correlation'], 3) == value
            for kind, value in [('travel', -.246), ('dwell', -.254)]},
        interpretation='Training-only dependence description, no fitting or independent confirmation. '
            'Prior delay and elapsed share the previous actual endpoint with opposite signs; '
            'measurement error or endpoint matching noise can induce negative correlation. '
            'Within-group centering does not remove trip, stop-segment, or time-of-day dependence; '
            'this coefficient alone does not identify recovery or explain the six V29 outcomes.')
    (HERE/'train_probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(frozen_fit_atoms_reproduced=True,
        route9={kind: {key: by_route['9'][kind][key] for key in
            ('included_rows', 'trips', 'included_groups', 'within_group_correlation')}
            for kind in ('travel', 'dwell')},
        rounds_to_report=result['route9_rounds_to_report']), indent=2))


if __name__ == '__main__':
    main()
