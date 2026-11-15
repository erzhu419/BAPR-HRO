"""Apply the frozen fourteen-date calibration screen, including unknowns.

``summarize(events, manifest)`` consumes already registered event rows and a
date manifest. Each row has day, incoming_trip, outgoing_trip, model and actual
(the existing P0/label_event dictionaries), plus optional frozen groups.
Manifest days declare available, denominator_known and expected_opportunities.
Missing dates or unknown denominators always leave the primary verdict unresolved.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from experiments.decision_gap_v1.panel import read, write

PARTITIONS = ('00', '01', '10', '11')
TARGET_DAYS, REPLICATES, SEED = 14, 20000, 20261115
MARGIN, LOWER_Q, UPPER_Q = .02, .00625, .99375
GROUPS = ('forecast_age', 'mct_difference', 'downstream_options')


def event_bounds(row):
    """Four indicator residuals and one four-class Brier score, never imputed."""
    model, actual = row.get('model', {}), row.get('actual', {})
    known_model = model.get('status') == 'resolved'
    if known_model:
        p = np.array([model['partitions'][key] for key in PARTITIONS], dtype=float)
        if not np.isfinite(p).all() or p.min() < -1e-9 or p.max() > 1+1e-9 or abs(p.sum()-1) > 1e-9:
            raise ValueError('Resolved P0 probabilities are not a four-class distribution')
        probability = np.column_stack((p, p))
    else:
        probability = np.tile([0., 1.], (4, 1))
    possible = actual.get('possible_partitions')
    if possible is None:
        if actual.get('actual_partition') in PARTITIONS:
            possible = [actual['actual_partition']]
        elif 'partition_bounds' in actual:
            possible = [key for key in PARTITIONS if actual['partition_bounds'][key][1] > 0]
        else:
            possible = list(PARTITIONS)
    possible = set(possible)
    if not possible or not possible.issubset(PARTITIONS):
        raise ValueError('Actual label has no feasible joint partition')
    indicator = np.array([[float(possible == {key}), float(key in possible)] for key in PARTITIONS])
    residual = np.column_stack((indicator[:, 0]-probability[:, 1], indicator[:, 1]-probability[:, 0]))
    if known_model:
        scores = [float(np.square(p-np.array([float(k == key) for k in PARTITIONS])).sum())
                  for key in possible]
        brier = np.array([min(scores), max(scores)])
    else:
        brier = np.array([0., 2.])  # A probability vector is unknown on the simplex.
    return dict(residual=residual, probability=probability, indicator=indicator,
                brier=brier, model_resolved=known_model, label_resolved=len(possible) == 1)


def incidents(rows):
    """Connected components join a trip in either role within one date."""
    parent = {}

    def find(trip):
        parent.setdefault(trip, trip)
        while parent[trip] != trip:
            parent[trip] = parent[parent[trip]]
            trip = parent[trip]
        return trip

    for row in rows:
        left, right = find(row['incoming_trip']), find(row['outgoing_trip'])
        parent[right] = left
    components = defaultdict(list)
    for i,row in enumerate(rows):
        components[find(row['incoming_trip'])].append(i)
    return sorted(components.values(), key=lambda indices:indices[0])


def average(values):
    return {key:np.mean([value[key] for value in values], axis=0)
            for key in ('residual', 'probability', 'indicator', 'brier')}


def described(value):
    return dict(bias_bounds={key:value['residual'][i].tolist() for i,key in enumerate(PARTITIONS)},
        predicted_probability_bounds={key:value['probability'][i].tolist() for i,key in enumerate(PARTITIONS)},
        observed_frequency_bounds={key:value['indicator'][i].tolist() for i,key in enumerate(PARTITIONS)},
        four_class_brier_bounds=value['brier'].tolist())


def unknown_day():
    return dict(residual=np.tile([-1., 1.], (4, 1)), probability=np.tile([0., 1.], (4, 1)),
                indicator=np.tile([0., 1.], (4, 1)), brier=np.array([0., 2.]))


def partition_verdict(low, high):
    if low > MARGIN or high < -MARGIN:
        return 'fail'
    if low >= -MARGIN and high <= MARGIN:
        return 'pass'
    return 'unresolved'


def summarize(events, manifest):
    dates = manifest['selected_dates']
    if len(dates) != len(set(dates)):
        raise ValueError('A date is repeated in the frozen manifest')
    by_day = {day:[] for day in dates}
    for row in events:
        if row['day'] not in by_day:
            raise ValueError('An event falls outside the frozen date panel')
        by_day[row['day']].append(row)
    metadata = {row['day']:row for row in manifest.get('days', [])}
    days, values, incomplete = [], [], []
    for day in dates:
        rows, declared = by_day[day], metadata.get(day, {})
        reasons = []
        if not declared.get('available', False):
            reasons.append('date_unavailable_or_unreported')
        if not declared.get('denominator_known', False):
            reasons.append('unknown_opportunity_denominator')
        expected = declared.get('expected_opportunities')
        if expected is None:
            reasons.append('unregistered_opportunity_count')
        elif expected != len(rows):
            reasons.append('event_rows_do_not_cover_registered_count')
        if not rows or expected == 0:
            reasons.append('no_structural_opportunity')
        bounds = [event_bounds(row) for row in rows]
        components = incidents(rows)
        incident_values = [average([bounds[i] for i in indices]) for indices in components]
        rows_complete = (declared.get('denominator_known', False)
                         and expected is not None and expected == len(rows))
        # A missing message archive blocks admission, but does not erase known
        # labels on a complete registered denominator. Their event bounds still
        # use unknown P0 in [0,1], exactly as the frozen missingness rule states.
        value = average(incident_values) if incident_values and rows_complete else unknown_day()
        values.append(value)
        if reasons:
            incomplete.append(day)
        days.append(dict(day=day, weekday=date.fromisoformat(day).strftime('%A'),
            available=declared.get('available', False), denominator_known=declared.get('denominator_known', False),
            expected_opportunities=expected, represented_event_rows=len(rows),
            model_resolved=sum(x['model_resolved'] for x in bounds),
            label_resolved=sum(x['label_resolved'] for x in bounds),
            status='unresolved' if reasons else 'represented', unresolved_reasons=reasons,
            incidents=[dict(index=i, event_row_indices=indices,
                trips=sorted({rows[j][role] for j in indices for role in ('incoming_trip','outgoing_trip')}, key=str),
                **described(incident_values[i])) for i,indices in enumerate(components)],
            **described(value)))
    admission = []
    if not manifest.get('dates_frozen', False):
        admission.append('date_panel_not_frozen')
    if len(dates) != TARGET_DAYS:
        admission.append('fourteen_dates_required')
    if incomplete:
        admission.append('missing_date_or_unknown_denominator')
    partition_results = {}
    if values:
        bias = np.array([value['residual'] for value in values])
        draws = np.random.default_rng(SEED).integers(0, len(dates), size=(REPLICATES,len(dates)))
        bootstrap = bias[draws].mean(axis=1)
        lower = np.quantile(bootstrap[:, :, 0], LOWER_Q, axis=0)
        upper = np.quantile(bootstrap[:, :, 1], UPPER_Q, axis=0)
        partition_results = {key:dict(interval=[float(lower[i]),float(upper[i])],
            interval_classification=partition_verdict(lower[i],upper[i])) for i,key in enumerate(PARTITIONS)}
        cohort = described(average(values))
    else:
        cohort = described(unknown_day())
    classifications = [row['interval_classification'] for row in partition_results.values()]
    verdict = ('unresolved' if admission else 'fail' if 'fail' in classifications else
               'pass' if classifications and all(x == 'pass' for x in classifications) else 'unresolved')
    represented = [event_bounds(row) for row in events]
    opportunity = dict(represented_event_rows=len(events),
        scope='Represented event rows only; full population claim requires complete registered day denominators',
        complete_registered_denominator=all(metadata.get(day, {}).get('denominator_known', False)
            and metadata.get(day, {}).get('expected_opportunities') == len(by_day[day]) for day in dates),
        primary_admission_satisfied=not admission,
        **(described(average(represented)) if represented else described(unknown_day())))
    groups = {}
    for name in GROUPS:
        labels = sorted({str(row.get('groups', {}).get(name, 'unavailable')) for row in events})
        groups[name] = {}
        for label in labels:
            selected = [i for i,row in enumerate(events) if str(row.get('groups', {}).get(name,'unavailable')) == label]
            groups[name][label] = dict(events=len(selected),
                dates=sorted({events[i]['day'] for i in selected}),
                by_day=dict(Counter(events[i]['day'] for i in selected)),
                scope='Descriptive opportunity-weighted represented rows; no subgroup equivalence verdict',
                **described(average([represented[i] for i in selected])))
    return dict(stage='decision_calibration_v2', verdict=verdict, admission_reasons=admission,
        declared_dates=len(dates), required_dates=TARGET_DAYS, incomplete_dates=incomplete,
        primary=dict(weighting='Equal event within incident, equal incident within date, equal date',
            equivalence_margin=MARGIN, nominal_family_confidence=.95,
            bootstrap=dict(replicates=REPLICATES, seed=SEED, quantiles=[LOWER_Q,UPPER_Q], unit='whole date'),
            partitions=partition_results, **cohort),
        days=days, opportunity_weighted=opportunity, groups=groups,
        scope='Frozen day-cluster screening with missingness bounds; approximate date resampling, not arbitrary conditional or finite-sample calibration certification',
        rules_source='../decision_calibration_v1/acceptance_rules.json', g2_passed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type=Path, required=True, help='JSON list of registered event rows')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(read(args.events), read(args.manifest))
    write(args.output, result)
    print(dict(verdict=result['verdict'], dates=result['declared_dates'],
               events=result['opportunity_weighted']['represented_event_rows'],
               reasons=result['admission_reasons']))
