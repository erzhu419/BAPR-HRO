"""Fit the single predeclared Gaussian error candidate using October 1--2 only."""

from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def fit_emissions(rows, *, training_dates, pseudocount=20., minimum_sd=30.):
    groups, pooled = defaultdict(list), defaultdict(list)
    counts_by_date, eligible_by_date = defaultdict(int), defaultdict(int)
    for row in rows:
        if row['split'] != 'training':
            continue
        day = row['date']
        if day not in training_dates:
            raise ValueError('A training-marked date lies outside the frozen training dates')
        counts_by_date[day] += 1
        if not row['error_eligible']:
            continue
        error, kind = float(row['error_seconds']), row['kind']
        groups[row['route_id'], kind].append(error)
        pooled[kind].append(error)
        eligible_by_date[day] += 1

    def moments(values, prior=None):
        values = np.asarray(values, dtype=float)
        if prior is None:
            mean, second = float(values.mean()), float(np.mean(values**2))
        else:
            denominator = len(values)+pseudocount
            mean = float((values.sum()+pseudocount*prior['bias_seconds'])/denominator)
            second = float((values@values+pseudocount*prior['second_moment_seconds_squared'])/denominator)
        variance = max(0., second-mean**2)
        return dict(n=len(values), bias_seconds=mean, sd_seconds=max(minimum_sd, float(np.sqrt(variance))),
                    second_moment_seconds_squared=second, unfloored_variance_seconds_squared=variance)

    by_kind = {}
    for kind in ('departure', 'arrival'):
        if not pooled[kind]:
            raise ValueError(f'No eligible training forecast errors for {kind}')
        by_kind[kind] = moments(pooled[kind])
    by_route = {}
    for (route, kind), values in sorted(groups.items()):
        by_route.setdefault(route, {})[kind] = moments(values, by_kind[kind])
    return dict(training_dates=list(training_dates), source='events.json.gz: split=training and error_eligible',
        error_definition='actual minus forecast', route_pseudocount=float(pseudocount),
        minimum_sd_seconds=float(minimum_sd), pooled=by_kind, route_kind=by_route,
        training_queries_by_date=dict(counts_by_date), eligible_errors_by_date=dict(eligible_by_date),
        forecast_joint_assumption='Latest per endpoint, conditionally independent errors given the unchanged physical chain; not validated by fitting.')


def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    with gzip.open(HERE/'events.json.gz', 'rt') as handle:
        events = json.load(handle)
    fitted = fit_emissions(events['rows'], training_dates=protocol['training_dates'],
        pseudocount=protocol['candidate']['route_pseudocount'],
        minimum_sd=protocol['candidate']['minimum_sd_seconds'])
    target = HERE/'fitted_emission.json'
    target.write_text(json.dumps(fitted, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(training_dates=fitted['training_dates'], pooled=fitted['pooled'],
                          eligible_errors_by_date=fitted['eligible_errors_by_date']), indent=2))


if __name__ == '__main__':
    main()
