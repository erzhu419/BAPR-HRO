"""Pool corrected route-ID forecast summaries for an explicit training period.

These are prediction-calibrated model inputs, not realized delay priors. Use
``build_route_priors_temporal`` for deployment-style past-only evaluation.
"""
import os
import pickle
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np


def pool_priors(per_day: dict, train_dates) -> dict:
    """Pool unique-event moments and observed trip-status counts by route ID."""
    grouped = {}
    for date in sorted(train_dates):
        for rid, item in per_day[date].items():
            group = grouped.setdefault(rid, [])
            group.append(item)
    priors = {}
    for rid, items in grouped.items():
        delay_items = [item for item in items if item['n_obs'] and item['mean'] is not None]
        if not delay_items:
            continue
        n = sum(item['n_obs'] for item in delay_items)
        mean = sum(item['mean'] * item['n_obs'] for item in delay_items) / n
        variance = sum(item['n_obs'] * (item['std'] ** 2 + (item['mean'] - mean) ** 2)
                       for item in delay_items) / n
        status_count = sum(item['n_status_trips'] for item in items)
        cancel_count = sum(item['n_canceled'] for item in items)
        priors[rid] = {
            'route_id': rid, 'agency_id': items[0]['agency_id'], 'name': items[0]['name'],
            'mean': mean, 'std': float(np.sqrt(variance)),
            'cancel_rate': cancel_count / status_count if status_count else None,
            'n_canceled': cancel_count, 'n_status_trips': status_count,
            'n_obs': n, 'n_days': len(items),
            'data_kind': 'latest_unique_predictions_not_realized_events',
            'cancel_source': 'explicit_status_among_observed_scheduled_trips',
        }
    return priors


def main(in_path='data/per_day_distributions_corrected.pkl',
         out_path='data/route_priors_corrected.pkl',
         exclude_dates=('2023-10-29',)):
    with open(in_path, 'rb') as handle:
        per_day = pickle.load(handle)
    train_dates = [date for date in per_day if date not in exclude_dates]
    priors = pool_priors(per_day, train_dates)
    with open(out_path, 'wb') as handle:
        pickle.dump(priors, handle)
    print(f'Saved {len(priors)} route-ID forecast priors to {out_path}; train_dates={sorted(train_dates)}')


if __name__ == '__main__':
    main()
