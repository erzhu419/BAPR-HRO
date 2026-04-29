"""Leave-one-day-out (LOO) per-route historical priors.

For each evaluation date d, build a prior excluding day d (and excluding
the disrupted Oct 29). The 35-day Swiss benchmark then evaluates day d
under a prior trained only on the other normal days, giving strict
out-of-sample evaluation of A4 instead of the original transductive
calibration on all 34 normal days.

Output: data/route_priors_loo.pkl
  {date: {route_short_name: {'mean','std','cancel_rate','n_days'}}}
"""

import os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from collections import defaultdict


def main(in_path: str = 'data/per_day_distributions.pkl',
         out_path: str = 'data/route_priors_loo.pkl',
         exclude_dates: list = ('2023-10-29',)):
    per_day = pickle.load(open(in_path, 'rb'))
    print(f'Loaded {len(per_day)} days')

    normal_days = [d for d in per_day if d not in exclude_dates]
    print(f'Normal days for prior pool: {len(normal_days)}')

    loo_priors = {}
    for target_date in per_day:
        train_dates = [d for d in normal_days if d != target_date]
        by_means = defaultdict(list)
        by_stds = defaultdict(list)
        by_cancels = defaultdict(list)
        for d in train_dates:
            for rname, dist in per_day[d].items():
                by_means[rname].append(dist['mean'])
                by_stds[rname].append(dist['std'])
                by_cancels[rname].append(dist['cancel_rate'])
        priors = {}
        for rname in by_means:
            priors[rname] = {
                'mean':        float(np.mean(by_means[rname])),
                'std':         float(np.mean(by_stds[rname])),
                'cancel_rate': float(np.mean(by_cancels[rname])),
                'n_days':      len(by_means[rname]),
            }
        loo_priors[target_date] = priors

    with open(out_path, 'wb') as f:
        pickle.dump(loo_priors, f)
    print(f'Saved LOO priors for {len(loo_priors)} dates to {out_path}')

    sample_date = sorted(per_day.keys())[0]
    n = len(loo_priors[sample_date])
    print(f'Sample: priors for date {sample_date} have {n} routes, '
          f'each based on n_days={list(loo_priors[sample_date].values())[0]["n_days"]} train days')


if __name__ == '__main__':
    main()
