"""Offline leave-one-day-out forecast priors; future dates may be in training.

This estimates offline generalization, not deployment-style past-only behavior.
Historical proxy-based pickles remain untouched.
"""
import os
import pickle
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from experiments.swiss_full.build_route_priors import pool_priors


def main(in_path='data/per_day_distributions_corrected.pkl',
         out_path='data/route_priors_loo_corrected.pkl',
         exclude_dates=('2023-10-29',)):
    with open(in_path, 'rb') as handle:
        per_day = pickle.load(handle)
    priors = {target: pool_priors(per_day, [date for date in per_day
                                         if date != target and date not in exclude_dates])
              for target in sorted(per_day)}
    with open(out_path, 'wb') as handle:
        pickle.dump(priors, handle)
    print(f'Saved offline LOO priors for {len(priors)} dates to {out_path}')


if __name__ == '__main__':
    main()
