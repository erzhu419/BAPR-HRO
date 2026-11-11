"""Past-only forecast priors; target and future dates never enter calibration."""
import os
import pickle
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from experiments.swiss_full.build_route_priors import pool_priors


def build_temporal_priors(per_day: dict, exclude_dates=('2023-10-29',)) -> dict:
    return {target: {'train_dates': sorted(date for date in per_day
                                          if date < target and date not in exclude_dates),
                     'priors': pool_priors(per_day, [date for date in per_day
                                                    if date < target and date not in exclude_dates])}
            for target in sorted(per_day)}


def main(in_path='data/per_day_distributions_corrected.pkl',
         out_path='data/route_priors_temporal_corrected.pkl'):
    with open(in_path, 'rb') as handle:
        per_day = pickle.load(handle)
    priors = build_temporal_priors(per_day)
    with open(out_path, 'wb') as handle:
        pickle.dump(priors, handle)
    print(f'Saved past-only forecast priors for {len(priors)} dates to {out_path}')


if __name__ == '__main__':
    main()
