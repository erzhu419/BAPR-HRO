"""A4 (GPT-5.5 review): Build per-route historical prior from
per_day_distributions.pkl.

For each route, average mean / std / cancel_rate across the 34 normal
days (excluding Oct 29 disrupted to avoid contamination). This gives
a route-level historical prior for cold-start initialization in
BanditRouter, replacing the uniform prior_mean=1.0, prior_var=2.

Output: data/route_priors.pkl
  {route_short_name: {'mean': float, 'std': float, 'cancel_rate': float, 'n_days': int}}
"""

import os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from collections import defaultdict


def main(in_path: str = 'data/per_day_distributions.pkl',
         out_path: str = 'data/route_priors.pkl',
         exclude_dates: list = ('2023-10-29',)):
    per_day = pickle.load(open(in_path, 'rb'))
    print(f'Loaded {len(per_day)} days; excluding {exclude_dates}')

    by_route_means = defaultdict(list)
    by_route_stds = defaultdict(list)
    by_route_cancels = defaultdict(list)

    for date, dist in per_day.items():
        if date in exclude_dates:
            continue
        for rname, d in dist.items():
            by_route_means[rname].append(d['mean'])
            by_route_stds[rname].append(d['std'])
            by_route_cancels[rname].append(d['cancel_rate'])

    priors = {}
    for rname in by_route_means:
        means = np.array(by_route_means[rname])
        stds = np.array(by_route_stds[rname])
        cancels = np.array(by_route_cancels[rname])
        priors[rname] = {
            'mean': float(np.mean(means)),
            'std':  float(np.mean(stds)),  # avg of per-day stds
            'cancel_rate': float(np.mean(cancels)),
            'n_days': len(means),
        }

    with open(out_path, 'wb') as f:
        pickle.dump(priors, f)
    print(f'Saved {len(priors)} route priors to {out_path}')

    # Sanity print: top-5 by sample size
    sorted_routes = sorted(priors.items(), key=lambda kv: -kv[1]['n_days'])
    print('\nFirst 10 priors:')
    print(f'{"route":<20} {"mean":>6} {"std":>6} {"cancel":>8} {"days":>5}')
    for r, p in sorted_routes[:10]:
        print(f'{str(r)[:19]:<20} {p["mean"]:>6.2f} {p["std"]:>6.2f}'
              f' {p["cancel_rate"]:>8.4f} {p["n_days"]:>5}')


if __name__ == '__main__':
    main()
