"""Proper distribution scores with missing labels kept out of denominators."""

import numpy as np


def distribution_score(distribution, actual, *, deadline=None):
    if distribution is None or actual is None:
        return None
    x = np.asarray(distribution['values'], dtype=float)
    p = np.asarray(distribution['probabilities'], dtype=float)
    successful_mass = float(p.sum())
    if deadline is not None:
        x = np.r_[np.minimum(x, deadline), deadline]
        p = np.r_[p, distribution['failure_mass']]
    if p.sum() <= 0:
        return None
    p = p / p.sum()
    order = np.argsort(x)
    x, p = x[order], p[order]
    mean = float(x @ p)
    cumulative = np.cumsum(p)
    # Half E|X-X'| = sum_i p_i (x_i P[X'<x_i] - E[X' 1(X'<x_i)]).
    pair_half = float(p @ (x*(cumulative-p)-(np.cumsum(x*p)-x*p)))
    crps = float(p @ np.abs(x-actual))-pair_half
    lo, hi = x[np.minimum(np.searchsorted(cumulative, [.05, .95]), len(x)-1)]
    return dict(actual=float(actual), mean=mean, bias=mean-actual,
                absolute_error=abs(mean-actual), crps=crps,
                covered90=bool(lo <= actual <= hi), width90=float(hi-lo),
                conditional_mass=successful_mass)


def score_forecast(forecast, actual, deadline, reference_tail=None):
    if forecast['status'] != 'supported':
        return {}
    distributions = forecast['distributions']
    scores = {name: distribution_score(distributions.get(name), actual.get(name))
              for name in ('first_arrival', 'remaining_wait', 'tail_wait')}
    scores['final_cost'] = distribution_score(distributions['final_arrival'], actual['cost'], deadline=deadline)
    scores['binary'] = {}
    arrival = distributions['first_arrival']
    if actual['first_arrival'] is not None or actual['first_path_completed'] is False:
        probability = sum(p for x, p in zip(arrival['values'], arrival['probabilities']) if x < deadline)
        label = actual['first_arrival'] is not None and actual['first_arrival'] < deadline
        scores['binary']['first_arrival_before_deadline'] = dict(probability=probability,
            actual=int(label), brier=(probability-label)**2)
    if 'departure' in distributions and (actual['root_departure'] is not None or actual['first_path_completed'] is False):
        probability = sum(distributions['departure']['probabilities'])
        label = actual['root_departure'] is not None
        scores['binary']['legal_target_appears'] = dict(probability=probability,
            actual=int(label), brier=(probability-label)**2)
    if reference_tail is not None and (actual['tail_offer_known'] is True or actual['first_path_completed'] is False):
        probability = forecast['tail_choice']['by_cid'].get(str(reference_tail), 0.)
        label = actual['tail_cid'] == reference_tail
        scores['binary']['reference_tail_chosen'] = dict(reference_tail=reference_tail,
            probability=probability, actual=int(label), brier=(probability-label)**2)
    return scores
