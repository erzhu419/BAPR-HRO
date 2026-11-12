"""Distributions for the frozen board and fixed-target continuation policies.

Successful physical arrival times remain untruncated in the distributions;
``cost`` clips final arrival at the common deadline and charges that deadline
for all missing probability mass. These are model failures, not missing data.
"""

import numpy as np


def _distribution(values, probabilities):
    values, probabilities = np.asarray(values), np.asarray(probabilities, dtype=float)
    positive = probabilities > 0
    values, probabilities = values[positive], probabilities[positive]
    unique, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=probabilities, minlength=len(unique))
    return dict(values=unique.tolist(), probabilities=mass.tolist(),
                failure_mass=float(max(0., 1.-mass.sum())))


def _propagate(compiler, path, weights):
    """Forward dual of the existing service-killing backward contraction."""
    trip, start, end = compiler._path_slots(path)
    law = compiler.bounds._law(trip)
    mass = np.asarray(weights, dtype=float) * ~law.canceled[start]
    for slot in range(start, end):
        mass = mass @ compiler.bounds._posterior_edge(trip, slot)
        mass *= ~law.canceled[slot+1]
    return law.times[end], mass


def _continuation(compiler, cid, weights):
    option = compiler.options[cid]
    arrivals, arrival_mass = _propagate(compiler, option.first_path, weights)
    first_distribution = _distribution(arrivals, arrival_mass)
    arrivals = np.asarray(first_distribution['values'])
    arrival_mass = np.asarray(first_distribution['probabilities'])
    distributions = dict(first_arrival=first_distribution, final_arrival=None, tail_wait=None)
    choices = dict(by_cid={}, no_tail=None)
    if not option.tails:
        final = first_distribution
    else:
        stop = compiler.graph.connections[option.first_path[-1]].arr_stop
        ready = arrivals+compiler.graph.stops[stop].min_transfer_time
        departures = {first: compiler.departure_distribution(first)
                      for first in option.tails}
        ranks = {first: (compiler.graph.connections[path[-1]].arr_time
                         -compiler.graph.connections[first].dep_time, first)
                 for first, path in option.tails.items()}
        final_times, final_mass, wait_times, wait_mass = [], [], [], []
        for first, path in option.tails.items():
            clocks, probabilities, _ = departures[first]
            legal = (clocks[None, :] >= ready[:, None]) & (clocks[None, :] < compiler.deadline)
            selected = np.broadcast_to(probabilities, legal.shape).copy()*legal
            for other, (other_clocks, other_mass, _) in departures.items():
                if first == other:
                    continue
                cdf = np.r_[0., np.cumsum(other_mass)]
                lower = cdf[np.searchsorted(other_clocks, ready, side='left')]
                side = 'right' if ranks[other] < ranks[first] else 'left'
                upper = cdf[np.searchsorted(other_clocks, clocks, side=side)]
                selected *= 1.-np.maximum(0., upper[None, :]-lower[:, None])
            joint = arrival_mass[:, None]*selected
            departure_weights = joint.sum(axis=0)
            choices['by_cid'][str(first)] = float(departure_weights.sum())
            positive = joint > 0
            wait_times.extend((clocks[None, :]-arrivals[:, None])[positive].tolist())
            wait_mass.extend(joint[positive].tolist())
            trip, start, _ = compiler._path_slots(path)
            law = compiler.bounds._law(trip)
            initial = np.zeros(len(law.times[start]))
            initial[~law.canceled[start]] = departure_weights
            tail_arrivals, tail_mass = _propagate(compiler, path, initial)
            final_times.extend(tail_arrivals.tolist())
            final_mass.extend(tail_mass.tolist())
        final = _distribution(final_times, final_mass)
        distributions['tail_wait'] = _distribution(wait_times, wait_mass)
        choices['no_tail'] = float(max(0., 1.-sum(choices['by_cid'].values())))
    distributions['final_arrival'] = final
    gain = sum(p*max(0., compiler.deadline-t)
               for t, p in zip(final['values'], final['probabilities'], strict=True))
    return dict(cost=float(compiler.deadline-gain), distributions=distributions,
                first_path_failure=first_distribution['failure_mass'], tail_choice=choices)


def forecast_board(compiler, cid, clock):
    """Observe current departure, retaining the existing off-grid projection."""
    trip, start, _ = compiler._path_slots(compiler.options[cid].first_path)
    law = compiler.bounds._law(trip)
    indices = np.flatnonzero((~law.canceled[start]) & (law.marginal(start) > 0))
    clocks = law.times[start][indices]
    index = np.searchsorted(clocks, clock)
    weights = np.zeros(len(law.times[start]))
    if index < len(clocks) and clocks[index] == clock:
        weights[indices[index]] = 1.
        method, endpoints = 'on_grid', [int(clock)]
    elif 0 < index < len(clocks):
        fraction = (clock-clocks[index-1])/(clocks[index]-clocks[index-1])
        weights[indices[index-1]], weights[indices[index]] = 1.-fraction, fraction
        method, endpoints = 'projection', [int(clocks[index-1]), int(clocks[index])]
    else:
        return dict(status='unsupported', reason='current_outside_score_support',
                    method=None, cost=None, distributions=None)
    result = _continuation(compiler, cid, weights)
    result.update(status='supported', reason=None, method=method,
                  support=dict(endpoints=endpoints, starting_mass=float(weights.sum())))
    return result


def forecast_target(compiler, cid, entry, clock, ready):
    """Wait only for this unobserved target, then use the fixed continuation."""
    clocks, probabilities, cancellation = compiler.departure_distribution(cid)
    before = float(probabilities[clocks < entry].sum())
    future = clocks > clock
    denominator = float(cancellation+before+probabilities[future].sum())
    legal = future & (clocks >= ready) & (clocks < compiler.deadline)
    support = dict(denominator=denominator, cancellation_mass=float(cancellation),
                   before_entry_mass=before,
                   after_deadline_mass=float(probabilities[future & (clocks >= compiler.deadline)].sum()))
    if denominator <= 0:
        return dict(status='unsupported', reason='target_zero_survival', method=None,
                    cost=None, distributions=None, support=support)
    trip, start, _ = compiler._path_slots(compiler.options[cid].first_path)
    law = compiler.bounds._law(trip)
    weights = np.zeros(len(law.times[start]))
    operating = probabilities*legal/denominator
    weights[~law.canceled[start]] = operating
    result = _continuation(compiler, cid, weights)
    result['distributions']['departure'] = _distribution(clocks, operating)
    result['distributions']['remaining_wait'] = _distribution(clocks-clock, operating)
    support['legal_future_probability'] = float(operating.sum())
    result.update(status='supported', reason=None, method='future_atom_integral', support=support)
    return result
