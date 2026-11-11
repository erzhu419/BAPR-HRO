"""Sensitivity to conditioning the same trip chain on geometric runtime floors."""
import argparse
from collections import defaultdict
import json
from statistics import median

import numpy as np

from physical_run import AUDIT, HERE, ROOT, joint_operating_edge, read
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from src.evolving_trip_belief import EvolvingTripBelief, _ChainLaw, logsumexp
from src.ordered_duration_belief import OrderedDurationBelief


class OrderedDurationCoarse(OrderedDurationBelief):
    """Hold the V5 elapsed-duration kernel and fit fixed on the 60s lattice."""
    _grid = EvolvingTripBelief._grid


def conditioned_trip(law, bounds):
    """Mask operating edges and normalize at the chain level, not row-by-row."""
    def transition(slot):
        original = law._transition(slot)
        if slot not in bounds:
            return original
        delta = law.times[slot + 1][None, :] - law.times[slot][:, None]
        operating = ~law.canceled[slot][:, None] & ~law.canceled[slot + 1][None, :]
        return np.where(operating & (delta < bounds[slot]), -np.inf, original)

    constrained = _ChainLaw(law.times, law.initial, transition, law.emissions, law.canceled,
                            tuple(value and slot not in bounds for slot, value in enumerate(law.stochastic)))
    log_mass = float(logsumexp(constrained.forward[-1]) - logsumexp(law.forward[-1]))
    if not np.isfinite(log_mass):
        raise ValueError('The declared sensitivity has no joint support on this trip')
    return constrained, float(np.exp(log_mass))


def score(law, slot, observed, floor_seconds):
    deltas, weights = joint_operating_edge(law, slot)
    atoms, inverse = np.unique(deltas[weights > 0], return_inverse=True)
    probability = np.bincount(inverse, weights=weights[weights > 0])
    probability /= probability.sum()
    cdf = probability.cumsum()
    mean = float(probability @ atoms)
    lower, upper = atoms[np.searchsorted(cdf, [.05, .95])]
    crps = probability @ abs(atoms - observed) - np.diff(atoms) @ (cdf[:-1] * (1 - cdf[:-1]))
    return dict(mean_runtime=mean, abs_error=abs(mean-observed), crps=float(crps),
                runtime_90=[int(lower), int(upper)], width=int(upper-lower),
                coverage=bool(lower <= observed <= upper), p_runtime_zero=float(probability[atoms == 0].sum()),
                p_runtime_under_60=float(probability[atoms < 60].sum()),
                p_below_floor=float(probability[atoms < floor_seconds].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('raw60', 'cdf60', 'cdf20'), default='raw60')
    model = parser.parse_args().model
    physical = read(HERE / 'physical_results.json')
    declared = [r for r in physical['city_runtime_rows'] if r['conditional_floors_seconds'].get('50', 0) > 0]
    data, _ = load_city(AUDIT / 'city_vbz_dated_replay_v2.json.gz')
    fit_path = ROOT / 'experiments/reconstruction_v5/ordered_duration_training_fit.json'
    fit = (fit_training(data, retain_no_pickup=True) if model == 'raw60'
           else read(fit_path)['continuous_training'])
    graph, _, _, _, _, _ = dated_graph(data, '2023-07-26', 'gtfs_station_max', True)
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    belief_type = {'raw60': EvolvingTripBelief, 'cdf60': OrderedDurationCoarse, 'cdf20': OrderedDurationBelief}[model]
    belief = belief_type(graph, minute=60, public_occurrences=False, **fit['increments'])
    groups = defaultdict(dict)
    for row in declared:
        trip, slot = belief._positions[row['connection'], 'departure']
        groups[trip][slot] = row
    rows, joint_acceptance = [], {}
    for trip, entries in groups.items():
        original = belief._law(trip, parameters)
        constrained, mass = conditioned_trip(original, {slot: r['conditional_floors_seconds']['50']
                                                         for slot, r in entries.items()})
        joint_acceptance[trip] = mass
        for slot, row in entries.items():
            observed = row['realized_runtime_seconds']
            floor_seconds = row['conditional_floors_seconds']['50']
            rows.append(dict(connection=row['connection'], trip=trip, observed=observed,
                             floor_seconds=floor_seconds,
                             original=score(original, slot, observed, floor_seconds),
                             conditional=score(constrained, slot, observed, floor_seconds)))
    short = [row for row in rows if row['observed'] < 60]
    result = dict(
        scope='Posthoc mechanism sensitivity. Full trip prior conditioned on all selected geometry floors simultaneously. '
              'No refit, no adaptive use of actual fleet or stopping-point labels, no reuse of the old model certificate.',
        inputs=[str((HERE / 'physical_results.json').relative_to(ROOT)),
                str((AUDIT / 'city_vbz_dated_replay_v2.json.gz').relative_to(ROOT))],
        model=model, fit='Raw July25 moments' if model == 'raw60' else str(fit_path.relative_to(ROOT)) + ':continuous_training',
        n_legs=len(rows), n_trips=len(groups), radius_per_endpoint_m=50, joint_acceptance=joint_acceptance,
        joint_acceptance_median=median(joint_acceptance.values()),
        summary={name: {key: sum(row[name][key] for row in rows)/len(rows)
                         for key in ('mean_runtime', 'abs_error', 'crps', 'width', 'coverage',
                                     'p_runtime_zero', 'p_runtime_under_60', 'p_below_floor')}
                 for name in ('original','conditional')},
        observed_under_60={'n': len(short), **{name: sum(row[name]['coverage'] for row in short)
                                              for name in ('original', 'conditional')}},
        rows=rows)
    assert all(row['conditional']['p_below_floor'] == 0 for row in rows)
    suffix = '' if model == 'raw60' else '_' + model
    (HERE / f'physical_conditioning{suffix}_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'joint_acceptance')}, indent=2))


if __name__ == '__main__':
    main()
