"""Training-only elapsed laws conditioned on route and scheduled duration."""

from collections import Counter, defaultdict

import numpy as np

from .evolving_trip_belief import EvolvingTripBelief, _ChainLaw, logsumexp


def _atoms(values):
    counts = Counter(values)
    clocks = sorted(counts)
    return dict(values=clocks, counts=[counts[t] for t in clocks], n=len(values))


def fit_empirical_durations(data, *, retain_no_pickup=True):
    """Use only declared training dates; never read evaluation outcomes.

    Observed elapsed seconds, including zero, are empirical atoms. A negative
    elapsed measurement is inconsistent with the ordered model and is counted
    separately rather than silently projected into its training support.
    """
    groups, route_groups, residuals, excluded = defaultdict(list), defaultdict(list), defaultdict(list), Counter()

    def add(route, kind, scheduled, elapsed):
        if elapsed < 0:
            excluded[kind+'_negative_elapsed'] += 1
            return
        groups[kind, scheduled].append(int(elapsed))
        route_groups[route, kind, scheduled].append(int(elapsed))
        residuals[kind].append(int(elapsed-scheduled))

    def same_vehicle(left, right, kind):
        identities = left.get('official_journey'), right.get('official_journey')
        if not all(identities) or identities[0] != identities[1]:
            excluded[kind+'_unmatched_vehicle_identity'] += 1
            return False
        return True

    for day in data['train_dates']:
        trips = defaultdict(list)
        for row in data['dates'][day]['connections']:
            if not row['dropoff_allowed'] or (not row['pickup_allowed'] and not retain_no_pickup):
                continue
            trips[row['trip_id']].append(row)
            if row['outcome'] == 'MEASURED' and same_vehicle(row['departure'], row['arrival'], 'travel'):
                add(row['route_id'], 'travel', row['scheduled_arrival_seconds']-row['scheduled_departure_seconds'],
                    row['arrival']['time_seconds']-row['departure']['time_seconds'])
        for rows in trips.values():
            rows.sort(key=lambda row: row['dep_sequence'])
            for before, after in zip(rows, rows[1:]):
                if before['arrival']['status'] != 'REAL' or after['departure']['status'] != 'REAL':
                    continue
                if before['arr_stop'] != after['dep_stop']:
                    kind = 'gap'
                elif before['arr_sequence'] == after['dep_sequence']:
                    kind = 'dwell'
                else:
                    excluded['nonadjacent_same_stop'] += 1
                    continue
                if not same_vehicle(before['arrival'], after['departure'], kind):
                    continue
                add(after['route_id'], kind, after['scheduled_departure_seconds']-before['scheduled_arrival_seconds'],
                    after['departure']['time_seconds']-before['arrival']['time_seconds'])
    return dict(train_dates=list(data['train_dates']), units='integer seconds',
        groups={kind: {str(delta): _atoms(values) for (k, delta), values in sorted(groups.items()) if k == kind}
                for kind in sorted(residuals)},
        route_groups={route: {kind: {str(delta): _atoms(values)
                    for (r, k, delta), values in sorted(route_groups.items()) if r == route and k == kind}
                for kind in sorted({k for r, k, _ in route_groups if r == route})}
            for route in sorted({r for r, _, _ in route_groups})},
        residuals={kind: _atoms(values) for kind, values in sorted(residuals.items())},
        excluded=dict(excluded), fallback='Unseen route group: pooled same-kind/schedule group; '
            'if absent, pooled same-kind residual plus schedule, clipped at zero.')


def empirical_clock_matrix(previous, following, elapsed, weights):
    """Split each elapsed atom between its two adjacent legal successor clocks.

    Zero atoms remain zero when the predecessor is a successor grid point.
    Interior projected clocks preserve the exact first moment. Atoms outside
    grid endpoints go to the nearest legal endpoint. A row with no legal
    successor stays zero, matching the inherited finite support.
    """
    previous, following = np.asarray(previous), np.asarray(following)
    elapsed, weights = np.asarray(elapsed), np.asarray(weights)
    first = np.searchsorted(following, previous, side='left')
    live = np.flatnonzero(first < len(following))
    matrix = np.zeros((len(previous), len(following)))
    desired = previous[live, None]+elapsed[None, :]
    desired = np.clip(desired, following[first[live, None]], following[-1])
    upper = np.searchsorted(following, desired, side='left')
    lower = np.maximum(first[live, None], upper-1)
    gap = following[upper]-following[lower]
    fraction = np.divide(desired-following[lower], gap, out=np.zeros_like(desired, dtype=float), where=gap > 0)
    rows = np.repeat(live, len(elapsed))
    np.add.at(matrix, (rows, lower.ravel()), (weights[None, :]*(1-fraction)).ravel())
    np.add.at(matrix, (rows, upper.ravel()), (weights[None, :]*fraction).ravel())
    return matrix


class EmpiricalDurationBelief(EvolvingTripBelief):
    """Preserve initial/service laws, replace Gaussian clock increments.

    Each observed route/kind/scheduled-duration group is used without a sample
    cutoff. Missing route groups use the pooled kind/duration distribution;
    missing durations then use an explicit same-kind residual fallback.
    This is a newly fitted finite model and has no inherited regret certificate.
    """

    def __init__(self, graph, *, duration_fit, minute=60, **kwargs):
        if minute != 60:
            raise ValueError('Empirical elapsed clocks use seconds, with minute=60')
        super().__init__(graph, minute=minute, **kwargs)
        self.duration_fit = duration_fit
        self._empirical_atoms = {}
        self._empirical_transition_cache = {}
        self.used_duration_groups, self.fallback_duration_groups = set(), set()
        self.pooled_duration_groups, self.residual_duration_groups = set(), set()

    def duration_atoms(self, kind, scheduled, *, route):
        key = route, kind, scheduled
        self.used_duration_groups.add(key)
        if key not in self._empirical_atoms:
            row = self.duration_fit['route_groups'].get(route, {}).get(kind, {}).get(str(scheduled))
            if row is None:
                row = self.duration_fit['groups'].get(kind, {}).get(str(scheduled))
                self.fallback_duration_groups.add(key)
                if row is not None:
                    self.pooled_duration_groups.add(key)
            if row is None:
                row = self.duration_fit['residuals'][kind]
                values = np.maximum(0, scheduled+np.asarray(row['values'], dtype=int))
                self.residual_duration_groups.add(key)
            else:
                values = np.asarray(row['values'], dtype=int)
            weights = np.asarray(row['counts'], dtype=float)/row['n']
            self._empirical_atoms[key] = values, weights
        return self._empirical_atoms[key]

    def _law(self, trip, parameters):
        common = super()._law(trip, parameters)
        if getattr(common, '_empirical_duration', False):
            return common
        slots = self._trips[trip]
        route = self.graph.connections[slots[0].cid].route
        pars = parameters.get(route, {'mean': 1., 'std': 2., 'cancel_rate': .01})
        q, persistence = pars['cancel_rate'], self.cancel_persistence
        times, canceled = common.times, common.canceled
        observed = any((slot.cid, slot.kind) in self._known_times for slot in slots)
        shared = None if observed else self._empirical_transition_cache
        offsets = [tuple(clock-slot.scheduled) for clock, slot in zip(times, slots)]

        def transition_kernel(index):
            following_slot = index+1
            travel = slots[following_slot].kind == 'arrival'
            kind = ('travel' if travel else
                    'gap' if self.graph.connections[slots[index].cid].arr_stop !=
                    self.graph.connections[slots[following_slot].cid].dep_stop else 'dwell')
            delta = slots[following_slot].scheduled-slots[index].scheduled
            elapsed, weights = self.duration_atoms(kind, delta, route=route)
            key = route, kind, delta, offsets[index], offsets[following_slot], q, persistence
            kernel = shared.get(key) if shared is not None else None
            if kernel is None:
                following = np.unique(times[following_slot])
                clock_mass = empirical_clock_matrix(times[index], following, elapsed, weights)
                clock_mass = clock_mass[:, np.searchsorted(following, times[following_slot])]
                if travel:
                    service = (canceled[index][:, None] == canceled[following_slot][None, :]).astype(float)
                else:
                    next_cancel = persistence*canceled[index][:, None]+(1-persistence)*q
                    service = np.where(canceled[following_slot][None, :], next_cancel, 1-next_cancel)
                with np.errstate(divide='ignore'):
                    kernel = np.log(clock_mass)+np.log(service)
                kernel.flags.writeable = False
                if shared is not None:
                    shared[key] = kernel
            return kernel

        law = _ChainLaw(times, common.initial, transition_kernel, common.emissions, canceled, common.stochastic)
        if any(np.any(emission != 0) for emission in common.emissions):
            normalizer = logsumexp(law._full_forward(len(times)-1))
            if not np.isfinite(normalizer):
                raise ValueError(f'Observations have no support under the empirical elapsed law for {trip}')
        law._empirical_duration = True
        memo = self._cache[trip]
        memo.law, memo.events = law, {}
        return law
