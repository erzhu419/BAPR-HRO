"""One frozen spatial refinement of the V25 empirical duration model."""

from collections import Counter, defaultdict

import numpy as np

from src.empirical_duration_belief import EmpiricalDurationBelief, empirical_clock_matrix
from src.evolving_trip_belief import EvolvingTripBelief, _ChainLaw, logsumexp


TRAIN_DATE = '2023-07-25'
SHRINKAGE = 20


def fit_segment_durations(data):
    """Retain the original duration-fit eligibility, splitting physical stops.

    V31 is a single development candidate trained only on the frozen July 25
    source. Other dates in ``data['dates']`` are not inspected.
    """
    if data['train_dates'] != [TRAIN_DATE]:
        raise ValueError('V31 duration fitting is frozen to 2023-07-25 only')
    groups, excluded = defaultdict(list), Counter()

    def add(route, kind, scheduled, left_stop, right_stop, left, right):
        identities = left.get('official_journey'), right.get('official_journey')
        if not all(identities) or identities[0] != identities[1]:
            excluded[kind+'_unmatched_vehicle_identity'] += 1
            return
        elapsed = right['time_seconds']-left['time_seconds']
        if elapsed < 0:
            excluded[kind+'_negative_elapsed'] += 1
            return
        groups[route, kind, scheduled, left_stop, right_stop].append(int(elapsed))

    trips = defaultdict(list)
    for row in data['dates'][TRAIN_DATE]['connections']:
        if not row['dropoff_allowed']:
            continue
        trips[row['trip_id']].append(row)
        if row['outcome'] == 'MEASURED':
            add(row['route_id'], 'travel',
                row['scheduled_arrival_seconds']-row['scheduled_departure_seconds'],
                row['dep_stop'], row['arr_stop'], row['departure'], row['arrival'])
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
            add(after['route_id'], kind,
                after['scheduled_departure_seconds']-before['scheduled_arrival_seconds'],
                before['arr_stop'], after['dep_stop'], before['arrival'], after['departure'])
    rows = []
    for (route, kind, scheduled, left, right), values in sorted(groups.items()):
        counts = Counter(values)
        clocks = sorted(counts)
        rows.append(dict(route=route, kind=kind, scheduled=scheduled, from_stop=left,
            to_stop=right, values=clocks, counts=[counts[t] for t in clocks], n=len(values)))
    return dict(train_dates=[TRAIN_DATE], shrinkage=SHRINKAGE, units='integer seconds',
        grouping='route, event kind, scheduled duration, physical from-stop, physical to-stop',
        groups=rows, excluded=dict(excluded),
        fallback='Unseen segment uses the unchanged V25 route/kind/schedule duration law.')


class SegmentDurationBelief(EmpiricalDurationBelief):
    """Mix each segment's empirical atoms with its unchanged V25 parent law.

    The segment weight is n/(n+20), fixed before evaluation. Initial clocks,
    service transitions, finite grids, and observed-history semantics remain
    those of the inherited model. ``stop_identities`` maps graph stop integers
    to the physical source IDs used in the training records.
    """

    def __init__(self, graph, *, duration_fit, segment_fit, stop_identities, **kwargs):
        super().__init__(graph, duration_fit=duration_fit, **kwargs)
        self.segment_fit, self.stop_identities = segment_fit, dict(stop_identities)
        self._segment_groups = {(r['route'], r['kind'], r['scheduled'], r['from_stop'], r['to_stop']): r
                                for r in segment_fit['groups']}
        self._segment_atoms, self._segment_transition_cache = {}, {}
        self.used_segment_groups, self.matched_segment_groups, self.fallback_segment_groups = set(), set(), set()

    def segment_atoms(self, kind, scheduled, *, route, segment):
        key = route, kind, scheduled, *segment
        self.used_segment_groups.add(key)
        if key not in self._segment_atoms:
            base_values, base_weights = self.duration_atoms(kind, scheduled, route=route)
            row = self._segment_groups.get(key)
            if row is None or row['n'] == 0:
                self.fallback_segment_groups.add(key)
                self._segment_atoms[key] = base_values, base_weights
            else:
                self.matched_segment_groups.add(key)
                weight = row['n']/(row['n']+SHRINKAGE)
                values = np.r_[base_values, row['values']]
                probabilities = np.r_[(1.-weight)*base_weights,
                    weight*np.asarray(row['counts'], dtype=float)/row['n']]
                unique, inverse = np.unique(values, return_inverse=True)
                mass = np.bincount(inverse, weights=probabilities, minlength=len(unique))
                self._segment_atoms[key] = unique, mass
        return self._segment_atoms[key]

    def _law(self, trip, parameters):
        # Obtain the shared support/initial/emission construction directly;
        # the parent empirical transition would otherwise be built and cached.
        common = EvolvingTripBelief._law(self, trip, parameters)
        if getattr(common, '_segment_duration', False):
            return common
        slots = self._trips[trip]
        route = self.graph.connections[slots[0].cid].route
        pars = parameters.get(route, {'mean': 1., 'std': 2., 'cancel_rate': .01})
        q, persistence = pars['cancel_rate'], self.cancel_persistence
        times, canceled = common.times, common.canceled
        observed = any((slot.cid, slot.kind) in self._known_times for slot in slots)
        shared = None if observed else self._segment_transition_cache
        offsets = [tuple(clock-slot.scheduled) for clock, slot in zip(times, slots)]

        def transition_kernel(index):
            next_index = index+1
            left = self.graph.connections[slots[index].cid]
            right = self.graph.connections[slots[next_index].cid]
            travel = slots[next_index].kind == 'arrival'
            kind = 'travel' if travel else 'gap' if left.arr_stop != right.dep_stop else 'dwell'
            segment = (self.stop_identities[left.dep_stop], self.stop_identities[left.arr_stop]) if travel else (
                self.stop_identities[left.arr_stop], self.stop_identities[right.dep_stop])
            delta = slots[next_index].scheduled-slots[index].scheduled
            elapsed, weights = self.segment_atoms(kind, delta, route=route, segment=segment)
            key = route, kind, delta, segment, offsets[index], offsets[next_index], q, persistence
            kernel = shared.get(key) if shared is not None else None
            if kernel is None:
                following = np.unique(times[next_index])
                clock_mass = empirical_clock_matrix(times[index], following, elapsed, weights)
                clock_mass = clock_mass[:, np.searchsorted(following, times[next_index])]
                if travel:
                    service = (canceled[index][:, None] == canceled[next_index][None, :]).astype(float)
                else:
                    next_cancel = persistence*canceled[index][:, None]+(1.-persistence)*q
                    service = np.where(canceled[next_index][None, :], next_cancel, 1.-next_cancel)
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
                raise ValueError(f'Observations have no support under the segment elapsed law for {trip}')
        law._segment_duration = law._empirical_duration = True
        memo = self._cache[trip]
        memo.law, memo.events = law, {}
        return law
