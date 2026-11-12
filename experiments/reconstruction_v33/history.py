"""Frozen trip-law conditioning for a causal journey across several visits.

Evidence belongs to physical event slots. A platform nonoccurrence interval
ends when the passenger leaves that visit; merely advancing the journey clock
never extends it. Signatures are valid within one fixed model/parameter set.
"""

from copy import copy

import numpy as np

from experiments.reconstruction_v32.conditional import departure_evidence, UnsupportedHistory


def _merge_windows(windows):
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


class ConditionalEngine:
    """Maintain public evidence without copying a graph or changing its law.

    ``observe`` records an operating departure/arrival. Off-grid clocks use
    adjacent active atoms of the law conditioned on prior other-slot evidence.
    Seeing an event replaces its earlier nonoccurrence intervals. Evidence at
    other slots remains, including observations on the same physical vehicle.

    ``observe_platform`` accepts only receipts from its [entry, clock] visit.
    Unseen local departures exclude that bounded interval and preserve both
    unobserved earlier departures and canceled service. ``clone`` shares frozen
    arrays and already compiled laws, with independent small evidence maps.
    """

    def __init__(self, belief, parameters):
        self.belief = belief
        self.parameters = {route: dict(row) for route, row in parameters.items()}
        self._records = {}
        self._laws = {}

    def _replace(self, trip, key, record):
        previous = self._records.get(trip, ())
        old = next((r for r in previous if r[:2] == key), None)
        if old == record:
            return False
        self._records[trip] = tuple(r for r in previous if r[:2] != key)+(record,)
        self._laws.pop(trip, None)
        return True

    def observe(self, cid, kind, clock):
        trip, _ = self.belief._positions[cid, kind]
        changed = self._replace(trip, (cid, kind), (cid, kind, 'observed', float(clock)))
        return dict(cid=cid, kind=kind, clock=float(clock), trip=trip, changed=changed)

    def observe_platform(self, cids, entry, clock, observed_times):
        if clock < entry:
            raise ValueError('A local observation window cannot end before entry')
        receipts = []
        for cid in cids:
            key = cid, 'departure'
            trip, _ = self.belief._positions[key]
            if cid in observed_times:
                observed = observed_times[cid]
                if not entry <= observed <= clock:
                    raise ValueError('A platform receipt must be inside its public visit prefix')
                receipts.append(self.observe(cid, 'departure', observed))
                continue
            old = next((r for r in self._records.get(trip, ()) if r[:2] == key), None)
            if old is not None and old[2] == 'observed':
                continue  # A known physical event cannot become unseen again.
            windows = () if old is None else old[3]
            windows = _merge_windows((*windows, (float(entry), float(clock))))
            changed = self._replace(trip, key, (cid, 'departure', 'unseen', windows))
            receipts.append(dict(cid=cid, kind='local_nonoccurrence', trip=trip,
                                 windows=windows, changed=changed))
        return receipts

    def clone(self):
        result = copy(self)
        result._records = dict(self._records)
        result._laws = dict(self._laws)
        return result

    def law(self, trip):
        if trip not in self._laws:
            law = self.belief._law(trip, self.parameters)
            for cid, kind, mode, value in self._records.get(trip, ()):
                _, slot = self.belief._positions[cid, kind]
                try:
                    if mode == 'observed':
                        # The frozen-slot helper operates identically for an
                        # arrival and a departure; only their physical slot differs.
                        law, _ = departure_evidence(law, slot, -np.inf, value, observed=value)
                    else:
                        for entry, clock in value:
                            law, _ = departure_evidence(law, slot, entry, clock)
                except UnsupportedHistory as error:
                    active = ~law.canceled[slot] & (law.marginal(slot) > 0)
                    clocks = law.times[slot][active]
                    support = (float(clocks[0]), float(clocks[-1])) if len(clocks) else None
                    raise UnsupportedHistory(f'{error}; trip={trip}, cid={cid}, kind={kind}, '
                        f'evidence={mode}:{value}, conditional_operating_support={support}') from error
            self._laws[trip] = law
        return self._laws[trip]

    def as_belief(self):
        result = copy(self.belief)
        result._law = lambda trip, parameters: self.law(trip)
        return result

    def signature(self, relevant_trips=None):
        trips = sorted(self._records if relevant_trips is None else set(relevant_trips))
        return tuple((trip, self._records.get(trip, ())) for trip in trips)

    def public_state(self):
        return [dict(trip=trip, cid=cid, kind=kind, evidence=mode,
                     **({'clock': value} if mode == 'observed' else {'windows': value}))
                for trip in sorted(self._records) for cid, kind, mode, value in self._records[trip]]
