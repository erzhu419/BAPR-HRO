"""Independent conditional draws for the local public platform contract."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import lru_cache

import numpy as np

from .evolving_trip_belief import _ChainLaw


class PlatformPublicParticles:
    """Prepare reusable trip laws from a frozen fitted platform belief.

    The planner binds one history per execution. Its simulations use fresh
    random numbers and never condition on sampled world identities. Local
    public receipts mask the original event grids. Waiting at a stop also
    rules out unobserved operating events there, without implying that a
    canceled vehicle would have generated an announcement.
    """

    def __init__(self, belief, parameters):
        self.belief, self.parameters = belief, parameters
        self.trips = {}
        self.layout = lru_cache(maxsize=128)(self._layout)
        self.prior_projection = lru_cache(maxsize=512)(self._conditioned_law)

    def bind(self):
        return _PublicHistory(self)

    def _layout(self, cids):
        groups = defaultdict(list)
        for column, cid in enumerate(cids):
            trip, slot = self.belief._positions[cid, 'departure']
            groups[trip].append((column, slot))
        return tuple((trip, min(slot for _, slot in columns), max(slot for _, slot in columns)+1,
                      columns) for trip, columns in groups.items())

    def prepare(self, cids):
        """Charge invariant packed-law setup before online conditioned draws."""
        for trip, first, last, _ in self.layout(tuple(cids)):
            law, _ = self.prior_projection(trip, first, last, (), ())
            _ = law.sampling_arrays

    def trip(self, trip):
        if trip not in self.trips:
            belief = self.belief
            law = belief._law(trip, self.parameters)
            by_stop = defaultdict(list)
            operating = []
            for i, slot in enumerate(belief._trips[trip]):
                connection = belief.graph.connections[slot.cid]
                stop = connection.dep_stop if slot.kind == 'departure' else connection.arr_stop
                by_stop[stop].append(i)
                indices = np.flatnonzero(~law.canceled[i])
                operating.append((indices, tuple(law.times[i][indices])))
            self.trips[trip] = law, by_stop, operating
        return self.trips[trip]

    def _conditioned_law(self, trip, first, last, exact, absent):
        base, _, operating = self.trip(trip)
        affected = (*[i for i, _ in exact], *[i for i, _ in absent])
        start = min(first, *affected) if affected else first
        end = max(last, base._last_effect, *affected)
        if not affected and start == 0 and end == len(base.times)-1:
            return base, start

        # Integrate the unseen prefix once using the fitted law's forward
        # message. Retain any later evidence and non-normalized transitions.
        emissions = list(base.emissions[start:end+1])
        emissions[0] = np.zeros_like(emissions[0])
        for slot, clock in exact:
            allowed = (~base.canceled[slot]) & (base.times[slot] == clock)
            emissions[slot-start] = np.where(allowed, emissions[slot-start], -np.inf)
        for slot, ranges in absent:
            blocked = np.concatenate([operating[slot][0][lo:hi] for lo, hi in ranges])
            updated = emissions[slot-start].copy()
            updated[blocked] = -np.inf
            emissions[slot-start] = updated
        law = _ChainLaw(base.times[start:end+1], base._full_forward(start),
                        lambda i: base._transition(start+i), tuple(emissions),
                        base.canceled[start:end+1], base.stochastic[start:end])
        if not np.isfinite(law._full_forward(end-start)).any():
            raise ValueError(f'Public platform history is outside the fitted support for trip {trip}')
        return law, start


class _PublicHistory:
    def __init__(self, compiler):
        self.compiler = compiler
        self.posterior = lru_cache(maxsize=128)(compiler._conditioned_law)
        self.visits = defaultdict(list)
        self.exact = defaultdict(dict)
        self.last_receipt = None
        self.last_log = ()

    def observe(self, state):
        """Record the public state reached after a wait, alight, or ride."""
        if not state.initialized:
            return
        windows = self.visits[state.stop]
        if windows and state.local_since <= windows[-1][1]+1:
            start, end = windows[-1]
            if state.time > end:
                windows[-1] = (start, state.time)
        else:
            windows.append((state.local_since, state.time))
        if state.public_log == self.last_log:
            return
        log = state.public_log
        start = 0 if self.last_receipt is None else bisect_left(log, (self.last_receipt,))
        positions = self.compiler.belief._positions
        for clock, cid, kind in log[start:]:
            if kind not in ('departure', 'arrival'):
                raise ValueError('The fitted platform sampler uses local occurrences, without exogenous publications')
            trip, slot = positions[cid, kind]
            self.exact[trip][slot] = clock
        self.last_log = log
        self.last_receipt = log[-1][0] if log else None

    def _evidence(self, trip):
        _, by_stop, operating = self.compiler.trip(trip)
        seen = self.exact.get(trip, {})
        absent = []
        for stop, intervals in self.visits.items():
            for slot in by_stop.get(stop, ()):
                if slot in seen:
                    continue
                times = operating[slot][1]
                spans = []
                for lo, hi in intervals:
                    start, end = bisect_left(times, lo), bisect_right(times, hi)
                    if end > start:
                        spans.append((start, end))
                if spans:
                    absent.append((slot, tuple(spans)))
        return tuple(sorted(seen.items())), tuple(sorted(absent))

    def sample(self, state, budget, rng, cids):
        """Draw packed departure, arrival and cancellation arrays in CID order."""
        self.observe(state)
        shape = budget, len(cids)
        departures = np.empty(shape, dtype=np.int32)
        arrivals = np.empty(shape, dtype=np.int32)
        canceled = np.empty(shape, dtype=np.uint8)
        for trip, first, last, columns in self.compiler.layout(tuple(cids)):
            exact, absent = self._evidence(trip)
            project = self.posterior if exact or absent else self.compiler.prior_projection
            law, start = project(trip, first, last, exact, absent)
            draws = law.draw_uniforms_batch(rng.random((budget, len(law.times)))[:, ::-1])
            for column, slot in columns:
                index = slot-start
                departures[:, column] = law.times[index][draws[:, index]]
                arrivals[:, column] = law.times[index+1][draws[:, index+1]]
                canceled[:, column] = law.canceled[index][draws[:, index]]
        return departures, arrivals, canceled
