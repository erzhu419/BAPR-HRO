"""Same-trip conditional futures on an existing ordered-chain law."""

from dataclasses import dataclass

import numpy as np

from .evolving_trip_belief import _ChainLaw
from .simulator import VehicleEvent


@dataclass(frozen=True)
class ConditionalEvent:
    """Operating clock mass and separate service cancellation probability."""

    clocks: np.ndarray
    probabilities: np.ndarray
    cancel_probability: float

    @property
    def operating_probability(self):
        return float(self.probabilities.sum())

    @property
    def mean_operating_clock(self):
        mass = self.operating_probability
        return float(self.clocks @ self.probabilities / mass) if mass > 0. else None


class PublicTripConditional:
    """Posterior suffix after a publicly observed operating departure."""

    def __init__(self, slots, cut, law):
        self._slots = slots
        self._cut = cut
        self._positions = {(slot.cid, slot.kind): i for i, slot in enumerate(slots) if i >= cut}
        self.law = law

    def marginal(self, cid, kind='arrival'):
        """Return the event law for a same-trip slot at or after the departure."""
        if (cid, kind) not in self._positions:
            raise ValueError('The event is outside the conditioned trip suffix')
        index = self._positions[cid, kind]
        probabilities = self.law.marginal(index)
        canceled = self.law.canceled[index]
        operating = ~canceled & (probabilities > 0.)
        return ConditionalEvent(self.law.times[index][operating], probabilities[operating],
                                float(probabilities[canceled].sum()))

    def sample_suffix(self, rng, n):
        """Draw independent complete suffixes, retaining dependence within a trip."""
        slots, law = self._slots, self.law
        states = law.draw_uniforms_batch(rng.random((n, len(slots)))[:, ::-1])
        draws = []
        for row in states:
            events = {}
            for i in range(self._cut, len(slots), 2):
                departure = int(law.times[i][row[i]])
                events[slots[i].cid] = VehicleEvent(
                    departure-slots[i].scheduled, bool(law.canceled[i][row[i]]),
                    departure, int(law.times[i+1][row[i+1]]))
            draws.append(events)
        return draws


def condition_on_departure(belief, parameters, cid, clock):
    """Condition the frozen law on one exact operating departure observation.

    The caller supplies the belief available before the public observation.
    Only the departure state is restricted. The clock grid, kernels and
    preexisting evidence stay fixed. An unsupported clock returns None.
    """
    trip, cut = belief._positions[cid, 'departure']
    law = belief._law(trip, parameters)
    allowed = (law.times[cut] == clock) & ~law.canceled[cut]
    if not allowed.any():
        return None
    anchor = np.where(allowed, law._full_forward(cut), -np.inf)
    future = law._backward_from(max(cut, law._last_effect), cut)
    if not np.isfinite(anchor+future).any():
        return None
    # Retain full indexing for existing itinerary propagation. Each branch
    # owns its new anchor emission and its inference caches. The other arrays
    # are read-only inputs shared with the root law.
    evidence = list(law.emissions)
    evidence[cut] = np.where(allowed, evidence[cut], -np.inf)
    conditioned = _ChainLaw(law.times, law.initial, law._transition,
                            tuple(evidence), law.canceled, law.stochastic)
    return PublicTripConditional(belief._trips[trip], cut, conditioned)
