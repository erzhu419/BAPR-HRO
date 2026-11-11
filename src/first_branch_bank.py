"""Common primitive random tapes for action-specific first-branch strata."""
import numpy as np

from .evolving_trip_belief import EvolvingTripBelief, _SampleBank, _LazyVehicleEvents
from .first_branch_belief import FirstBranchSampler, _ConditionalEvents, _ConditionalTripDraws
from .simulator import LazyTripScenario


class FirstBranchBank:
    """Reuse untouched trip worlds and couple conditional draws across actions.

    One original uniform tape is allocated per trip/world/event. For a tested
    action, only its trip uses conditional FFBS on that same tape; other trips
    are shared exactly across all actions. Each action's weighted strata still
    integrate its own full-posterior expected loss. These action-specific
    transformed worlds are not one common physical finite scenario bank.
    """

    def __init__(self, belief, parameters, rng, n):
        if not isinstance(belief, EvolvingTripBelief):
            raise TypeError('FirstBranchBank requires an ordered independent-trip belief')
        if n < 1:
            raise ValueError('At least one private world is required')
        self.belief = belief.copy()
        self.parameters = {route: dict(row) for route, row in parameters.items()}
        self.n = n
        for trip in self.belief._revision:
            self.belief._law(trip, self.parameters)
        self.uniforms = {}
        for trip, slots in self.belief._trips.items():
            values = rng.random((n, len(slots)))[:, ::-1]
            values.flags.writeable = False
            self.uniforms[trip] = values
        bank = _SampleBank(self.belief, self.parameters, self.uniforms)
        self.original = [LazyTripScenario(_LazyVehicleEvents(bank, i), minute=belief.minute,
                        publications=None if belief.public_occurrences else {}) for i in range(n)]
        self._samplers = {}

    def sampler(self, cid, ready, deadline, patience=None):
        key = cid, ready, deadline, patience
        if key not in self._samplers:
            self._samplers[key] = FirstBranchSampler(self.belief, self.parameters,
                                                     cid, ready, deadline, patience)
        return self._samplers[key]

    def for_action(self, cid, ready, deadline, *, n_catch=None, patience=None):
        """Return weighted worlds, with a pre-draw allocation of both strata.

        The default allocation rounds n*p and keeps at least one sample for
        each positive-mass branch. A supplied n_catch fixes another allocation
        before seeing any rollout loss. Zero-mass branches use no worlds.
        """
        sampler = self.sampler(cid, ready, deadline, patience)
        p = sampler.probability(True)
        if not p:
            n_catch = 0
        elif not sampler.probability(False):
            n_catch = self.n
        elif n_catch is None:
            n_catch = max(1, min(self.n - 1, round(self.n * p)))
        n_fail = self.n - n_catch
        if (p and n_catch < 1) or (sampler.probability(False) and n_fail < 1):
            raise ValueError('Each positive-mass branch needs a private world')
        result, offset = [], 0
        for caught, count in ((True, n_catch), (False, n_fail)):
            probability = sampler.probability(caught)
            if not probability:
                continue
            law = sampler.conditional_law(caught)
            uniforms = self.uniforms[sampler.trip][offset:offset + count]
            conditional = _ConditionalTripDraws(self.belief, sampler.trip, law, uniforms)
            for local in range(count):
                original = self.original[offset + local]
                scene = LazyTripScenario(_ConditionalEvents(original, conditional, local),
                                         minute=original.minute, publications=original.publications)
                result.append((probability / count, scene))
            offset += count
        return result
