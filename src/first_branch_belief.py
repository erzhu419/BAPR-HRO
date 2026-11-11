"""Exact first-attempt strata of one fixed public ordered-trip posterior.

The branch mask belongs only to private scenario generation. A rollout tail
must start from the original public belief and learn the actual action result;
the sampler does not give the traveller a future catch/fail observation.
"""
from collections.abc import Mapping

import numpy as np

from .evolving_trip_belief import EvolvingTripBelief, _ChainLaw
from .simulator import LazyTripScenario, VehicleEvent
from .trip_marginal_tilt import TripMarginalTilt


class _ConditionalTripDraws:
    def __init__(self, source, trip, law, uniforms):
        self.source, self.trip, self.law, self.uniforms = source, trip, law, uniforms
        self.events = None

    def materialize(self):
        if self.events is None:
            slots = self.source._trips[self.trip]
            worlds = []
            for uniforms in self.uniforms:
                indices = self.law.draw_uniforms(uniforms)
                events = {}
                for i in range(0, len(slots), 2):
                    dep, arr = slots[i:i + 2]
                    canceled = bool(self.law.canceled[i][indices[i]])
                    departure = int(self.law.times[i][indices[i]])
                    arrival = int(self.law.times[i + 1][indices[i + 1]])
                    if canceled and self.source.public_occurrences:
                        departure, arrival = dep.scheduled, arr.scheduled
                    events[dep.cid] = VehicleEvent(999 if canceled else departure - dep.scheduled,
                                                  canceled, departure, arrival)
                worlds.append(events)
            self.events = worlds
        return self.events


class _ConditionalEvents(Mapping):
    def __init__(self, original, conditional, index):
        self.original, self.conditional, self.index = original, conditional, index

    def __getitem__(self, cid):
        source = self.conditional.source
        if source._trip_by_cid[cid] == self.conditional.trip:
            return self.conditional.materialize()[self.index][cid]
        return self.original.events[cid]

    def __iter__(self):
        return iter(self.original.events)

    def __len__(self):
        return len(self.original.events)


class FirstBranchSampler:
    """Rejection-free catch/fail FFBS with the original joint trip law.

    This supports EvolvingTripBelief, its OrderedDurationBelief subclass, and
    the common independent-trip posterior projection TripMarginalTilt,
    whose trips are independent conditional on fixed public parameters. A
    shared-day mixture needs a branch-dependent latent-day posterior and is
    intentionally outside this sampler's interface.

    All clock values use the belief's ticks; patience defaults to12minutes.
    ``ready`` includes platform transfer time. ``deadline`` is absolute.
    """

    def __init__(self, belief, parameters, cid, ready, deadline, patience=None):
        if not isinstance(belief, (EvolvingTripBelief, TripMarginalTilt)):
            raise TypeError('FirstBranchSampler requires an ordered independent-trip belief')
        self.belief = belief.copy()
        self.parameters = {route: dict(row) for route, row in parameters.items()}
        self.cid, self.ready, self.deadline = cid, ready, deadline
        self.patience = 12 * belief.minute if patience is None else patience
        self.trip, self.index = belief._positions[cid, 'departure']
        self.law = self.belief._law(self.trip, self.parameters)
        clocks, canceled = self.law.times[self.index], self.law.canceled[self.index]
        self.catch_mask = ((~canceled) & (ready < deadline) & (clocks >= ready)
                           & (clocks <= min(deadline, ready + self.patience)))
        marginal = self.law.marginal(self.index)
        # Sum both raw masses. In a very unlikely operating tail, cancellation
        # may round to1 while its operating mass remains representable.
        masses = np.array([marginal[self.catch_mask].sum(), marginal[~self.catch_mask].sum()])
        masses /= masses.sum()
        self.probabilities = tuple(float(value) for value in masses)
        self._conditionals = {}

    def probability(self, caught):
        return self.probabilities[0 if caught else 1]

    def conditional_law(self, caught):
        """Private fixed-grid chain; never write its mask into public history."""
        caught = bool(caught)
        if not self.probability(caught):
            raise ValueError('The requested first-attempt branch has zero posterior mass')
        if caught not in self._conditionals:
            mask = self.catch_mask if caught else ~self.catch_mask
            emissions = list(self.law.emissions)
            emissions[self.index] = np.where(mask, emissions[self.index], -np.inf)
            result = _ChainLaw(self.law.times, self.law.initial, self.law._transition,
                               emissions, self.law.canceled, self.law.stochastic)
            # Earlier forward messages are unchanged. Reuse them without
            # traversing them again for every root action/branch.
            self.law._full_forward(self.index)
            result._forward = list(self.law._forward[:self.index + 1])
            result._forward[-1] = np.where(mask, result._forward[-1], -np.inf)
            self._conditionals[caught] = result
        return self._conditionals[caught]

    def sample(self, caught, rng, n):
        """Draw n complete worlds, conditional on this action's one branch."""
        law = self.conditional_law(caught)
        # Other trips retain their independent original posterior and lazy
        # draws. The unneeded original draw for this trip is never materialized.
        original = self.belief.sample(self.parameters, rng, n)
        uniforms = rng.random((n, len(law.times)))[:, ::-1]
        uniforms.flags.writeable = False
        conditional = _ConditionalTripDraws(self.belief, self.trip, law, uniforms)
        return [LazyTripScenario(_ConditionalEvents(scene, conditional, i), minute=scene.minute,
                                 publications=scene.publications)
                for i, scene in enumerate(original)]

    def conditional_distribution(self, cid, kind, caught=True):
        """Operating clock CDF atoms/weights and cancellation, on this stratum."""
        if not self.probability(caught):
            raise ValueError('The requested first-attempt branch has zero posterior mass')
        trip, index = self.belief._positions[cid, kind]
        law = (self.conditional_law(caught) if trip == self.trip
               else self.belief._law(trip, self.parameters))
        return self.belief._event_distribution(law, index)

    def stratified(self, rng, n_catch=2, n_fail=2):
        """Weighted complete worlds; both positive-probability strata included.

        These weights define an unbiased stratified Monte Carlo estimator for
        any fixed causal tail, not a deterministic certificate for the full
        posterior or the physical travel day.
        """
        bank = []
        if not self.probability(True):
            n_fail += n_catch
        elif not self.probability(False):
            n_catch += n_fail
        for caught, n in ((True, n_catch), (False, n_fail)):
            probability = self.probability(caught)
            if not probability:
                continue
            if n < 1:
                raise ValueError('Each positive-mass stratum needs at least one sample')
            bank.extend((probability / n, world) for world in self.sample(caught, rng, n))
        return bank
