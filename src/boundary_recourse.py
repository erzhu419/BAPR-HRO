"""Exact arrival-conditioned choice among frozen continuation witnesses."""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from .evolving_trip_belief import _ChainLaw
from .risk_bounds import CommitmentBounds
from .tail_rescue import TailRescueBounds, TailRescuePlan, freeze_tail_plan


@dataclass(frozen=True)
class ArrivalPhase:
    arrival: int
    probability: float
    conditional_uppers: tuple[float, ...]
    selected: int


@dataclass(frozen=True)
class BoundaryRecoursePlan:
    prefix: TailRescuePlan
    tails: tuple[TailRescuePlan, ...]
    choices: tuple[tuple[int, int], ...]

    def tail_for(self, arrival):
        # The first frozen tail also specifies behavior outside the model grid.
        return self.tails[dict(self.choices).get(arrival, 0)]


@dataclass(frozen=True)
class BoundaryRecourseValue:
    upper: float
    fixed_tail_uppers: tuple[float, ...]
    prefix_success_probability: float
    phases: tuple[ArrivalPhase, ...]
    seconds: float


@dataclass(frozen=True)
class BoundaryTailSelection:
    selected: int
    conditional_uppers: tuple[float, ...]
    seconds: float

    @property
    def upper(self):
        return self.conditional_uppers[self.selected]


def choose_boundary_tail(bounds, observed_state, tails, *, lookahead=None):
    """Choose a frozen tail using the externally updated public posterior.

    ``bounds`` must belong to the new observed-history context. This also
    supports arrival clocks outside the earlier planning grid; its value is
    conditional on the supplied new posterior, not certified by the old grid's
    integrated upper. Candidate paths and their issued thresholds stay fixed.
    """
    started = perf_counter()
    tails = tuple(tails)
    if not tails:
        raise ValueError('Boundary selection needs a frozen tail family')
    if any(tail.first_action_now for tail in tails):
        raise ValueError('Boundary tails must use frozen post-arrival waiting thresholds')
    values = tuple(TailRescueBounds.evaluate(bounds, observed_state, tail,
                                            lookahead=lookahead).upper for tail in tails)
    return BoundaryTailSelection(int(np.argmin(values)), values, perf_counter() - started)


def _arrival_conditioned(law, slot, clock):
    """Condition on an observed operating arrival without rebuilding its grid."""
    mask = (law.times[slot] == clock) & ~law.canceled[slot]
    emissions = list(law.emissions)
    emissions[slot] = np.where(mask, emissions[slot], -np.inf)
    result = _ChainLaw(law.times, law.initial, law._transition,
                       emissions, law.canceled, law.stochastic)
    law._full_forward(slot)
    result._forward = list(law._forward[:slot + 1])
    result._forward[-1] = np.where(mask, result._forward[-1], -np.inf)
    return result


class BoundaryRecourseBounds(TailRescueBounds):
    """Value of a restricted causal policy under the full current finite grid.

    Follow a same-trip prefix; on successful arrival choose the best expected
    frozen tail for that observed clock. Any prefix failure incurs the deadline
    loss. This is a feasible-policy upper, not the full causal optimum.

    The observed last arrival is a Markov separator: all prefix catch events
    precede it, so future slots of that trip can use its exact conditional law.
    Other trips retain their independent root posterior. No future realized
    tail clock participates in the choice.
    """

    def prefix_arrivals(self, state, prefix, *, first_action_now=True,
                        departures=None, lookahead=None):
        """Unnormalized successful-arrival masses for frozen candidate generation."""
        prefix = tuple(prefix)
        if not prefix:
            raise ValueError('A boundary policy needs a nonempty prefix')
        trip = self.belief._trip_by_cid[prefix[0]]
        if any(self.belief._trip_by_cid[cid] != trip for cid in prefix):
            raise ValueError('The boundary prefix must stay on one trip')
        boundary_stop = self.graph.connections[prefix[-1]].arr_stop
        prefix_plan = freeze_tail_plan(self, prefix, first_action_now=first_action_now)
        if departures is not None:
            prefix_plan = replace(prefix_plan, departures=tuple(dict(departures).items()))
        prefix_bounds = CommitmentBounds(self.belief, self.parameters, self.graph,
                                         boundary_stop, self.deadline)
        prefix_bounds._laws, prefix_bounds._posterior_edges = self._laws, self._posterior_edges
        prefix_bounds.stats = self.stats
        return prefix_bounds._itinerary(state, prefix, first_action_now=first_action_now,
            lookahead=lookahead, departures=dict(prefix_plan.departures))

    def arrival_bounds(self, state, prefix, clock):
        """Return child state and analytic bounds conditional on this arrival.

        The child overrides analytic trip laws; its belief remains the frozen
        root and is not a new public-history context for nominal path searches.
        """
        trip, slot = self.belief._positions[prefix[-1], 'arrival']
        law = self._law(trip)
        mask = (law.times[slot] == clock) & ~law.canceled[slot]
        if not law.marginal(slot)[mask].sum():
            raise ValueError('The arrival has zero operating mass on the frozen grid')
        c = self.graph.connections[prefix[-1]]
        child_state = replace(state, stop=c.arr_stop, time=int(clock), last_arrival=int(clock),
                              current_trip=c.trip_id,
                              unavailable_ids=state.unavailable_ids | frozenset(prefix))
        child = TailRescueBounds(self.belief, self.parameters, self.graph, self.destination, self.deadline)
        child._laws = self._laws.copy()
        child._laws[trip] = _arrival_conditioned(law, slot, clock)
        # Usable incoming-trip edges follow the separator, so their backward
        # likelihoods are unchanged. Other-trip edges also reuse their cache.
        child._posterior_edges, child.stats = self._posterior_edges, self.stats
        return child_state, child

    def evaluate(self, state, prefix, tails, *, first_action_now=True,
                 departures=None, lookahead=None):
        started = perf_counter()
        prefix, tails = tuple(prefix), tuple(tails)
        if not prefix or not tails:
            raise ValueError('A boundary policy needs a nonempty prefix and frozen tail family')
        trip, arrival_slot = self.belief._positions[prefix[-1], 'arrival']
        boundary_stop = self.graph.connections[prefix[-1]].arr_stop
        for tail in tails:
            if tail.first_action_now:
                raise ValueError('Boundary tails must use frozen post-arrival waiting thresholds')
            for cid in tail.primary + tail.rescue:
                target_trip, slot = self.belief._positions[cid, 'departure']
                if cid in prefix or (target_trip == trip and slot <= arrival_slot):
                    raise ValueError('A boundary tail cannot reuse past incoming-trip slots')
                self._law(target_trip)
        prefix_plan = freeze_tail_plan(self, prefix, first_action_now=first_action_now)
        if departures is not None:
            prefix_plan = replace(prefix_plan, departures=tuple(dict(departures).items()))
        incoming = self.prefix_arrivals(state, prefix, first_action_now=first_action_now,
                                       lookahead=lookahead, departures=prefix_plan.departures)
        cap = self.deadline - state.time
        fixed_gain = np.zeros(len(tails))
        adaptive_gain, phases = 0., []
        # The prefix propagator retains service states; only operating arrivals
        # have positive success mass. Combine equal public clocks explicitly.
        for clock in np.unique(incoming.arrival_times[incoming.arrival_mass > 0.]):
            probability = float(incoming.arrival_mass[incoming.arrival_times == clock].sum())
            child_state, child = self.arrival_bounds(state, prefix, clock)
            if boundary_stop == self.destination:
                values = np.zeros(len(tails))
            else:
                values = np.array([child.evaluate(child_state, tail, lookahead=lookahead).upper for tail in tails])
            selected = int(np.argmin(values))
            gain = self.deadline - clock - values
            fixed_gain += probability * gain
            adaptive_gain += probability * gain[selected]
            phases.append(ArrivalPhase(int(clock), probability, tuple(map(float, values)), selected))
        plan = BoundaryRecoursePlan(prefix_plan, tails, tuple((p.arrival, p.selected) for p in phases))
        value = BoundaryRecourseValue(float(cap - adaptive_gain), tuple(map(float, cap - fixed_gain)),
            incoming.success_probability, tuple(phases), perf_counter() - started)
        return plan, value
