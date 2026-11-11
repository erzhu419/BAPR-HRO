"""A frozen continuation with one causal downstream transfer rescue."""

from dataclasses import dataclass, replace

import numpy as np

from .committed_transit import WAIT
from .recourse_itinerary_router import nominal_itinerary
from .risk_bounds import CommitmentBounds, ItineraryValue


@dataclass(frozen=True)
class TailRescuePlan:
    """Remaining fixed itinerary, with a rescue on one indexed failure.

    Thresholds are frozen when the plan is issued. Observed boarding advances
    the primary; only a failed attempt at the designated transfer opens rescue.
    """

    primary: tuple[int, ...]
    failure_index: int | None = None
    rescue: tuple[int, ...] = ()
    first_action_now: bool = True
    departures: tuple[tuple[int, int], ...] = ()

    def action(self, state, graph, *, minute=60, lookahead=None):
        if not self.primary:
            return WAIT
        cid = self.primary[0]
        c = graph.connections[cid]
        lookahead = 25 * minute if lookahead is None else lookahead
        if (c.dep_stop != state.stop or cid in state.unavailable_ids
                or c.dep_time > state.time + lookahead
                or (not c.pickup_allowed and c.trip_id != state.current_trip)
                or (c.dep_distribution is not None and c.dep_distribution.support_max < state.time)):
            return WAIT
        if not self.first_action_now:
            transfer = (0 if state.current_trip is None or c.trip_id == state.current_trip
                        else graph.stops[state.stop].min_transfer_time)
            ready = max(state.time, state.last_arrival + transfer)
            if dict(self.departures).get(cid, c.dep_time) > ready + 12 * minute:
                return WAIT
        return ('select', cid)

    def advance(self, action, caught):
        if action[0] != 'select':
            return self
        if not self.primary or action[1] != self.primary[0]:
            raise ValueError('Attempt must belong to the currently issued continuation')
        if not caught:
            return TailRescuePlan(self.rescue if self.failure_index == 0 else (),
                                  first_action_now=False, departures=self.departures)
        if self.failure_index is None or self.failure_index == 0:
            return TailRescuePlan(self.primary[1:], first_action_now=False, departures=self.departures)
        return replace(self, primary=self.primary[1:], failure_index=self.failure_index - 1,
                       first_action_now=False)


@dataclass(frozen=True)
class TailRescueValue:
    upper: float
    primary: ItineraryValue
    rescue_success_gain: float
    failed_probability: float
    rescue_completion_probability: float


@dataclass(frozen=True)
class _TransferFailure:
    index: int
    stop: int
    previous_trip: str | None
    last_arrivals: np.ndarray
    clocks: np.ndarray
    mass: np.ndarray


class TailRescueBounds(CommitmentBounds):
    """Exact fixed-plan loss under the frozen no-occurrence committed model.

    The failure mass at each observable arrival phase integrates the failed
    trip's departure marginal against its legal catch window. Rescue vehicles
    are disjoint from vehicles attempted up to and including that failure, so
    the rescue law is unchanged by reaching the branch. Different phases need
    different clocks, but a single weighted pass evaluates all of them.
    """

    def _primary(self, state, plan, lookahead, indices):
        failures = {}

        def entry(index, last, mass, previous_trip, law, slot, ready):
            if index not in indices:
                return
            clocks = ready + self.patience
            live = (mass > 0.) & (clocks < self.deadline)
            if not np.any(live):
                return
            ready, clocks, incoming_mass = ready[live], clocks[live], mass[live]
            operating = ~law.canceled[slot]
            departures, marginal = law.times[slot][operating], law.marginal(slot)
            operating_mass = marginal[operating]
            before = np.r_[0., np.cumsum(operating_mass)]
            after = np.r_[np.cumsum(operating_mass[::-1])[::-1], 0.]
            failure_mass = incoming_mass * (marginal[~operating].sum()
                + before[np.searchsorted(departures, ready, side='left')]
                + after[np.searchsorted(departures, clocks, side='right')])
            positive = failure_mass > 0.
            if np.any(positive):
                c = self.graph.connections[plan.primary[index]]
                phases = np.broadcast_to(last, mass.shape)[live][positive]
                failures[index] = _TransferFailure(index, c.dep_stop, previous_trip, phases,
                                                    clocks[positive], failure_mass[positive])

        value = self._itinerary(state, plan.primary, first_action_now=plan.first_action_now,
                                lookahead=lookahead, departures=dict(plan.departures),
                                on_entry=entry if indices else None)
        return value, failures

    def _validate_rescue(self, plan):
        index = plan.failure_index
        if index is None or not 0 <= index < len(plan.primary):
            raise ValueError('Rescue requires a designated primary failure')
        trip_by_cid = self.belief._trip_by_cid
        if index and trip_by_cid[plan.primary[index - 1]] == trip_by_cid[plan.primary[index]]:
            raise ValueError('Rescue must follow a transfer or the remaining root attempt')
        attempted = {trip_by_cid[cid] for cid in plan.primary[:index + 1]}
        if any(trip_by_cid[cid] in attempted for cid in plan.rescue):
            raise ValueError('Independent rescue must exclude all attempted primary trips')

    def _combine(self, state, plan, primary, failure, lookahead):
        if failure is None or not plan.rescue:
            return TailRescueValue(primary.upper, primary, 0., 0., 0.)
        failed = replace(state, stop=failure.stop, current_trip=failure.previous_trip)
        rescue = self._itinerary(failed, plan.rescue, first_action_now=False, lookahead=lookahead,
                                 departures=dict(plan.departures),
                                 incoming=(failure.clocks, failure.mass, failure.last_arrivals))
        return TailRescueValue(primary.upper - rescue.success_gain, primary, rescue.success_gain,
                               float(failure.mass.sum()), rescue.success_probability)

    def evaluate(self, state, plan, *, lookahead=None):
        if plan.rescue:
            self._validate_rescue(plan)
        primary, failures = self._primary(state, plan, lookahead, {plan.failure_index} if plan.rescue else ())
        return self._combine(state, plan, primary, failures.get(plan.failure_index), lookahead)


def freeze_tail_plan(bounds, primary, *, first_action_now=True, failure_index=None, rescue=()):
    primary, rescue = tuple(primary), tuple(rescue)
    nominal = bounds.belief.mean_events(bounds.parameters)
    departures = []
    for cid in dict.fromkeys(primary + rescue):
        departure = nominal[cid].departure
        departures.append((cid, bounds.graph.connections[cid].dep_time if departure is None else round(departure)))
    return TailRescuePlan(primary, failure_index, rescue, first_action_now, tuple(departures))


def compile_tail_rescue(bounds, state, primary, *, first_action_now=True, trip_blocks=None, lookahead=None):
    """Compile at most one nominal rescue search, then integrate its actual phases."""
    lookahead = 25 * bounds.minute if lookahead is None else lookahead
    plan = freeze_tail_plan(bounds, primary, first_action_now=first_action_now)
    trip_by_cid = bounds.belief._trip_by_cid
    indices = {i for i in range(1, len(plan.primary))
               if trip_by_cid[plan.primary[i - 1]] != trip_by_cid[plan.primary[i]]}
    primary_value, failures = bounds._primary(state, plan, lookahead, indices)
    if not failures:
        return plan, bounds._combine(state, plan, primary_value, None, lookahead)
    failure = max(failures.values(), key=lambda f: float(f.mass @ (bounds.deadline - f.clocks)))
    order = np.argsort(failure.clocks, kind='stable')
    position = order[np.searchsorted(np.cumsum(failure.mass[order]), failure.mass.sum() / 2)]
    excluded = {trip_by_cid[cid] for cid in plan.primary[:failure.index + 1]}
    representative = replace(state, stop=failure.stop, current_trip=failure.previous_trip,
                             time=int(failure.clocks[position]), last_arrival=int(failure.last_arrivals[position]))
    rescue = nominal_itinerary(bounds.graph, representative, bounds.destination,
                               bounds.belief.mean_events(bounds.parameters),
                               range(len(bounds.graph.connections)), bounds.deadline, lookahead,
                               minute=bounds.minute, excluded_trips=excluded, trip_blocks=trip_blocks)
    if rescue:
        plan = freeze_tail_plan(bounds, plan.primary, first_action_now=first_action_now,
                                failure_index=failure.index, rescue=rescue)
    return plan, bounds._combine(state, plan, primary_value, failure, lookahead)
