"""Public-event rescue rule scored by ordered-chain itinerary integration."""

from dataclasses import dataclass

import numpy as np

from .background_policy import IssuedPolicy
from .committed_transit import CommittedState
from .interruptible_recourse import InterruptibleRecourse, RecourseCursor
from .public_trip_conditionals import condition_on_departure
from .risk_bounds import CommitmentBounds


@dataclass(frozen=True)
class DepartureScore:
    clock: int
    probability: float
    expected_arrival_or_deadline: float
    success_probability: float
    next_arrival_support: int


class ConditionalItineraries:
    """Compile event-indexed scores from the original law, without sample worlds."""

    def __init__(self, belief, parameters, graph, root, goal, deadline, paths, *, point_forecast=False):
        self.graph, self.root, self.deadline = graph, root, deadline
        self.ready = max(root.time, root.last_arrival + graph.stops[root.stop].min_transfer_time)
        self.paths, self.rows, self.excluded = paths, {}, {}
        bounds = CommitmentBounds(belief, parameters, graph, goal, deadline)
        nominal = belief.mean_events(parameters) if point_forecast else None
        for cid, plan in paths.items():
            trip, slot = belief._positions[cid, 'departure']
            prior = bounds._law(trip)
            probability = prior.marginal(slot)
            departures = dict(plan.departures)
            rows = {}
            try:
                for i in np.flatnonzero(~prior.canceled[slot] & (probability > 0.)):
                    clock = int(prior.times[slot][i])
                    if not root.time <= clock < deadline:
                        continue
                    condition = condition_on_departure(belief, parameters, cid, clock)
                    assert condition is not None
                    state = CommittedState(root.stop, clock, None, root.last_arrival, root.used_connections)
                    bounds._laws[trip] = condition.law
                    # The conditioned mask is at the first departure only.
                    # Edges after it depend on future messages, not on that mask.
                    value = bounds._itinerary(state, plan.primary, first_action_now=True,
                                              lookahead=max(1500, graph.connections[cid].dep_time-clock),
                                              departures=departures)
                    event = condition.marginal(cid)
                    score = (clock+value.upper if nominal is None else self._mean_clock_score(
                             condition, bounds, nominal, trip, plan.primary, departures, clock))
                    rows[clock] = DepartureScore(clock, float(probability[i]), score,
                        value.success_probability, int(np.count_nonzero(event.probabilities)))
            except ValueError as error:
                self.excluded[cid] = str(error)
            finally:
                bounds._laws[trip] = prior
            if rows and cid not in self.excluded:
                self.rows[cid] = rows

    def _mean_clock_score(self, condition, bounds, nominal, trip, path, departures, clock):
        """Diagnostic: replace random operating clocks by their means on the same path."""
        now, previous_trip = clock, None
        for i, cid in enumerate(path):
            c = self.graph.connections[cid]
            if c.trip_id == trip:
                dep = condition.marginal(cid,'departure').mean_operating_clock
                arr = condition.marginal(cid,'arrival').mean_operating_clock
            else:
                dep, arr = nominal[cid].departure, nominal[cid].arrival
            if dep is None or arr is None:
                return self.deadline
            dep, arr = round(dep), max(round(dep),round(arr))
            ready = int(bounds._ready(np.array([now]), cid, previous_trip, first_action_now=i==0,
                                      departures=departures)[0])
            if not ready <= dep <= ready+bounds.patience or arr >= self.deadline:
                return self.deadline
            now, previous_trip = arr, c.trip_id
        return now

    def future(self, cid, state, earliest, latest):
        if any(target == cid and kind in ('departure', 'published_departure', 'published_cancel')
               for _, target, kind in state.public_log):
            return ()
        values = self.rows.get(cid, {}).values()
        surviving_mass = 1. - sum(row.probability for row in values if row.clock <= state.time)
        if surviving_mass <= 0.:
            return ()
        return tuple((row.clock, row.probability/surviving_mass, row.expected_arrival_or_deadline)
                     for row in values if earliest <= row.clock <= latest)

    def waiting_score(self, state, cursor):
        """A/B waiting proxy for selecting a causal action.

        Competing targets are independently filtered by their own departure
        evidence. After the current V13 window, the rescue may still depart.
        This proxy lets it compete early once physically ready, without the
        original rescue window; extra public events are ignored for scoring.
        """
        if not cursor.plan.primary:
            return self.deadline
        current = cursor.plan.primary[0]
        if current not in self.rows:
            return None
        lo = max(self.ready, state.time+1, cursor.ready or state.time)
        hi = cursor.expiry if cursor.expiry is not None else self.deadline
        first = self.future(current, state, lo, hi)
        rescue = ()
        if cursor.plan.failure_index == 0 and cursor.plan.rescue:
            cid = cursor.plan.rescue[0]
            if cid not in self.rows:
                return None
            rescue = self.future(cid, state, max(self.ready, state.time+1), self.deadline)
        first += ((self.deadline, max(0., 1-sum(p for _, p, _ in first)), self.deadline),)
        rescue += ((self.deadline, max(0., 1-sum(p for _, p, _ in rescue)), self.deadline),)
        return sum(pa*pb*(va if ta <= tb else vb)
                   for ta, pa, va in first for tb, pb, vb in rescue)


class ConditionalRescuePolicy:
    def __init__(self, graph, plan, root, deadline, scores, extras, margin):
        self.original = InterruptibleRecourse(graph, plan, root, deadline, mode='first_visible')
        self.scores, self.extras, self.margin = scores, extras, margin
        self.root = root
        self.suffixes = {cid: InterruptibleRecourse(graph, scores.paths[cid], root, deadline,
                         mode='public_failure') for cid in extras if cid in scores.rows}

    def bind(self, name):
        return _BoundConditionalRescue(self, name)


class _BoundConditionalRescue:
    def __init__(self, controller, name):
        self.controller = controller
        self.original = controller.original.bind(name)
        self.trace, self.stats = self.original.trace, self.original.stats
        self.stats.update(conditional_comparisons=0, conditional_boardings=0)
        self.follow = None
        self.policy = IssuedPolicy(self.action, self.original.policy.applicable, name)

    def targets(self, state):
        if self.follow is not None:
            return self.follow.targets(state)
        return self.original.targets(state)

    def action(self, state):
        if self.follow is not None:
            return self.follow.action(state)
        default = self.original.action(state)
        c = self.controller
        if state.stop != c.root.stop or state.used_connections != c.root.used_connections:
            return default
        available = [cid for cid in state.visible_connections if cid in c.suffixes
                     and state.time in c.scores.rows[cid]]
        if not available:
            return default
        self.stats['conditional_comparisons'] += 1
        if default[0] == 'board':
            row = c.scores.rows.get(default[1], {}).get(state.time)
            reference = row.expected_arrival_or_deadline if row is not None else None
        else:
            reference = c.scores.waiting_score(state, self.original.cursor)
        row, cid = min(((c.scores.rows[cid][state.time], cid) for cid in available),
                       key=lambda item: (item[0].expected_arrival_or_deadline, item[1]))
        if reference is None or row.expected_arrival_or_deadline+c.margin >= reference:
            return default
        self.follow = c.suffixes[cid].bind(self.policy.name)
        self.follow.cursor = RecourseCursor(self.follow.controller.plan, state.time)
        self.follow.previous = state
        self.stats['conditional_boardings'] = 1
        self.trace.append(dict(time=state.time, stop=state.stop, action=('board', cid),
            branch='conditional_rescue', reason='conditional_path_score',
            score=row.expected_arrival_or_deadline, reference=reference,
            next_arrival_support=row.next_arrival_support,
            success_probability=row.success_probability))
        return ('board', cid)
