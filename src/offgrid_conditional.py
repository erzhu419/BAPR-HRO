"""Use actual-second departure evidence while keeping platform actions exact."""

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from .committed_transit import CommittedState
from .conditional_rescue_policy import ConditionalRescuePolicy, _BoundConditionalRescue
from .interruptible_recourse import RecourseCursor
from .risk_bounds import CommitmentBounds


@dataclass(frozen=True)
class ActualClockScore:
    clock: int
    expected_arrival_or_deadline: float
    success_probability: float
    next_arrival_support: int | None
    method: str
    bracket: tuple


class ActualClockScores:
    """Fill an observed-clock score, without adding it to the old predictive PMF.

    Projection linearly interpolates the two neighbouring model scores. It
    only projects a cost, never the physical event or the legality of boarding.
    Anchoring rebuilds the observed trip with the existing exact-receipt grid
    update and evaluates its conditional suffix. An off-grid observation has
    no likelihood under the old discrete support: the new anchor changes that
    trip's grid and normalizers and does not preserve an old certificate.
    Other trip laws and the original waiting proxy stay frozen.
    """

    def __init__(self, belief, parameters, scores, goal):
        self.belief, self.parameters = belief, parameters
        self.scores, self.goal = scores, goal
        self.clocks = {cid: tuple(sorted(rows)) for cid, rows in scores.rows.items()}

    def at_visible(self, cid, state, mode):
        time = state.time
        if (state.stop != self.scores.root.stop or state.current_trip is not None
                or state.used_connections != self.scores.root.used_connections
                or not self.scores.ready <= time < self.scores.deadline
                or cid not in state.visible_connections
                or (time, cid, 'departure') not in state.public_log
                or cid not in self.clocks):
            return None
        rows = self.scores.rows[cid]
        row = rows.get(time)
        if row is not None:
            return ActualClockScore(time, row.expected_arrival_or_deadline, row.success_probability,
                                    row.next_arrival_support, 'on_grid', (time, time))
        clocks = self.clocks[cid]
        index = bisect_left(clocks, time)
        if not 0 < index < len(clocks):
            return None  # The prototype compares only observed clocks bracketed by the original support.
        lo, hi = bracket = (clocks[index-1], clocks[index])
        if mode == 'projection':
            w = (time-lo) / (hi-lo)
            low, high = rows[lo], rows[hi]
            score = (1-w) * low.expected_arrival_or_deadline + w * high.expected_arrival_or_deadline
            success = (1-w) * low.success_probability + w * high.success_probability
            return ActualClockScore(time, score, success, None, mode, bracket)
        if mode != 'anchor':
            raise ValueError(f'Unknown off-grid scoring mode: {mode}')

        graph, deadline = self.scores.graph, self.scores.deadline
        branch = self.belief.copy()
        branch.observe_connection_event(cid, 'departure', time-graph.connections[cid].dep_time)
        bounds = CommitmentBounds(branch, self.parameters, graph, self.goal, deadline)
        plan = self.scores.paths[cid]
        exact = CommittedState(state.stop, time, None, state.last_arrival, state.used_connections)
        value = bounds._itinerary(exact, plan.primary, first_action_now=True,
                                  lookahead=max(1500, graph.connections[cid].dep_time-time),
                                  departures=dict(plan.departures))
        trip, slot = branch._positions[cid, 'departure']
        law = bounds._law(trip)
        departures = law.marginal(slot)
        assert np.isclose(departures[(law.times[slot] == time) & ~law.canceled[slot]].sum(), 1.)
        support = np.count_nonzero((law.marginal(slot+1) > 0.) & ~law.canceled[slot+1])
        return ActualClockScore(time, time+value.upper, value.success_probability, int(support), mode, bracket)


class ActualClockRescue(ConditionalRescuePolicy):
    def __init__(self, graph, plan, root, deadline, scores, extras, margin, clock_scores, *, mode,
                 censor_unknown_extras=True, wait_value=None):
        super().__init__(graph, plan, root, deadline, scores, extras, margin)
        if mode not in ('anchor', 'projection'):
            raise ValueError(f'Unknown off-grid scoring mode: {mode}')
        self.clock_scores, self.mode = clock_scores, mode
        self.censor_unknown_extras = censor_unknown_extras
        self.waiting_score = (scores if wait_value is None else wait_value).waiting_score

    def bind(self, name):
        return _BoundActualClockRescue(self, name)

class _BoundActualClockRescue(_BoundConditionalRescue):
    def targets(self, state):
        targets = super().targets(state)
        c = self.controller
        if (self.follow is None and c.censor_unknown_extras
                and state.stop == c.root.stop and state.used_connections == c.root.used_connections
                and state.time >= c.scores.ready):
            targets += tuple(c.suffixes)
        return targets

    def action(self, state):
        if self.follow is not None:
            return self.follow.action(state)
        default = self.original.action(state)
        c = self.controller
        if state.stop != c.root.stop or state.used_connections != c.root.used_connections:
            return default
        choices = [(row, cid) for cid in state.visible_connections if cid in c.suffixes
                   if (row := c.clock_scores.at_visible(cid, state, c.mode)) is not None]
        if not choices:
            return default
        self.stats['conditional_comparisons'] += 1
        if default[0] == 'board':
            comparator = c.clock_scores.at_visible(default[1], state, c.mode)
            reference = comparator.expected_arrival_or_deadline if comparator is not None else None
        else:
            reference = c.waiting_score(state, self.original.cursor)
        row, cid = min(choices, key=lambda item: (item[0].expected_arrival_or_deadline, item[1]))
        if reference is None or row.expected_arrival_or_deadline+c.margin >= reference:
            return default
        self.start_suffix(state, c.suffixes[cid])
        self.stats['conditional_boardings'] = 1
        self.trace.append(dict(time=state.time, stop=state.stop, action=('board', cid),
            branch='conditional_rescue', reason='actual_clock_path_score',
            score=row.expected_arrival_or_deadline, reference=reference, clock_method=row.method,
            original_bracket=row.bracket, next_arrival_support=row.next_arrival_support,
            success_probability=row.success_probability))
        return ('board', cid)

    def start_suffix(self, state, controller):
        self.follow = controller.bind(self.policy.name)
        self.follow.cursor = RecourseCursor(self.follow.controller.plan, state.time)
        self.follow.previous = state
