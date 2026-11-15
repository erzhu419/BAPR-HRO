"""Exact terminal-platform policy values on independent named-trip factors.

The subset state is sufficient only after the scorer's semantic read test.
Physical terminal values, hypothetical WAIT marks and observed BOARD scores
are different quantities. All work uses the hosting reference's budget.
"""
import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v33.nodes import Platform
from experiments.reconstruction_v41.physical_posterior import posterior_transition
from experiments.reconstruction_v52.reference import point
from experiments.reconstruction_v53.relevance import history_key
from src.plan_waiting_controls import CombinedWaitingOpportunity, SingleTargetOpportunity


def _slots(model, ride):
    path = []
    for cid in ride.path:
        path.append(cid)
        if cid in ride.exits:
            if ride.exits[cid] is not None:
                return None
            break
    else:
        return None
    positions = [model._positions[cid, kind] for cid in path
                 for kind in ('departure', 'arrival')]
    trip = positions[0][0]
    slots = tuple(slot for _, slot in positions)
    if (any(other != trip for other, _ in positions)
            or any(right <= left for left, right in zip(slots, slots[1:]))):
        return None
    return trip, slots


def _physical_marks(host, ride, history, specification):
    """Batch V53's executed-slot integral conditional on each departure atom."""
    trip, slots = specification
    law = host.root_law.law(trip, history)
    deadline = host.deadline
    last = slots[-1]
    value = np.where(~law.canceled[last] & (law.times[last] < deadline),
                     law.times[last], deadline)
    for left, right in reversed(tuple(zip(slots, slots[1:]))):
        kernel = posterior_transition(law, left)
        for hidden in range(left+1, right):
            kernel = kernel @ posterior_transition(law, hidden)
        legal = (~law.canceled[right] & (law.times[right] < deadline))[None, :]
        legal = legal & (law.times[right][None, :] >= law.times[left][:, None])
        value = deadline + (kernel*legal) @ (value-deadline)
    first = slots[0]
    mass, times = law.marginal(first), law.times[first]
    operating = ~law.canceled[first]
    clocks = np.unique(times[operating & (mass > 0)])
    probabilities = np.array([mass[operating & (times == t)].sum() for t in clocks])
    marks = np.array([mass[operating & (times == t)] @ value[operating & (times == t)] / p
                      for t, p in zip(clocks, probabilities)])
    return clocks, probabilities, float(mass[~operating].sum()), marks


class _Terminal:
    def __init__(self, host, platform, entry, history, depth, after, specifications):
        self.host, self.platform, self.history = host, platform, history
        self.entry, self.after, self.depth = float(entry), float(after), depth
        self.ready = self.entry+host.graph.stops[platform.stop].min_transfer_time
        self.deadline = host.deadline
        self.specifications = specifications
        seen = {cid for cid in platform.rides if any(
            r[:3] == (cid, 'departure', 'observed') and entry <= r[3] <= after
            for r in history._records.get(host.model._positions[cid, 'departure'][0], ()))}
        self.cids = tuple(cid for cid in platform.rides if cid not in seen)
        self.ranks = host.planner.ranks(platform)
        self.physical, self.board, self.wait = {}, {}, {}
        self.combined = None
        self.memo = {}

    def spend(self, kind):
        if not self.host._available():
            return False
        self.host.stats['states'] += 1
        self.host.stats[kind] = self.host.stats.get(kind, 0)+1
        return True

    def unknown(self, clock):
        self.host.stats['terminal_dp_budget_stops'] = self.host.stats.get('terminal_dp_budget_stops', 0)+1
        return self.host._unknown(clock)

    def marks(self, cid):
        if cid not in self.physical:
            if not self.spend('terminal_platform_integrations'):
                return None
            self.physical[cid] = _physical_marks(self.host, self.platform.rides[cid],
                                                self.history, self.specifications[cid])
        return self.physical[cid]

    def solve(self):
        if not self.cids or self.ready >= self.deadline:
            return point(self.deadline)
        for cid in self.cids:
            if self.marks(cid) is None:
                return self.unknown(self.entry)
        if len(self.cids) == 1:
            # Every policy takes the only legal offer: BOARD <= H = WAIT.
            times, mass, _, marks = self.physical[self.cids[0]]
            legal = (times >= self.ready) & (times > self.after) & (times < self.deadline)
            value = self.deadline+mass[legal] @ (marks[legal]-self.deadline)
            self.host.stats['terminal_singleton_integrations'] = self.host.stats.get(
                'terminal_singleton_integrations', 0)+1
            return point(value)
        self.times = np.array(sorted({float(t) for cid in self.cids
            for t, p in zip(*self.physical[cid][:2])
            if p > 0 and t >= self.entry and t > self.after and t < self.deadline}))
        self.hazards = np.zeros((len(self.times), len(self.cids)))
        self.actual = np.full_like(self.hazards, self.deadline)
        for i, cid in enumerate(self.cids):
            times, mass, cancel, marks = self.physical[cid]
            absent = cancel+mass[times < self.entry].sum()
            tail = np.r_[np.cumsum(mass[::-1])[::-1], 0.]
            denominator = absent+tail[np.searchsorted(times, self.times, side='left')]
            for t, p, value in zip(times, mass, marks):
                if t in self.times:
                    j = int(np.searchsorted(self.times, t))
                    self.hazards[j, i] = p/denominator[j] if denominator[j] > 0 else 0.
                    self.actual[j, i] = value
        return self.event(0, (1 << len(self.cids))-1, self.depth)

    def board_score(self, cid, clock):
        key = cid, float(clock)
        if key not in self.board:
            if not self.spend('terminal_platform_board_scores'):
                return None
            h = self.history.clone()
            h.observe(cid, 'departure', clock)
            self.board[key] = self.host.planner.compile_ride(self.platform.rides[cid],
                self.host._score_history(h), 'rolling_fixed').score(clock)
            if self.board[key] is None:
                raise UnsupportedHistory('Terminal BOARD lies outside conditional support')
        return self.board[key]

    def wait_score(self, clock, remaining):
        key = float(clock), remaining
        if key not in self.wait:
            if self.combined is None:
                if not self.spend('terminal_platform_wait_compiles'):
                    return None
                platform = Platform(self.platform.stop,
                    {cid:self.platform.rides[cid] for cid in self.cids})
                reference = self.host.planner.platform_reference(platform,
                    self.host._score_history(self.history), self.entry, self.ready, 'rolling_fixed')
                # Preserve the original menu's scheduled tie metadata.
                reference.ranks = {cid:self.ranks[cid] for cid in self.cids}
                self.combined = CombinedWaitingOpportunity(SingleTargetOpportunity(reference))
            if not self.spend('terminal_platform_wait_scores'):
                return None
            seen = {cid for i, cid in enumerate(self.cids) if not remaining & (1 << i)}
            self.wait[key] = self.combined.wait_score(clock, seen)
        return self.wait[key]

    def event(self, j, mask, depth):
        if not mask or j == len(self.times):
            return point(self.deadline)
        key = j, mask, depth
        if key in self.memo:
            self.host.stats['cache_hits'] += 1
            return self.memo[key]
        clock = float(self.times[j])
        if not self.spend('terminal_platform_states'):
            return self.unknown(clock)
        self.host.stats['terminal_dp_states'] = self.host.stats.get('terminal_dp_states', 0)+1
        outcomes = [(1., 0)]
        for i in range(len(self.cids)):
            if mask & (1 << i):
                hazard = self.hazards[j, i]
                outcomes = [(p*q, departed | bit) for p, departed in outcomes
                            for q, bit in ((1-hazard, 0), (hazard, 1 << i)) if p*q > 0]
        # Match V53's mass-first traversal while retaining the entire support.
        outcomes.sort(key=lambda row:(-row[0], row[1]))
        lo = hi = consumed = 0.
        complete = True
        for mass, departed in outcomes:
            if not self.host._available():
                missing = max(0., 1.-consumed)
                lo += missing*clock
                hi += missing*self.deadline
                complete = False
                self.unknown(clock)
                break
            remaining = mask ^ departed
            if not departed or clock < self.ready:
                value = self.event(j+1, remaining, depth)
            else:
                self.host.stats['decisions'] += 1
                active = [i for i in range(len(self.cids)) if departed & (1 << i)]
                if self.host.mode == 'optimal' or depth > 0:
                    following_depth = depth if self.host.mode == 'optimal' else depth-1
                    waiting = self.event(j+1, remaining, following_depth)
                    boarding = min(self.actual[j, i] for i in active)
                    value = min(boarding, waiting[0]), min(boarding, waiting[1]), waiting[2]
                elif not remaining:
                    # WAIT costs H. Still choose among simultaneous BOARD offers
                    # using the declared scorer, not the physical expectation.
                    scores = [(self.board_score(self.cids[i], clock), i) for i in active]
                    if any(score is None for score, _ in scores):
                        value = self.unknown(clock)
                    else:
                        best = min(scores, key=lambda row:(row[0], self.ranks[self.cids[row[1]]]))[1]
                        value = point(self.actual[j, best])
                else:
                    scores = [(self.board_score(self.cids[i], clock), i) for i in active]
                    waiting = self.wait_score(clock, remaining)
                    if waiting is None or any(score is None for score, _ in scores):
                        value = self.unknown(clock)
                    else:
                        score, best = min(scores, key=lambda row:(row[0], self.ranks[self.cids[row[1]]]))
                        value = (point(self.actual[j, best]) if score <= waiting+1e-9
                                 else self.event(j+1, remaining, depth))
            lo += mass*value[0]
            hi += mass*value[1]
            consumed += mass
            complete &= value[2]
            self.host.stats['chance_outcomes'] += 1
        result = max(clock, lo), min(self.deadline, hi), complete
        self.memo[key] = result
        return result


def terminal_platform(reference, platform, entry, history, depth, after=float('-inf')):
    """Return absolute value bounds; None means the structure is inadmissible."""
    reference.stats['terminal_dp_calls'] = reference.stats.get('terminal_dp_calls', 0)+1
    if entry >= reference.deadline:
        return point(reference.deadline)
    specifications = {}
    trips = set()
    for cid, ride in platform.rides.items():
        specification = _slots(reference.model, ride)
        if specification is None:
            return None
        trip, _ = specification
        if trip in trips or reference.reads.ride_trips(ride) != frozenset((trip,)):
            return None
        trips.add(trip)
        specifications[cid] = specification
    if set(reference.reads.observation_cids(platform.stop, trips)) != set(platform.rides):
        return None
    reference.stats['terminal_dp_admitted'] = reference.stats.get('terminal_dp_admitted', 0)+1
    key = ('terminal_platform', reference._structure(platform), float(entry), float(after),
           history_key(history, trips), depth, reference.mode)
    old = reference.memo.get(key)
    if old is not None and old[2]:
        reference.stats['cache_hits'] += 1
        reference.stats['terminal_dp_hits'] = reference.stats.get('terminal_dp_hits', 0)+1
        return old
    if not reference._available():
        reference.stats['terminal_dp_budget_stops'] = reference.stats.get('terminal_dp_budget_stops', 0)+1
        return old if old is not None else reference._unknown(entry)
    reference.stats['states'] += 1
    reference.stats['terminal_platform_calls'] = reference.stats.get('terminal_platform_calls', 0)+1
    try:
        result = _Terminal(reference, platform, entry, history, depth, after, specifications).solve()
    except UnsupportedHistory as error:
        result = reference._unknown(entry, str(error))
    if old is not None:
        result = reference._intersect(result, min(float(entry), reference.deadline), old)
    if result[2]:
        reference.stats['terminal_dp_complete'] = reference.stats.get('terminal_dp_complete', 0)+1
    reference.memo[key] = result
    return result
