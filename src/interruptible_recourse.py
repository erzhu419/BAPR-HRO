"""Public-event interpreters for one frozen primary/failure-rescue itinerary."""

from dataclasses import dataclass, replace
from time import perf_counter

from .background_policy import IssuedPolicy
from .platform_transit import ALIGHT, OBSERVE, WAIT


@dataclass(frozen=True)
class RecourseCursor:
    plan: object
    epoch: int
    branch: str = 'primary'
    opening: int | None = None
    ready: int | None = None
    expiry: int | None = None
    reference_window: tuple | None = None


@dataclass(frozen=True)
class RecourseDecision:
    action: tuple
    cursor: RecourseCursor
    reason: str
    events: tuple = ()


class InterruptibleRecourse:
    """Compile once; bind a fresh execution cursor for each posterior world.

    fixed_timeout retains each declared attempt through its inclusive expiry.
    public_failure can abandon a publicly failed target immediately, retaining
    the first-rescue boarding window of its fixed-timeout counterpart.
    first_visible additionally takes an admissible visible rescue before failure;
    that is a different opportunistic policy and can be worse than waiting.

    The pure decide/advance core reads only the public PlatformState. There is
    no scenario, future-clock lookup, posterior inference or candidate search.
    Stop-skipping is not a publication type in the current platform model.
    """

    def __init__(self, graph, plan, root, deadline, *, mode='public_failure', minute=60,
                 lookahead=None):
        if mode not in ('fixed_timeout', 'public_failure', 'first_visible'):
            raise ValueError('Unknown interruptible policy mode')
        self.graph, self.plan, self.root, self.deadline = graph, plan, root, deadline
        self.mode, self.minute = mode, minute
        self.lookahead = 25 * minute if lookahead is None else lookahead
        self.patience = 12 * minute

    def _progress(self, state):
        remaining, count = self.plan, 0
        while remaining.primary and remaining.primary[0] in state.used_connections:
            remaining = remaining.advance(('select', remaining.primary[0]), True)
            count += 1
        return remaining, count

    def applicable(self, state):
        remaining, count = self._progress(state)
        if not remaining.primary:
            return False
        expected = frozenset(self.plan.primary[:count])
        previous = self.plan.primary[count - 1] if count else self.root.last_connection
        stop = self.graph.connections[remaining.primary[0]].dep_stop
        previous_trip = (self.graph.connections[previous].trip_id if count
                         else self.root.current_trip)
        return (state.used_connections - self.root.used_connections == expected
                and state.stop == stop and state.last_connection == previous
                and state.current_trip in (None, previous_trip))

    def initial(self, state):
        plan, count = self._progress(state)
        # Compilation/publication latency cannot restart an already issued root
        # timer. After a ridden prefix, local_since is the actual arrival phase.
        return RecourseCursor(plan, state.local_since if count else self.root.time)

    def _prepare(self, state, cursor):
        if cursor.opening is not None:
            return cursor
        cid = cursor.plan.primary[0]
        c = self.graph.connections[cid]
        transfer = (self.graph.stops[state.stop].min_transfer_time
                    if state.current_trip is None and state.last_connection is not None else 0)
        platform_ready = state.last_arrival + transfer
        if cursor.plan.first_action_now:
            opening = cursor.epoch
        else:
            def aligned(clock):
                return cursor.epoch + self.minute * ((max(0, clock - cursor.epoch)
                                                      + self.minute - 1) // self.minute)
            opening = aligned(c.dep_time - self.lookahead)
            nominal = dict(cursor.plan.departures).get(cid, c.dep_time)
            if nominal > max(opening, platform_ready) + self.patience:
                opening = aligned(nominal - self.patience)
        if c.dep_distribution is not None and opening > c.dep_distribution.support_max:
            opening = self.deadline
        ready = min(self.deadline, max(opening, platform_ready))
        expiry = min(self.deadline, ready + self.patience)
        if cursor.reference_window is not None:
            old_opening, old_ready, old_expiry = cursor.reference_window
            opening = min(opening, old_opening)
            ready = max(platform_ready, min(ready, old_ready))
            expiry = max(expiry, old_expiry)
        return replace(cursor, opening=opening, ready=ready, expiry=expiry)

    @staticmethod
    def _public_failure(state, cid):
        for clock, target, kind in state.public_log:
            if target != cid or clock > state.time:
                continue
            if kind == 'published_cancel':
                return 'published_cancel'
            if kind == 'published_departure':
                return 'published_departure'
            if kind == 'departure':
                # Called only after checking whether this departure is still
                # boardable under the issued window and physical transfer rule.
                return 'local_departure_unboardable'
        return None

    def _failed(self, cursor, state, *, preserve_window=False):
        cid = cursor.plan.primary[0]
        plan = cursor.plan.advance(('select', cid), False)
        result = RecourseCursor(plan, state.time, 'rescue' if plan.primary else 'stopped')
        if preserve_window and plan.primary and state.time < cursor.expiry:
            c = self.graph.connections[plan.primary[0]]
            alights = state.current_trip is not None and state.current_trip != c.trip_id
            reference = replace(state, time=cursor.expiry,
                current_trip=None if alights else state.current_trip,
                last_arrival=cursor.expiry if alights else state.last_arrival)
            old = self._prepare(reference, RecourseCursor(plan, cursor.expiry, 'rescue'))
            result = replace(result, reference_window=(old.opening, old.ready, old.expiry))
        return result

    def decide(self, state, cursor):
        if not state.initialized:
            return RecourseDecision(OBSERVE, cursor, 'initial_observation')
        events = []
        while cursor.plan.primary:
            cid = cursor.plan.primary[0]
            c = self.graph.connections[cid]
            if c.dep_stop != state.stop:
                cursor = replace(cursor, plan=replace(cursor.plan, primary=()), branch='stopped')
                events.append(('off_path', cid, state.time))
                break
            # Alighting starts physical transfer time immediately, including
            # while the frozen nominal attempt opening is still in the future.
            if state.current_trip is not None and state.current_trip != c.trip_id:
                return RecourseDecision(ALIGHT, cursor, 'transfer_alight', tuple(events))
            cursor = self._prepare(state, cursor)
            if cid in state.visible_connections and cursor.ready <= state.time <= cursor.expiry:
                return RecourseDecision(('board', cid), cursor, 'board_target', tuple(events))
            reason = (self._public_failure(state, cid) if self.mode != 'fixed_timeout' else None)
            if reason is None and state.time >= cursor.expiry:
                reason = 'attempt_expired'
            if reason is not None:
                events.append((reason, cid, state.time))
                cursor = self._failed(cursor, state, preserve_window=reason != 'attempt_expired')
                continue
            if (self.mode == 'first_visible' and cursor.plan.failure_index == 0
                    and cursor.plan.rescue and cursor.plan.rescue[0] in state.visible_connections):
                rescue = self._prepare(state, self._failed(cursor, state))
                if rescue.ready <= state.time <= rescue.expiry:
                    events.append(('visible_rescue', cid, state.time))
                    cursor = rescue
                    continue
            staying = any(self.graph.connections[target].trip_id == state.current_trip
                          for target in state.visible_connections) if state.current_trip is not None else False
            if staying:
                # WAIT would physically ride this vehicle. A frozen target
                # outside its opening cannot be skipped by pretending to wait.
                return RecourseDecision(ALIGHT, cursor, 'avoid_unissued_inertia', tuple(events))
            reason = 'wait_opening' if state.time < cursor.ready else 'wait_target'
            return RecourseDecision(WAIT, cursor, reason, tuple(events))
        action = ALIGHT if state.current_trip is not None else WAIT
        return RecourseDecision(action, cursor, 'stop_at_deadline', tuple(events))

    def advance(self, before, after, cursor):
        boarded = after.used_connections - before.used_connections
        if not boarded:
            return cursor
        if cursor.plan.primary and cursor.plan.primary[0] in boarded:
            return RecourseCursor(cursor.plan.advance(('select', cursor.plan.primary[0]), True),
                                  after.time, cursor.branch)
        return RecourseCursor(replace(cursor.plan, primary=()), after.time, 'stopped')

    def bind(self, name=None):
        return _BoundRecourse(self, self.mode if name is None else name)


class _BoundRecourse:
    """One world's execution adapter; all memory updates use the pure core."""

    def __init__(self, controller, name):
        self.controller, self.cursor, self.previous = controller, None, None
        self.trace, self.last_reason = [], None
        self.stats = dict(action_calls=0, action_seconds=0., public_failures=0,
                          expiries=0, visible_rescues=0)
        self.policy = IssuedPolicy(self.action, controller.applicable, name)

    def action(self, state):
        started = perf_counter()
        if self.cursor is None:
            self.cursor = self.controller.initial(state)
        elif self.previous is not None:
            self.cursor = self.controller.advance(self.previous, state, self.cursor)
        decision = self.controller.decide(state, self.cursor)
        self.cursor, self.previous = decision.cursor, state
        for event, _, _ in decision.events:
            if event == 'attempt_expired':
                self.stats['expiries'] += 1
            elif event == 'visible_rescue':
                self.stats['visible_rescues'] += 1
            elif event in ('published_cancel', 'published_departure', 'local_departure_unboardable'):
                self.stats['public_failures'] += 1
        if decision.reason != self.last_reason or decision.action != WAIT or decision.events:
            self.trace.append(dict(time=state.time, stop=state.stop, action=decision.action,
                branch=self.cursor.branch, target=self.cursor.plan.primary[:1],
                opening=self.cursor.opening, ready=self.cursor.ready, expiry=self.cursor.expiry,
                reference_window=self.cursor.reference_window, reason=decision.reason, events=decision.events))
        self.last_reason = decision.reason
        self.stats['action_calls'] += 1
        self.stats['action_seconds'] += perf_counter() - started
        return decision.action

    def targets(self, state):
        cursor = self.cursor if self.cursor is not None else self.controller.initial(state)
        targets = cursor.plan.primary[:1]
        if self.controller.mode == 'first_visible' and cursor.plan.failure_index == 0:
            targets += cursor.plan.rescue[:1]
        return tuple(cid for cid in targets if self.controller.graph.connections[cid].dep_stop == state.stop)
