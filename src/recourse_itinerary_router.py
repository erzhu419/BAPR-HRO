"""Compile and integrate causal itineraries before dispatching their actions."""
from dataclasses import dataclass, replace
from collections.abc import Mapping
import heapq
import itertools
from time import perf_counter

from .causal_planning_router import CausalPlanningRouter
from .committed_transit import WAIT
from .risk_bounds import CommitmentBounds
from .first_branch_belief import FirstBranchSampler
from .transit_graph import StopLabel


def trip_key(connection):
    return connection.trip_id or f'connection:{connection.id}'


def nominal_itinerary(graph, state, destination, nominal, menu, deadline, lookahead,
                      *, minute=60, excluded_trips=(), trip_blocks=None):
    """A deterministic nominal path; complete same-trip blocks are expanded.

    The witness class uses each trip in one contiguous block. Unsupported
    repeated-trip or boundary-crossing paths return None, rather than having
    their dependence silently removed.
    """
    excluded_trips = set(excluded_trips)
    serial = itertools.count()
    queue = [(state.time, next(serial), state.stop, state.current_trip, ())]
    best = {(state.stop, state.current_trip): state.time}
    path = None
    while queue:
        time, _, stop, trip, prefix = heapq.heappop(queue)
        if time > best[stop, trip]:
            continue
        if stop == destination:
            path = prefix
            break
        for c in graph.get_connections_from(stop):
            same = trip is not None and c.trip_id == trip
            if (c.id not in menu or c.id in state.unavailable_ids or trip_key(c) in excluded_trips
                    or (not c.pickup_allowed and not same)):
                continue
            event = nominal[c.id]
            if event.departure is None:
                continue
            departure = round(event.departure)
            arrival = departure + round(event.arrival - event.departure)
            transfer = 0 if trip is None or same else graph.stops[stop].min_transfer_time
            ready = (state.last_arrival if not prefix else time) + transfer
            attempt = max(time, c.dep_time - lookahead)
            if departure > max(attempt, ready) + 12 * minute:
                attempt = departure - 12 * minute
            if (departure < max(attempt, ready) or attempt >= deadline or arrival > deadline
                    or (c.dep_distribution is not None and c.dep_distribution.support_max < attempt)):
                continue
            key = c.arr_stop, trip_key(c)
            if arrival < best.get(key, float('inf')):
                best[key] = arrival
                heapq.heappush(queue, (arrival, next(serial), *key, prefix + (c.id,)))
    if path is None:
        return None
    return expand_trip_blocks(graph, path, destination, state.unavailable_ids, trip_blocks)


def build_trip_blocks(graph):
    by_trip = {}
    for c in sorted(graph.connections, key=lambda c: (c.dep_time, c.arr_time, c.id)):
        by_trip.setdefault(trip_key(c), []).append(c.id)
    positions = {cid: i for cids in by_trip.values() for i, cid in enumerate(cids)}
    return by_trip, positions


def expand_trip_blocks(graph, path, destination, unavailable=(), trip_blocks=None):
    by_trip, positions = build_trip_blocks(graph) if trip_blocks is None else trip_blocks
    result, seen, previous = [], set(), None
    for cid in path:
        c = graph.connections[cid]
        trip = trip_key(c)
        if trip != previous:
            if trip in seen:
                return None
            seen.add(trip)
            append = [cid]
        else:
            before, after = positions[result[-1]], positions[cid]
            if after <= before:
                return None
            append = by_trip[trip][before + 1:after + 1]
        for member in append:
            edge = graph.connections[member]
            if member in unavailable or (result and graph.connections[result[-1]].arr_stop != edge.dep_stop):
                return None
            result.append(member)
            if edge.arr_stop == destination:
                return tuple(result)
        previous = trip
    return tuple(result)


@dataclass(frozen=True)
class CompiledPolicy:
    primary: tuple
    rescue: tuple
    upper: float
    root_action: tuple


class _ConditionalNominal(Mapping):
    def __init__(self, original, belief, trip, law):
        self.original, self.belief, self.trip, self.law = original, belief, trip, law
        self.cache = {}

    def __getitem__(self, cid):
        if self.belief._trip_by_cid[cid] != self.trip:
            return self.original[cid]
        if cid not in self.cache:
            self.cache[cid] = self.belief._mean_event(cid, self.law)
        return self.cache[cid]

    def __iter__(self):
        return iter(self.original)

    def __len__(self):
        return len(self.original)


class RecourseItineraryRouter(CausalPlanningRouter):
    """Exact fixed-grid witness values, followed by issued itinerary actions.

    At most max_itineraries candidate paths are integrated per fresh decision.
    Family screening uses the chosen fixed rescue and is logged separately
    from the full causal problem's commitment-cost lower bound.
    """

    def __init__(self, *args, max_itineraries=8, optimize=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_itineraries, self.optimize = max_itineraries, optimize
        self._path, self._rescue = (), ()
        self._fixed_departures = {}
        self._first_attempt = False
        self._root_pending = False
        self._trip_blocks = build_trip_blocks(self.graph)

    def _can_attempt(self, state):
        if not self._path:
            return False
        c = self.graph.connections[self._path[0]]
        if c.dep_stop != state.stop or c.id in state.unavailable_ids:
            return False
        if c.dep_time > state.time + self.lookahead:
            return False
        transfer = (0 if state.current_trip is None or c.trip_id == state.current_trip
                    else self.graph.stops[state.stop].min_transfer_time)
        ready = max(state.time, state.last_arrival + transfer)
        return self._first_attempt or self._fixed_departures[c.id] <= ready + 12 * self.minute

    def precommitted_connection(self, stop, time, current_trip):
        state = self.information_state
        if self._path and not self._first_attempt and self._can_attempt(state):
            c = self.graph.connections[self._path[0]]
            if current_trip is not None and c.trip_id == current_trip:
                return c.id
        return None

    def _install(self, policy, nominal):
        self._path, self._rescue = policy.primary, policy.rescue
        self._fixed_departures = {
            cid: (self.graph.connections[cid].dep_time if nominal[cid].departure is None
                  else round(nominal[cid].departure))
            for cid in (*policy.primary, *policy.rescue)}
        self._first_attempt = policy.root_action != WAIT
        self._root_pending = policy.root_action != WAIT

    def observe_attempt_outcome(self, cid, kind, started, time):
        super().observe_attempt_outcome(cid, kind, started, time)
        if not self._path or self._path[0] != cid:
            return
        if kind == 'board':
            self._path = self._path[1:]
            self._rescue = ()
        elif self._root_pending:
            self._path, self._rescue = self._rescue, ()
        else:
            # A failed later leg has deadline cost in the witness. Replanning
            # here can only improve that terminal cap in the same world.
            self._path, self._rescue = (), ()
        self._first_attempt = self._root_pending = False

    def _record(self, state, action, started, **details):
        self.predictions.append(dict(connection=None if action == WAIT else action[1],
            stop=state.stop, time=state.time, planning_ms=(perf_counter() - started) * 1000,
            work=0, completed_rollouts=0, root_actions=details.pop('root_actions', 0),
            scored_root_actions=details.pop('scored_root_actions', 0),
            complete_common_rounds=0, tree_nodes=0, menu_size=len(self.menu), **details))
        if action == WAIT:
            return None
        c = self.graph.connections[action[1]]
        return StopLabel(c.id, c.arr_distribution, c.arr_time, 0.), details.get('value')

    def _dispatch_policy(self, best, policies, bounds, state, started):
        return best, {}

    def select_connection(self, current_stop, current_time, rng, top_k=5):
        started = perf_counter()
        state = self.information_state
        if self._path:
            c = self.graph.connections[self._path[0]]
            if c.dep_stop == state.stop and c.id not in state.unavailable_ids:
                action = ('select', c.id) if self._can_attempt(state) else WAIT
                return self._record(state, action, started, compiled=True, value=None)
            self._path, self._rescue = (), ()
        parameters = self._parameters()
        nominal = self.trip_belief.mean_events(parameters)
        bounds = CommitmentBounds(self.trip_belief, parameters, self.graph,
                                  self.destination, self.journey_deadline)
        def path_from(context, excluded=(), events=nominal):
            return nominal_itinerary(self.graph, context, self.destination, events,
                self.menu, self.journey_deadline, self.lookahead, minute=self.minute,
                excluded_trips=excluded, trip_blocks=self._trip_blocks)
        path = path_from(state)
        cap = self.journey_deadline - state.time
        best = CompiledPolicy((), (), float(cap), WAIT)
        if path:
            # The nominal witness may legally WAIT before its first attempt.
            first = self.graph.connections[path[0]]
            transfer = (0 if state.current_trip is None or state.current_trip == first.trip_id
                        else self.graph.stops[state.stop].min_transfer_time)
            ready = max(state.time, state.last_arrival + transfer)
            now = first.dep_time <= state.time + self.lookahead and (
                nominal[first.id].departure <= ready + 12 * self.minute)
            value = bounds.itinerary(state, path, first_action_now=now, lookahead=self.lookahead)
            best = CompiledPolicy(path, (), value.upper, ('select', path[0]) if now else WAIT)
        if not self.optimize:
            self._install(best, nominal)
            action = best.root_action
            return self._record(state, action, started, compiled=False, value=best.upper,
                                primary=list(best.primary), rescue=[])
        # Construct only the public legal root menu, without any sampled bank.
        # Physical menu is independent of world clocks. The same declared
        # rules are used directly so no dummy vehicle scene is introduced.
        actions = [WAIT]
        for c in self.graph.get_connections_from(state.stop):
            if (c.id in self.menu and c.id not in state.unavailable_ids
                    and c.dep_time <= state.time + self.lookahead
                    and (c.pickup_allowed or c.trip_id == state.current_trip)
                    and (c.dep_distribution is None or c.dep_distribution.support_max >= state.time)):
                actions.append(('select', c.id))
        lower = {WAIT: min(self.minute, cap)}
        risks, rescue_cache, candidates = {}, {}, []
        screened = global_removed = 0
        for action in actions[1:]:
            cid = action[1]
            risk = bounds.commit(state, cid)
            risks[cid], lower[action] = risk, risk.expected_first_cost
            if risk.expected_first_cost > best.upper:
                global_removed += 1
                continue
            failed_trip = self.trip_belief._trip_by_cid[cid]
            failed = replace(state, time=risk.failure_clock)
            key = risk.failure_clock, failed_trip
            # Most actions have the same failure clock; paths not using the
            # failed trip can share their exact rescue evaluation.
            if (risk.failure_clock, None) not in rescue_cache:
                general_path = path_from(failed) or ()
                general_value = bounds.itinerary(failed, general_path, first_action_now=False,
                                                  lookahead=self.lookahead)
                rescue_cache[risk.failure_clock, None] = general_path, general_value
            general = rescue_cache[risk.failure_clock, None]
            if all(self.trip_belief._trip_by_cid[x] != failed_trip for x in general[0]):
                rescue, rescue_value = general
            else:
                if key not in rescue_cache:
                    rescue = path_from(failed, (failed_trip,)) or ()
                    rescue_value = bounds.itinerary(failed, rescue, first_action_now=False,
                                                     lookahead=self.lookahead)
                    rescue_cache[key] = rescue, rescue_value
                    rescue_cache.setdefault((risk.failure_clock, None), (rescue, rescue_value))
                rescue, rescue_value = rescue_cache[key]
            family_lower = risk.expected_first_cost + risk.failure_probability * rescue_value.upper
            if family_lower >= best.upper:
                screened += 1
                continue
            candidates.append((family_lower, cid, rescue, risk))
        evaluations = 0
        policies = [best]
        for family_lower, cid, rescue, risk in sorted(candidates, key=lambda item: item[0]):
            if family_lower >= best.upper:
                screened += 1
                continue
            if evaluations >= self.max_itineraries:
                break
            c = self.graph.connections[cid]
            if c.arr_stop == self.destination:
                primary = (cid,)
            elif risk.catch_probability:
                conditional = FirstBranchSampler(self.trip_belief, parameters, cid,
                    risk.ready, self.journey_deadline).conditional_law(True)
                conditional_nominal = _ConditionalNominal(nominal, self.trip_belief,
                    self.trip_belief._trip_by_cid[cid], conditional)
                conditional_time = conditional_nominal[cid].arrival
                arrival = min(self.journey_deadline, round(conditional_time))
                after = replace(state, stop=c.arr_stop, time=arrival, last_arrival=arrival,
                                current_trip=trip_key(c), unavailable_ids=state.unavailable_ids | {cid})
                tail = path_from(after, events=conditional_nominal)
                primary = expand_trip_blocks(self.graph, (cid,) + tail, self.destination,
                                              state.unavailable_ids, self._trip_blocks) if tail is not None else None
            else:
                primary = None
            if primary is None:
                continue
            value = bounds.with_rescue(state, primary, rescue, lookahead=self.lookahead)
            evaluations += 1
            policies.append(CompiledPolicy(primary, rescue, value.upper, ('select', cid)))
            if value.upper < best.upper:
                best = CompiledPolicy(primary, rescue, value.upper, ('select', cid))
        best, dispatch = self._dispatch_policy(best, policies, bounds, state, started)
        self._install(best, nominal)
        lower_value = dispatch.pop('dispatch_lower', min(lower.values()))
        return self._record(state, best.root_action, started, compiled=False, value=best.upper,
            root_actions=len(actions), scored_root_actions=evaluations,
            full_model_lower=lower_value, full_model_gap=best.upper - lower_value,
            globally_removed=global_removed, family_screened=screened,
            itinerary_evaluations=evaluations, analytic_work=dict(bounds.stats),
            primary=list(best.primary), rescue=list(best.rescue), **dispatch)
