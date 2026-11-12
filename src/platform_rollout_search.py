"""Common-random-number rollout improvement on the public platform clock."""

from dataclasses import dataclass

import numpy as np

from .background_policy import IssuedPolicy
from .interruptible_recourse import RecourseCursor
from .local_transfer_rescue import compile_local_transfer_rescue
from .platform_transit import OBSERVE, WAIT
from .recourse_itinerary_router import build_trip_blocks, trip_key
from .tail_rescue import TailRescuePlan


@dataclass(frozen=True)
class _Leg:
    cid: int
    transfer: int | None
    stray: tuple[int, ...]
    physical_mct: int


@dataclass(frozen=True)
class _Route:
    legs: tuple[_Leg, ...]
    transfer_index: int | None
    fallbacks: tuple


class PlatformRolloutSearch:
    """Improve one current action over a compiled nominal public-event policy.

    All candidate starts are executable public offers. Future WAIT rollouts run
    the nominal policy at each next offer and stop when that policy boards.
    Both board and WAIT subsequently use the same fixed-path interpreter and
    its first-transfer-failure fallback. Simulation clocks can reveal a future
    occurrence only when the rollout reaches its public time. Draws are fresh
    at each actual decision, shared across its alternatives.
    """

    def __init__(self, graph, belief, parameters, root, goal, deadline, paths, particles):
        self.graph, self.root, self.goal, self.deadline = graph, root, goal, deadline
        self.particles = particles
        self.ready = max(root.time, root.last_arrival+graph.stops[root.stop].min_transfer_time)
        self.options = tuple(sorted(paths))
        self.option_index = {cid: i for i, cid in enumerate(self.options)}
        blocks = build_trip_blocks(graph)
        self.continuations = {}
        self.routes = []
        for cid in self.options:
            path = tuple(paths[cid])
            if path[0] != cid or graph.connections[cid].dep_stop != root.stop:
                raise ValueError('The option must start with its identified platform departure')
            continuation = compile_local_transfer_rescue(graph, TailRescuePlan(path), root,
                                                         goal, deadline, trip_blocks=blocks)
            self.continuations[cid] = continuation
            fallbacks = tuple(self._route(fallback, None, ()) for fallback in continuation.fallback_paths.values())
            self.routes.append(self._route(path, continuation.transfer_index, fallbacks))
        needed = set(self.options)
        for route in self.routes:
            for path in (route, *route.fallbacks):
                for leg in path.legs:
                    needed.add(leg.cid)
                    needed.update(leg.stray)
        self.sample_cids = tuple(sorted(needed))
        self.sample_index = {cid: i for i, cid in enumerate(self.sample_cids)}
        self.first_cols = [self.sample_index[cid] for cid in self.options]
        self.nominal_board, self.nominal_wait = self._compile_marks(belief, parameters)
        if self.particles is not None:
            self.particles.prepare(self.sample_cids)

    def _route(self, path, transfer_index, fallbacks):
        used = set(self.root.used_connections)
        previous, legs = None, []
        for cid in path:
            c = self.graph.connections[cid]
            mct = self.graph.stops[c.dep_stop].min_transfer_time
            trip = trip_key(c)
            transfer = mct if previous is not None and previous != trip else None
            stray = ()
            if previous is not None and transfer is None:
                stray = tuple(other.id for other in self.graph.get_connections_from(c.dep_stop)
                              if trip_key(other) == previous and other.id != cid and other.id not in used)
            legs.append(_Leg(cid, transfer, stray, mct))
            used.add(cid)
            previous = trip
        return _Route(tuple(legs), transfer_index, fallbacks)

    def _ready(self, cid, at, platform):
        c = self.graph.connections[cid]
        opening = at+60*((np.maximum(0, c.dep_time-1500-at)+59)//60)
        late = c.dep_time > np.maximum(opening, platform)+720
        opening = np.where(late, at+60*((np.maximum(0, c.dep_time-720-at)+59)//60), opening)
        if c.dep_distribution is not None:
            opening = np.where(opening > c.dep_distribution.support_max, self.deadline, opening)
        return np.minimum(self.deadline, np.maximum(opening, platform))

    def _route_arrival(self, route, get):
        """Run a fixed path in parallel under the sampled or nominal clocks."""
        _, first_arrival, canceled = get(route.legs[0].cid)
        at = np.where(canceled, self.deadline, np.minimum(first_arrival, self.deadline))
        rescue = np.full(at.shape, self.deadline, dtype=np.int32)
        for index, leg in enumerate(route.legs[1:], start=1):
            depart, arrival, canceled = get(leg.cid)
            live = at < self.deadline
            platform = at if leg.transfer is None else at+leg.transfer
            ready = self._ready(leg.cid, at, platform)
            expiry = np.minimum(self.deadline, ready+720)
            if leg.stray:
                earliest_stray = np.full(at.shape, self.deadline, dtype=np.int32)
                for cid in leg.stray:
                    stray, _, gone = get(cid)
                    # The interpreter explicitly alights to avoid taking an
                    # unissued same-vehicle departure before the planned one.
                    eligible = ~gone & (at <= stray) & (stray < depart) & (stray < expiry)
                    earliest_stray = np.minimum(earliest_stray, np.where(eligible, stray, self.deadline))
                platform = np.where(earliest_stray < self.deadline,
                                    earliest_stray+leg.physical_mct, platform)
            caught = live & ~canceled & (np.maximum(ready, platform) <= depart) & (depart <= expiry)
            if index == route.transfer_index and route.fallbacks:
                early_seen = ~canceled & (at <= depart) & (depart < ready)
                start = np.where(early_seen, np.maximum(depart, platform), expiry)
                start = np.where(live & ~caught, start, self.deadline)
                rescue = self._rescue_arrival(route.fallbacks, get, start)
            at = np.where(caught, np.minimum(arrival, self.deadline), self.deadline)
        return np.minimum(at, rescue)

    def _rescue_arrival(self, routes, get, start):
        earliest = np.full(start.shape, self.deadline, dtype=np.int32)
        selected = np.full(start.shape, self.deadline, dtype=np.int32)
        for route in routes:
            depart, _, canceled = get(route.legs[0].cid)
            wins = ~canceled & (start <= depart) & (depart < earliest)
            if np.any(wins):
                selected = np.where(wins, self._route_arrival(route, get), selected)
                earliest = np.where(wins, depart, earliest)
        return selected

    def _compile_marks(self, belief, parameters):
        """Cache a causal nominal policy, using no conditioned rollout draws."""
        nominal = belief.mean_events(parameters)
        clock = np.arange(self.root.time, self.deadline, dtype=np.int32)
        board = np.empty((clock.size, len(self.options)), dtype=np.float64)
        wait = np.full_like(board, self.deadline)
        means = {}
        for cid in self.sample_cids:
            event = nominal[cid]
            c = self.graph.connections[cid]
            d = c.dep_time if event.departure is None else round(event.departure)
            a = c.arr_time if event.arrival is None else round(event.arrival)
            means[cid] = d, max(d, a), event.departure is None
        for col, (cid, route) in enumerate(zip(self.options, self.routes, strict=True)):
            trip = trip_key(self.graph.connections[cid])
            shift = clock-means[cid][0]

            def get(leg):
                d, a, x = means[leg]
                drift = shift if trip_key(self.graph.connections[leg]) == trip else 0
                return d+drift, a+drift, np.asarray(x)

            board[:, col] = self._route_arrival(route, get)
            departures, masses, cancel = belief.event_distribution(cid, 'departure', parameters)
            masses = masses*(1-cancel)
            gain = np.zeros(masses.size)
            live = (departures >= self.ready) & (departures < self.deadline)
            gain[live] = self.deadline-board[departures[live]-self.root.time, col]
            # A no-show by clock t has no hidden future ETA. Integrate the
            # remaining public offer times against frozen nominal path marks.
            i = np.searchsorted(departures, clock, side='right')
            survival = cancel+np.r_[np.cumsum(masses[::-1])[::-1], 0.][i]
            saving = np.r_[np.cumsum((masses*gain)[::-1])[::-1], 0.][i]
            wait[:, col] -= np.divide(saving, survival, out=np.zeros_like(saving), where=survival > 0)
        return board, wait

    def _seen(self, state):
        return {self.option_index[cid] for _, cid, kind in state.public_log
                if cid in self.option_index and kind in ('departure', 'published_departure', 'published_cancel')}

    def _nominal(self, state, visible, seen):
        phase = state.time-self.root.time
        offer = min(visible, key=lambda col: (self.nominal_board[phase, col], col))
        wait = min((value for col, value in enumerate(self.nominal_wait[phase])
                    if col not in seen and col not in visible), default=self.deadline)
        return offer if self.nominal_board[phase, offer] <= wait else None

    def _rollout_wait(self, clock, departures, arrivals, canceled, seen):
        first_dep = departures[:, self.first_cols]
        can_offer = ~canceled[:, self.first_cols] & (first_dep > clock)
        if seen:
            can_offer[:, list(seen)] = False
        ticks = np.where(can_offer & (first_dep < self.deadline), first_dep, self.deadline)
        # After each tick the remaining offer set is the causal event history;
        # only this binary nonoccurrence is used, never a future realized ETA.
        ordered = np.sort(ticks, axis=1)
        picked = np.full(len(departures), -1)
        previous = np.full(len(departures), clock)
        for times in ordered.T:
            active = (picked < 0) & (times < self.deadline) & (times != previous)
            if not np.any(active):
                continue
            seen_now = times[active, None]
            future = ticks[active] > seen_now
            if seen:
                future[:, list(seen)] = False
            phase = times[active]-self.root.time
            offers = np.where(ticks[active] == seen_now, self.nominal_board[phase], np.inf)
            wait = np.min(np.where(future, self.nominal_wait[phase], self.deadline), axis=1)
            choice = offers.argmin(axis=1)
            picked[active] = np.where(np.min(offers, axis=1) <= wait, choice, -1)
            previous[active] = times[active]
        loss = np.full(len(departures), self.deadline, dtype=np.int32)
        for col in np.unique(picked[picked >= 0]):
            selected = picked == col
            loss[selected] = self._arrival(self.routes[col], departures[selected], arrivals[selected], canceled[selected])
        return loss

    def _arrival(self, route, departures, arrivals, canceled):
        def get(cid):
            col = self.sample_index[cid]
            return departures[:, col], arrivals[:, col], canceled[:, col]

        return self._route_arrival(route, get)

    def choose(self, state, binding, budget, rng, visible):
        seen = self._seen(state) | set(visible)
        nominal = self._nominal(state, visible, seen)
        if not budget:
            return nominal, ()
        dep, arr, canceled = binding.sample(state, budget, rng, self.sample_cids)
        canceled = canceled.astype(bool, copy=False)
        wait = self._rollout_wait(state.time, dep, arr, canceled, seen)
        marks = [(None, float(np.mean(wait)))]
        for col in visible:
            assert np.all((dep[:, self.first_cols[col]] == state.time) & ~canceled[:, self.first_cols[col]])
            marks.append((col, float(np.mean(self._arrival(self.routes[col], dep, arr, canceled)))))
        # Prefer the underlying causal policy at a sampling tie.
        choice, _ = min(marks, key=lambda item: (item[1], item[0] != nominal,
                                                  -1 if item[0] is None else item[0]))
        return choice, tuple((None if col is None else self.options[col], score) for col, score in marks)

    def for_budget(self, budget, seed):
        return _Budget(self, budget, seed)


@dataclass(frozen=True)
class _Budget:
    search: PlatformRolloutSearch
    budget: int
    seed: object

    def bind(self, name):
        return _BoundRollout(self.search, self.budget, self.seed, name)


class _BoundRollout:
    def __init__(self, search, budget, seed, name):
        self.search, self.budget = search, budget
        self.rng = np.random.default_rng(seed)
        self.particles = search.particles.bind() if budget else None
        self.trace, self.stats = [], dict(search_calls=0, rollout_worlds=0, rollout_trajectories=0)
        self.follow = None
        self.policy = IssuedPolicy(self.action, self.applicable, name)

    def applicable(self, state):
        root = self.search.root
        return (state.stop == root.stop and state.current_trip == root.current_trip
                and state.used_connections == root.used_connections)

    def targets(self, state):
        return self.follow.targets(state) if self.follow is not None else self.search.options

    def action(self, state):
        if not state.initialized:
            return OBSERVE
        if self.follow is not None:
            return self.follow.action(state)
        search = self.search
        if self.particles is not None:
            self.particles.observe(state)
        visible = tuple(search.option_index[cid] for cid in state.visible_connections if cid in search.option_index)
        if not visible:
            return WAIT
        choice, marks = search.choose(state, self.particles, self.budget, self.rng, visible)
        self.stats['search_calls'] += 1
        self.stats['rollout_worlds'] += self.budget
        self.stats['rollout_trajectories'] += self.budget*(len(visible)+1)
        action = WAIT if choice is None else ('board', search.options[choice])
        self.trace.append(dict(time=state.time, stop=state.stop, action=action, branch='rollout_search', marks=marks))
        if choice is not None:
            continuation = search.continuations[action[1]]
            self.follow = continuation.bind(self.policy.name)
            self.follow.cursor = RecourseCursor(continuation.plan, state.time)
            self.follow.previous = state
        return action
