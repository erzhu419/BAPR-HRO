"""An exact event-time reduction of the V13 integer-clock platform model."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache

from .causal_prefix import causal_prefix_lower
from .information_planner import Outcome
from .platform_transit import WAIT, build_platform_model


class EventPlatform:
    """Prepare reusable event schedules and preserve the original bank indices.

    Idle WAIT jumps to the earliest possible next local occurrence, explicit
    publication, or clock boundary over the current support. No observation
    can differ before that time. There is no new board action in the open
    interval. If onboard, a planned ALIGHT during that deterministic interval
    can move to its start: the same local history remains available and the
    fixed station transfer can only finish earlier. A decision to stay onboard
    until another observation remains available via WAIT. Same-trip inertia
    at a visible departure is always handled by the original transition.

    This reduction uses V13 station transfer times and has no footpaths or
    exogenous polls other than its idle one-second timers. A supplied tail
    oracle must respect earlier station readiness. A custom stopping predicate
    must depend on ride outcomes, with elapsed_horizon for clock boundaries.
    """

    def __init__(self, graph, bank, initial, goal, deadline):
        bank = tuple(bank)
        self.initial = initial
        self.deadline = deadline
        self.model = build_platform_model(graph, bank, initial.stop, goal, initial.time, deadline,
                                          initial_state=initial, poll_interval=1)
        self.connections = {connection.id: connection for connection in graph.connections}
        schedules = [self._event_ticks(scene) for _, scene in bank]

        @lru_cache(maxsize=100_000)
        def next_tick(index, stop, clock):
            local, publications = schedules[index]
            return min(self._after(local.get(stop, ()), clock), self._after(publications, clock))

        self.next_tick = next_tick

    def _event_ticks(self, scene):
        local = defaultdict(set)
        for cid, event in scene.events.items():
            if cid in self.connections and not event.canceled:
                connection = self.connections[cid]
                local[connection.dep_stop].add(event.departure)
                local[connection.arr_stop].add(event.arrival)
        publications = set()
        for key, tick in (scene.publications or {}).items():
            raw_cid, kind = key.split(':', 1)
            cid = int(raw_cid)
            if cid in self.connections and cid in scene.events and kind in ('cancel', 'departure', 'arrival'):
                publications.add(tick)
        return {stop: tuple(sorted(ticks)) for stop, ticks in local.items()}, tuple(sorted(publications))

    def _after(self, ticks, clock):
        position = bisect_right(ticks, clock)
        return ticks[position] if position < len(ticks) else self.deadline

    def outcomes(self, state, indices, action, cutoff):
        """Yield the original outcome for events and one shared quiet outcome."""
        if action != WAIT or (state.current_trip is not None and any(
                self.connections[cid].trip_id == state.current_trip for cid in state.visible_connections)):
            for index in indices:
                yield index, self.model.transition(self.model.scenarios[index], state, action)
            return

        next_ticks = tuple(self.next_tick(index, state.stop, state.time) for index in indices)
        tick = min(cutoff, *next_ticks)
        duration = tick-state.time
        idle = replace(state, time=tick, visible_connections=())
        quiet = Outcome(idle, idle.public_log, duration, tick >= self.deadline)
        # There are no intermediate events in any remaining world. Reuse the
        # original one-second transition at the jump endpoint to publish it.
        pre_tick = replace(idle, time=tick-1)
        for index, next_tick in zip(indices, next_ticks):
            if next_tick == tick:
                event = self.model.transition(self.model.scenarios[index], pre_tick, action)
                yield index, replace(event, cost=duration)
            else:
                yield index, quiet

    def solve(self, oracle_terminal, *, elapsed_horizon=None, max_nodes=50_000,
              stop_after_first_ride=True, stop_when=None, factor_labels=None, trace=None, weights=None,
              singleton_exact=None, prune_with_oracle=False):
        """Evaluate one finite distribution on the prepared worlds.

        For a full deadline solve set stop_after_first_ride=False and omit
        elapsed_horizon. A zero weight masks an original bank row.
        On a full solve only, singleton_exact(index, state) can supply an
        independently certified exact deterministic continuation, or None.
        A trace marks that delegated tail with the exact_continuation boundary.
        With prune_with_oracle, the supplied oracle must also lower-bound each
        child state for the declared objective. Tracing disables action pruning.
        """
        horizon = self.deadline-self.initial.time if elapsed_horizon is None else elapsed_horizon
        if singleton_exact is not None and (
                stop_after_first_ride or stop_when is not None or horizon < self.deadline-self.initial.time):
            raise ValueError('The exact singleton continuation is for full deadline solves')
        cutoff = min(self.deadline, self.initial.time+horizon)
        model = self.model
        if weights is not None:
            weights = tuple(weights)
            if len(weights) != len(model.scenarios):
                raise ValueError('Provide one weight for each prepared bank row')
            model = replace(model, scenarios=tuple(replace(world, probability=weight)
                                                  for world, weight in zip(model.scenarios, weights)))
        if stop_when is None and not stop_after_first_ride:
            stop_when = lambda state: False
        return causal_prefix_lower(model, oracle_terminal, elapsed_horizon=horizon, max_nodes=max_nodes,
            stop_when=stop_when, factor_labels=factor_labels, trace=trace,
            action_outcomes=lambda state, indices, action: self.outcomes(state, indices, action, cutoff),
            singleton_exact=singleton_exact, prune_with_oracle=prune_with_oracle)


def event_platform_lower(graph, bank, initial, goal, deadline, oracle_terminal, **kwargs):
    """Solve a platform bank once. Reuse EventPlatform for multiple weight vectors."""
    return EventPlatform(graph, bank, initial, goal, deadline).solve(oracle_terminal, **kwargs)
