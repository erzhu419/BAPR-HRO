"""Journey simulator for the Bandit Router.

Separate from the main simulator to keep things clean.
The key difference: instead of the generic replan() pattern,
the bandit router uses observe() + select() at each stop.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .transit_graph import TransitGraph, Connection
from .bandit_router import BanditRouter
from .bandit_router_v2 import BanditRouterV2
from .adaptive_bandit_router import AdaptiveBetaBanditRouter
from .ssp_mdp import PosteriorSamplingRouter
from .bamcp_router import BAMCPRouter
from .dro_router import DRORouter
from .router import StaticRouter
from .durner.topocsa import HyperpathResult
from .simulator import (
    JourneyEvent, JourneyResult, RegimeSchedule,
    sample_actual_delay, generate_delay_observations,
    set_regime_dist_fn, _regime_dist_cache,
)


def simulate_bandit_journey(
    graph: TransitGraph,
    router,  # BanditRouter or StaticRouter
    s_source: int,
    s_dest: int,
    t_depart: int,
    regime_schedule: RegimeSchedule,
    rng: np.random.Generator,
    max_time: int = 180,
) -> JourneyResult:
    """Simulate a journey with bandit-based or static boarding decisions.

    For BanditRouter:
    - At each stop, feed nearby delay observations to update beliefs
    - Use Thompson Sampling to pick which bus to try
    - After cancel/boarding, update belief with actual outcome

    For StaticRouter:
    - At each stop, pick the first available bus from hyperpath ranking
    """
    events = []
    current_stop = s_source
    current_time = t_depart
    n_transfers = 0
    total_comp_ms = 0.0

    is_bandit = isinstance(router, (BanditRouter, BanditRouterV2, AdaptiveBetaBanditRouter,
                                    PosteriorSamplingRouter, BAMCPRouter, DRORouter))

    # Adaptive-β: signal start of journey
    if isinstance(router, AdaptiveBetaBanditRouter):
        router.begin_journey()

    # Initial route computation
    if is_bandit:
        hyperpath = router.route(s_source, s_dest, t_depart)
    else:
        decision = router.route(s_source, s_dest, t_depart)
        hyperpath = decision.hyperpath

    events.append(JourneyEvent(t_depart, s_source, "depart",
                               details="Initial route computed"))

    canceled_at_stop: set[str] = set()
    prev_stop = -1

    while current_stop != s_dest and current_time < t_depart + max_time:
        if current_stop != prev_stop:
            canceled_at_stop = set()
            prev_stop = current_stop

        current_regime = regime_schedule.get_regime(current_time)

        # Feed observations to bandit (learn from nearby delays)
        if is_bandit:
            obs = generate_delay_observations(
                graph, current_stop, current_time, current_regime, rng, n_obs=5)
            for o in obs:
                if o.actual_delay > 25:  # likely cancel signal
                    router.observe_cancel(o.route)
                else:
                    router.observe_delay(o.route, o.actual_delay)

        # Get candidate labels at this stop
        labels = hyperpath.stop_labels.get(current_stop, [])
        if not labels:
            events.append(JourneyEvent(current_time, current_stop, "stuck"))
            break

        # Select a connection to try
        if is_bandit:
            # Thompson Sampling: pick based on posterior beliefs
            result = router.select_connection(current_stop, current_time, rng, top_k=5)
            if result is None:
                current_time += 2
                continue
            label, sampled_arr = result
            c = graph.connections[label.connection_id]

            # Skip known-canceled routes at this stop
            if c.route in canceled_at_stop:
                # TS picked a canceled route — update posterior and retry
                router.observe_cancel(c.route)
                current_time += 1
                continue
        else:
            # Static: pick first available by ranking (same as before)
            found = False
            for label in reversed(labels):
                c = graph.connections[label.connection_id]
                if c.dep_time < current_time - 1:
                    continue
                if c.dep_time > current_time + 25:
                    continue
                if c.route in canceled_at_stop:
                    continue
                found = True
                break
            if not found:
                current_time += 2
                continue

        # Try to board this connection
        delay = sample_actual_delay(c, current_regime, rng)

        if delay == 999:
            # Canceled
            canceled_at_stop.add(c.route)
            if is_bandit:
                router.observe_cancel(c.route)
            events.append(JourneyEvent(
                current_time, current_stop, "cancel",
                route=c.route,
                details=f"{c.route} canceled (bandit will learn)"))
            current_time += 3  # brief wait before trying next
            continue

        actual_dep = c.dep_time + delay
        if actual_dep < current_time:
            continue  # already left
        if actual_dep > current_time + 12:
            # Too long to wait — try another option
            if is_bandit:
                router.observe_delay(c.route, delay)
            current_time += 2
            continue

        # Board!
        actual_arr = c.arr_time + delay
        if is_bandit:
            router.observe_delay(c.route, delay)

        events.append(JourneyEvent(
            actual_dep, current_stop, "board",
            route=c.route,
            details=f"→{c.arr_stop} (sched={c.dep_time} actual={actual_dep})"))

        current_time = actual_arr
        current_stop = c.arr_stop

        if current_stop != s_dest:
            n_transfers += 1
            events.append(JourneyEvent(
                current_time, current_stop, "arrive",
                route=c.route, details=f"delay={delay}min"))

    status = "arrived" if current_stop == s_dest else "timeout"
    events.append(JourneyEvent(current_time, current_stop, "finish", details=status))

    # Adaptive-β: signal end of journey with total travel time
    if isinstance(router, AdaptiveBetaBanditRouter):
        tt = (current_time if current_stop == s_dest else t_depart + max_time) - t_depart
        router.end_journey(tt)

    return JourneyResult(
        origin=s_source,
        destination=s_dest,
        departure_time=t_depart,
        arrival_time=current_time if current_stop == s_dest else t_depart + max_time,
        events=events,
        n_transfers=n_transfers,
        n_replans=router.total_observations if is_bandit else 0,
    )
