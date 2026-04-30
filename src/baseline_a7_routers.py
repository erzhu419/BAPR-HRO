"""Baseline routers with the two A7 hyperpath-structural risk terms retrofitted.

These adapters answer the controlled-experiment question: how much of
the deployed Swiss gain comes from the two A7 penalties (StopLabel
feasibility and destination on-time PMF) versus from the LCB family's
posterior-pessimism core?

Two variants are exposed:

  StaticA7Router:  no learning, no posterior, no cancel penalty.
                   Score = mean_dest_arrival + 60(1-feasibility)
                                          + 60(1-on_time_PMF).
  SWLCBA7Router:   sliding-window LCB plus the same two A7 terms.
                   Score = SWLCB.score(label, route) + 60(1-feasibility)
                                                    + 60(1-on_time_PMF).

Both implement the bandit-interface that ``simulate_bandit_journey``
expects (``begin_journey`` / ``observe_delay`` / ``observe_cancel`` /
``select_connection`` / ``end_journey`` / ``total_observations`` /
``route``). They are accepted by the simulator's ``is_bandit`` check
because they inherit from existing bandit-interface classes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .bandit_router import BanditRouter
from .durner.topocsa import topocsa, HyperpathResult
from .sw_lcb_router import SWLCBRouter


class StaticA7Router(BanditRouter):
    """Static hyperpath ranking augmented with the two A7 penalties only.

    No posterior, no cancel penalty, no disruption gate, no
    hierarchical prior. We inherit from BanditRouter only to satisfy
    the simulator's ``is_bandit`` isinstance check. The base score is
    the hyperpath label's nominal mean destination arrival; the A7
    layered penalties (infeasibility and timeout) are added on top.
    """

    def __init__(self, graph,
                 infeasibility_weight: float = 60.0,
                 timeout_weight: float = 60.0,
                 max_time: int = 120):
        super().__init__(
            graph,
            disruption_gate=False,
            use_hierarchical_prior=False,
            infeasibility_weight=infeasibility_weight,
            timeout_weight=timeout_weight,
            max_time=max_time,
        )

    def _get_belief(self, route):
        # Override to return a fresh (un-updated) belief: this disables
        # the posterior contribution at decision time, leaving only the
        # nominal hyperpath ranking + A7 penalties.
        from .bandit_router import RouteBeliefState
        if route not in self.route_beliefs:
            self.route_beliefs[route] = RouteBeliefState()
        return self.route_beliefs[route]

    # Disable observation effects: keep counters for total_observations
    # consistency, but never update the belief state.
    def observe_delay(self, route, delay):
        self.total_observations += 1

    def observe_cancel(self, route, kind='true'):
        self.total_observations += 1

    def select_connection(self, stop_id, current_time, rng, top_k=5, beta=0.0):
        # Force beta=0 so std_penalty drops out; with the un-updated
        # belief, delay_adj=0 and cancel_penalty=0 too. The remaining
        # scoring is exactly mean_dest_arrival + A7 penalties.
        return super().select_connection(stop_id, current_time, rng,
                                          top_k=top_k, beta=0.0)


class SWLCBA7Router(SWLCBRouter):
    """SW-LCB augmented with the two A7 hyperpath-structural penalties."""

    def __init__(self, graph,
                 window_size: int = 20,
                 beta: float = 1.5,
                 gamma: float = 60.0,
                 infeasibility_weight: float = 60.0,
                 timeout_weight: float = 60.0,
                 max_time: int = 120):
        super().__init__(graph, window_size=window_size, beta=beta, gamma=gamma)
        self.infeasibility_weight = float(infeasibility_weight)
        self.timeout_weight = float(timeout_weight)
        self.max_time = max_time
        self.journey_deadline: Optional[int] = None

    def route(self, s_source: int, s_dest: int, t_source: int) -> HyperpathResult:
        self.journey_deadline = t_source + self.max_time
        return super().route(s_source, s_dest, t_source)

    def lcb_score(self, label, route):
        base = super().lcb_score(label, route)
        infeasibility_penalty = self.infeasibility_weight * (1.0 - label.feasibility)
        if label.dest_arrival is not None and self.timeout_weight > 0.0:
            deadline = self.journey_deadline if self.journey_deadline is not None else 999
            p_on_time = label.dest_arrival.prob_le(deadline)
            timeout_penalty = self.timeout_weight * (1.0 - p_on_time)
        else:
            timeout_penalty = 0.0
        return base + infeasibility_penalty + timeout_penalty
