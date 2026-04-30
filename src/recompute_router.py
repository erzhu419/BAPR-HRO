"""BOCD + topocsa hyperpath-recompute router.

Bandit-interface adapter for the BOCD-style strategy. It stores both
the normal-day and the disrupted-day per-route delay distributions;
when the journey simulator delivers a cancellation observation it flips
the graph's per-connection delay PMFs to the disrupted set and reruns
the topological connection-scan algorithm from the current stop. This
implements the strategy the paper Section ``Why Hyperpath Recomputation
Fails'' compares against: BOCD detects a regime shift, the optimizer
recomputes the hyperpath under the post-disruption distributions, and
the recomputed hyperpath drops fallback routes that the static
hyperpath kept.

We inherit from BanditRouter purely to satisfy the ``isinstance(router,
(BanditRouter, ...))`` check inside ``simulate_bandit_journey``;
otherwise the simulator would treat us as a static router and never
deliver observations. We override the four bandit-interface methods
(``observe_cancel``, ``observe_delay``, ``select_connection``,
``end_journey``) and the ``begin_journey`` hook (no-op).
"""

from __future__ import annotations

import copy
import numpy as np
from typing import Optional

from .bandit_router import BanditRouter
from .durner.topocsa import topocsa, HyperpathResult


class RecomputeRouter(BanditRouter):
    def __init__(
        self,
        graph,
        normal_dist: dict,
        disrupted_dist: dict,
    ):
        # Bypass BanditRouter's belief-state cold-start cost (we do not
        # use posteriors) by disabling A4/A7/gate; the parent's other
        # state is harmless to instantiate.
        super().__init__(graph,
                         disruption_gate=False,
                         use_hierarchical_prior=False,
                         infeasibility_weight=0.0,
                         timeout_weight=0.0)
        self._normal_dist = normal_dist
        self._disrupted_dist = disrupted_dist
        # Apply normal distribution to the graph initially.
        self.graph.assign_distributions(self._build_topocsa_dist(normal_dist))
        self._current_regime = 'normal'
        self._n_cancel_obs = 0
        self._initial_t_source: Optional[int] = None
        self._initial_dest: Optional[int] = None

    # ------------------------------------------------------------------
    # Distribution conversion: harness passes per-route mean/std/cancel,
    # topocsa needs per-route ``delay_probs`` arrays.
    # ------------------------------------------------------------------
    @staticmethod
    def _build_topocsa_dist(per_route):
        out = {}
        for rname, d in per_route.items():
            mean, std = d['mean'], max(d['std'], 0.5)
            delays = np.arange(-5, 65)
            probs = np.exp(-0.5 * ((delays - mean) / std) ** 2)
            probs = probs / probs.sum()
            cancel = d.get('cancel_rate', 0.0)
            if cancel > 0:
                probs = probs * (1.0 - cancel)
            out[rname] = {'delay_probs': probs, 'delay_offset': -5}
        return out

    # ------------------------------------------------------------------
    # Bandit interface
    # ------------------------------------------------------------------
    def begin_journey(self):
        pass

    def end_journey(self, total_time):
        pass

    def route(self, s_source: int, s_dest: int, t_source: int) -> HyperpathResult:
        self._initial_t_source = t_source
        self._initial_dest = s_dest
        self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)
        return self.cached_result

    def observe_delay(self, route: str, delay: float):
        self.total_observations += 1

    def observe_cancel(self, route: str, kind: str = 'true'):
        self.total_observations += 1
        self._n_cancel_obs += 1
        # First cancellation triggers regime detection: switch to the
        # disrupted-regime distributions. (BOCD would arrive at the
        # same conclusion after one or two strong observations; we
        # collapse it to a single threshold for clarity, matching the
        # paper's claim that ``the disruption-aware optimizer drops
        # fallback routes once it knows the corridor is hit''.)
        if self._current_regime == 'normal':
            self._current_regime = 'disrupted'
            self.graph.assign_distributions(
                self._build_topocsa_dist(self._disrupted_dist))
            self._dirty = True
        else:
            self._dirty = True

    def select_connection(self, current_stop: int, current_time: int,
                          rng, top_k: int = 5):
        # Recompute the hyperpath under the current distributions
        # whenever a regime change has been detected since the last
        # call (or every call after the first cancel).
        if getattr(self, '_dirty', False):
            self.cached_result = topocsa(
                self.graph, current_stop, self._initial_dest, current_time)
            self._dirty = False
        labels = self.cached_result.stop_labels.get(current_stop, [])
        if not labels:
            return None
        # Pick the lowest-mean-arrival label among those whose
        # connection is feasible (dep_time within the patience window
        # relative to current_time). This matches the Static router's
        # selection rule on the recomputed hyperpath.
        feasible = []
        for lab in labels:
            c = self.graph.connections[lab.connection_id]
            if c.dep_time < current_time - 1:
                continue
            if c.dep_time > current_time + 25:
                continue
            feasible.append(lab)
        if not feasible:
            return None
        best = min(feasible, key=lambda lab: lab.mean_dest_arrival)
        return best, best.mean_dest_arrival
