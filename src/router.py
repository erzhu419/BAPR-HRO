"""Adaptive routing engine: combines BOCD regime detection with Durner's TopoCSA.

This is the core of BAPR-HRO. At each decision point (transfer stop),
the router:
1. Feeds recent delay observations to the RegimeDetector
2. If regime has changed, updates the delay distributions
3. Re-runs TopoCSA with updated distributions
4. Returns ranked alternatives for the current stop
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

from .transit_graph import TransitGraph
from .durner.topocsa import topocsa, HyperpathResult
from .durner.reweight import reweight_hyperpath
from .bocd.regime_detector import RegimeDetector, DelayObservation
from .synthetic_network import create_regime_distributions as _default_regime_dists


@dataclass
class RoutingDecision:
    """Output of the adaptive router at a decision point."""
    hyperpath: HyperpathResult
    regime_id: int
    regime_name: str
    confidence: float
    surprise: float
    recomputed: bool    # whether hyperpath was recomputed this step
    computation_ms: float


class StaticRouter:
    """Baseline: compute hyperpath once at origin, never re-plan."""

    def __init__(self, graph: TransitGraph):
        self.graph = graph
        self.cached_result: Optional[HyperpathResult] = None

    def route(self, s_source: int, s_dest: int, t_source: int) -> RoutingDecision:
        if self.cached_result is None:
            t0 = time.time()
            self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)
            elapsed = (time.time() - t0) * 1000
        else:
            elapsed = 0.0

        return RoutingDecision(
            hyperpath=self.cached_result,
            regime_id=0,
            regime_name="normal",
            confidence=1.0,
            surprise=0.0,
            recomputed=elapsed > 0,
            computation_ms=elapsed,
        )

    def replan(self, current_stop: int, s_dest: int, current_time: int,
               observations: list[DelayObservation]) -> RoutingDecision:
        """Static router never re-plans."""
        return RoutingDecision(
            hyperpath=self.cached_result,
            regime_id=0,
            regime_name="normal",
            confidence=1.0,
            surprise=0.0,
            recomputed=False,
            computation_ms=0.0,
        )


class AdaptiveRouter:
    """BAPR-HRO adaptive router with BOCD regime detection.

    At each transfer point:
    1. Feed delay observations to RegimeDetector
    2. If regime changed or confidence is low, re-run TopoCSA with
       updated delay distributions
    3. Otherwise, reuse cached hyperpath
    """

    def __init__(
        self,
        graph: TransitGraph,
        regime_names: list[str] = None,
        recompute_threshold: float = 0.5,
        hazard_rate: float = 0.05,
        regime_dist_fn=None,
    ):
        self.graph = graph
        self.regime_names = regime_names or ["normal", "rush_hour", "disrupted_402", "weather"]
        self.regime_dist_fn = regime_dist_fn or _default_regime_dists
        self.detector = RegimeDetector(
            n_regimes=len(self.regime_names),
            regime_names=self.regime_names,
            hazard_rate=hazard_rate,
        )
        self.recompute_threshold = recompute_threshold
        self.prev_regime: int = 0
        self.cached_result: Optional[HyperpathResult] = None
        self.total_recomputes: int = 0

    def route(self, s_source: int, s_dest: int, t_source: int) -> RoutingDecision:
        """Initial route computation (no observations yet)."""
        t0 = time.time()
        self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)
        elapsed = (time.time() - t0) * 1000
        self.total_recomputes += 1

        return RoutingDecision(
            hyperpath=self.cached_result,
            regime_id=0,
            regime_name="normal",
            confidence=1.0,
            surprise=0.0,
            recomputed=True,
            computation_ms=elapsed,
        )

    def replan(
        self,
        current_stop: int,
        s_dest: int,
        current_time: int,
        observations: list[DelayObservation],
    ) -> RoutingDecision:
        """Re-evaluate route at a transfer point with new observations.

        This is where BAPR-HRO's adaptive behavior happens:
        - Feed observations to BOCD
        - Detect if regime has shifted
        - If shifted or uncertain, recompute hyperpath with new distributions
        """
        # Update regime detector
        detection = self.detector.update(observations)
        regime_id = detection["regime_id"]
        confidence = detection["confidence"]
        surprise = detection["surprise"]

        # Only recompute when regime actually changes (not on low confidence alone)
        regime_changed = (regime_id != self.prev_regime)
        should_recompute = regime_changed

        if should_recompute and current_stop != s_dest:
            # Update delay distributions for new regime
            regime_name = self.regime_names[regime_id]
            regime_dists = self.regime_dist_fn(regime_name)
            self.graph.assign_distributions(regime_dists)

            # Full recompute from current position with updated distributions.
            # This allows discovering NEW routes not in the original hyperpath
            # (the key advantage over static + reweight).
            t0 = time.time()
            self.cached_result = topocsa(
                self.graph, current_stop, s_dest, current_time)
            elapsed = (time.time() - t0) * 1000
            self.total_recomputes += 1
            self.prev_regime = regime_id
            recomputed = True
        else:
            elapsed = 0.0
            recomputed = False

        return RoutingDecision(
            hyperpath=self.cached_result,
            regime_id=regime_id,
            regime_name=self.regime_names[regime_id],
            confidence=confidence,
            surprise=surprise,
            recomputed=recomputed,
            computation_ms=elapsed,
        )


class PeriodicRouter:
    """Baseline: recompute hyperpath every N minutes regardless of regime."""

    def __init__(self, graph: TransitGraph, recompute_interval: int = 5):
        self.graph = graph
        self.interval = recompute_interval
        self.cached_result: Optional[HyperpathResult] = None
        self.last_recompute_time: int = 0
        self.total_recomputes: int = 0

    def route(self, s_source: int, s_dest: int, t_source: int) -> RoutingDecision:
        t0 = time.time()
        self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)
        elapsed = (time.time() - t0) * 1000
        self.last_recompute_time = t_source
        self.total_recomputes += 1
        return RoutingDecision(
            hyperpath=self.cached_result, regime_id=0, regime_name="normal",
            confidence=1.0, surprise=0.0, recomputed=True, computation_ms=elapsed)

    def replan(self, current_stop: int, s_dest: int, current_time: int,
               observations: list[DelayObservation]) -> RoutingDecision:
        if current_time - self.last_recompute_time >= self.interval:
            t0 = time.time()
            self.cached_result = topocsa(self.graph, current_stop, s_dest, current_time)
            elapsed = (time.time() - t0) * 1000
            self.last_recompute_time = current_time
            self.total_recomputes += 1
            return RoutingDecision(
                hyperpath=self.cached_result, regime_id=0, regime_name="normal",
                confidence=1.0, surprise=0.0, recomputed=True, computation_ms=elapsed)
        return RoutingDecision(
            hyperpath=self.cached_result, regime_id=0, regime_name="normal",
            confidence=1.0, surprise=0.0, recomputed=False, computation_ms=0.0)
