"""Online regime detection for transit delay streams.

Combines BOCD (BeliefTracker) with a surprise signal derived from
GTFS-RT observations: |actual_delay - predicted_delay|.

When a bus is much later (or earlier) than predicted, surprise is high,
triggering BOCD to shift belief toward h=0 (recent changepoint).
The RegimeDetector then classifies the current regime and selects
the appropriate delay distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Optional

from .belief_tracker import BeliefTracker


@dataclass
class DelayObservation:
    """A single real-time delay observation from GTFS-RT."""
    route: str
    stop_id: int
    scheduled_time: int      # minutes from midnight
    predicted_time: int      # GTFS-RT predicted time
    actual_time: int         # actual observed time (if available)
    timestamp: int           # observation time

    @property
    def predicted_delay(self) -> int:
        return self.predicted_time - self.scheduled_time

    @property
    def actual_delay(self) -> int:
        return self.actual_time - self.scheduled_time

    @property
    def prediction_error(self) -> float:
        """How wrong the prediction was — this is the surprise signal."""
        return abs(self.actual_time - self.predicted_time)


class TransitSurpriseComputer:
    """Compute surprise signal from transit delay observations.

    Adapted from BAPR's SurpriseComputer. Instead of reward z-score and
    Q-std, we use:
    - Signal 1: Prediction error z-score (|actual - predicted| normalized)
    - Signal 2: Delay magnitude spike (sudden large delay)

    EMA smoothing prevents noise from single observations.
    """

    def __init__(self, ema_alpha: float = 0.3, window: int = 20):
        self.ema_alpha = ema_alpha
        self.window = window
        self.ema_surprise = 0.0
        self.error_history: list[float] = []
        self.error_ema = 0.0
        self.error_var_ema = 1.0
        self.prev_mean_delay: Optional[float] = None

    def reset(self):
        self.ema_surprise = 0.0
        self.error_history = []
        self.error_ema = 0.0
        self.error_var_ema = 1.0
        self.prev_mean_delay = None

    def compute(self, observations: list[DelayObservation]) -> float:
        """Compute surprise from a batch of delay observations.

        Args:
            observations: Recent GTFS-RT observations.

        Returns:
            EMA-smoothed surprise value (higher = more surprising).
        """
        if not observations:
            return self.ema_surprise

        signals = []

        # Signal 1: Prediction error z-score
        errors = [obs.prediction_error for obs in observations]
        mean_error = np.mean(errors)
        self.error_history.append(mean_error)
        if len(self.error_history) > self.window:
            self.error_history = self.error_history[-self.window:]

        self.error_ema = 0.9 * self.error_ema + 0.1 * mean_error
        deviation = (mean_error - self.error_ema) ** 2
        self.error_var_ema = 0.9 * self.error_var_ema + 0.1 * deviation
        error_std = max(self.error_var_ema ** 0.5, 0.1)
        error_zscore = abs(mean_error - self.error_ema) / error_std
        signals.append(error_zscore)

        # Signal 2: Delay magnitude spike
        mean_delay = np.mean([obs.actual_delay for obs in observations])
        if self.prev_mean_delay is not None:
            delay_change = abs(mean_delay - self.prev_mean_delay)
            signals.append(delay_change / max(abs(self.prev_mean_delay), 1.0))
        self.prev_mean_delay = mean_delay

        # Combine (max-pooling, same as BAPR)
        raw_surprise = max(signals) if signals else 0.0

        self.ema_surprise = (self.ema_alpha * raw_surprise +
                             (1 - self.ema_alpha) * self.ema_surprise)
        return self.ema_surprise


class RegimeDetector:
    """Detect regime shifts in transit delay streams.

    Combines TransitSurpriseComputer (what to measure) with
    BeliefTracker (how to interpret measurements) to output
    a regime classification and confidence score.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        regime_names: Optional[list[str]] = None,
        hazard_rate: float = 0.05,
        max_run_length: int = 20,
    ):
        self.n_regimes = n_regimes
        self.regime_names = regime_names or [f"regime_{i}" for i in range(n_regimes)]
        self.belief_tracker = BeliefTracker(
            max_run_length=max_run_length,
            hazard_rate=hazard_rate,
        )
        self.surprise_computer = TransitSurpriseComputer()

        # Regime classification state
        self.current_regime: int = 0  # start with "normal"
        self.regime_history: list[int] = []
        # Per-regime delay statistics (updated online)
        self.regime_delay_stats: list[dict] = [
            {"mean": 1.0, "std": 2.0} for _ in range(n_regimes)
        ]

    def reset(self):
        self.belief_tracker.reset()
        self.surprise_computer.reset()
        self.current_regime = 0
        self.regime_history = []

    def update(self, observations: list[DelayObservation]) -> dict:
        """Process new delay observations and detect regime.

        Args:
            observations: Batch of recent GTFS-RT delay observations.

        Returns:
            Dict with regime_id, regime_name, confidence, surprise,
            effective_window, changepoint_prob.
        """
        # Compute surprise
        surprise = self.surprise_computer.compute(observations)

        # Update BOCD belief
        self.belief_tracker.update(surprise)

        # Classify regime based on delay patterns
        if observations:
            delays = [obs.actual_delay for obs in observations]
            mean_delay = np.mean(delays)
            max_delay = np.max(delays)
            self.current_regime = self._classify_regime(mean_delay, max_delay, surprise)

        self.regime_history.append(self.current_regime)

        return {
            "regime_id": self.current_regime,
            "regime_name": self.regime_names[self.current_regime],
            "confidence": self.belief_tracker.confidence,
            "surprise": surprise,
            "effective_window": self.belief_tracker.effective_window,
            "changepoint_prob": self.belief_tracker.changepoint_probability,
        }

    def _classify_regime(self, mean_delay: float, max_delay: float, surprise: float) -> int:
        """Classify current regime from delay statistics.

        Uses both mean and max delay: a single 30-min no-show (cancel)
        should trigger disruption detection even if other lines are fine.

        Regimes:
            0: normal (delays small)
            1: rush_hour (moderate delays)
            2: disrupted (any large delay or cancellation signal)
            3: weather (moderate delay + high variance)
        """
        # Determine candidate regime
        if max_delay >= 20 or surprise > 2.0:
            candidate = 2  # disrupted (cancel detected)
        elif mean_delay >= 5:
            candidate = 1  # rush_hour
        else:
            candidate = 0  # normal

        # Switch when BOCD has moderate evidence of changepoint.
        if candidate != self.current_regime:
            if self.belief_tracker.changepoint_probability > 0.35:
                return candidate
            else:
                return self.current_regime  # keep current, not enough evidence
        return candidate
