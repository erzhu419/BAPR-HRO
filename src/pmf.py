"""Discrete probability mass function (PMF) over integer time bins.

Durner's algorithm propagates entire arrival time distributions through the
transit network. Each distribution is a PMF over discrete time bins
(e.g., minutes from midnight). This module implements the core PMF operations
needed for Algorithm 5.3 (Topological CSA).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PMF:
    """Discrete probability distribution over integer time values.

    Represents P(T = offset + i) = probs[i] for i = 0, ..., len(probs)-1.
    The offset allows efficient storage: a distribution centered around
    t=480 (8:00 AM) only stores the non-negligible bins.

    Attributes:
        probs: Array of probabilities, sums to 1 (or less if partial).
        offset: The time value corresponding to probs[0].
    """
    probs: np.ndarray
    offset: int

    def __post_init__(self):
        self.probs = np.asarray(self.probs, dtype=np.float64)

    @staticmethod
    def deterministic(t: int) -> PMF:
        """A distribution concentrated at a single time point."""
        return PMF(probs=np.array([1.0]), offset=t)

    @staticmethod
    def from_delays(scheduled: int, delay_probs: np.ndarray, delay_offset: int = 0) -> PMF:
        """Build an arrival/departure PMF from a delay distribution.

        Args:
            scheduled: Scheduled time (minutes from midnight).
            delay_probs: P(delay = delay_offset + i) for i=0,...
            delay_offset: Minimum delay value (can be negative).
        """
        return PMF(probs=delay_probs.copy(), offset=scheduled + delay_offset)

    @property
    def support_min(self) -> int:
        return self.offset

    @property
    def support_max(self) -> int:
        return self.offset + len(self.probs) - 1

    @property
    def total_prob(self) -> float:
        return float(self.probs.sum())

    def mean(self) -> float:
        """Expected value E[T]."""
        if self.total_prob < 1e-12:
            return float('inf')
        times = np.arange(len(self.probs)) + self.offset
        return float(np.dot(times, self.probs) / self.total_prob)

    def prob_le(self, t: int) -> float:
        """P(T <= t)."""
        idx = t - self.offset
        if idx < 0:
            return 0.0
        if idx >= len(self.probs):
            return self.total_prob
        return float(self.probs[:idx + 1].sum())

    def scale(self, factor: float) -> PMF:
        """Return a new PMF with all probabilities multiplied by factor."""
        return PMF(probs=self.probs * factor, offset=self.offset)

    def normalize(self) -> PMF:
        """Normalize so probabilities sum to 1."""
        s = self.total_prob
        if s < 1e-12:
            return PMF(probs=self.probs.copy(), offset=self.offset)
        return PMF(probs=self.probs / s, offset=self.offset)

    def trim(self, threshold: float = 1e-10) -> PMF:
        """Remove leading/trailing near-zero bins."""
        mask = self.probs > threshold
        if not mask.any():
            return PMF(probs=np.array([0.0]), offset=self.offset)
        first = int(mask.argmax())
        last = len(self.probs) - 1 - int(mask[::-1].argmax())
        return PMF(probs=self.probs[first:last + 1].copy(), offset=self.offset + first)


def componentwise_sum(a: PMF, b: PMF) -> PMF:
    """Component-wise sum of two PMFs (Durner's ⊕ operator, Alg 5.3 line 15).

    This is NOT convolution. It adds the probability masses at each time bin:
        result(t) = a(t) + b(t)

    Used to accumulate T_temp across continuing connections.
    """
    lo = min(a.offset, b.offset)
    hi = max(a.support_max, b.support_max)
    size = hi - lo + 1
    result = np.zeros(size, dtype=np.float64)
    # Add a
    a_start = a.offset - lo
    result[a_start:a_start + len(a.probs)] += a.probs
    # Add b
    b_start = b.offset - lo
    result[b_start:b_start + len(b.probs)] += b.probs
    return PMF(probs=result, offset=lo)


def convolve_pmfs(a: PMF, b: PMF) -> PMF:
    """Convolution of two independent PMFs: P(A + B = t).

    Used for computing arrival time = departure time + travel time.
    """
    probs = np.convolve(a.probs, b.probs)
    return PMF(probs=probs, offset=a.offset + b.offset)


def prob_reachable(arr_pmf: PMF, dep_pmf: PMF, transfer_time: int) -> float:
    """P(T_arr(c) + transfer_time <= T_dep(c')).

    Probability that a passenger arriving via connection c can reach
    the departure of connection c' at the same stop, given minimum
    transfer time (walking time between platforms, etc.).

    Durner Section 3.2: P_reachable(c, c').
    """
    # For each arrival time t_a, we need P(dep >= t_a + transfer_time)
    total = 0.0
    for i, p_arr in enumerate(arr_pmf.probs):
        if p_arr < 1e-15:
            continue
        t_arr = arr_pmf.offset + i
        needed = t_arr + transfer_time
        # P(dep >= needed)
        p_dep_ok = 1.0 - dep_pmf.prob_le(needed - 1)
        total += p_arr * p_dep_ok
    return total
