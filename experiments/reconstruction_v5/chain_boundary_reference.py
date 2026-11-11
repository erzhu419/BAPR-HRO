"""Reference inference on fixed clock/service chain factors, outside production.

Exact observations separate a chain only when the entire (clock, service)
state is fixed. All adaptive-grid metadata remains part of these factors.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from src.evolving_trip_belief import logsumexp


def normalized(log_weights):
    normalizer = logsumexp(log_weights)
    if not np.isfinite(normalizer):
        raise ValueError('No consistent chain path')
    return log_weights - normalizer


@dataclass(frozen=True)
class BoundaryResult:
    probabilities: np.ndarray
    left_separator: int | None
    right_separator: int | None
    forward_steps: int
    backward_steps: int


class BoundaryChain:
    """Query one marginal using nearest exact separators and neutral suffixes.

    The chain must have positive evidence likelihood. This is a fixed-factor
    reference: regridding or parameter changes require new factors. Checking
    row stochasticity here scans supplied kernels once; production can inherit
    that property from the ordered-kernel construction instead.
    """

    def __init__(self, initial, transitions, emissions):
        self.initial = initial
        self.transitions = tuple(transitions)
        self.emissions = tuple(emissions)
        self.exact = {i: int(np.flatnonzero(np.isfinite(e))[0]) for i, e in enumerate(emissions)
                      if np.count_nonzero(np.isfinite(e)) == 1}
        self.stochastic = tuple(bool(np.allclose(logsumexp(k, axis=1), 0, atol=1e-12, rtol=0))
                                for k in transitions)

    def marginal(self, query):
        if query in self.exact:
            probabilities = np.zeros(len(self.emissions[query]))
            probabilities[self.exact[query]] = 1
            return BoundaryResult(probabilities, query, query, 0, 0)
        left = max((i for i in self.exact if i < query), default=None)
        right = min((i for i in self.exact if i > query), default=None)
        if left is None:
            forward = normalized(self.initial + self.emissions[0])
            first = 0
        else:
            forward = np.full(len(self.emissions[left]), -np.inf)
            forward[self.exact[left]] = 0
            first = left
        for slot in range(first + 1, query + 1):
            forward = normalized(logsumexp(forward[:, None] + self.transitions[slot - 1], axis=0)
                                 + self.emissions[slot])
        if right is not None:
            end = right
            backward = np.full(len(self.emissions[end]), -np.inf)
            backward[self.exact[end]] = 0
        else:
            # A stochastic transition with no later likelihood has backward
            # message one. Substochastic/dead rows must not be skipped.
            end = max([query, *(i for i in range(query + 1, len(self.emissions))
                                if np.any(self.emissions[i] != 0)),
                       *(i + 1 for i in range(query, len(self.transitions))
                         if not self.stochastic[i])])
            backward = np.zeros(len(self.emissions[end]))
        for slot in range(end - 1, query - 1, -1):
            backward = normalized(logsumexp(self.transitions[slot] +
                                            self.emissions[slot + 1] + backward, axis=1))
        probability = np.exp(normalized(forward + backward))
        return BoundaryResult(probability, left, right, query - first, end - query)


def full_marginals(initial, transitions, emissions):
    forward = [normalized(initial + emissions[0])]
    for kernel, emission in zip(transitions, emissions[1:]):
        forward.append(normalized(logsumexp(forward[-1][:, None] + kernel, axis=0) + emission))
    backwards = np.zeros(len(emissions[-1]))
    result = [None] * len(emissions)
    for slot in range(len(emissions) - 1, -1, -1):
        result[slot] = np.exp(normalized(forward[slot] + backwards))
        if slot:
            backwards = normalized(logsumexp(transitions[slot - 1] + emissions[slot] + backwards, axis=1))
    return result


def ordered_service_feasible(times, canceled, emissions, initial_allowed, service_steps):
    """Linear support reachability for ordered Gaussian clock/service chains.

    ``service_steps[i][a,b]`` states whether the availability transition from
    slot i to i+1 has positive probability, with 0 operating and 1 canceled.
    Every allowed ordered clock pair has a strictly positive clock potential.
    """
    earliest = [math.inf, math.inf]
    inspected = 0
    for slot, (clocks, states, emission) in enumerate(zip(times, canceled, emissions)):
        next_earliest = [math.inf, math.inf]
        inspected += len(clocks)
        allowed = np.isfinite(emission)
        for state in (0, 1):
            threshold = ((-math.inf if initial_allowed[state] else math.inf) if slot == 0 else
                         min((earliest[previous] for previous in (0, 1)
                              if service_steps[slot - 1][previous, state]), default=math.inf))
            reachable = allowed & (states == state) & (clocks >= threshold)
            if reachable.any():
                next_earliest[state] = int(clocks[reachable].min())
        earliest = next_earliest
        if min(earliest) == math.inf:
            return False, inspected
    return True, inspected
