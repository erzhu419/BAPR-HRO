"""Finite joint-outcome values for a committed transit connection attempt.

Incoming arrival is observed before the attempt. Departure and arrival of the
next vehicle retain their joint dependence. Cancellation has explicit mass and
an optional public announcement time. A timeout reveals only absence during the
waiting window; hidden future departures and cancellations remain pooled.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Iterable, Mapping

from .pmf import PMF


@dataclass(frozen=True)
class JointOutcome:
    probability: float
    incoming_arrival: float
    next_departure: float | None
    next_arrival: float | None
    next_canceled: bool = False
    cancel_announcement: float | None = None
    departure_reported_at: float | None = None  # receipt of an exact public departure report


@dataclass(frozen=True)
class FailureState:
    """Information observed when the passenger may choose a rescue policy."""
    time: float
    kind: str
    incoming_arrival: float | None
    observed_departure: float | None = None
    cancellation_time: float | None = None


@dataclass(frozen=True)
class FailureBranch:
    state: FailureState
    probability: float
    # Index and conditional probability in the caller's JointOutcome sequence.
    outcome_weights: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class TransitionSummary:
    catch_probability: float
    conditional_catch_loss: float | None
    catch_outcome_weights: tuple[tuple[int, float], ...]
    failure_branches: tuple[FailureBranch, ...]
    cap: float

    @property
    def failure_probability(self) -> float:
        return sum(branch.probability for branch in self.failure_branches)


def shared_trip_outcomes(
    delay_pmf: PMF,
    *,
    scheduled_incoming_arrival: float,
    scheduled_next_departure: float,
    scheduled_next_arrival: float,
) -> tuple[JointOutcome, ...]:
    """Condition on the current trip operating, then use one shared delay.

    Missing input mass is conditioned away: a passenger already riding this
    trip cannot independently redraw a cancellation on its next segment.
    """
    if delay_pmf.total_prob <= 0:
        raise ValueError('The boarded trip needs positive operation probability')
    return tuple(JointOutcome(
        float(mass / delay_pmf.total_prob),
        scheduled_incoming_arrival + delay_pmf.offset + index,
        scheduled_next_departure + delay_pmf.offset + index,
        scheduled_next_arrival + delay_pmf.offset + index,
    ) for index, mass in enumerate(delay_pmf.probs) if mass > 0)


def independent_trip_outcomes(
    incoming_delay_pmf: PMF,
    next_delay_pmf: PMF,
    *,
    scheduled_incoming_arrival: float,
    scheduled_next_departure: float,
    scheduled_next_arrival: float,
    cancel_announcement: float | None,
) -> tuple[JointOutcome, ...]:
    """Independent trip delays, conditional on the incoming trip operating.

    The next delay PMF is a sub-probability distribution; its missing mass is
    an explicit canceled outcome. ``None`` means no cancellation announcement.
    """
    incoming_mass, next_mass = incoming_delay_pmf.total_prob, next_delay_pmf.total_prob
    if incoming_mass <= 0 or next_mass > 1.0 + 1e-10:
        raise ValueError('Invalid conditional incoming or next operation probability')
    outcomes = []
    for index, mass in enumerate(incoming_delay_pmf.probs):
        if mass <= 0:
            continue
        incoming_arrival = scheduled_incoming_arrival + incoming_delay_pmf.offset + index
        p_incoming = float(mass / incoming_mass)
        for next_index, next_probability in enumerate(next_delay_pmf.probs):
            if next_probability <= 0:
                continue
            delay = next_delay_pmf.offset + next_index
            outcomes.append(JointOutcome(
                p_incoming * float(next_probability), incoming_arrival,
                scheduled_next_departure + delay, scheduled_next_arrival + delay,
            ))
        if next_mass < 1.0:
            outcomes.append(JointOutcome(
                p_incoming * (1.0 - next_mass), incoming_arrival, None, None,
                next_canceled=True, cancel_announcement=cancel_announcement,
            ))
    return tuple(outcomes)


def summarize_transition(
    outcomes: Iterable[JointOutcome],
    *,
    min_transfer_time: float,
    same_trip: bool,
    journey_departure: float,
    cap: float,
    wait_limit: float = 12,
    decision_time: float | None = None,
) -> TransitionSummary:
    """Value one chosen connection, separating catch and observable failure.

    The passenger commits to waiting until boarding, a public cancellation,
    or the end of the waiting window. Transfers first consume the known
    minimum transfer time. Conditional terminal loss on catch is the actual
    capped journey duration, using only the outcomes in which boarding occurs.
    This function chooses no rescue actions.

    Call this kernel within one current observable history, with joint
    outcomes already conditioned on that history. FailureState records only
    the new local observations; its fields are not a sufficient state for an
    arbitrary network. A caller caching across histories must retain its own
    history key together with the returned branch.

    A missed attempt reveals that the vehicle left before readiness. It gives
    an exact departure time only if a public report of the completed event
    has been received by that point, as specified by departure_reported_at.
    """
    outcomes = tuple(outcomes)
    total = sum(outcome.probability for outcome in outcomes)
    if not isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Joint outcome probabilities must sum to one')
    if cap <= 0 or wait_limit <= 0 or min_transfer_time < 0:
        raise ValueError('cap and wait_limit must be positive; transfer time nonnegative')
    deadline = journey_departure + cap
    caught = []
    weighted_catch_loss = 0.0
    failures: dict[FailureState, list[tuple[int, float]]] = {}
    for index, outcome in enumerate(outcomes):
        p = outcome.probability / total
        if p < 0:
            raise ValueError('Outcome probabilities must be nonnegative')
        if p == 0:
            continue
        ready = outcome.incoming_arrival + (0 if same_trip else min_transfer_time)
        if decision_time is not None:
            ready = max(ready, decision_time)
        if ready >= deadline:
            # Arrival beyond the horizon has not yet become an observation.
            observed_arrival = outcome.incoming_arrival if outcome.incoming_arrival <= deadline else None
            state = FailureState(deadline, 'horizon', observed_arrival)
        else:
            wait_end = min(deadline, ready + wait_limit)
            if outcome.next_canceled:
                announcement = outcome.cancel_announcement
                if announcement is not None and announcement <= wait_end:
                    state = FailureState(max(ready, announcement), 'canceled',
                                         outcome.incoming_arrival, cancellation_time=announcement)
                else:
                    state = FailureState(wait_end, 'wait_timeout', outcome.incoming_arrival)
            else:
                dep, arr = outcome.next_departure, outcome.next_arrival
                if dep is None or arr is None or arr < dep:
                    raise ValueError('Operating outcomes need ordered departure and arrival times')
                if dep < ready:
                    receipt = outcome.departure_reported_at
                    observed_departure = dep if receipt is not None and dep <= receipt <= ready else None
                    state = FailureState(ready, 'missed_departure', outcome.incoming_arrival,
                                         observed_departure=observed_departure)
                elif dep <= wait_end:
                    caught.append((index, p))
                    loss = min(cap, max(0.0, arr - journey_departure))
                    weighted_catch_loss += p * loss
                    continue
                else:
                    state = FailureState(wait_end, 'wait_timeout', outcome.incoming_arrival)
        failures.setdefault(state, []).append((index, p))
    p_catch = sum(probability for _, probability in caught)
    branches = []
    for state, weights in failures.items():
        probability = sum(p for _, p in weights)
        branches.append(FailureBranch(state, probability,
                                      tuple((index, p / probability) for index, p in weights)))
    return TransitionSummary(
        p_catch, weighted_catch_loss / p_catch if p_catch else None,
        tuple((index, p / p_catch) for index, p in caught), tuple(branches), cap,
    )


def combine_rescue_costs(
    summary: TransitionSummary,
    rescue_losses: Mapping[FailureState, float],
) -> float:
    """Combine one causal rescue value per observable failure state.

    Values are expected capped journey losses from a rescue policy executable
    with that state's information. They include elapsed time. The caller may
    use ``outcome_weights`` to evaluate one shared action or causal policy over
    a conditional branch; taking the best action separately in each hidden
    outcome would be a clairvoyant value and is not a causal rescue value.
    """
    result = summary.catch_probability * (summary.conditional_catch_loss or 0.0)
    for branch in summary.failure_branches:
        loss = float(rescue_losses[branch.state])
        if not 0 <= loss <= summary.cap:
            raise ValueError('Rescue losses must use the same capped journey objective')
        result += branch.probability * loss
    return result
