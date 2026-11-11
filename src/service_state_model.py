"""A fixed joint law for segment completion, including stop and restoration.

Three independent Beta variables per route govern first-retained-segment
failure, operating->failed, and failed->operating. They are drawn once per
world and shared by that route's trips. This is an explicit generative model,
not a claim that correlated historical service records are iid observations.
Completion failure does not specify its physical time/place or announcement.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from math import exp, lgamma
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class BetaRate:
    alpha: float
    beta: float

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def log_moment(self, events, nonevents):
        a, b = self.alpha, self.beta
        return (lgamma(a + events) + lgamma(b + nonevents) - lgamma(a + b + events + nonevents)
                - lgamma(a) - lgamma(b) + lgamma(a + b))


@dataclass(frozen=True)
class RouteCompletion:
    initial_failure: BetaRate
    stop: BetaRate
    restore: BetaRate


@dataclass(frozen=True)
class ServiceTrip:
    route: str
    connections: tuple


@dataclass(frozen=True)
class ServiceFact:
    connection: int
    can_complete: bool
    received_at: int


@dataclass(frozen=True)
class CompletionLabel:
    can_complete: bool | None
    journey: str | None
    reason: str


def completion_label(record):
    """Offline traversal label; it provides neither failure nor receipt time."""
    departure, arrival = record['departure'], record['arrival']
    identities = {event.get('official_journey') for event in (departure, arrival)
                  if event.get('official_journey') is not None}
    if len(identities) != 1:
        return CompletionLabel(None, None, 'unresolved_service_identity')
    journey = next(iter(identities))
    if any(event.get('status') == 'CANCELED' for event in (departure, arrival)):
        return CompletionLabel(False, journey, 'explicit_segment_cancellation')
    if all(event.get('status') == 'REAL' for event in (departure, arrival)):
        return CompletionLabel(True, journey, 'measured_completed_segment')
    return CompletionLabel(None, journey, 'unresolved_completion')


def fit_completion_parameters(days, prior=.5):
    """Known completion labels only; unresolved counts remain in the audit.

    The posterior interpretation assumes fixed route hazards and conditionally
    Bernoulli transitions. A date-block frequentist guarantee is not supplied.
    """
    counts = defaultdict(Counter)
    audit = {}
    for date, rows in days.items():
        trips = defaultdict(list)
        day = Counter()
        for row in rows:
            trips[row['trip_id']].append(row)
            counts[row['route_id']]
        for records in trips.values():
            records.sort(key=lambda row: (row['dep_sequence'], row['arr_sequence'], row['id']))
            first = completion_label(records[0])
            route = records[0]['route_id']
            if first.can_complete is None:
                counts[route]['initial_unknown'] += 1
            else:
                counts[route]['initial_' + ('success' if first.can_complete else 'failure')] += 1
            for before, after in zip(records, records[1:]):
                if (before['arr_stop'] != after['dep_stop']
                        or before['arr_sequence'] != after['dep_sequence']):
                    day['nonadjacent_retained_segments'] += 1
                    continue
                left, right = completion_label(before), completion_label(after)
                route = before['route_id']
                if (left.can_complete is None or right.can_complete is None
                        or left.journey != right.journey):
                    counts[route]['transition_unknown'] += 1
                    day['transition_unknown'] += 1
                    continue
                key = ('operating_' if left.can_complete else 'failed_')
                key += 'operating' if right.can_complete else 'failed'
                counts[route][key] += 1
                day[key] += 1
        audit[date] = dict(day)
    parameters = {}
    for route, row in counts.items():
        parameters[route] = RouteCompletion(
            BetaRate(prior + row['initial_failure'], prior + row['initial_success']),
            BetaRate(prior + row['operating_failed'], prior + row['operating_operating']),
            BetaRate(prior + row['failed_operating'], prior + row['failed_failed']))
    return parameters, {'by_route': {route: dict(row) for route, row in counts.items()}, 'by_date': audit}


class ServiceStateModel:
    """One whole-journey generative law, with immutable model parameters.

    Sampling is linear in retained segments. Exact enumeration is reserved
    for the small-model audit and is exponential in their number.
    """
    def __init__(self, trips, parameters):
        self.trips = tuple(trips)
        self.parameters = MappingProxyType(dict(parameters))
        self.connections = tuple(cid for trip in self.trips for cid in trip.connections)
        self.positions = {cid: i for i, cid in enumerate(self.connections)}

    def path_probability(self, states):
        """Integrate shared route hazards, rather than multiply marginal means."""
        counts = defaultdict(Counter)
        for trip in self.trips:
            complete = [bool(states[cid]) for cid in trip.connections]
            row = counts[trip.route]
            row['initial_failure' if not complete[0] else 'initial_success'] += 1
            for left, right in zip(complete, complete[1:]):
                row[('stop' if left else 'restore') + ('_event' if left != right else '_nonevent')] += 1
        value = 0.
        for route, row in counts.items():
            params = self.parameters[route]
            value += params.initial_failure.log_moment(row['initial_failure'], row['initial_success'])
            for name in ('stop', 'restore'):
                value += getattr(params, name).log_moment(row[name + '_event'], row[name + '_nonevent'])
        return exp(value)

    def exact(self):
        worlds = tuple(product((True, False), repeat=len(self.connections)))
        weights = np.array([self.path_probability(dict(zip(self.connections, states))) for states in worlds])
        weights /= weights.sum()
        return ServicePosterior(self, worlds, weights)

    def sample(self, rng, n):
        result = []
        for _ in range(n):
            rates = {route: tuple(rng.beta(rate.alpha, rate.beta) for rate in
                     (params.initial_failure, params.stop, params.restore))
                     for route, params in self.parameters.items()}
            states = {}
            for trip in self.trips:
                initial, stop, restore = rates[trip.route]
                operating = rng.random() >= initial
                for index, cid in enumerate(trip.connections):
                    if index:
                        operating = (rng.random() >= stop) if operating else (rng.random() < restore)
                    states[cid] = operating
            result.append(MappingProxyType(states))
        return result


class ServicePosterior:
    """Condition the same original worlds at every history; never reproject."""
    def __init__(self, model, worlds, weights, evidence=None):
        self.model, self.worlds = model, worlds
        self.weights = np.array(weights, copy=True)
        self.weights.flags.writeable = False
        self.evidence = MappingProxyType(dict(evidence or {}))

    def condition(self, facts):
        evidence = dict(self.evidence)
        if any(cid in evidence and evidence[cid] != complete for cid, complete in facts.items()):
            raise ValueError('Contradictory completion facts')
        evidence.update(facts)
        mask = np.array([all(states[self.model.positions[cid]] == complete for cid, complete in facts.items())
                         for states in self.worlds])
        weights = self.weights * mask
        if not weights.sum():
            raise ValueError('Completion evidence has zero probability')
        return ServicePosterior(self.model, self.worlds, weights / weights.sum(), evidence)

    def condition_received(self, facts, time):
        return self.condition({fact.connection: fact.can_complete for fact in facts if fact.received_at <= time})

    def completion_probability(self, cid):
        index = self.model.positions[cid]
        return float(sum(weight for states, weight in zip(self.worlds, self.weights) if states[index]))

    def sample(self, rng, n):
        indices = rng.choice(len(self.worlds), size=n, p=self.weights)
        return [MappingProxyType(dict(zip(self.model.connections, self.worlds[index]))) for index in indices]
