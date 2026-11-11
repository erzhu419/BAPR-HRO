from itertools import product

import numpy as np
import pytest

from src.evolving_trip_belief import EvolvingTripBelief, logsumexp
from src.information_planner import ExactPlanner
from src.opportunity_bounds import ExitWindow, FirstExitOpportunities, OrderedTripOpportunities
from src.platform_transit import build_platform_model
from src.risk_bounds import CommitmentBounds
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


class SmallGrid(EvolvingTripBelief):
    def _grid(self, slot, shifts, anchors, cancellation_possible):
        return super()._grid(slot, np.array([-1, 0, 1]), anchors, cancellation_possible)


def network(rows):
    graph = TransitGraph()
    for sid in sorted({s for row in rows for s in row[2:4]}):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=0))
    for row in rows:
        graph.add_connection(Connection(-1, *row))
    return graph


def enumerate_trip(law, slots):
    paths = []
    for states in product(*(range(len(t)) for t in law.times)):
        log_mass = law.initial[states[0]] + law.emissions[0][states[0]]
        for i, state in enumerate(states[1:], 1):
            log_mass += law._transition(i - 1)[states[i - 1], state] + law.emissions[i][state]
        if np.isfinite(log_mass):
            paths.append((log_mass, {slots[i].cid: VehicleEvent(
                0, bool(law.canceled[i][states[i]]), int(law.times[i][states[i]]),
                int(law.times[i + 1][states[i + 1]])) for i in range(0, len(states), 2)}))
    norm = logsumexp(np.array([p for p, _ in paths]))
    return [(float(np.exp(p - norm)), events) for p, events in paths]


def test_chain_masks_match_exact_joint_service_and_clock_enumeration():
    graph = network([('R', 'X', 0, 1, 2, 3), ('R', 'X', 1, 0, 3, 4),
                     ('R', 'X', 0, 2, 4, 5)])
    belief = SmallGrid(graph, minute=1, public_occurrences=False)
    belief.observe_connection_event(1, 'arrival', 0)
    parameters = {'R': dict(mean=0., std=.6, cancel_rate=.25)}
    bounds = CommitmentBounds(belief, parameters, graph, 2, 8)
    calculation = OrderedTripOpportunities(bounds)
    windows = [ExitWindow(0, 2, 1), ExitWindow(2, 4, 0)]
    cutoffs = np.arange(0, 8)
    bank = enumerate_trip(bounds._law('X'), belief._trips['X'])

    def eligible(event, window, cutoff):
        return not event.canceled and event.departure >= window.ready and event.arrival + window.remaining <= cutoff

    expected = [sum(p for p, events in bank if not any(
        eligible(events[w.connection], w, z) for w in windows)) for z in cutoffs]
    assert calculation.no_event('X', windows, cutoffs) == pytest.approx(expected)
    for w in windows:
        direct = [sum(p for p, events in bank if eligible(events[w.connection], w, z)) for z in cutoffs]
        assert calculation.connection_probability(w, cutoffs) == pytest.approx(direct)


def test_integer_tail_sum_is_tight_despite_zero_static_runtime_floor():
    graph = network([('A', 'A', 0, 1, 0, 2), ('B', 'B', 1, 2, 3, 5)])
    belief = SmallGrid(graph, minute=1, public_occurrences=False)
    for c in graph.connections:
        belief.observe_connection_event(c.id, 'departure', 0)
        belief.observe_connection_event(c.id, 'arrival', 0)
    parameters = {route: dict(mean=0., std=.6, cancel_rate=0.) for route in ('A', 'B')}
    bounds = CommitmentBounds(belief, parameters, graph, 2, 8)
    local = {0: VehicleEvent(0, False, 0, 2)}
    world = JourneyScenario({**local, 1: VehicleEvent(0, False, 3, 5)}, minute=1, publications={})
    exact = ExactPlanner(build_platform_model(graph, [(1., world)], 0, 2, 0, 8)).solve().value
    envelope = FirstExitOpportunities(OrderedTripOpportunities(bounds), graph, 2)
    value = envelope.evaluate([(1., local)], (0,), 0, 8, range(1, 9))
    assert value.static_lower == 2
    assert value.lower == exact == 5
    coarse = envelope.evaluate([(1., local)], (0,), 0, 8, [2, 4, 6, 8])
    assert 2 <= coarse.lower < value.lower


def test_walk_shortcut_and_same_trip_early_alternative_remain_in_bound():
    graph = network([('R', 'A', 0, 1, 0, 4), ('R', 'A', 0, 2, 2, 3),
                     ('R', 'B', 1, 2, 7, 8)])
    belief = SmallGrid(graph, minute=1, public_occurrences=False)
    bounds = CommitmentBounds(belief, {'R': dict(mean=0., std=.6, cancel_rate=0.)}, graph, 2, 10)
    envelope = FirstExitOpportunities(OrderedTripOpportunities(bounds), graph, 2,
                                     footpaths=((0, 2, 2),))
    local = {0: VehicleEvent(0, False, 0, 4), 1: VehicleEvent(0, False, 2, 3)}
    value = envelope.evaluate([(1., local)], (0,), 0, 10, range(1, 11))
    assert value.lower <= 2


def test_cross_trip_independence_is_required_for_multiplication():
    # Coupled opposite cancellations: at least one vehicle always operates.
    # Marginal no-event probabilities are both .5, so multiplying them .25
    # would incorrectly lower-bound the true zero survival probability.
    operating = [(True, False), (False, True)]
    no_opportunity = sum(not any(world) for world in operating) / len(operating)
    marginal_product = np.prod([sum(not world[i] for world in operating) / len(operating)
                                for i in range(2)])
    assert no_opportunity == 0 < marginal_product
