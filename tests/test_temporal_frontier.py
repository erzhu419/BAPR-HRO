from itertools import product

import numpy as np
import pytest

from src.committed_transit import CommittedState
from src.evolving_trip_belief import EvolvingTripBelief
from src.information_planner import ExactPlanner
from src.platform_transit import build_platform_model
from src.risk_bounds import CommitmentBounds
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_frontier import TemporalFrontier, integrate_survival, time_labels
from src.transit_graph import Connection, Stop, TransitGraph


class FixedClocks(EvolvingTripBelief):
    def _grid(self, slot, shifts, anchors, cancellation_possible):
        return super()._grid(slot, np.array([0]), anchors, cancellation_possible)


def network(rows, probabilities):
    graph = TransitGraph()
    for stop in {s for row in rows for s in row[2:4]}:
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for row in rows:
        graph.add_connection(Connection(-1, *row))
    belief = FixedClocks(graph, minute=1, public_occurrences=False, cancel_persistence=1.)
    pars = {r: dict(mean=0., std=1., cancel_rate=q) for r, q in probabilities.items()}
    start = CommittedState(0, 0, None, 0, initialized=True)
    return graph, CommitmentBounds(belief, pars, graph, 2, 12), start


def test_second_ride_clock_and_walking_restrict_the_frontier():
    graph, bounds, state = network([('A','A',0,1,0,2), ('B','B',1,2,1,1), ('C','C',1,2,8,9)],
                                   {'A':0., 'B':0., 'C':0.})
    windows = {c.id: (c.dep_time, c.arr_time) for c in graph.connections}
    labels = time_labels(graph, state, 2, 12, windows)
    results, _ = TemporalFrontier(bounds, state).evaluate(labels)
    assert [(r.threshold, r.survival) for r in results] == [(9,1.), (9,1.)]
    walks = ((1,2,3),)
    with pytest.raises(ValueError, match='deterministic walk'):
        TemporalFrontier(bounds, state, walks=walks).evaluate(labels)
    valid, _ = TemporalFrontier(bounds, state, walks=walks).evaluate(time_labels(graph, state, 2, 12, windows, walks=walks))
    assert [(r.threshold, r.survival) for r in valid] == [(5,1.), (5,1.)]


def test_same_trip_crossings_count_once_and_arrival_equal_to_label_is_allowed():
    rates = {'A':0., 'B':.6, 'C':.8, 'D':0.}
    graph, bounds, state = network([('A','A',0,1,0,2), ('B','B',1,2,3,4),
        ('B','B',2,1,4,5), ('B','B',1,2,6,7), ('C','C',1,2,3,4), ('D','D',1,2,9,10)], rates)
    value, _ = TemporalFrontier(bounds, state).evaluate(({0:0,1:2,2:10}, {0:0,1:2,2:4}))
    assert value[0].survival == pytest.approx(.48)
    assert value[0].union_survival == pytest.approx(.4)
    assert value[0].repeated_trips == 1
    assert value[1].survival == 1.
    bank = []
    for b, c in product((True,False), repeat=2):
        canceled = {'A':False,'B':b,'C':c,'D':False}
        weight = (rates['B'] if b else 1-rates['B'])*(rates['C'] if c else 1-rates['C'])
        events = {e.id:VehicleEvent(0,canceled[e.trip_id],e.dep_time,e.arr_time) for e in graph.connections}
        bank.append((weight, JourneyScenario(events, minute=1, publications={})))
    exact = ExactPlanner(build_platform_model(graph, bank, 0, 2, 0, 12)).solve_compressed().value
    assert integrate_survival(0, [(r.threshold,r.survival) for r in value]) == pytest.approx(exact)
