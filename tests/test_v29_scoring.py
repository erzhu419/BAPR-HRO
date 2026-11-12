from itertools import product

import numpy as np
import pytest

from experiments.reconstruction_v29.scoring import evaluate_pair
from src.evolving_trip_belief import EvolvingTripBelief, _ChainLaw
from src.transfer_opportunity import RouteOption, TransferOpportunityScores
from src.transit_graph import Connection, Stop, TransitGraph


def law(states, weights, transition):
    with np.errstate(divide='ignore'):
        kernel = np.log(np.asarray(transition, dtype=float))
        initial = np.log(weights)
    return _ChainLaw([np.array([t for t, _ in row]) for row in states], initial,
        lambda _: kernel, [np.zeros(len(row)) for row in states],
        [np.array([c for _, c in row], dtype=bool) for row in states], (True,))


def fixture():
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=5 if sid == 1 else 0))
    rows = ((0, 'current', 0, 1, 0, 15), (1, 'target', 0, 1, 15, 20),
            (2, 'a', 1, 2, 15, 25), (3, 'b', 1, 2, 15, 23),
            (4, 'c', 1, 2, 30, 45), (5, 'd', 1, 2, 45, 55))
    for cid, trip, dep, arr, clock, end in rows:
        graph.add_connection(Connection(cid, 'R', trip, dep, arr, clock, end))
    laws = {'current': law([[(0, False), (20, False)],
                           [(10, False), (20, False), (30, False), (40, False)]],
                          [.5, .5], [[.5, .5, 0, 0], [0, 0, .5, .5]]),
            'target': law([[(-5, False), (5, False), (15, False), (35, False), (70, False), (5, True)],
                           [(0, False), (10, False), (20, False), (40, False), (75, False), (10, True)]],
                          [.2, .1, .2, .2, .1, .2], np.eye(6)),
            'b': law([[(15, False), (15, True)], [(23, False), (23, True)]], [.75, .25], np.eye(2))}
    for trip, clock, end in (('a', 15, 25), ('c', 30, 45), ('d', 45, 55)):
        laws[trip] = law([[(clock, False)], [(end, False)]], [1.], [[1.]])
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    belief._law = lambda trip, parameters: laws[trip]
    tails = {cid: (cid,) for cid in range(2, 6)}
    compiler = TransferOpportunityScores(graph, belief, {}, 50,
                                         {cid: RouteOption((cid,), tails) for cid in (0, 1)})
    return graph, compiler, laws


def atoms(law):
    for i, j in product(range(len(law.times[0])), range(len(law.times[1]))):
        probability = np.exp(law.initial[i]+law._transition(0)[i, j])
        if probability:
            yield (int(law.times[0][i]), int(law.times[1][j]), bool(law.canceled[0][i])), probability


def physical_cost(graph, world, chosen, attempt):
    departure, arrival, canceled = world[chosen]
    if canceled or departure < attempt or departure >= 50:
        return 50
    offers = [(world[cid][0], graph.connections[cid].arr_time-graph.connections[cid].dep_time, cid)
              for cid in range(2, 6) if not world[cid][2] and arrival+5 <= world[cid][0] < 50]
    return min(world[min(offers)[2]][1], 50) if offers else 50


def test_integrated_current_and_target_match_independent_joint_worlds_with_mct_ties_and_deadline():
    graph, compiler, laws = fixture()
    names = ('current', 'target', 'a', 'b', 'c', 'd')
    current, target, total = 0., 0., 0.
    for combination in product(*(tuple(atoms(laws[name])) for name in names)):
        world = {cid: row[0] for cid, row in enumerate(combination)}
        probability = np.prod([row[1] for row in combination])
        if world[0][0] != 0:
            continue
        current += probability*physical_cost(graph, world, 0, 0)
        target += probability*physical_cost(graph, world, 1, 1)
        total += probability
    result = evaluate_pair(compiler, 0, 1, clock=0, entry=0, ready=0)
    assert result['status'] == 'supported'
    assert result['integrated']['current'] == pytest.approx(current/total)
    assert result['integrated']['wait'] == pytest.approx(target/total)
    assert result['integrated']['current'] == pytest.approx(34.25)
    assert result['integrated']['wait'] == pytest.approx(46.35)
    assert result['support']['before_entry_mass'] == pytest.approx(.2)
    assert result['support']['cancellation_mass'] == pytest.approx(.2)
    assert result['support']['after_deadline_mass'] == pytest.approx(.1)


def test_off_grid_projection_reports_endpoint_ranking_against_same_actual_wait():
    _, compiler, _ = fixture()
    result = evaluate_pair(compiler, 0, 1, clock=10, entry=0, ready=0)
    assert result['current_clock_method'] == 'projection'
    assert result['support']['denominator'] == pytest.approx(.9)  # Exclude witnessed-window atom5.
    assert result['integrated']['wait'] == pytest.approx(50-1/.9)
    assert result['integrated']['current'] == pytest.approx((34.25+50)/2)
    bracket = result['current_bracket']
    assert [row['clock'] for row in bracket['endpoints']] == [0, 20]
    assert bracket['integrated_ranking_sensitive']
    assert all(row['integrated_comparison']['wait'] == result['integrated']['wait'] for row in bracket['endpoints'])
    assert result['nominal']['current'] == compiler.clock_nominal_score(0, 10)


def test_zero_survival_or_unsupported_current_returns_no_invented_ordering():
    _, compiler, _ = fixture()
    result = evaluate_pair(compiler, 0, 1, clock=30, entry=0, ready=0)
    assert result['status'] == 'unsupported' and result['reason'] == 'current_outside_score_support'
    assert result['integrated'] is None
    compiler._departures[1] = np.array([5]), np.array([1.]), 0.
    result = evaluate_pair(compiler, 0, 1, clock=10, entry=0, ready=0)
    assert result['status'] == 'unsupported' and result['reason'] == 'target_zero_survival'
    assert result['nominal'] is None and result['integrated'] is None
