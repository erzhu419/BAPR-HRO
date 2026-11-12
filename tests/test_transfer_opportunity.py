from itertools import product

import numpy as np
import pytest

from src.evolving_trip_belief import EvolvingTripBelief, _ChainLaw
from src.transfer_opportunity import RouteOption, TransferOpportunityScores
from src.transit_graph import Connection, Stop, TransitGraph


def finite_law(states, initial, transitions):
    times = [np.array([t for t, _ in values]) for values in states]
    canceled = [np.array([c for _, c in values], dtype=bool) for values in states]
    with np.errstate(divide='ignore'):
        logs = [np.log(np.asarray(p)) for p in transitions]
        initial = np.log(np.asarray(initial))
    return _ChainLaw(times, initial, logs.__getitem__, [np.zeros(len(t)) for t in times],
                     canceled, (True,) * (len(times) - 1))


def two_event(departures, mass, arrivals):
    return finite_law([[(t, False) for t in departures], [(t, False) for t in arrivals]],
                      mass, [np.eye(len(departures))])


def fixture(first=None, tails=None, deadline=100):
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    graph.add_connection(Connection(0, 'R', 'first', 0, 1, 0, 15))
    graph.add_connection(Connection(1, 'R', 'a', 1, 2, 15, 25))
    graph.add_connection(Connection(2, 'R', 'b', 1, 2, 30, 40))
    laws = {'first': first or finite_law([[(0, False)], [(10, False), (20, False)]], [1.], [[[.5, .5]]]),
            'a': two_event((15,), (1.,), (25,)), 'b': two_event((30,), (1.,), (40,))}
    laws.update(tails or {})
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    belief._law = lambda trip, pars: laws[trip]
    compiler = TransferOpportunityScores(graph, belief, {}, deadline, {0: RouteOption((0,), {1: (1,), 2: (2,)})})
    return graph, compiler, laws


def enumerate_trip(law):
    for indices in product(*(range(len(times)) for times in law.times)):
        p = np.exp(law.initial[indices[0]])
        for slot, (left, right) in enumerate(zip(indices, indices[1:])):
            p *= np.exp(law._transition(slot)[left, right])
        if p:
            yield tuple(int(law.times[i][j]) for i, j in enumerate(indices)), tuple(
                bool(law.canceled[i][j]) for i, j in enumerate(indices)), p


def test_transfer_score_matches_full_independent_chain_enumeration_and_mean_is_not_exact():
    graph, compiler, laws = fixture()
    expected = 0.
    for first, a, b in product(*(tuple(enumerate_trip(laws[name])) for name in ('first', 'a', 'b'))):
        arrival = first[0][-1]
        offers = [(tail[0][0], graph.connections[cid].arr_time - graph.connections[cid].dep_time,
                   cid, tail) for cid, tail in ((1, a), (2, b))
                  if not tail[1][0] and arrival <= tail[0][0] < 100]
        tail = min(offers)[-1] if offers else None
        cost = 100 if tail is None or any(tail[1]) else min(tail[0][-1], 100)
        expected += first[2] * a[2] * b[2] * cost
    score = compiler.board_score(0, 0)
    assert expected == score.exact == 32.5
    assert score.mean_arrival == 25.
    assert score.clock_nominal == 25.
    assert score.mean_transfer_arrival == 15.
    assert score.success_probability == 1.


def test_tail_departed_before_entry_stays_in_failure_mass_and_late_rescue_has_no_timeout():
    first = two_event((0,), (1.,), (10,))
    graph, compiler, _ = fixture(first, {'a': two_event((5, 20), (.5, .5), (15, 30)),
                                       'b': two_event((900,), (1.,), (920,))}, deadline=1000)
    # Half the worlds missed a before entering and must wait beyond 720 seconds.
    assert compiler.board_score(0, 0).exact == 475.
    assert compiler.tail_value(0, [10, 21, 901]).tolist() == [475., 920., 1000.]


def test_two_random_tail_survivals_match_full_world_product_without_future_peeking():
    a = finite_law([[(5, False), (15, False), (15, True), (30, False)],
                    [(15, False), (25, False), (25, True), (40, False)]],
                   [.2, .3, .1, .4], [np.eye(4)])
    b = two_event((15, 30), (.4, .6), (45, 50))
    graph, compiler, laws = fixture(tails={'a': a, 'b': b})
    expected = 0.
    for first, a, b in product(*(tuple(enumerate_trip(laws[name])) for name in ('first', 'a', 'b'))):
        offers = [(tail[0][0], graph.connections[cid].arr_time - graph.connections[cid].dep_time,
                   cid, tail) for cid, tail in ((1, a), (2, b))
                  if not tail[1][0] and first[0][-1] <= tail[0][0] < 100]
        tail = min(offers)[-1] if offers else None
        cost = 100 if tail is None or any(tail[1]) else min(tail[0][-1], 100)
        expected += first[2] * a[2] * b[2] * cost
    assert compiler.board_score(0, 0).exact == pytest.approx(expected)


def test_same_second_tails_use_scheduled_rank_and_retain_deadline_and_cancellation():
    first = two_event((0,), (1.,), (10,))
    canceled_a = finite_law([[(15, False), (15, True)], [(25, False), (25, True)]],
                           [.75, .25], [np.eye(2)])
    graph, compiler, _ = fixture(first, {'a': canceled_a, 'b': two_event((15,), (1.,), (150,))})
    # Scheduled durations tie, so a wins if it operates. b's arrival is clipped.
    assert compiler.board_score(0, 0).exact == pytest.approx(43.75)
    assert compiler.tail_value(0, 16) == 100


def test_internal_service_failure_does_not_resurrect_on_a_later_operating_segment():
    graph = TransitGraph()
    for stop in range(5):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for cid, trip, dep, arr, t, u in ((0, 'first', 0, 1, 0, 10),
                                      (1, 'first', 1, 2, 11, 20),
                                      (2, 'first', 2, 3, 21, 30), (3, 'tail', 3, 4, 40, 50)):
        graph.add_connection(Connection(cid, 'R', trip, dep, arr, t, u))
    first = finite_law([[(0, False)], [(10, False)], [(11, False), (11, True)],
                        [(20, False), (20, True)], [(21, False)], [(30, False)]],
                       [1.], [[[1.]], [[.6, .4]], np.eye(2), [[1.], [1.]], [[1.]]])
    laws = {'first': first, 'tail': two_event((40,), (1.,), (50,))}
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    belief._law = lambda trip, pars: laws[trip]
    compiler = TransferOpportunityScores(graph, belief, {}, 100, {0: RouteOption((0, 1, 2), {3: (3,)})})
    score = compiler.board_score(0, 0)
    assert score.success_probability == .6
    assert score.exact == score.mean_arrival == 70.


def test_nominal_control_does_not_prepare_exact_scores_and_projection_is_explicit():
    first = two_event((0, 2), (.5, .5), (10, 20))
    _, compiler, _ = fixture(first)
    assert compiler.clock_nominal_score(0, 0) == 25.
    assert not compiler._tables and not compiler.bounds._posterior_edges
    score = compiler.board_score(0, 1)
    assert score.method == 'projection'
    assert score.exact == 32.5
    assert compiler.board_score(0, -1) is None


def test_root_and_tail_trip_overlap_is_rejected_before_using_independence():
    graph, compiler, _ = fixture()
    with pytest.raises(ValueError, match='independent'):
        TransferOpportunityScores(graph, compiler.belief, {}, 100,
            {0: RouteOption((0,), {1: (1,)}), 1: RouteOption((1,), {})})
