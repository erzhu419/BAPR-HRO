from collections import defaultdict
from itertools import product

import numpy as np
import pytest

from experiments.reconstruction_v30.forecast import forecast_board, forecast_target
from src.evolving_trip_belief import EvolvingTripBelief, _ChainLaw
from src.transfer_opportunity import RouteOption, TransferOpportunityScores
from src.transit_graph import Connection, Stop, TransitGraph


def law(states, weights, transition):
    with np.errstate(divide='ignore'):
        kernel, initial = np.log(transition), np.log(weights)
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
    return graph, TransferOpportunityScores(graph, belief, {}, 50,
        {cid: RouteOption((cid,), tails) for cid in (0, 1)}), laws


def atoms(model):
    for i, j in product(range(len(model.times[0])), range(len(model.times[1]))):
        p = np.exp(model.initial[i]+model._transition(0)[i, j])
        if p:
            yield (int(model.times[0][i]), int(model.times[1][j]),
                   bool(model.canceled[0][i]), bool(model.canceled[1][j])), p


def independent_enumeration(graph, laws, chosen, clock, entry=0):
    """Enumerate independent complete vehicle worlds, not score recursion."""
    names = ('current', 'target', 'a', 'b', 'c', 'd')
    expected = {key: defaultdict(float) for key in ('first_arrival', 'final_arrival', 'tail_wait', 'tail_choice')}
    normalizer, cost = 0., 0.
    for combination in product(*(tuple(atoms(laws[name])) for name in names)):
        world = {cid: row[0] for cid, row in enumerate(combination)}
        p = float(np.prod([row[1] for row in combination]))
        first = world[chosen]
        observed = not first[2] and entry <= first[0] <= clock
        if chosen == 0 and (not observed or first[0] != clock):
            continue
        if chosen == 1 and observed:
            continue
        normalizer += p
        first_arrival = final_arrival = wait = tail_id = None
        legal = chosen == 0 or clock < first[0] < 50
        if legal and not first[2] and not first[3]:
            first_arrival = first[1]
            offers = [(world[cid][0], graph.connections[cid].arr_time-graph.connections[cid].dep_time, cid)
                for cid in range(2, 6) if not world[cid][2] and first_arrival+5 <= world[cid][0] < 50]
            if offers:
                tail_id = min(offers)[2]
                wait = world[tail_id][0]-first_arrival
                if not world[tail_id][3]:
                    final_arrival = world[tail_id][1]
        for key, value in [('first_arrival', first_arrival), ('final_arrival', final_arrival),
                           ('tail_wait', wait), ('tail_choice', tail_id)]:
            expected[key][value] += p
        cost += p*(50 if final_arrival is None else min(final_arrival, 50))
    return cost/normalizer, {key: {value: p/normalizer for value, p in row.items()}
                            for key, row in expected.items()}


@pytest.mark.parametrize('chosen,clock', [(0, 0), (1, 0), (1, 10)])
def test_full_distributions_match_independent_worlds_including_mct_ties_cancel_and_unseen(chosen, clock):
    graph, compiler, laws = fixture()
    actual = (forecast_board(compiler, chosen, clock) if chosen == 0 else
              forecast_target(compiler, chosen, 0, clock, 0))
    cost, expected = independent_enumeration(graph, laws, chosen, clock)
    assert actual['status'] == 'supported'
    assert actual['cost'] == pytest.approx(cost)
    for key in ('first_arrival', 'final_arrival', 'tail_wait'):
        distribution = actual['distributions'][key]
        found = dict(zip(distribution['values'], distribution['probabilities']))
        assert found == pytest.approx({x: p for x, p in expected[key].items() if x is not None})
        assert distribution['failure_mass'] == pytest.approx(expected[key].get(None, 0.))
    assert actual['tail_choice']['no_tail'] == pytest.approx(expected['tail_choice'].get(None, 0.))
    for cid, probability in actual['tail_choice']['by_cid'].items():
        assert probability == pytest.approx(expected['tail_choice'].get(int(cid), 0.))
    if chosen == 1:
        assert actual['support']['cancellation_mass'] == pytest.approx(.2)
        assert actual['support']['before_entry_mass'] == pytest.approx(.2)
        assert actual['support']['after_deadline_mass'] == pytest.approx(.1)


def test_offgrid_projection_matches_existing_score_and_both_endpoint_distributions():
    _, compiler, _ = fixture()
    actual = forecast_board(compiler, 0, 10)
    left, right = forecast_board(compiler, 0, 0), forecast_board(compiler, 0, 20)
    assert actual['method'] == 'projection'
    assert actual['support']['endpoints'] == [0, 20]
    assert actual['cost'] == pytest.approx(compiler.board_score(0, 10).exact)
    for key in ('first_arrival', 'final_arrival', 'tail_wait'):
        expected = defaultdict(float)
        for endpoint in (left, right):
            dist = endpoint['distributions'][key]
            for value, p in zip(dist['values'], dist['probabilities']):
                expected[value] += p/2
        actual_dist = actual['distributions'][key]
        assert dict(zip(actual_dist['values'], actual_dist['probabilities'])) == pytest.approx(dict(expected))


def test_direct_target_keeps_no_offer_probability_and_physical_late_arrival_separate():
    _, compiler, _ = fixture()
    compiler.options[1] = RouteOption((1,), {})
    actual = forecast_target(compiler, 1, 0, 0, 0)
    waiting = actual['distributions']['remaining_wait']
    assert waiting['values'] == [5, 15, 35]
    assert waiting['probabilities'] == pytest.approx([.1, .2, .2])
    assert waiting['failure_mass'] == pytest.approx(.5)
    assert actual['distributions']['tail_wait'] is None
    assert actual['cost'] == pytest.approx(38.)
    assert forecast_board(compiler, 1, 70)['cost'] == 50
    assert forecast_board(compiler, 1, 70)['distributions']['final_arrival']['values'] == [75]


def test_unsupported_observations_are_not_converted_to_a_probability_forecast():
    _, compiler, _ = fixture()
    actual = forecast_board(compiler, 0, 25)
    assert actual['status'] == 'unsupported' and actual['distributions'] is None
    compiler._departures[1] = np.array([5]), np.array([1.]), 0.
    actual = forecast_target(compiler, 1, 0, 10, 0)
    assert actual['status'] == 'unsupported' and actual['reason'] == 'target_zero_survival'


def test_same_vehicle_service_interruption_does_not_reappear_as_success_later():
    graph = TransitGraph()
    for sid in range(4):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=0))
    for cid, start, end in ((0, 0, 10), (1, 11, 20), (2, 21, 30)):
        graph.add_connection(Connection(cid, 'R', 'first', cid, cid+1, start, end))
    times = [np.array(x) for x in ([0], [10], [11, 11], [20, 20], [21], [30])]
    canceled = [np.array(x, dtype=bool) for x in ([0], [0], [0, 1], [0, 1], [0], [0])]
    with np.errstate(divide='ignore'):
        transitions = [np.log(x) for x in ([[1.]], [[.6, .4]], np.eye(2), [[1.], [1.]], [[1.]])]
    model = _ChainLaw(times, np.array([0.]), transitions.__getitem__,
                      [np.zeros(len(x)) for x in times], canceled, (True,)*5)
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    belief._law = lambda trip, parameters: model
    compiler = TransferOpportunityScores(graph, belief, {}, 50,
        {0: RouteOption((0, 1, 2), {})})
    actual = forecast_board(compiler, 0, 0)
    assert actual['first_path_failure'] == pytest.approx(.4)
    assert actual['distributions']['final_arrival']['probabilities'] == pytest.approx([.6])
    assert actual['cost'] == pytest.approx(.6*30+.4*50)
    assert actual['cost'] == pytest.approx(compiler.board_score(0, 0).exact)
