from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v32.journey import TailOutcomes, TripWorlds
from experiments.reconstruction_v32.run import actual_policy, first_chain, sampled_policy
from experiments.reconstruction_v32.tail_values import integrate
from src.evolving_trip_belief import EvolvingTripBelief, _ChainLaw
from src.full_menu_opportunity import FullMenuOpportunity
from src.onboard_stopping import OnboardStopping
from src.transit_graph import Connection, Stop, TransitGraph


def finite_law(states, mass, transitions):
    with np.errstate(divide='ignore'):
        initial = np.log(np.asarray(mass))
        edges = [np.log(np.asarray(k)) for k in transitions]
    times = [np.array([t for t, _ in row]) for row in states]
    return _ChainLaw(times, initial, edges.__getitem__,
                     [np.zeros(len(t)) for t in times],
                     [np.array([c for _, c in row]) for row in states],
                     (True,) * len(edges))


def graph_and_belief():
    graph = TransitGraph()
    for i in range(3):
        graph.add_stop(Stop(i, str(i), min_transfer_time=5))
    graph.add_connection(Connection(0, 'R', 'a', 0, 1, 10, 20))
    graph.add_connection(Connection(1, 'R', 'a', 1, 2, 21, 30))
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    return graph, belief


def world_events(events):
    worlds = TripWorlds.__new__(TripWorlds)
    worlds.count = len(next(iter(events.values()))[0])
    worlds.events = {key: (np.asarray(t), np.asarray(c, bool)) for key, (t, c) in events.items()}
    worlds._paths = {}
    return worlds


def test_a_shared_vehicle_is_drawn_once_for_all_candidate_occurrences():
    graph, belief = graph_and_belief()
    law = finite_law([[(10, False), (15, False)], [(20, False), (25, False)],
                      [(21, False), (26, False)], [(30, False), (35, False)]],
                     [.5, .5], [np.eye(2)] * 3)
    calls = []
    belief._law = lambda trip, parameters: calls.append(trip) or law
    worlds = TripWorlds(belief, {}, [(0, 1), (1,), (0, 1)], 32, 42)
    assert calls == ['a']
    assert len(np.unique(worlds.events[0, 'departure'][0])) == 2
    assert np.array_equal(worlds.events[1, 'departure'][0]-worlds.events[0, 'departure'][0],
                          np.full(32, 11))
    assert np.array_equal(worlds.path((0, 1), 100), worlds.path((1,), 100))


def test_first_chain_initial_mass_and_internal_service_killing_are_not_conditioned_away():
    graph, belief = graph_and_belief()
    law = finite_law([[(0, False), (10, False), (10, True)],
                      [(20, False), (20, True)], [(21, False), (21, True)], [(30, False)]],
                     [.25, .5, .25], [[[.5, .5], [1., 0.], [0., 1.]],
                                      [[.6, .4], [0., 1.]], [[1.], [1.]]])
    belief._law = lambda trip, parameters: law
    root = dict(first_cid=0, start=5, deadline=100,
                hubs=[dict(arrival_cid=0), dict(arrival_cid=1)])
    times, kernels, initial = first_chain(graph, belief, {}, root)
    assert initial == pytest.approx([.5])
    assert kernels[0] == pytest.approx(np.array([[.6]]))
    chain = OnboardStopping(times, ([100.], [40.]), kernels, 100.)
    assert chain.solve(initial).policies['optimal'].root_value == pytest.approx(82.)


def test_sampled_execution_retains_source_failure_and_cannot_resurrect_after_canceled_leg():
    worlds = world_events({
        (0, 'departure'): ([0, 10, 10, 10], [0, 0, 0, 1]),
        (0, 'arrival'): ([20] * 4, [0] * 4),
        (1, 'departure'): ([21] * 4, [0, 0, 1, 0]),
        (1, 'arrival'): ([30] * 4, [0] * 4)})
    root = dict(first_cid=0, start=5, deadline=100,
                hubs=[dict(prefix=[0]), dict(prefix=[0, 1])])
    times = (np.array([20]), np.array([30]))
    policy = OnboardStopping(times, ([100.], [40.]), ([[1.]],), 100.).solve().policies['optimal']
    losses, chosen = sampled_policy(root, worlds, times, policy,
                                    [np.full((1, 4), 100.), np.full((1, 4), 40.)])
    assert losses == pytest.approx([95., 35., 95., 95.])
    assert chosen.tolist() == [-1, 1, -1, -1]


class ToyMenu:
    deadline = 100
    cids = (0, 1)
    ranks = {0: (20, 0), 1: (30, 1)}

    def __init__(self):
        self.graph = SimpleNamespace(stops={0: SimpleNamespace(min_transfer_time=5)})
        self.hub = dict(options=[dict(first_path=[0], tail_paths=[]),
                                 dict(first_path=[1], tail_paths=[])])

    def departure_distribution(self, cid):
        return ((np.array([1, 6, 10]), np.ones(3)/3, 0.) if cid == 0
                else (np.array([10, 15]), np.array([.5, .5]), 0.))

    def best(self, cid, clock):
        return (20. if cid == 0 else 30.), cid

    def clock_nominal_score(self, cid, clock):
        return self.best(cid, clock)[0]


def test_tail_world_execution_respects_mct_and_observes_pre_ready_but_not_pre_entry_offers():
    menu = ToyMenu()
    reference = FullMenuOpportunity(menu, menu.ranks, entry=5, ready=10, deadline=100)
    _, tables = integrate(reference)
    worlds = world_events({
        (0, 'departure'): ([6, 1, 10], [0] * 3),
        (0, 'arrival'): ([20] * 3, [0] * 3),
        (1, 'departure'): ([10, 10, 15], [0] * 3),
        (1, 'arrival'): ([30] * 3, [0] * 3)})
    results = TailOutcomes(menu, worlds).evaluate(tables, 5, 10, ['first'])
    assert results['first'] == pytest.approx([30., 30., 20.])
    # At B's clock 10, A at time 6 has been observed but A at time 1 has not.
    j = int(np.searchsorted(tables['times'], 10))
    assert tables['choices']['first'][j, 0, 2] == 1
    assert tables['choices']['first'][j, 1, 2] == 1


def test_tail_policy_table_cannot_choose_using_the_sampled_future_arrival():
    menu = ToyMenu()
    _, tables = integrate(FullMenuOpportunity(menu, menu.ranks, 5, 10, 100))
    worlds = world_events({
        (0, 'departure'): ([10, 10], [0, 0]),
        (0, 'arrival'): ([20, 90], [0, 0]),
        (1, 'departure'): ([15, 15], [0, 0]),
        (1, 'arrival'): ([30, 30], [0, 0])})
    outcomes = TailOutcomes(menu, worlds).evaluate(tables, 5, 10, ['pair', 'exact'])
    # Both worlds have the same public departure prefix. The slow hidden
    # arrival is charged after selecting A; the controller cannot switch to B.
    assert outcomes['pair'] == pytest.approx([20., 90.])
    assert outcomes['exact'] == pytest.approx([20., 90.])


def test_final_platform_mct_and_same_second_tie_use_the_common_frozen_rule():
    graph = SimpleNamespace(stops={1: SimpleNamespace(min_transfer_time=5)}, connections=[
        SimpleNamespace(dep_time=0, arr_time=10),
        SimpleNamespace(dep_time=14, arr_time=19),
        SimpleNamespace(dep_time=15, arr_time=25)])
    worlds = world_events({
        (0, 'departure'): ([0, 0], [0, 0]), (0, 'arrival'): ([10, 10], [0, 0]),
        (1, 'departure'): ([14, 15], [0, 0]), (1, 'arrival'): ([19, 20], [0, 0]),
        (2, 'departure'): ([15, 15], [0, 0]), (2, 'arrival'): ([25, 25], [0, 0])})
    row = dict(first_path=[0], transfer_stop=1, tail_paths=[[1], [2]])
    assert worlds.option(graph, row, 100) == pytest.approx([25., 20.])


def test_actual_onboard_action_never_reads_an_unselected_future_prefix(monkeypatch):
    def endpoint(t):
        return dict(status='REAL', time_seconds=t, official_journey='A')
    metadata = {0: dict(departure=endpoint(0), arrival=endpoint(10)),
                1: dict(departure=endpoint(11), arrival=endpoint(20))}
    root = dict(first_cid=0, start=0, deadline=100, hubs=[
        dict(prefix=[0], stop=1, arrival_cid=0),
        dict(prefix=[0, 1], stop=2, arrival_cid=1)])
    times = (np.array([10]), np.array([20, 30]))
    chain = OnboardStopping(times, ([20.], [60., 70.]), ([[.5, .5]],), 100.)
    policy = chain.solve().policies['optimal']
    monkeypatch.setattr('experiments.reconstruction_v32.run.actual_tail',
                        lambda menu, metadata, entry, method: (20, 'arrived', []))
    before = actual_policy(None, metadata, root, [None, None], times, policy, 'pair')
    changed = deepcopy(metadata)
    changed[1]['arrival'] = dict(status='MISSING')
    after = actual_policy(None, changed, root, [None, None], times, policy, 'pair')
    assert before['loss'] == after['loss'] == 20
    assert before['trace'] == after['trace']
    assert len(before['trace']) == 1
