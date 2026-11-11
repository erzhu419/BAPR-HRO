from dataclasses import replace
from fractions import Fraction

import pytest

from src.candidate_certificate import ArrivalRelaxation
from src.committed_transit import CommittedState, OBSERVE, WAIT, build_committed_model
from src.information_planner import ExactPlanner
from src.simulator import JourneyScenario, VehicleEvent
from src.sparse_certificate import SparseCertificateIndex
from src.transit_graph import Connection, Stop, TransitGraph


def make_graph(rows):
    graph = TransitGraph()
    for stop in {s for row in rows for s in row[:2]}:
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=2))
    for cid, (source, target, departure, arrival, trip) in enumerate(rows):
        graph.add_connection(Connection(cid, trip, trip, source, target, departure, arrival))
    return graph


def scenario(graph, changes=None, publications=None):
    events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections}
    events.update(changes or {})
    return JourneyScenario(events, publications=publications)


@pytest.mark.parametrize('publications', [None, {}, {'0:cancel': 5, '1:departure': 5}])
@pytest.mark.parametrize('deadline', [9, 25])
def test_scalar_first_projection_matches_committed_transition(publications, deadline):
    graph = make_graph([(0, 2, 5, 8, 'A'), (0, 2, 0, 2, 'B'),
                        (0, 2, 20, 22, 'C'), (0, 2, 6, 12, 'D')])
    scene = scenario(graph, {0: VehicleEvent(999, True, 5, 8)}, publications)
    bank = [(1, scene)]
    state = CommittedState(0, 3, 'previous', 1, initialized=True)
    model = build_committed_model(graph, bank, 0, 2, 0, deadline, initial_state=state)
    index = SparseCertificateIndex(graph, bank, 2, 0, deadline)
    for action in model.actions(state):
        actual = model.transition(model.scenarios[0], state, action).state
        assert index.first_projection(0, state, action) == (actual.stop, actual.time)


def test_platform_transfer_and_same_trip_are_resolved_before_the_relaxation():
    graph = make_graph([(0, 2, 2, 9, 'A'), (0, 2, 4, 5, 'B')])
    bank = [(1, scenario(graph))]
    index = SparseCertificateIndex(graph, bank, 2, 0, 10)
    transfer = CommittedState(0, 1, 'incoming', 1, initialized=True)
    through = replace(transfer, current_trip='A')
    assert index.first_projection(0, transfer, ('select', 0)) == (0, 3)
    assert index.first_projection(0, through, ('select', 0)) == (2, 9)
    assert index.action_lower(transfer, [('select', 0)])['select', 0] == 4
    assert index.action_lower(through, [('select', 0)])['select', 0] == 8


def test_zero_time_cycles_do_not_require_a_stop_topological_order():
    graph = make_graph([(0, 1, 0, 0, 'A'), (1, 0, 0, 0, 'A'),
                        (1, 2, 0, 0, 'A'), (0, 2, 3, 4, 'B')])
    bank = [(1, scenario(graph, publications={}))]
    index = SparseCertificateIndex(graph, bank, 2, 0, 8)
    assert index.arrival(0, 0, 0) == 0
    assert index.arrival(0, 0, 1) == 4
    assert index.stats['maximum_search_labels'] <= 3


def test_sparse_arrivals_match_dense_reference_without_allocating_clock_labels():
    graph = make_graph([(0, 1, 1, 3, 'A'), (1, 2, 4, 6, 'B'), (0, 2, 3, 8, 'C')])
    bank = [(.4, scenario(graph, publications={})),
            (.6, scenario(graph, {1: VehicleEvent(999, True, 4, 6)}, {}))]
    sparse = SparseCertificateIndex(graph, bank, 2, 0, 10)
    dense = ArrivalRelaxation(graph, bank, 2, 0, 10)
    for world in range(2):
        for stop in graph.stops:
            for time in range(10):
                assert sparse.arrival(world, stop, time) == dense.arrival(world, stop, time)
    assert sparse.stats['ea_cache_hits'] > 0


def test_all_action_bounds_and_all_future_screen_preserve_same_information_optimum():
    graph = make_graph([(0, 2, 1, 4, 'A'), (0, 2, 1, 8, 'B')])
    bank = [(.5, scenario(graph, publications={})),
            (.5, scenario(graph, {0: VehicleEvent(1, False, 2, 5)}, {}))]
    state = CommittedState(0, 0, None, 0, initialized=True)
    model = build_committed_model(graph, bank, 0, 2, 0, 10, initial_state=state)
    index = SparseCertificateIndex(graph, bank, 2, 0, 10)
    planner = ExactPlanner(model)
    optimum = planner.solve()
    lower = index.action_lower(state, model.actions(state))
    assert all(float(lower[action]) <= value for action, value in optimum.action_values.items())
    assert lower['select', 0] == Fraction(9, 2)
    assert index.action_lower(state, [('select', 0)], support=[1])['select', 0] == 5

    def witness_upper(node):
        total = Fraction(0)
        for world, probability in zip(node.support, node.belief):
            current, elapsed = node.state, 0
            for _ in range(model.horizon):
                actions = model.actions(current)
                if not actions:
                    break
                action = next((a for a in actions if a == ('select', 0)), actions[0])
                outcome = model.transition(model.scenarios[world], current, action)
                elapsed += outcome.cost
                current = outcome.state
                if outcome.terminal:
                    break
            total += Fraction(probability) * (elapsed + model.terminal_cost(model.scenarios[world], current))
        return total

    removed = []
    def menu(node):
        actions = model.actions(node.state)
        bounds = index.action_lower(node.state, actions, support=node.support)
        upper = witness_upper(node)
        kept = tuple(action for action in actions if bounds[action] <= upper)
        removed.extend(set(actions) - set(kept))
        return kept
    assert ExactPlanner(model, menu).solve().value == optimum.value == 4.5
    assert ('select', 1) in removed


def test_interval_cache_reuses_the_same_first_departure_and_counts_work():
    graph = make_graph([(0, 2, 5, 8, 'A')])
    index = SparseCertificateIndex(graph, [(1, scenario(graph))], 2, 0, 10)
    assert index.arrival(0, 0, 0) == 8
    scans = index.stats['connection_scans']
    assert index.arrival(0, 0, 5) == 8
    assert index.stats['connection_scans'] == scans
    assert index.arrival(0, 0, 6) == 10
