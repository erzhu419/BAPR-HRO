"""Recursive alighting and relevant-history cache tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v33.planner import Planner
from experiments.reconstruction_v33.history import ConditionalEngine
from src.evolving_trip_belief import _ChainLaw
from src.transit_graph import Connection, Stop, TransitGraph


def chain(times, transitions, initial=None, canceled=None):
    times = [np.asarray(row) for row in times]
    initial = np.ones(len(times[0]))/len(times[0]) if initial is None else np.asarray(initial)
    with np.errstate(divide='ignore'):
        matrices = [np.log(np.asarray(k, float)) for k in transitions]
        first = np.log(initial)
    return _ChainLaw(times, first, lambda i: matrices[i], [np.zeros(len(t)) for t in times],
                     [np.zeros(len(t), bool) for t in times] if canceled is None else canceled,
                     (True,)*len(matrices))


class History:
    def __init__(self, laws, revisions=None):
        self.laws, self.revisions = laws, revisions or {}

    def law(self, trip):
        return self.laws[trip]

    def signature(self, relevant):
        return tuple((trip, self.revisions.get(trip, 0)) for trip in sorted(relevant))


def example():
    graph = TransitGraph()
    for stop in range(6):
        graph.add_stop(Stop(stop, str(stop), 0., 0., 0))
    rows = [(0, 'source', 0, 1, 0, 0), (1, 'second', 1, 2, 1, 10),
            (2, 'second', 2, 3, 11, 25), (3, 'fast', 2, 5, 12, 15),
            (4, 'slow', 2, 5, 22, 60), (5, 'later', 3, 5, 30, 40)]
    for cid, trip, dep, arr, start, end in rows:
        graph.add_connection(Connection(cid, trip, trip, dep, arr, start, end))
    positions = {(0, 'departure'): ('source', 0), (0, 'arrival'): ('source', 1),
                 (1, 'departure'): ('second', 0), (1, 'arrival'): ('second', 1),
                 (2, 'departure'): ('second', 2), (2, 'arrival'): ('second', 3)}
    for cid, name in ((3, 'fast'), (4, 'slow'), (5, 'later')):
        positions[cid, 'departure'], positions[cid, 'arrival'] = (name, 0), (name, 1)
    laws = dict(source=chain([[0], [0]], [[[1]]]),
        second=chain([[1], [10, 20], [11, 21], [25, 26]], [[[.5, .5]], np.eye(2), np.eye(2)]),
        fast=chain([[12], [15]], [[[1]]]), slow=chain([[22], [60]], [[[1]]]),
        later=chain([[30], [40]], [[[1]]]))
    first_exit = Platform(2, {3: Ride(3, (3,), {3: None}), 4: Ride(4, (4,), {4: None})})
    last_exit = Platform(3, {5: Ride(5, (5,), {5: None})})
    second = Ride(1, (1, 2), {1: first_exit, 2: last_exit})
    source = Ride(0, (0,), {0: Platform(1, {1: second})})
    return graph, SimpleNamespace(_positions=positions), History(laws), source, second


def test_second_ride_keeps_both_alighting_choices_instead_of_locking_a_variant():
    graph, model, history, source, second = example()
    planner = Planner(graph, model, {}, 100)
    inner = planner.compile_ride(second, history)
    assert inner.arrival_cids == (1, 2)
    assert inner.decision(1, 10)[0] is True
    assert inner.decision(1, 20)[0] is False
    assert inner.score(1) == pytest.approx(27.5)
    # Fixing the first exit before its random arrival would cost 37.5.
    assert inner.score(1) < (15+60)/2
    assert planner.compile_ride(source, history).score(0) == pytest.approx(27.5)


@pytest.mark.parametrize('method', ['optimal', 'rolling_fixed', 'one_step_rollout', 'nominal'])
def test_all_controls_receive_the_same_recursive_action_menu(method):
    graph, model, history, source, _ = example()
    compiled = Planner(graph, model, {}, 100).compile_ride(source, history, method)
    assert compiled.score(0) == pytest.approx(27.5)
    assert 'not a joint information-tree optimum' in compiled.conditional_scope


def test_cache_reuses_unrelated_history_but_recomputes_relevant_tail_evidence():
    graph, model, history, source, _ = example()
    planner = Planner(graph, model, {}, 100)
    original = planner.compile_ride(source, history)
    unrelated = History(history.laws, {'outside_menu': 1})
    assert planner.compile_ride(source, unrelated) is original
    changed_laws = dict(history.laws)
    changed_laws['fast'] = chain([[12], [55]], [[[1]]])
    changed = History(changed_laws, {'fast': 1})
    updated = planner.compile_ride(source, changed)
    assert updated is not original
    assert updated.score(0) == pytest.approx(40.)
    assert planner.stats['ride_hits'] > 0


def test_platform_reference_uses_conditional_child_chain_board_values():
    graph, model, history, source, _ = example()
    planner = Planner(graph, model, {}, 100)
    reference = planner.platform_reference(source.exits[0], history, 0, 0, 'optimal')
    assert reference.score(1, 1) == pytest.approx(27.5)
    assert reference.choose([1], [1], 1)[0] == 1


def test_first_opportunity_respects_entry_mct_and_unobserved_past_mass():
    graph, model, history, _, second = example()
    planner = Planner(graph, model, {}, 100)
    costs = planner.first_offer_values(second.exits[1], history, np.array([10, 12, 13, 22, 23]))
    assert costs == pytest.approx([15., 15., 60., 60., 100.])


def test_distinct_structural_nodes_with_same_menu_share_compilation():
    graph, model, history, source, second = example()
    planner = Planner(graph, model, {}, 100)
    original = planner.compile_ride(second, history)
    duplicate = Ride(second.first_cid, second.path, dict(second.exits))
    assert planner.compile_ride(duplicate, history) is original


def test_warm_platform_cache_does_not_retain_a_mutable_journey_history():
    graph, model, history, source, _ = example()
    planner = Planner(graph, model, {}, 100)
    platform = source.exits[0]
    cold = planner.platform_compiler(platform, history, 'optimal')
    prior_laws = dict(history.laws)
    history.laws['second'] = chain([[8], [10, 20], [11, 21], [25, 26]],
                                   [[[.5, .5]], np.eye(2), np.eye(2)])
    history.revisions['second'] = 1
    warm = planner.platform_compiler(platform, History(prior_laws), 'optimal')
    assert warm is cold
    assert warm.departure_distribution(1)[0] == pytest.approx([1])
    assert warm.clock_nominal_score(1, 1) == pytest.approx(27.5)


def test_future_kernels_reuse_prior_after_own_arrival_but_not_before_that_evidence():
    graph, model, base, _, second = example()
    model._law = lambda trip, parameters: base.laws[trip]
    history = ConditionalEngine(model, {})
    planner = Planner(graph, model, {}, 100)
    future = planner._kernel('second', 1, 3, history)
    original = planner._edge('second', 0, history)
    history.observe(1, 'arrival', 10)
    assert planner._kernel('second', 1, 3, history) is future
    conditioned = planner._edge('second', 0, history)
    assert conditioned is not original
    assert original == pytest.approx(np.array([[.5, .5]]))
    assert conditioned == pytest.approx(np.array([[1., 0.]]))
    assert planner.compile_ride(second, history).score(1) == pytest.approx(15.)
