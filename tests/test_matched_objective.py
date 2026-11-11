"""Check the actual objective, independently of the posterior/scan recursion."""
import numpy as np
import pytest
from src.matched_router import MatchedRouter
from src.pmf import PMF
from src.transit_graph import TransitGraph, Stop, Connection, StopLabel


def test_failure_mass_is_charged_once_and_can_change_the_best_option():
    graph = TransitGraph()
    for sid in range(2):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=5))
    for cid, arrival, success in [(0, 10, .2), (1, 30, 1.)]:
        graph.add_connection(Connection(cid, str(cid), str(cid), 0, 1, 1, arrival,
                                        PMF(np.array([success]), 1), PMF.deterministic(arrival)))
    router = MatchedRouter(graph, {})
    router.departure = 0
    fast = StopLabel(0, PMF.deterministic(10), 10, 1.)
    slow = StopLabel(1, PMF.deterministic(30), 30, 1.)
    assert router._score(fast, 0) == pytest.approx(.2 * 10 + .8 * 120)
    assert router._score(slow, 0) == 30
    assert router._score(slow, 0) < router._score(fast, 0)
    # Conditioning on already being aboard removes this trip's cancellation.
    router.current_trip = "0"
    assert router._score(fast, 1) == 10


def test_successful_arrivals_after_deadline_have_the_same_capped_loss_as_failure():
    graph = TransitGraph()
    graph.add_stop(Stop(0, "origin"))
    graph.add_stop(Stop(1, "destination"))
    graph.add_connection(Connection(0, "r", "t", 0, 1, 1, 150,
                                    PMF.deterministic(1), PMF.deterministic(150)))
    router = MatchedRouter(graph, {})
    router.departure = 0
    label = StopLabel(0, PMF.deterministic(150), 150, .5)
    assert router._score(label, 0) == 120


def test_reestimated_failure_does_not_delete_a_retained_connection():
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=2))
    for cid, route, dep_stop, arr_stop, dep, arr in [
            (0, "A", 0, 1, 0, 10), (1, "B", 1, 2, 12, 20)]:
        graph.add_connection(Connection(cid, route, route, dep_stop, arr_stop, dep, arr))
    priors = {r: {"mean": 0., "std": .5, "cancel_rate": .1} for r in ("A", "B")}
    retained = MatchedRouter(graph, priors, candidate_mode="retain", online_mean=True)
    retained.route(0, 2, 0)
    assert 0 in retained.initial_menu
    retained.observe_delay("A", 20)
    selected = retained.select_connection(0, 0, np.random.default_rng(1))
    assert 0 not in retained._fresh_menu
    assert selected is not None and selected[0].connection_id == 0
    assert selected[0].feasibility == 0
    assert selected[1] == 120
    retained.candidate_mode = "refresh"
    assert retained.select_connection(0, 0, np.random.default_rng(1)) is None


def test_direct_trip_catch_and_arrival_share_the_same_delay():
    graph = TransitGraph()
    graph.add_stop(Stop(0, "origin"))
    graph.add_stop(Stop(1, "destination"))
    graph.add_connection(Connection(0, "r", "t", 0, 1, 0, 10,
                                    PMF(np.array([.5] + [0.] * 9 + [.5]), 0),
                                    PMF(np.array([.5] + [0.] * 9 + [.5]), 10)))
    router = MatchedRouter(graph, {})
    router.departure, router.destination = 0, 1
    label = StopLabel(0, graph.connections[0].arr_distribution, 15, 1.)
    # The on-time half has left; only the +10 half can be caught at time 5.
    assert router._score(label, 5) == .5 * 20 + .5 * 120
