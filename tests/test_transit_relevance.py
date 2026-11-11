from dataclasses import replace
import random

import pytest

from src.committed_transit import OBSERVE, WAIT, build_committed_model
from src.evolving_trip_belief import EvolvingTripBelief
from src.information_planner import ExactPlanner
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph
from src.transit_relevance import RelevanceIndex, compute_relevance, product_coupling_bound


def graph_of(rows):
    graph = TransitGraph()
    for stop in {s for row in rows for s in row[:2]}:
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for cid, (source, target, departure, arrival, trip) in enumerate(rows):
        graph.add_connection(Connection(cid, trip, trip, source, target, departure, arrival))
    return graph


def supports_from_bank(bank):
    def support(cid, kind):
        values = [getattr(scene.events[cid], kind) for _, scene in bank if not scene.events[cid].canceled]
        return (min(values), max(values)) if values else None
    return support


def restrict(model, retained):
    return lambda info: tuple(action for action in model.actions(info.state)
                              if action in (OBSERVE, WAIT) or action[1] in retained)


def independent_fixture():
    graph = graph_of([(0, 1, 0, 1, 'incoming'), (1, 2, 2, 4, 'target'),
                      (0, 2, 0, 7, 'direct'), (0, 3, 0, 1, 'irrelevant')])
    bank = []
    for target_cancel in (False, True):
        for nuisance_cancel in (False, True):
            events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections}
            events[1] = replace(events[1], canceled=target_cancel)
            events[3] = replace(events[3], canceled=nuisance_cancel)
            bank.append((.25, JourneyScenario(events, publications={})))
    return graph, bank


def test_independent_trip_elimination_preserves_all_history_optimum():
    graph, bank = independent_fixture()
    result = compute_relevance(graph, 0, 2, 0, 8, supports_from_bank(bank))
    assert result.transport_edge_ids == result.retained_connection_ids == frozenset((0, 1, 2))
    model = build_committed_model(graph, bank, 0, 2, 0, 8)
    full = ExactPlanner(model).solve()
    screened = ExactPlanner(model, restrict(model, result.retained_connection_ids)).solve()
    assert full.value == screened.value == 6
    assert result.affected_inference({'target', 'irrelevant'}) == {'target'}


def test_whole_trip_retains_remote_evidence_that_changes_the_decision():
    graph = graph_of([(3, 0, -1, 0, 'useful'), (0, 2, 2, 4, 'useful'),
                      (0, 2, 1, 6, 'direct')])
    bank = []
    for shift in (0, 1):
        events = {0: VehicleEvent(shift, False, -1 + shift, shift),
                  1: VehicleEvent(0, False, 2, 4 if shift == 0 else 8),
                  2: VehicleEvent(0, False, 1, 6)}
        bank.append((.5, JourneyScenario(events)))
    result = compute_relevance(graph, 0, 2, 1, 9, supports_from_bank(bank),
                               public_occurrences=True)
    assert 0 not in result.transport_edge_ids
    assert 0 in result.retained_connection_ids
    model = build_committed_model(graph, bank, 0, 2, 1, 9)
    assert ExactPlanner(model).solve().value == 4  # remaining: (.5*4+.5*6)-1
    assert ExactPlanner(model, restrict(model, result.retained_connection_ids)).solve().value == 4
    # Suppressing the remote segment's same-trip evidence changes the value.
    no_feed = [(probability, JourneyScenario(scene.events, publications={})) for probability, scene in bank]
    assert ExactPlanner(build_committed_model(graph, no_feed, 0, 2, 1, 9)).solve().value == 5


def test_public_extension_keeps_offpath_first_five_masking_factors():
    graph = graph_of([(0, 3, 0, 1, f'nuisance{i}') for i in range(5)] + [(0, 2, 0, 2, 'useful')])
    events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections}
    scene = JourneyScenario(events)
    private = compute_relevance(graph, 0, 2, 0, 3, supports_from_bank([(1, scene)]))
    public = compute_relevance(graph, 0, 2, 0, 3, supports_from_bank([(1, scene)]),
                               public_occurrences=True)
    assert private.required_trip_ids == {'useful'}
    assert len(public.required_trip_ids) == 6
    model = build_committed_model(graph, [(1, scene)], 0, 2, 0, 3)
    observed = model.transition(model.scenarios[0], model.initial_state, OBSERVE).state
    assert len(observed.messages) == 5
    assert all(message.event_id != '5:departure' for message in observed.messages)


def test_optimistic_labels_include_waiting_and_relax_transfer_duration():
    graph = graph_of([(0, 1, 0, 3, 'A'), (1, 2, 4, 6, 'B'), (0, 2, 8, 9, 'C')])
    bounds = {0: ((0, 2), (3, 5)), 1: ((4, 4), (6, 6)), 2: ((8, 8), (9, 9))}
    result = compute_relevance(graph, 0, 2, 1, 7, lambda cid, kind: bounds[cid][kind == 'arrival'],
                               minimum_travel=lambda cid: (3, 2, 1)[cid])
    assert result.earliest[1] == 4
    assert result.earliest[2] == 6
    assert result.latest[1] == 4
    assert result.latest[0] == 1
    assert result.transport_edge_ids == {0, 1}


def test_support_refresh_reuses_only_unchanged_trip_metadata_and_rebuilds_labels():
    graph, bank = independent_fixture()
    bounds = {c.id: ((c.dep_time, c.dep_time), (c.arr_time, c.arr_time)) for c in graph.connections}
    callback = lambda cid, kind: bounds[cid][kind == 'arrival']
    index = RelevanceIndex(graph, callback)
    assert index.solve(0, 2, 0, 8).required_trip_ids == {'incoming', 'target', 'direct'}
    bounds[1] = ((10, 10), (12, 12))
    assert index.refresh({'target'}) == 2
    repaired = index.solve(0, 2, 0, 8)
    rebuilt = compute_relevance(graph, 0, 2, 0, 8, callback)
    assert repaired.required_trip_ids == rebuilt.required_trip_ids == {'direct'}
    assert repaired.earliest == rebuilt.earliest
    assert repaired.latest == rebuilt.latest


def test_adaptive_absolute_anchor_support_is_retained_without_hmm_evaluation():
    graph = graph_of([(0, 1, 0, 1, 'same'), (1, 2, 100, 101, 'same')])
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    belief.observe_connection_event(0, 'departure', 0)
    belief.observe_connection_event(0, 'arrival', 0)
    parameters = {'same': {'mean': 0, 'std': 1, 'cancel_rate': 0}}
    assert belief.hard_support(1, 'departure', parameters)[0] < 95
    result = compute_relevance(graph, 1, 2, 1, 10,
                               lambda cid, kind: belief.hard_support(cid, kind, parameters))
    assert result.required_trip_ids == {'same'}
    assert all(memo.law is None for memo in belief._cache.values())


def test_supplied_product_error_bound_can_be_useful_or_vacuous():
    bound = product_coupling_bound(60, [.01, .02], 1)
    assert bound['world_coupling_error'] == pytest.approx(.0298)
    assert bound['policy_regret'] == pytest.approx(4.576)
    assert product_coupling_bound(60, [.5, .5])['policy_regret'] == 60


def test_every_feasible_world_path_is_inside_the_optimistic_witness_region():
    rng = random.Random(29)
    for repetition in range(12):
        graph = graph_of([(rng.randrange(3), rng.randrange(1, 4), departure,
                           departure + rng.randrange(1, 3), str(cid))
                          for cid, departure in enumerate(rng.randrange(6) for _ in range(9))])
        # Include all stops even when this random graph omits an endpoint.
        for stop in range(4):
            graph.stops.setdefault(stop, Stop(stop, str(stop)))
        bank = []
        for world in range(2):
            events = {}
            for c in graph.connections:
                shift = rng.randrange(2)
                events[c.id] = VehicleEvent(shift, rng.random() < .15,
                                            c.dep_time + shift, c.arr_time + shift)
            bank.append((.5, JourneyScenario(events, publications={})))
        result = compute_relevance(graph, 0, 3, 0, 7, supports_from_bank(bank),
                                   minimum_travel=lambda cid: 1)
        for _, scene in bank:
            successful = set()

            def enumerate_paths(stop, time, prefix):
                if stop == 3:
                    successful.update(prefix)
                    return
                for c in graph.get_connections_from(stop):
                    event = scene.events[c.id]
                    if not event.canceled and time <= event.departure and event.arrival <= 7:
                        enumerate_paths(c.arr_stop, event.arrival, prefix + (c.id,))

            enumerate_paths(0, 0, ())
            assert successful <= result.transport_edge_ids
