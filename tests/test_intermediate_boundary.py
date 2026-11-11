"""Complete physical counterexamples, not a restatement of the bound formula."""
from itertools import product

import pytest

from src.information_planner import ExactPlanner
from src.intermediate_boundary import FirstExitEnvelope, evaluate_causal_policy, macro_advantage_reference
from src.platform_transit import OBSERVE, WAIT, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def make_graph(rows):
    graph = TransitGraph()
    for stop in {s for row in rows for s in row[2:4]}:
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for row in rows:
        graph.add_connection(Connection(-1, *row))
    return graph


def scheduled(graph, *, publications=None):
    return JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                            for c in graph.connections}, minute=1,
                           publications={} if publications is None else publications)


def fixed_policy(model, path, graph):
    def policy(node):
        state = node.state
        actions = tuple(model.actions(state))
        if OBSERVE in actions:
            return OBSERVE
        next_cid = next((cid for cid in path if cid not in state.used_connections), None)
        if (('alight',) in actions and next_cid is not None
                and graph.connections[next_cid].trip_id != state.current_trip):
            return ('alight',)
        return ('board', next_cid) if ('board', next_cid) in actions else WAIT
    return policy


def basic(*, footpaths=()):
    graph = make_graph([('R', 'A', 0, 1, 0, 10), ('R', 'A', 1, 2, 10, 20),
                        ('B', 'B', 0, 2, 25, 30)])
    scene = scheduled(graph)
    model = build_platform_model(graph, [(1., scene)], 0, 2, 0, 40,
                                 footpaths=footpaths, poll_interval=60)
    envelope = FirstExitEnvelope(graph, 2, {0: 10, 1: 10, 2: 5},
                                  {0: 0, 1: 10, 2: 25}, footpaths=footpaths)
    return graph, scene, model, envelope


def test_exact_middle_certificate_does_not_need_the_macro_to_reach_goal():
    graph, scene, model, envelope = basic()
    witness = evaluate_causal_policy(model, fixed_policy(model, (0, 1), graph))
    certificate = envelope.certify([(1., {0: scene.events[0]})], (0,), 0, witness.value,
                                   deadline=40, tolerance=0)
    assert graph.connections[0].arr_stop != 2
    assert certificate.accepted
    assert certificate.lower == witness.value == ExactPlanner(model).solve().value == 20


def test_walk_shortcut_must_be_in_both_physical_model_and_boundary():
    graph, scene, model, envelope = basic(footpaths=((0, 2, 3),))
    witness = evaluate_causal_policy(model, fixed_policy(model, (0, 1), graph))
    full = ExactPlanner(model).solve().value
    certificate = envelope.certify([(1., {0: scene.events[0]})], (0,), 0, witness.value,
                                   deadline=40, tolerance=1)
    assert full == 3
    assert certificate.lower == 3 and not certificate.accepted
    _, _, _, missing_walk = basic()
    unsafe = missing_walk.certify([(1., {0: scene.events[0]})], (0,), 0, witness.value,
                                   deadline=40, tolerance=0)
    assert unsafe.lower > full  # The deliberately mismatched contract is false.


def test_correlated_external_notice_invalidates_less_information_value_lower():
    graph = make_graph([('R', 'A', 0, 1, 0, 10), ('B', 'B', 1, 2, 10, 11),
                        ('C', 'C', 1, 2, 10, 11), ('N', 'notice', 3, 4, 5, 6)])
    worlds = []
    for first_fast in (True, False):
        scene = scheduled(graph, publications={'3:departure': 5 if first_fast else 15})
        events = dict(scene.events)
        events[1] = VehicleEvent(0, False, 10, 11 if first_fast else 21)
        events[2] = VehicleEvent(0, False, 10, 21 if first_fast else 11)
        worlds.append((.5, JourneyScenario(events, minute=1, publications=scene.publications)))
    model = build_platform_model(graph, worlds, 0, 2, 0, 40)
    full = ExactPlanner(model).solve().value
    upper = evaluate_causal_policy(model, fixed_policy(model, (0, 1), graph)).value
    envelope = FirstExitEnvelope(graph, 2, {0: 10, 1: 1, 2: 1, 3: 1},
                                  {0: 0, 1: 10, 2: 10, 3: 5})
    result = envelope.certify([(1., {0: worlds[0][1].events[0]})], (0,), 0, upper,
                              deadline=40, tolerance=1)
    assert full == result.lower == 11
    assert upper == 16 and not result.accepted
    no_notice = [(p, JourneyScenario(scene.events, minute=1, publications={})) for p, scene in worlds]
    reduced = build_platform_model(graph, no_notice, 0, 2, 0, 40)
    assert ExactPlanner(reduced).solve().value == 16 > full


def test_service_failure_and_later_same_trip_recovery_remain_possible():
    graph = make_graph([('R', 'A', 0, 1, 0, 1), ('R', 'A', 1, 2, 3, 4),
                        ('B', 'B', 0, 1, 1, 2)])
    active = scheduled(graph)
    stopped = dict(active.events)
    stopped[0] = VehicleEvent(999, True, 0, 1)
    bank = [(.5, active), (.5, JourneyScenario(stopped, minute=1, publications={}))]
    model = build_platform_model(graph, bank, 0, 2, 0, 40)
    upper = evaluate_causal_policy(model, fixed_policy(model, (0, 1), graph)).value
    envelope = FirstExitEnvelope(graph, 2, {0: 1, 1: 1, 2: 1}, {0: 0, 1: 3, 2: 1})
    result = envelope.certify([(p, {0: s.events[0]}) for p, s in bank], (0,), 0, upper,
                              deadline=40, tolerance=1)
    assert ExactPlanner(model).solve().value == 4
    assert result.lower <= 4 <= upper and not result.accepted
    assert any(e.kind == 'service_failure' for e in envelope.inventory((0,), 0))


def test_same_trip_return_and_alighting_are_not_silently_deleted():
    graph = make_graph([('R', 'A', 0, 1, 0, 10), ('R', 'A', 1, 0, 10, 20),
                        ('R', 'A', 0, 2, 20, 21)])
    envelope = FirstExitEnvelope(graph, 2, {0: 10, 1: 10, 2: 1}, {0: 0, 1: 10, 2: 20})
    inventory = envelope.inventory((0, 1), 0)
    assert any(e.kind == 'same_trip_loop' and e.connection == 2 for e in inventory)
    assert sum(e.kind == 'alight_skip' for e in inventory) == 2
    assert inventory[-1].kind == 'decision_boundary' and inventory[-1].stop == 0
    scene = scheduled(graph)
    model = build_platform_model(graph, [(1., scene)], 0, 2, 0, 40)
    upper = evaluate_causal_policy(model, fixed_policy(model, (0, 1, 2), graph)).value
    result = envelope.certify([(1., scene.events)], (0, 1), 0, upper, deadline=40, tolerance=0)
    assert result.lower <= ExactPlanner(model).solve().value <= result.upper


def test_policy_only_upper_avoids_expanding_unchosen_actions():
    graph, _, model, _ = basic(footpaths=((0, 2, 3),))
    policy = fixed_policy(model, (0, 1), graph)
    independent = evaluate_causal_policy(model, policy)
    full = ExactPlanner(model)
    assert independent.value == full.evaluate(policy) == 20
    assert independent.transitions < full.stats['transition_evaluations']


def test_missing_support_and_inconsistent_upper_do_not_produce_a_certificate():
    graph, scene, _, envelope = basic()
    with pytest.raises(ValueError, match='every physical connection'):
        FirstExitEnvelope(graph, 2, {0: 10}, {0: 0})
    with pytest.raises(ValueError, match='inconsistent models'):
        envelope.certify([(1., scene.events)], (0,), 0, 1., deadline=40, tolerance=1)


def test_optimistic_zero_runtime_floor_can_remove_all_middle_tightness():
    graph, scene, model, _ = basic()
    broad = FirstExitEnvelope(graph, 2, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0})
    upper = evaluate_causal_policy(model, fixed_policy(model, (0, 1), graph)).value
    result = broad.certify([(1., scene.events)], (0,), 0, upper, deadline=40, tolerance=1)
    assert result.lower == 0 and result.gap == 20 and not result.accepted


def test_two_intermediate_macros_telescope_incremental_cost_on_one_common_law():
    graph = make_graph([('R', 'A', 0, 1, 0, 10), ('B', 'B', 1, 2, 10, 20),
                        ('C', 'C', 1, 2, 10, 15)])
    scene = scheduled(graph)
    model = build_platform_model(graph, [(1., scene)], 0, 2, 0, 40)
    planner = ExactPlanner(model)
    policy = fixed_policy(model, (0, 1), graph)
    first = macro_advantage_reference(planner, policy, lambda node: 0 in node.state.used_connections)
    later = sum(p * macro_advantage_reference(planner, policy, lambda node: False, child).advantage
                for p, child, terminal in first.exits if not terminal)
    actual_gap = evaluate_causal_policy(model, policy).value - planner.solve().value
    assert first.incremental_cost == 10
    assert first.boundary_optimum == 5
    assert first.advantage == 0
    assert later == actual_gap == 5
    assert first.advantage + later == actual_gap


def test_first_exit_lower_holds_in_every_coupled_physical_world():
    graph = make_graph([('R', 'A', 0, 1, 0, 4), ('R', 'A', 1, 2, 6, 9),
                        ('B', 'B', 0, 2, 0, 3)])
    envelope = FirstExitEnvelope(graph, 2, {0: 4, 1: 3, 2: 3}, {0: 0, 1: 6, 2: 0},
                                  footpaths=((1, 2, 2),))
    for delay, tail, departure, canceled in product((0, 2), (0, 3), (0, 2), (False, True)):
        events = {0: VehicleEvent(999 if canceled else 0, canceled, 0, 4 + delay),
                  1: VehicleEvent(0, False, 6, 9 + tail),
                  2: VehicleEvent(departure, False, departure, departure + 3)}
        scene = JourneyScenario(events, minute=1, publications={})
        lower, _ = envelope.local_lower(events, (0,), 0, deadline=40)
        clairvoyant = ExactPlanner(build_platform_model(graph, [(1., scene)], 0, 2, 0, 40,
            footpaths=((1, 2, 2),))).solve().value
        assert lower <= clairvoyant
