from src.background_policy import IssuedPolicy, execute_background
from src.platform_transit import build_platform_model
from src.transit_graph import TransitGraph, Stop, Connection
from src.simulator import JourneyScenario, VehicleEvent


def setup():
    graph = TransitGraph()
    for i in range(2):
        graph.add_stop(Stop(i, str(i)))
    for i, departure in enumerate((5, 20)):
        graph.add_connection(Connection(-1, 'R', str(i), 0, 1, departure, departure + 10))
    scene = JourneyScenario({i: VehicleEvent(0, False, c.dep_time, c.arr_time)
        for i, c in enumerate(graph.connections)}, minute=60, publications={})
    model = build_platform_model(graph, [(1., scene)], 0, 1, 0, 60, poll_interval=1)
    return graph, model


def rule(cid):
    return IssuedPolicy(lambda state: ('board', cid) if cid in state.visible_connections else ('wait',),
        applicable=lambda state: state.stop == 0 and not any(c == cid and kind == 'departure'
            and t < state.time for t, c, kind in state.public_log), name=str(cid))


def test_all_methods_can_keep_their_incumbent_during_computation():
    _, model = setup()
    background = execute_background(model, model.scenarios[0], rule(0), rule(1), 8)
    blocking = execute_background(model, model.scenarios[0], rule(0), rule(1), 8, blocking=True)
    assert background['cost'] == 15 and background['incumbent_actions_during_compute'] == 1
    assert blocking['cost'] == 30


def test_stale_plan_is_rejected_using_received_departure_not_hidden_world():
    _, model = setup()
    result = execute_background(model, model.scenarios[0], rule(1), rule(0), 8)
    assert result['rejected'] and not result['published'] and result['cost'] == 30


def test_fast_replacement_is_published_before_the_next_opportunity():
    _, model = setup()
    result = execute_background(model, model.scenarios[0], rule(1), rule(0), 2)
    assert result['published'] and result['cost'] == 15


def test_unresolved_evidence_preserves_elapsed_time_without_a_point_loss():
    _, model = setup()
    result = execute_background(model, model.scenarios[0], rule(1), rule(0), 2,
        unresolved=lambda state, policy, action: (0,) if policy.name == '0' else ())
    assert result['published'] and result['state'].time == 2
    assert result['cost'] is None and result['missing_connections'] == (0,)
