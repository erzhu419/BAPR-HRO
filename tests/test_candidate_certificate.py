from fractions import Fraction

import pytest

from src.candidate_certificate import ArrivalRelaxation, RepairWork, certify
from src.committed_transit import CommittedState, WAIT
from src.information_planner import ExactPlanner
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def timetable(rows):
    graph = TransitGraph()
    for stop in sorted({stop for row in rows for stop in row[:2]}):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=2))
    for cid, (u, v, dep, arr) in enumerate(rows):
        graph.add_connection(Connection(cid, str(cid), str(cid), u, v, dep, arr))
    return graph


def scene(graph, edits=None):
    events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections}
    events.update(edits or {})
    return JourneyScenario(events)


def prefer(*boardings):
    def policy(_, actions):
        return next((a for a in boardings if a in actions), actions[0])
    return policy


def test_action_certificate_preserves_optimum_on_all_future_histories():
    graph = timetable([(0, 1, 1, 4), (0, 1, 1, 8)])
    bank = [(0.75, scene(graph)), (0.25, scene(graph, {0: VehicleEvent(2, False, 3, 6)}))]
    initial = CommittedState(0, 0, None, 0, initialized=True)
    relaxation = ArrivalRelaxation(graph, bank, 1, 0, 9)
    policy = prefer(("select", 0))
    result = certify(relaxation, initial, policy)
    assert result.lower == result.upper == Fraction(9, 2)
    assert result.removed == (("select", 1),)
    assert result.transitions == 6

    model = relaxation.committed_model(initial)
    full = ExactPlanner(model)
    decisions = {}

    def menu(info):
        certificate = certify(relaxation, info.state, policy, support=info.support)
        decisions[info.state.time] = certificate.removed
        return tuple(a for a in model.actions(info.state) if a not in certificate.removed)

    screened = ExactPlanner(model, menu)
    assert screened.solve().value == full.solve().value == float(result.upper)
    assert any(time > 0 for time in decisions)


def test_optimistic_transfer_and_clairvoyance_leave_an_explicit_gap():
    graph = timetable([(0, 1, 1, 2), (1, 2, 2, 3), (1, 2, 5, 7), (0, 2, 2, 6)])
    canceled = VehicleEvent(999, True, 5, 7)
    bank = [(0.5, scene(graph)), (0.5, scene(graph, {2: canceled}))]
    initial = CommittedState(0, 0, None, 0, initialized=True)
    relaxation = ArrivalRelaxation(graph, bank, 2, 0, 8)
    policy = prefer(("select", 3))
    cert = certify(relaxation, initial, policy)
    planner = ExactPlanner(relaxation.committed_model(initial))
    assert cert.lower == 3  # free transfer from the feeder is physically optimistic
    assert cert.upper == planner.solve().value == 6
    assert cert.gap == 3
    assert ("select", 0) not in cert.removed
    assert all(cert.action_lower[a] > planner.solve().value for a in cert.removed)
    for action, value in planner.solve().action_values.items():
        assert cert.action_lower[action] <= value


def test_repair_updates_zero_time_and_parallel_dependencies_and_matches_rebuild():
    graph = timetable([(0, 1, 1, 1), (1, 2, 1, 4), (1, 2, 1, 4), (2, 3, 4, 6), (0, 3, 0, 9)])
    bank = [(0.5, scene(graph)), (0.5, scene(graph))]
    relaxation = ArrivalRelaxation(graph, bank, 3, 0, 10)
    cancel = VehicleEvent(999, True, 1, 4)
    assert relaxation.repair(0, {1: cancel}) == RepairWork(1, 1, 0)
    assert relaxation.arrival(0, 0, 0) == 6
    work = relaxation.repair(0, {2: cancel})
    assert 1 < work.changed_labels <= work.recomputed_labels < relaxation.label_count // 2
    assert relaxation.arrival(0, 0, 0) == 9
    assert relaxation.arrival(1, 0, 0) == 6

    shifted = VehicleEvent(1, False, 2, 5)
    relaxation.repair(0, {1: shifted, 3: VehicleEvent(1, False, 5, 7)})
    edits = {1: shifted, 2: cancel, 3: VehicleEvent(1, False, 5, 7)}
    rebuilt = ArrivalRelaxation(graph, [(0.5, scene(graph, edits)), bank[1]], 3, 0, 10)
    for world in range(2):
        for stop in graph.stops:
            for minute in range(11):
                assert relaxation.arrival(world, stop, minute) == rebuilt.arrival(world, stop, minute)
    state = CommittedState(0, 0, None, 0, initialized=True)
    assert relaxation.arrival(0, 0, 0) == 7
    assert certify(relaxation, state, prefer(("select", 4))) == certify(rebuilt, state, prefer(("select", 4)))


def test_edit_breaking_same_minute_order_requires_rebuild_before_application():
    graph = timetable([(0, 1, 1, 1), (1, 0, 2, 3), (0, 2, 5, 7)])
    relaxation = ArrivalRelaxation(graph, [(1., scene(graph))], 2, 0, 9)
    with pytest.raises(ValueError, match="same-minute ride"):
        relaxation.repair(0, {1: VehicleEvent(0, False, 2, 2)})
    assert relaxation.arrival(0, 1, 2) == 7


def test_certificate_preserves_seconds_clock_and_unpublished_service_status():
    graph = timetable([(0, 1, 600, 840)])
    hidden_cancel = JourneyScenario({0: VehicleEvent(999, True, 600, 840)}, minute=60, publications={})
    relaxation = ArrivalRelaxation(graph, [(1., hidden_cancel)], 1, 0, 1000)
    state = CommittedState(0, 0, None, 0, initialized=True)
    model = relaxation.committed_model(state)
    outcome = model.transition(model.scenarios[0], state, ('select', 0))
    assert outcome.state.time == 720
    assert outcome.state.feedback == (('wait_late', 0, 720),)
    assert not outcome.state.messages
    certificate = certify(relaxation, state, prefer(('select', 0)))
    assert certificate.lower == certificate.upper == 1000
