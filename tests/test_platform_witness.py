from dataclasses import replace

import numpy as np
import pytest

from experiments.reconstruction_v10.platform_sensitivity import adapter
from src.background_policy import execute_background
from src.committed_transit import CommittedState, build_committed_model
from src.information_planner import ExactPlanner
from src.platform_transit import PlatformState, build_platform_model
from src.platform_witness import platform_witness_conditions
from src.tail_rescue import TailRescueBounds, freeze_tail_plan
from src.temporal_scan import TemporalScan, build_temporal_kernel
from src.transit_graph import Connection
from tests.test_risk_bounds import SmallGrid, bank, graph
from tests.test_tail_rescue import run


def models(timetable, belief, parameters, path, deadline):
    state = CommittedState(0, 0, 'A', 0, frozenset({0}), initialized=True)
    bounds = TailRescueBounds(belief, parameters, timetable, 2, deadline)
    plan = freeze_tail_plan(bounds, path)
    worlds = bank(belief, parameters)
    committed = build_committed_model(timetable, worlds, 0, 2, 0, deadline, initial_state=state)
    initial = PlatformState(0, 0, 'A', 0, last_connection=0,
                            used_connections=state.unavailable_ids)
    platform = build_platform_model(timetable, worlds, 0, 2, 0, deadline,
                                    initial_state=initial, poll_interval=1)
    policy, _, _ = adapter(timetable, path, initial, 'primary')
    return bounds, state, plan, worlds, committed, platform, policy


@pytest.mark.parametrize('same_trip', [False, True])
def test_coupled_posterior_worlds_bound_platform_policy_and_optimum(tmp_path, same_trip):
    timetable = graph([('R', 'A', 3, 0, -2, 0), ('R', 'A', 0, 1, 1, 2),
                       ('R', 'A' if same_trip else 'B', 1, 2, 4, 5)], transfer=2)
    belief = SmallGrid(timetable, minute=1, public_occurrences=False)
    belief.observe_connection_event(0, 'departure', 0)
    belief.observe_connection_event(0, 'arrival', 0)
    parameters = {'R': dict(mean=0., std=.7, cancel_rate=.2)}
    bounds, state, plan, worlds, committed, platform, policy = models(
        timetable, belief, parameters, (1, 2), 8)
    assert not platform_witness_conditions(bounds, state, plan)
    upper = bounds.evaluate(state, plan).upper
    committed_losses = [run(plan, committed, world, timetable)[0] for world in committed.scenarios]
    platform_losses = [execute_background(platform, world, policy, policy, 9)['cost']
                       for world in platform.scenarios]
    assert min(committed_losses) < 8 and max(committed_losses) == 8
    assert all(p <= c for p, c in zip(platform_losses, committed_losses))
    probabilities = np.array([p for p, _ in worlds])
    assert probabilities @ committed_losses == pytest.approx(upper)

    # This is the complete finite posterior, not a sampled scenario estimate.
    optimum = ExactPlanner(platform).solve_compressed().value
    scan = TemporalScan(timetable, build_temporal_kernel(tmp_path / 'temporal.so'))
    departures = np.array([[scene.events[c.id].departure for c in timetable.connections]
                           for _, scene in worlds], dtype=np.int32)
    arrivals = np.array([[scene.events[c.id].arrival for c in timetable.connections]
                        for _, scene in worlds], dtype=np.int32)
    canceled = np.array([[scene.events[c.id].canceled for c in timetable.connections]
                        for _, scene in worlds], dtype=np.uint8)
    oracle = scan.arrivals(departures, arrivals, canceled, state, 2, 8, transfer=2)
    assert probabilities @ oracle <= optimum + 1e-9
    platform_cost = float(probabilities @ platform_losses)
    assert optimum <= platform_cost + 1e-9
    assert platform_cost <= upper + 1e-9


def test_platform_can_catch_a_target_after_the_committed_patience_window():
    timetable = graph([('R', 'A', 3, 0, -2, 0), ('R', 'A', 0, 2, 13, 14)], transfer=2)
    belief = SmallGrid(timetable, minute=1, public_occurrences=False)
    for c in timetable.connections:
        belief.observe_connection_event(c.id, 'departure', 0)
        belief.observe_connection_event(c.id, 'arrival', 0)
    parameters = {'R': dict(mean=0., std=1., cancel_rate=0.)}
    bounds, state, plan, _, committed, platform, policy = models(timetable, belief, parameters, (1,), 20)
    assert not platform_witness_conditions(bounds, state, plan)
    assert bounds.evaluate(state, plan).upper == 20
    assert run(plan, committed, committed.scenarios[0], timetable)[0] == 20
    assert execute_background(platform, platform.scenarios[0], policy, policy, 21)['cost'] == 14


def test_bridge_rejects_rescue_delayed_alighting_and_same_trip_diversion():
    timetable = graph([('R', 'A', 3, 0, -2, 0), ('R', 'A', 0, 2, 2, 3)], transfer=2)
    belief = SmallGrid(timetable, minute=1, public_occurrences=False)
    bounds = TailRescueBounds(belief, {'R': dict(mean=0., std=1., cancel_rate=0.)}, timetable, 2, 20)
    state = CommittedState(0, 0, 'A', 0, frozenset({0}), initialized=True)
    plan = freeze_tail_plan(bounds, (1,))
    assert not platform_witness_conditions(bounds, state, plan)
    assert platform_witness_conditions(bounds, state, replace(plan, rescue=(1,)))
    assert platform_witness_conditions(bounds, replace(state, time=1), plan)
    assert platform_witness_conditions(bounds, state, plan, footpaths=((0, 2, 1),))
    assert platform_witness_conditions(bounds, state, plan, transfer_time=lambda state, cid: None)
    timetable.connections[1].trip_id = None
    assert any('named trip' in issue for issue in platform_witness_conditions(bounds, state, plan))
    timetable.connections[1].trip_id = 'A'
    timetable.add_connection(Connection(-1, 'R', 'A', 0, 3, 1, 2))
    assert any('inertia' in issue for issue in platform_witness_conditions(bounds, state, plan))
