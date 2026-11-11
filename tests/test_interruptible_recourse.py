from dataclasses import replace
import json

import pytest

from src.background_policy import execute_background
from src.interruptible_recourse import InterruptibleRecourse
from src.platform_transit import PlatformState, build_platform_model
from src.pmf import PMF
from src.simulator import JourneyScenario, VehicleEvent
from src.tail_rescue import TailRescuePlan
from src.transit_graph import Connection, Stop, TransitGraph


def fixture_worlds(clocks, *, transfer=0, minute=1, deadline=40, publications=None,
                   primary_schedule=5, rescue_schedule=8, incoming_trip='IN', onboard=False,
                   departures=None):
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=transfer))
    for trip, origin, target, dep, arr in [(incoming_trip, 2, 0, -1, 0),
            ('A', 0, 1, primary_schedule, primary_schedule + 1),
            ('B', 0, 1, rescue_schedule, rescue_schedule + 1)]:
        graph.add_connection(Connection(-1, 'R', trip, origin, target, dep, arr))
    worlds = []
    for index, (primary, rescue) in enumerate(clocks):
        events = {0: VehicleEvent(0, False, -1, 0)}
        for cid, (dep, arr, canceled) in enumerate((primary, rescue), 1):
            events[cid] = VehicleEvent(999 if canceled else dep - graph.connections[cid].dep_time,
                                       canceled, dep, arr)
        worlds.append((1 / len(clocks), JourneyScenario(events, minute=minute,
                       publications=(publications or [{}] * len(clocks))[index])))
    root = PlatformState(0, 0, incoming_trip if onboard else None, 0, 0,
                          frozenset({0}), local_since=0)
    model = build_platform_model(graph, worlds, 0, 1, 0, deadline, poll_interval=1, initial_state=root)
    plan = TailRescuePlan((1,), 0, (2,), True,
                         tuple(departures or ((1, primary_schedule), (2, rescue_schedule))))
    return graph, model, plan, root


def replay(graph, model, plan, root, mode, *, minute=1):
    controller = InterruptibleRecourse(graph, plan, root, root.time + model.terminal_cost(model.scenarios[0], root),
                                       mode=mode, minute=minute)
    results = []
    for world in model.scenarios:
        binding = controller.bind(mode)
        result = execute_background(model, world, binding.policy, binding.policy, root.time)
        json.dumps(dict(stats=binding.stats, trace=binding.trace))
        results.append((result, binding))
    return results


def test_departure_during_transfer_triggers_rescue_before_blind_timeout():
    graph, model, plan, root = fixture_worlds([((5, 6, False), (12, 14, False))], transfer=10)
    fixed, = replay(graph, model, plan, root, 'fixed_timeout')
    early, = replay(graph, model, plan, root, 'public_failure')
    assert fixed[0]['cost'] == 40
    assert early[0]['cost'] == 14
    assert early[1].stats['public_failures'] == 1
    trigger = next(row for row in early[1].trace if row['events'])
    assert trigger['time'] == 5 and trigger['branch'] == 'rescue'
    assert trigger['ready'] == 10
    assert next(row for row in fixed[1].trace if row['events'])['time'] == 22


@pytest.mark.parametrize('primary,publication', [((5, 6, True), 'cancel'), ((-5, -4, False), 'departure')])
def test_published_failure_is_used_only_when_received_and_absence_does_not_reveal_it(primary, publication):
    clocks = [(primary, (7, 9, False))] * 2
    graph, model, plan, root = fixture_worlds(clocks, publications=[{'1:' + publication: 3}, {}])
    announced, missing = replay(graph, model, plan, root, 'public_failure')
    assert announced[0]['cost'] == 9
    assert missing[0]['cost'] == 40
    assert next(row for row in announced[1].trace if row['events'])['time'] == 3
    assert next(row for row in missing[1].trace if row['events'])['time'] == 12
    assert missing[1].trace[0]['expiry'] == 12


def test_visible_rescue_can_be_worse_than_waiting_for_the_unknown_primary():
    clocks = [((8, 9, False), (4, 20, False)), ((8, 9, True), (4, 20, False))]
    graph, model, plan, root = fixture_worlds(clocks, publications=[{}, {'1:cancel': 2}])
    fixed = replay(graph, model, plan, root, 'fixed_timeout')
    public = replay(graph, model, plan, root, 'public_failure')
    visible = replay(graph, model, plan, root, 'first_visible')
    assert [row[0]['cost'] for row in fixed] == [9, 40]
    assert [row[0]['cost'] for row in public] == [9, 20]
    assert [row[0]['cost'] for row in visible] == [20, 20]
    assert visible[0][1].stats['visible_rescues'] == 1
    assert all(a[0]['cost'] <= b[0]['cost'] for a, b in zip(public, fixed))


def test_early_failure_preserves_late_fixed_timeout_rescue_opportunity():
    graph, model, plan, root = fixture_worlds([((5, 6, True), (23, 24, False))],
        publications=[{'1:cancel': 3}], rescue_schedule=20, departures=((1, 5), (2, 0)))
    fixed, = replay(graph, model, plan, root, 'fixed_timeout')
    public, = replay(graph, model, plan, root, 'public_failure')
    assert fixed[0]['cost'] == public[0]['cost'] == 24
    first_rescue = next(row for row in public[1].trace if row['branch'] == 'rescue')
    assert first_rescue['ready'] == 3
    assert first_rescue['expiry'] == 24  # a newly reset twelve-tick window would end at15
    assert first_rescue['reference_window'] == (12, 12, 24)


def test_window_envelope_also_preserves_the_original_minute_phase_opening():
    graph, model, plan, root = fixture_worlds([((5, 6, True), (725, 730, False))],
        publications=[{'1:cancel': 619}], minute=60, deadline=1600, rescue_schedule=1420)
    fixed, = replay(graph, model, plan, root, 'fixed_timeout', minute=60)
    public, = replay(graph, model, plan, root, 'public_failure', minute=60)
    assert fixed[0]['cost'] == public[0]['cost'] == 730
    first_rescue = next(row for row in public[1].trace if row['branch'] == 'rescue')
    assert first_rescue['opening'] == 720  # independent early-phase rounding gives739


def test_waiting_root_owes_transfer_and_opening_is_frozen_not_reset_at_publication():
    graph, model, plan, root = fixture_worlds([((1000, 1001, False), (1100, 1101, False))],
        transfer=61, minute=60, deadline=2000, primary_schedule=1000)
    root = replace(root, time=3, last_arrival=3, local_since=3, initialized=True)
    plan = replace(plan, first_action_now=False)
    controller = InterruptibleRecourse(graph, plan, root, 2000, mode='fixed_timeout')
    at_publication = replace(root, time=104)
    decision = controller.decide(at_publication, controller.initial(at_publication))
    assert (decision.cursor.opening, decision.cursor.ready, decision.cursor.expiry) == (303, 303, 1023)
    early = replace(plan, departures=((1, 10), (2, 1100)))
    other = InterruptibleRecourse(graph, early, root, 2000, mode='fixed_timeout')
    decision = other.decide(root, other.initial(root))
    assert decision.cursor.ready == 64


def test_same_vehicle_is_retained_and_transfer_alighting_is_immediate():
    graph, model, plan, root = fixture_worlds([((5, 7, False), (12, 14, False))],
        transfer=100, incoming_trip='A', onboard=True)
    same, = replay(graph, model, plan, root, 'fixed_timeout')
    assert same[0]['cost'] == 7
    assert all(row['action'] != ('alight',) for row in same[1].trace)
    graph, model, plan, root = fixture_worlds([((5, 7, False), (12, 14, False))],
        transfer=10, onboard=True)
    transfer, = replay(graph, model, plan, root, 'public_failure')
    assert transfer[1].trace[0]['action'] == ('alight',)
    assert transfer[1].trace[0]['time'] == 0
    assert transfer[0]['cost'] == 14


def test_unknown_future_changes_neither_public_actions_nor_shared_controller_state():
    graph, model, plan, root = fixture_worlds([((20, 21, False), (7, 9, False)),
                                             ((30, 35, False), (7, 9, False))])
    controller = InterruptibleRecourse(graph, plan, root, 40, mode='public_failure', minute=1)
    traces = []
    for index in (0, 1, 0):
        binding = controller.bind('public')
        result = execute_background(model, model.scenarios[index], binding.policy, binding.policy, 0)
        traces.append([row for row in result['trace'] if row['time'] < 12])
        before = dict(binding.stats)
        binding.targets(result['state'])
        assert binding.stats == before
    assert traces[0] == traces[1] == traces[2]


def test_nominal_support_limits_attempt_opening_not_later_boarding_in_started_window():
    graph, model, plan, root = fixture_worlds([((8, 9, False), (20, 21, False))])
    graph.connections[1].dep_distribution = PMF([1.], 5)
    fixed, = replay(graph, model, plan, root, 'fixed_timeout')
    assert fixed[0]['cost'] == 9
    assert fixed[1].trace[0]['opening'] == 0
    assert next(row for row in fixed[1].trace if row['action'] == ('board', 1))['time'] == 8


@pytest.mark.parametrize('with_rescue', [False, True])
def test_primary_departing_before_frozen_opening_with_or_without_rescue(with_rescue):
    graph, model, plan, root = fixture_worlds([((3, 4, False), (6, 7, False))],
                                            primary_schedule=20, rescue_schedule=6)
    plan = replace(plan, first_action_now=False)
    if not with_rescue:
        plan = replace(plan, failure_index=None, rescue=())
    fixed, = replay(graph, model, plan, root, 'fixed_timeout')
    public, = replay(graph, model, plan, root, 'public_failure')
    assert fixed[0]['cost'] == 40
    assert public[0]['cost'] == (7 if with_rescue else 40)
    assert public[1].trace[0]['opening'] == 8


def test_incorrect_cancellation_is_outside_truthful_event_dominance():
    graph, model, plan, root = fixture_worlds([((8, 9, False), (4, 20, False))],
                                            publications=[{'1:cancel': 2}])
    fixed, = replay(graph, model, plan, root, 'fixed_timeout')
    public, = replay(graph, model, plan, root, 'public_failure')
    assert fixed[0]['cost'] == 9
    assert public[0]['cost'] == 20
