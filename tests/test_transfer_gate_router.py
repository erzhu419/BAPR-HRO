from unittest.mock import patch

import numpy as np

from src.causal_planning_router import CausalPlanningRouter
from src.evolving_trip_belief import EvolvingTripBelief
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import JourneyScenario, RegimeSchedule, VehicleEvent
from src.transfer_gate_router import GateUCTRouter
from src.transit_graph import Connection, Stop, TransitGraph


def network(transfer=120):
    result = TransitGraph()
    for sid in range(4):
        result.add_stop(Stop(sid, str(sid), min_transfer_time=transfer))
    for row in [('R', 'A', 0, 1, 5, 60), ('R', 'A', 1, 2, 65, 90),
                ('R', 'A', 2, 3, 90, 110)]:
        result.add_connection(Connection(-1, *row))
    return result


def router(cls, timetable):
    belief = EvolvingTripBelief(timetable, minute=60, public_occurrences=False,
                                travel_sd=.001, dwell_sd=.001)
    belief.observe_forecast(0, 'departure', 5, 1.)
    result = cls(timetable, {'R': dict(mean=0., std=.5, cancel_rate=0.)},
                 event_belief=belief, algorithm='uct', work_budget=64, samples=4,
                 max_time=600, seed=17)
    result.online_mean = result.online_cancel = False
    result._initial_cancels = {'R': 0.}
    return result


def test_gate_compiles_immediate_suffix_and_only_later_leg_is_preissued():
    timetable = network()
    policy = router(GateUCTRouter, timetable)
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                             for c in timetable.connections}, minute=60, publications={})
    with patch('src.simulate_bandit.perf_counter', side_effect=[0., 2., 2.1, 4.1, 4.2, 6.2, 6.3]):
        result = simulate_bandit_journey(timetable, policy, 0, 3, 0,
            RegimeSchedule([(0, '')]), np.random.default_rng(0), max_time=600,
            scenario=scene, charge_computation=True)
    assert result.arrival_time == 110
    assert [p['gate_used'] for p in policy.predictions] == [False, True, True]
    assert [p['gate_cached'] for p in policy.predictions] == [False, False, True]
    assert [p['decision_ordinal'] for p in policy.predictions] == [0, 1, 2]
    assert policy.query_number == 3
    events = [event for event in result.events if event.event_type == 'compute']
    assert [event.time for event in events] == [2, 62, 90]
    assert 'preissued_stay=False' in events[1].details
    assert 'preissued_stay=True' in events[2].details
    assert policy.predictions[1]['gate_certificate_at_issue']['regret_bound'] <= 1


def test_new_certificate_first_choice_cannot_claim_prior_issue_and_failure_clears():
    timetable = network()
    policy = router(GateUCTRouter, timetable)
    policy.route(0, 3, 0)
    policy.observe_connection_event(0, 'departure', 0)
    policy.observe_connection_event(0, 'arrival', 0)
    policy.set_information_context(1, 60, 'A', 60, {0}, set(), {'A'})
    assert policy.precommitted_connection(1, 60, 'A') is None
    selected = policy.select_connection(1, 60, np.random.default_rng(0))
    assert selected[0].connection_id == 1
    assert policy.precommitted_connection(1, 60, 'A') is None
    policy.observe_connection_event(1, 'departure', 0)
    policy.observe_connection_event(1, 'arrival', 0)
    policy.observe_attempt_outcome(1, 'board', 60, 90)
    policy.set_information_context(2, 90, 'A', 90, {0, 1}, set(), {'A'})
    assert policy.precommitted_connection(2, 90, 'A') == 2
    policy.observe_attempt_outcome(2, 'wait_late', 90, 600)
    assert not policy._gate_path
    assert policy.precommitted_connection(2, 90, 'A') is None


def test_fallback_has_identical_private_query_ordinals_and_choices_without_gate():
    gate, plain = router(GateUCTRouter, network()), router(CausalPlanningRouter, network())
    for policy in (gate, plain):
        policy.route(0, 3, 0)
    for clock in (0, 60):
        for policy in (gate, plain):
            policy.set_information_context(0, clock, None, 0, set(), set(), set())
        a = gate.select_connection(0, clock, np.random.default_rng(0))
        b = plain.select_connection(0, clock, np.random.default_rng(0))
        assert (None if a is None else a[0].connection_id) == (None if b is None else b[0].connection_id)
        assert gate.predictions[-1]['work'] == plain.predictions[-1]['work']
        assert gate.predictions[-1]['value'] == plain.predictions[-1]['value']
        assert gate.query_number == plain.query_number
        assert not gate.predictions[-1]['gate_used']


def test_small_transfer_buffer_rejects_gate_and_keeps_original_uct():
    timetable = network(transfer=1)
    policy = router(GateUCTRouter, timetable)
    policy.route(0, 3, 0)
    policy.observe_connection_event(0, 'departure', 0)
    policy.observe_connection_event(0, 'arrival', 0)
    policy.set_information_context(1, 60, 'A', 60, {0}, set(), {'A'})
    policy.select_connection(1, 60, np.random.default_rng(0))
    p = policy.predictions[-1]
    assert p['gate_checked'] and not p['gate_used']
    assert p['gate_rejected_certificate']['regret_bound'] > 1
    assert p['work'] <= 64
