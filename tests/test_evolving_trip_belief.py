"""Measured event clocks, partial service, and causal second-clock replay."""
import numpy as np
import pytest

from src.committed_transit import OBSERVE, build_committed_model
from src.evolving_trip_belief import EvolvingTripBelief
from src.simulator import JourneyScenario, VehicleEvent, RegimeSchedule, generate_delay_observations
from src.simulate_bandit import simulate_bandit_journey
from src.causal_planning_router import CausalPlanningRouter
from src.transit_graph import TransitGraph, Stop, Connection


PARAMETERS = {'A': {'mean': 1., 'std': 2., 'cancel_rate': .1}}


def graph():
    network = TransitGraph()
    for sid in range(4):
        network.add_stop(Stop(sid, str(sid), min_transfer_time=120))
    for cid in range(3):
        network.add_connection(Connection(cid, 'A', 'trip', cid, cid + 1,
                                          600 + cid * 300, 840 + cid * 300))
    return network


def test_distinct_exact_departure_arrival_and_partial_service_are_supported():
    network = graph()
    belief = EvolvingTripBelief(network, public_occurrences=False)
    belief.observe_connection_event(0, 'departure', 37)
    belief.observe_connection_event(0, 'arrival', 102)
    belief.observe_connection_event(1, 'cancel', 0)
    belief.observe_connection_event(2, 'departure', 23)
    belief.observe_connection_event(2, 'arrival', 65)
    for scene in belief.sample(PARAMETERS, np.random.default_rng(3), 12):
        assert scene.events[0].departure == 637
        assert scene.events[0].arrival == 942
        assert scene.events[1].canceled
        assert scene.events[2].departure == 1223
        assert scene.events[2].arrival == 1505
    assert belief.mean_events(PARAMETERS)[1].cancel_probability == 1
    assert belief.mean_events(PARAMETERS)[2].arrival == pytest.approx(1505)


def test_all_sampled_trip_clocks_are_ordered_with_variable_running_delays():
    belief = EvolvingTripBelief(graph())
    scenes = belief.sample(PARAMETERS, np.random.default_rng(4), 30)
    for scene in scenes:
        clocks = [t for event in scene.events.values() if not event.canceled
                  for t in (event.departure, event.arrival)]
        assert clocks == sorted(clocks)
    assert any(e.arrival - e.departure != 240 for scene in scenes for e in scene.events.values())


def test_empty_poll_without_occurrence_feed_has_no_negative_information():
    belief = EvolvingTripBelief(graph(), public_occurrences=False)
    before = belief.event_distribution(0, 'departure', PARAMETERS)
    belief.observe_poll(0, 600, [], set())
    after = belief.event_distribution(0, 'departure', PARAMETERS)
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2] == after[2]


def test_wait_without_occurrence_feed_retains_past_future_and_canceled():
    belief = EvolvingTripBelief(graph(), public_occurrences=False)
    belief.observe_wait(0, 600, 720)
    clocks, weights, cancel = belief.event_distribution(0, 'departure', PARAMETERS)
    assert 0 < cancel < 1
    assert weights[clocks < 600].sum() > 0
    assert weights[clocks > 720].sum() > 0
    assert weights[(600 <= clocks) & (clocks <= 720)].sum() == 0


def test_latest_forecast_replaces_previous_likelihood_and_copy_is_private():
    belief = EvolvingTripBelief(graph(), public_occurrences=False)
    branch = belief.copy()
    branch.observe_forecast(0, 'departure', 720, 60)
    branch.observe_forecast(0, 'departure', 780, 60)
    direct = belief.copy()
    direct.observe_forecast(0, 'departure', 780, 60)
    np.testing.assert_allclose(branch.event_distribution(0, 'departure', PARAMETERS)[1],
                               direct.event_distribution(0, 'departure', PARAMETERS)[1])
    assert not belief._forecasts
    branch.observe_forecast(0, 'departure', None, 60)
    np.testing.assert_allclose(branch.event_distribution(0, 'departure', PARAMETERS)[1],
                               belief.event_distribution(0, 'departure', PARAMETERS)[1])


def test_actual_arrival_conditions_the_following_departure():
    belief = EvolvingTripBelief(graph())
    belief.observe_connection_event(0, 'arrival', 200)
    times, weights, _ = belief.event_distribution(1, 'departure', PARAMETERS)
    assert weights[times < 1040].sum() == 0


@pytest.mark.parametrize('event', [VehicleEvent(999, True, 600, 840),
                                  VehicleEvent(-120, False, 480, 720)])
def test_hidden_cancel_and_already_departed_have_only_wait_feedback(event):
    network = graph()
    events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in network.connections}
    events[0] = event
    scene = JourneyScenario(events, minute=60, publications={})
    model = build_committed_model(network, [(1, scene)], 0, 3, 600, 1800)
    world = model.scenarios[0]
    state = model.transition(world, model.initial_state, OBSERVE).state
    outcome = model.transition(world, state, ('select', 0))
    assert outcome.state.time == 1320
    assert outcome.state.feedback == (('wait_late', 0, 1320),)
    assert not outcome.state.messages
    assert not outcome.state.learning_log


def test_future_and_unpublished_actual_endpoints_are_not_public_receipts():
    network = graph()
    event = VehicleEvent(0, False, 600, 840)
    scene = JourneyScenario({0: event}, minute=60, publications={'0:departure': 660})
    assert not generate_delay_observations(network, 0, 620, '', None, scenario=scene)
    messages = generate_delay_observations(network, 0, 660, '', None, scenario=scene)
    assert [(m.actual_time, m.timestamp) for m in messages] == [(600, 660)]


def test_seconds_entry_point_records_distinct_arrival_and_censors_unknown():
    network = graph()
    events = {0: VehicleEvent(37, False, 637, 942), 1: VehicleEvent(102, False, 1002, 1142)}
    belief = EvolvingTripBelief(network, public_occurrences=False)
    router = CausalPlanningRouter(network, PARAMETERS, menu='full', algorithm='nominal',
                                  max_time=2400, event_belief=belief)
    result = simulate_bandit_journey(network, router, 0, 3, 600,
                                     RegimeSchedule([(0, 'unused')]), np.random.default_rng(1),
                                     max_time=2400, scenario=JourneyScenario(events, minute=60, publications={}))
    assert result.events[-1].details == 'censored'
    assert belief._known_times[0, 'departure'] == 637
    assert belief._known_times[0, 'arrival'] == 942
    assert not result.observations


def test_skipped_station_gap_uses_its_own_trained_increment():
    network = graph()
    network.connections[1].dep_stop = 0
    belief = EvolvingTripBelief(network, gap_drift=3., gap_sd=.5, dwell_drift=-1., dwell_sd=.5)
    ordinary = EvolvingTripBelief(network, gap_drift=-1., gap_sd=.5, dwell_drift=-1., dwell_sd=.5)
    for model in (belief, ordinary):
        model.observe_connection_event(0, 'arrival', 0)
    assert belief.mean_events(PARAMETERS)[1].departure > ordinary.mean_events(PARAMETERS)[1].departure + 100


def test_shared_transition_kernels_keep_distinct_trip_evidence_private():
    network = graph()
    network.add_connection(Connection(3, 'A', 'second-trip', 0, 1, 600, 840))
    belief = EvolvingTripBelief(network, public_occurrences=False)
    first = belief._law('trip', PARAMETERS)
    second = belief._law('second-trip', PARAMETERS)
    reusable_kernel_count = len(belief._transition_cache)
    assert first.transitions[0] is second.transitions[0]
    assert not first.transitions[0].flags.writeable
    branch = belief.copy()
    before = belief.mean_events(PARAMETERS)
    branch.observe_connection_event(0, 'arrival', 37)
    assert not belief._known_times
    assert branch.mean_events(PARAMETERS)[0].arrival == pytest.approx(877)
    assert len(belief._transition_cache) == reusable_kernel_count
    assert before[3] is belief.mean_events(PARAMETERS)[3]
    assert before[0].arrival == belief.mean_events(PARAMETERS)[0].arrival
    np.testing.assert_array_equal(belief.event_distribution(3, 'arrival', PARAMETERS)[1],
                                   branch.event_distribution(3, 'arrival', PARAMETERS)[1])


def test_pickup_restriction_preserves_same_vehicle_continuation():
    network = graph()
    network.connections[1].pickup_allowed = False
    events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in network.connections}
    model = build_committed_model(network, [(1., JourneyScenario(events, minute=60, publications={}))],
                                  0, 3, 600, 2400)
    world = model.scenarios[0]
    initial = model.transition(world, model.initial_state, OBSERVE).state
    aboard = model.transition(world, initial, ('select', 0)).state
    assert ('select', 1) in model.actions(aboard)
    from dataclasses import replace
    assert ('select', 1) not in model.actions(replace(aboard, current_trip=None))
    belief = EvolvingTripBelief(network, public_occurrences=False)
    router = CausalPlanningRouter(network, PARAMETERS, algorithm='nominal', max_time=1800,
                                  event_belief=belief)
    result = simulate_bandit_journey(network, router, 0, 3, 600, RegimeSchedule([(0, '')]),
                                     np.random.default_rng(1), max_time=1800,
                                     scenario=JourneyScenario(events, minute=60, publications={}))
    assert result.events[-1].details == 'arrived'


def test_log_weight_reduction_matches_scipy_with_impossible_rows_and_narrow_tails():
    from scipy.special import logsumexp as scipy_logsumexp
    from src.evolving_trip_belief import logsumexp
    values = np.array([[-np.inf, -np.inf, -np.inf], [-100000., -100001., -np.inf],
                       [-5., -3., -1.]])
    for axis in (None, 0, 1):
        for keepdims in (True, False):
            np.testing.assert_allclose(logsumexp(values, axis, keepdims),
                                       scipy_logsumexp(values, axis=axis, keepdims=keepdims),
                                       rtol=1e-13, atol=1e-13)


def test_reused_sampling_weights_preserve_the_exact_seeded_draw_sequence():
    from src.evolving_trip_belief import logsumexp
    belief = EvolvingTripBelief(graph(), public_occurrences=False)
    law = belief._law('trip', PARAMETERS)
    expected_rng, cached_rng = np.random.default_rng(83), np.random.default_rng(83)
    for _ in range(20):
        expected = np.empty(len(law.times), dtype=int)
        next_index = None
        for i in range(len(expected) - 1, -1, -1):
            weights = law.forward[i]
            if next_index is not None:
                weights = weights + law.transitions[i][:, next_index]
            probabilities = np.exp(weights - logsumexp(weights))
            next_index = expected_rng.choice(len(probabilities), p=probabilities)
            expected[i] = next_index
        np.testing.assert_array_equal(law.draw(cached_rng), expected)
