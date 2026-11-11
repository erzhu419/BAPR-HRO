"""Public-history conditioning for the actual finite polling protocol."""

import numpy as np
import pytest

from src.bocd.regime_detector import DelayObservation
from src.simulator import generate_delay_observations
from src.transit_graph import TransitGraph, Stop, Connection
from src.trip_belief import TripBelief, DELAYS


PARAMS = {'A': {'mean': 1., 'std': 2., 'cancel_rate': .2}}


def graph_with_connections(n=1, dep=10):
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'source'))
    graph.add_stop(Stop(1, 'destination'))
    for i in range(n):
        graph.add_connection(Connection(i, 'A', f'trip-{i}', 0, 1, dep, dep + 10))
    return graph


def as_dict(belief, cid=0, parameters=PARAMS):
    values, probabilities = belief.posterior(cid, parameters)
    return dict(zip(map(int, values), probabilities))


def test_empty_poll_conditions_on_absence_of_every_unseen_message():
    graph = graph_with_connections()
    belief = TripBelief(graph)
    prior = as_dict(belief)
    belief.observe_poll(0, 10, [], set())
    after = as_dict(belief)
    assert all(after[d] == 0 for d in DELAYS if d <= 0)
    assert after[999] == 0  # cancellation would have been announced at 10
    surviving = sum(prior[d] for d in DELAYS if d > 0)
    assert all(after[d] == pytest.approx(prior[d] / surviving) for d in DELAYS if d > 0)
    assert sum(after.values()) == pytest.approx(1)


def test_absence_uses_only_the_last_ten_minutes():
    belief = TripBelief(graph_with_connections())
    belief.observe_poll(0, 25, [], set())
    after = as_dict(belief)
    assert all(after[d] == 0 for d in DELAYS if 5 <= d <= 15)
    assert after[0] > 0  # departed before the polling window
    assert after[20] > 0  # still in the future
    assert after[999] > 0  # cancellation announcement was before the window


def test_five_message_cutoff_keeps_unreported_events_after_the_fifth_key():
    graph = graph_with_connections(n=6, dep=0)
    belief = TripBelief(graph)
    messages = [DelayObservation('A', 0, 0, 0, 0, 0, event_id=f'{i}:departure')
                for i in range(5)]
    belief.observe_poll(0, 5, messages, set())
    after = as_dict(belief, 5)
    assert after[-1] == 0  # earlier than the fifth returned event
    assert after[0] > 0  # equal time, greater connection ID: outside the top five
    assert after[1] > 0  # later event, also allowed
    assert after[999] > 0  # cancel key (0,5,cancel) is also after (0,4,departure)
    assert as_dict(belief, 0) == {0: 1.0}  # every returned event is exact


def test_already_seen_event_is_not_treated_as_missing_again():
    no_cancels = {'A': {'mean': 1., 'std': 2., 'cancel_rate': 0.}}
    belief = TripBelief(graph_with_connections())
    before = as_dict(belief, parameters=no_cancels)
    belief.observe_poll(0, 10, [], {'0:departure'})
    assert as_dict(belief, parameters=no_cancels) == before


def test_exact_message_conditions_the_entire_trip_and_bank_is_immutable():
    graph = graph_with_connections()
    graph.add_connection(Connection(1, 'A', 'trip-0', 1, 0, 20, 30))
    belief = TripBelief(graph)
    belief.observe_vehicle_event(0, False, 3)
    assert as_dict(belief, 1) == {3: 1.0}
    scenes = belief.sample(PARAMS, np.random.default_rng(0), 3)
    for scene in scenes:
        assert scene.event(graph.connections[0]).departure == 13
        assert scene.event(graph.connections[1]).departure == 23
        with pytest.raises(TypeError):
            scene.events[0] = scene.events[1]
    nominal = belief.mean_events(PARAMS)
    assert nominal[0].mean_delay == nominal[1].mean_delay == 3
    assert nominal[1].cancel_probability == 0


def test_timeout_keeps_canceled_and_late_worlds_until_announcement():
    graph = graph_with_connections(dep=20)
    belief = TripBelief(graph)
    before = as_dict(belief)
    belief.observe_attempt(0, 'timeout', 12)
    assert as_dict(belief) == before
    # Once the scheduled cancellation announcement would have happened,
    # another timeout rules cancellation out and retains only late vehicles.
    belief.observe_attempt(0, 'timeout', 22)
    after = as_dict(belief)
    assert after[999] == 0
    assert all(after[d] == 0 for d in DELAYS if d <= 2)
    assert after[3] > 0


def test_missed_attempt_is_an_operating_departure_strictly_before_the_time():
    belief = TripBelief(graph_with_connections())
    belief.observe_attempt(0, 'missed', 12)
    probabilities = as_dict(belief)
    assert probabilities[999] == 0
    assert probabilities[1] > 0
    assert all(probabilities[d] == 0 for d in DELAYS if d >= 2)


def test_conditioning_rare_gaussian_tail_does_not_underflow():
    graph = graph_with_connections(dep=0)
    belief = TripBelief(graph)
    belief.observe_attempt(0, 'timeout', 63)
    narrow = {'A': {'mean': 1., 'std': .5, 'cancel_rate': .2}}
    probabilities = as_dict(belief, parameters=narrow)
    assert probabilities[64] == 1
    assert sum(probabilities.values()) == 1
    assert belief.mean_events(narrow)[0].mean_delay == 64
    assert all(scene.event(graph.connections[0]).delay == 64
               for scene in belief.sample(narrow, np.random.default_rng(1), 5))


def test_explicit_cancel_has_no_conditional_operating_mean():
    graph = graph_with_connections()
    belief = TripBelief(graph)
    belief.observe_vehicle_event(0, True, 0)
    nominal = belief.mean_events(PARAMS)[0]
    assert nominal.mean_delay is nominal.departure is nominal.arrival is None
    assert nominal.cancel_probability == 1
    assert all(scene.event(graph.connections[0]).canceled
               for scene in belief.sample(PARAMS, np.random.default_rng(0), 3))


def test_poll_conditioned_samples_obey_the_same_message_protocol():
    graph = graph_with_connections(n=6, dep=0)
    belief = TripBelief(graph)
    messages = [DelayObservation('A', 0, 0, 0, 0, 0, event_id=f'{i}:departure')
                for i in range(5)]
    belief.observe_poll(0, 5, messages, set())
    for scene in belief.sample(PARAMS, np.random.default_rng(2), 20):
        polled = generate_delay_observations(graph, 0, 5, 'unused', np.random.default_rng(0),
                                              scenario=scene, seen_events=set())
        assert [(m.event_id, m.actual_time) for m in polled] == [
            (m.event_id, m.actual_time) for m in messages]


def test_rollout_history_copy_does_not_change_the_real_history():
    graph = graph_with_connections(n=2)
    belief = TripBelief(graph)
    belief.observe_poll(0, 10, [], set())
    before = as_dict(belief)
    branch = belief.copy()
    another_candidate = belief.copy()
    branch.observe_vehicle_event(0, False, 3)
    branch.observe_attempt(1, 'timeout', 20)
    assert as_dict(belief) == before
    assert as_dict(another_candidate) == before
    assert as_dict(belief, 1)[5] > 0
    assert as_dict(another_candidate, 1)[5] > 0
    assert as_dict(branch, 1)[5] == 0
    assert branch.graph is belief.graph


def test_nominal_cache_tracks_known_poll_and_timeout_history_per_trip():
    graph = graph_with_connections(n=2)
    graph.connections[1].dep_time = 20
    graph.connections[1].arr_time = 30
    belief = TripBelief(graph)
    warm = belief.mean_events(PARAMS)
    assert list(warm) == [0, 1]
    assert all(belief.mean_events(PARAMS)[cid] is event for cid, event in warm.items())

    belief.observe_vehicle_event(0, False, 3)
    known = belief.mean_events(PARAMS)
    assert known[0].mean_delay == 3
    assert known[0] is not warm[0]
    assert known[1] is warm[1]
    belief.observe_vehicle_event(0, False, 3)  # repeated exact fact preserves cache
    assert belief.mean_events(PARAMS)[0] is known[0]

    belief.observe_poll(0, 20, [], set())
    polled = belief.mean_events(PARAMS)
    assert polled[1].mean_delay > warm[1].mean_delay
    assert polled[1].cancel_probability == 0
    assert polled[0] is known[0]
    belief.observe_attempt(1, 'timeout', 32)
    timed_out = belief.mean_events(PARAMS)
    assert timed_out[1].mean_delay >= 13
    assert timed_out[1] is not polled[1]
    assert timed_out[0] is known[0]
    # A fresh cache with the same public history gives bit-identical values.
    fresh = TripBelief(graph)
    fresh.observe_vehicle_event(0, False, 3)
    fresh.observe_poll(0, 20, [], set())
    fresh.observe_attempt(1, 'timeout', 32)
    assert timed_out == fresh.mean_events(PARAMS)


def test_nominal_cache_uses_current_route_parameters():
    graph = graph_with_connections()
    belief = TripBelief(graph)
    warm = belief.mean_events(PARAMS)[0]
    changed = {'A': {'mean': 6., 'std': 4., 'cancel_rate': .4}}
    updated = belief.mean_events(changed)[0]
    assert updated.mean_delay != warm.mean_delay
    assert updated.cancel_probability == pytest.approx(.4)
    assert updated == TripBelief(graph).mean_events(changed)[0]
    assert belief.mean_events(changed)[0] is updated
    assert belief.mean_events(PARAMS)[0] == warm


def test_nominal_cache_copy_reuses_immutable_values_without_history_leakage():
    belief = TripBelief(graph_with_connections(n=2))
    warm = belief.mean_events(PARAMS)
    candidate = belief.copy()
    other_candidate = belief.copy()
    assert candidate.mean_events(PARAMS)[0] is warm[0]
    candidate.observe_vehicle_event(0, False, 3)
    candidate.observe_attempt(1, 'timeout', 20)
    branch_events = candidate.mean_events(PARAMS)
    assert branch_events[0].mean_delay == 3
    assert branch_events[1].mean_delay >= 11
    assert all(belief.mean_events(PARAMS)[cid] is event for cid, event in warm.items())
    assert all(other_candidate.mean_events(PARAMS)[cid] is event for cid, event in warm.items())
