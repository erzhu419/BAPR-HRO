"""Finite shared-day mixture identities and causal branch histories."""
from itertools import product
from unittest.mock import patch

import numpy as np

from src.evolving_trip_belief import logsumexp
from src.joint_day_belief import JointDayBelief, cauchy_bin_weights
from src.transit_graph import Connection, Stop, TransitGraph


def graph():
    result = TransitGraph()
    for sid in range(4):
        result.add_stop(Stop(sid, str(sid), min_transfer_time=120))
    result.add_connection(Connection(0, 'R', 'A', 0, 1, 600, 720))
    result.add_connection(Connection(1, 'R', 'B', 2, 3, 900, 1020))
    return result


PARAMETERS = {'R': {'mean': 0., 'std': .5, 'cancel_rate': 0.}}


def test_symmetric_prior_and_cross_trip_posterior_movement():
    offsets = np.arange(-360, 361, 60)
    prior = cauchy_bin_weights(offsets, 43.52903486620926)
    np.testing.assert_allclose(prior, prior[::-1])
    assert np.isclose(prior.sum(), 1.)
    belief = JointDayBelief(graph(), public_occurrences=False, travel_sd=.1)
    original = belief.mean_events(PARAMETERS)[1].departure
    belief.observe_connection_event(0, 'arrival', 180)
    assert belief.mean_events(PARAMETERS)[1].departure > original + 80
    offsets, weights = belief.day_posterior(PARAMETERS)
    assert offsets @ weights > 80
    worlds = belief.sample(PARAMETERS, np.random.default_rng(19), 1000)
    sampled = np.mean([world.events[1].departure for world in worlds])
    assert abs(sampled - belief.mean_events(PARAMETERS)[1].departure) < 8


def test_shared_offset_samples_correlate_distinct_trips_and_preserve_clocks():
    belief = JointDayBelief(graph(), public_occurrences=False, travel_sd=.1,
                           deltas_seconds=[-180, 180], prior_weights=[.5, .5])
    worlds = belief.sample(PARAMETERS, np.random.default_rng(17), 1500)
    delays = np.array([[world.events[cid].delay for cid in (0, 1)] for world in worlds])
    assert np.corrcoef(delays.T)[0, 1] > .8
    assert all(world.events[cid].arrival >= world.events[cid].departure for world in worlds for cid in (0, 1))


def enumerate_trip(law, departure_index=0):
    total, mass = 0., {}
    for indices in product(*(range(len(times)) for times in law.times)):
        log_probability = law.initial[indices[0]] + law.emissions[0][indices[0]]
        for i in range(1, len(indices)):
            log_probability += law.transitions[i - 1][indices[i - 1], indices[i]] + law.emissions[i][indices[i]]
        probability = float(np.exp(log_probability))
        total += probability
        slot = indices[departure_index]
        key = int(law.times[departure_index][slot]), bool(law.canceled[departure_index][slot])
        mass[key] = mass.get(key, 0.) + probability
    return total, mass


def test_two_trip_likelihood_and_cancel_weighting_match_brute_force():
    parameters = {'R': {'mean': 0., 'std': 1., 'cancel_rate': .2}}
    with patch('src.evolving_trip_belief.DELAYS', np.array([-1, 0, 1])):
        belief = JointDayBelief(graph(), public_occurrences=False, deltas_seconds=[-60, 0, 60],
                               prior_weights=[.2, .5, .3], travel_sd=.5)
        belief.observe_connection_event(0, 'arrival', 60)
        belief.observe_forecast(0, 'departure', 630, 40)
        belief.observe_wait(1, 890, 910)
        offsets, weights = belief.day_posterior(parameters)
        joint, component_masses = [], []
        for index, component in enumerate(belief.components):
            a = component._law('A', belief._shifted_parameters[index])
            b = component._law('B', belief._shifted_parameters[index])
            za, _ = enumerate_trip(a)
            zb, mass = enumerate_trip(b)
            np.testing.assert_allclose(np.log(za), logsumexp(a.forward[-1]))
            joint.append(belief.prior_weights[index] * za * zb)
            component_masses.append((zb, mass))
        expected = np.asarray(joint) / sum(joint)
        np.testing.assert_allclose(weights, expected, atol=1e-12)
        expected_mass, expected_cancel = {}, 0.
        for weight, (normalizer, masses) in zip(expected, component_masses):
            for (clock, canceled), probability in masses.items():
                if canceled:
                    expected_cancel += weight * probability / normalizer
                else:
                    expected_mass[clock] = expected_mass.get(clock, 0.) + weight * probability / normalizer
        clocks, actual, cancel = belief.event_distribution(1, 'departure', parameters)
        assert np.isclose(cancel, expected_cancel)
        np.testing.assert_allclose(actual, [expected_mass[int(clock)] / (1 - expected_cancel) for clock in clocks], atol=1e-12)


def test_branch_history_is_private_and_order_repetition_replacement_agree():
    base = JointDayBelief(graph(), public_occurrences=False)
    initial = base.day_posterior(PARAMETERS)[1].copy()
    left, right = base.copy(), base.copy()
    left.observe_connection_event(0, 'arrival', 70)
    left.observe_connection_event(0, 'departure', 60)
    left.observe_forecast(1, 'departure', 960, 60)
    left.observe_forecast(1, 'departure', 1020, 60)
    right.observe_forecast(1, 'departure', 1020, 60)
    right.observe_connection_event(0, 'departure', 60)
    right.observe_connection_event(0, 'arrival', 70)
    np.testing.assert_allclose(left.day_posterior(PARAMETERS)[1], right.day_posterior(PARAMETERS)[1])
    repeated = left.day_posterior(PARAMETERS)[1].copy()
    left.observe_connection_event(0, 'arrival', 70)
    np.testing.assert_array_equal(left.day_posterior(PARAMETERS)[1], repeated)
    np.testing.assert_array_equal(base.day_posterior(PARAMETERS)[1], initial)
    np.testing.assert_allclose(left.event_distribution(1, 'departure', PARAMETERS)[1],
                               right.event_distribution(1, 'departure', PARAMETERS)[1])
    changed = {'R': dict(PARAMETERS['R'], mean=2.)}
    assert not np.allclose(left.day_posterior(changed)[1], repeated)
    np.testing.assert_array_equal(base.day_posterior(PARAMETERS)[1], initial)


def test_unobserved_trip_evidence_is_normalized_in_each_component():
    belief = JointDayBelief(graph(), public_occurrences=False)
    belief.day_posterior(PARAMETERS)
    for component, parameters in zip(belief.components, belief._shifted_parameters):
        assert abs(float(logsumexp(component._law('A', parameters).forward[-1]))) < 1e-10


def test_twelve_minute_wait_retains_finite_tiny_operating_distribution():
    network = TransitGraph()
    network.add_stop(Stop(0, '0'))
    network.add_stop(Stop(1, '1'))
    network.add_connection(Connection(0, 'R', 'T', 0, 1, 0, 1))
    belief = JointDayBelief(network, minute=1, deltas_seconds=[0], prior_weights=[1], public_occurrences=False)
    belief.observe_wait(0, -5, 7)
    parameters = {'R': {'mean': 0., 'std': .5, 'cancel_rate': .1}}
    clocks, mass, cancel = belief.event_distribution(0, 'departure', parameters)
    assert np.isfinite(mass).all()
    assert np.isclose(mass.sum(), 1.)
    assert cancel == 1.
    assert mass[(clocks >= -5) & (clocks <= 7)].sum() == 0
    assert belief.mean_events(parameters)[0].departure is None
