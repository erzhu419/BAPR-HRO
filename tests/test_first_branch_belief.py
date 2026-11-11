from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

from src.evolving_trip_belief import EvolvingTripBelief
from src.first_branch_belief import FirstBranchSampler
from src.ordered_duration_belief import OrderedDurationBelief
from src.transit_graph import TransitGraph, Stop, Connection


def network():
    graph = TransitGraph()
    for sid in range(5):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=120))
    for row in [('R', 'A', 0, 1, 600, 720), ('R', 'A', 1, 2, 720, 840),
                ('R', 'B', 3, 4, 900, 1020)]:
        graph.add_connection(Connection(-1, *row))
    return graph


def paths(law):
    for states in product(*(range(len(times)) for times in law.times)):
        weight = law.initial[states[0]] + law.emissions[0][states[0]]
        for i in range(1, len(states)):
            weight += law._transition(i - 1)[states[i - 1], states[i]] + law.emissions[i][states[i]]
        if np.isfinite(weight):
            yield states, np.exp(weight)


@pytest.mark.parametrize('q', [0., .2])
def test_exact_catch_mass_and_every_conditional_marginal_match_enumeration(q):
    with patch('src.evolving_trip_belief.DELAYS', np.array([-1, 0, 1])):
        belief = EvolvingTripBelief(network(), public_occurrences=False, travel_sd=.5, dwell_sd=.5)
        belief.observe_connection_event(0, 'arrival', 7)
        belief.observe_forecast(1, 'arrival', 880, 35)
        belief.observe_wait(1, 660, 700)
        pars = {'R': {'mean': .2, 'std': 1., 'cancel_rate': q}}
        sampler = FirstBranchSampler(belief, pars, 1, 750, 900)
        all_paths = list(paths(sampler.law))
        total = sum(probability for _, probability in all_paths)
        for caught in (True, False):
            selected = [(states, p) for states, p in all_paths
                        if bool(sampler.catch_mask[states[sampler.index]]) == caught]
            mass = sum(p for _, p in selected)
            assert np.isclose(sampler.probability(caught), mass / total, atol=1e-12)
            conditional = sampler.conditional_law(caught)
            assert all(a is b for a, b in zip(conditional.times, sampler.law.times))
            for index in range(len(conditional.times)):
                expected = np.zeros(len(conditional.times[index]))
                for states, probability in selected:
                    expected[states[index]] += probability / mass
                np.testing.assert_allclose(conditional.marginal(index), expected, atol=1e-12)


def test_sample_preserves_correlated_trip_other_trip_law_and_public_information():
    belief = EvolvingTripBelief(network(), public_occurrences=False, travel_sd=.3, dwell_sd=.3)
    parameters = {'R': {'mean': 0., 'std': .7, 'cancel_rate': .1}}
    sampler = FirstBranchSampler(belief, parameters, 0, 600, 800)
    original = belief.mean_events(parameters)[0]
    untouched = belief.mean_events(parameters)[2]
    rng = np.random.default_rng(591)
    for caught in (True, False):
        scenes = sampler.sample(caught, rng, 1800)
        observed = []
        for scene in scenes:
            event = scene.events[0]
            assert ((not event.canceled) and 600 <= event.departure <= 800) == caught
            assert event.departure <= event.arrival <= scene.events[1].departure <= scene.events[1].arrival
            observed.append(scene.events[2])
        operating = [event for event in observed if not event.canceled]
        assert abs(np.mean([event.departure for event in operating]) - untouched.departure) < 4
        assert abs(np.mean([event.canceled for event in observed]) - untouched.cancel_probability) < .025
    assert belief._known_times == belief._known_status == belief._excluded == {}
    assert belief.mean_events(parameters)[0] == original
    with pytest.raises(TypeError):
        scenes[0].events[0] = scenes[0].events[0]
    belief.observe_forecast(0, 'departure', 2000, 10)
    parameters['R']['mean'] = 25.
    assert sampler.parameters['R']['mean'] == 0
    assert all(scene.events[0].departure <= 800 for scene in sampler.sample(True, rng, 10))


def test_zero_mass_branch_and_deadline_keep_four_weighted_worlds():
    belief = EvolvingTripBelief(network(), public_occurrences=False)
    pars = {'R': {'mean': 0., 'std': .5, 'cancel_rate': 0.}}
    sampler = FirstBranchSampler(belief, pars, 0, 800, 800)
    assert sampler.probability(True) == 0
    with pytest.raises(ValueError, match='zero posterior'):
        sampler.sample(True, np.random.default_rng(1), 1)
    with pytest.raises(ValueError, match='zero posterior'):
        sampler.conditional_distribution(2, 'departure', caught=True)
    bank = sampler.stratified(np.random.default_rng(2))
    assert len(bank) == 4 and sum(p for p, _ in bank) == 1
    with pytest.raises(ValueError, match='at least one'):
        FirstBranchSampler(belief, pars, 0, 600, 900).stratified(np.random.default_rng(3), n_catch=0)


def test_tiny_operating_mass_is_not_lost_by_one_minus_cancel():
    graph = TransitGraph()
    graph.add_stop(Stop(0, '0'))
    graph.add_stop(Stop(1, '1'))
    graph.add_connection(Connection(0, 'R', 'A', 0, 1, 0, 1))
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    belief.observe_wait(0, -5, 7)
    pars = {'R': {'mean': 0., 'std': .5, 'cancel_rate': .1}}
    sampler = FirstBranchSampler(belief, pars, 0, 8, 20)
    assert 0 < sampler.probability(True) < 1e-20
    for scene in sampler.sample(True, np.random.default_rng(50), 8):
        assert not scene.events[0].canceled
        assert 8 <= scene.events[0].departure <= 20


def test_ordered_duration_law_and_no_rejection_even_for_unlikely_branch():
    belief = OrderedDurationBelief(network(), public_occurrences=False, travel_sd=.1)
    pars = {'R': {'mean': 0., 'std': .5, 'cancel_rate': 0.}}
    sampler = FirstBranchSampler(belief, pars, 0, 960, 1200)
    law = sampler.conditional_law(True)
    with patch.object(law, 'draw_uniforms', wraps=law.draw_uniforms) as draw:
        scenes = sampler.sample(True, np.random.default_rng(43), 4)
        for scene in scenes:
            assert 960 <= scene.events[0].departure <= 1200
        assert draw.call_count == 4  # no1/p rejection loop


def test_shared_day_model_requires_its_own_branch_conditioning():
    from src.joint_day_belief import JointDayBelief
    from src.first_branch_bank import FirstBranchBank

    joint = JointDayBelief(network(), public_occurrences=False)
    pars = {'R': {'mean': 0., 'std': .5, 'cancel_rate': 0.}}
    with pytest.raises(TypeError, match='independent-trip'):
        FirstBranchSampler(joint, pars, 0, 600, 800)
    with pytest.raises(TypeError, match='independent-trip'):
        FirstBranchBank(joint, pars, np.random.default_rng(1), 4)
