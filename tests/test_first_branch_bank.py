from unittest.mock import patch

import numpy as np

from src.evolving_trip_belief import EvolvingTripBelief
from src.first_branch_bank import FirstBranchBank
from tests.test_first_branch_belief import network


PARAMETERS = {'R': {'mean': 0., 'std': .7, 'cancel_rate': .1}}


def test_original_worlds_and_rng_exactly_match_original_sampler():
    belief = EvolvingTripBelief(network(), public_occurrences=False)
    left, right = np.random.default_rng(70), np.random.default_rng(70)
    original = belief.sample(PARAMETERS, left, 8)
    bank = FirstBranchBank(belief, PARAMETERS, right, 8)
    assert left.bit_generator.state == right.bit_generator.state
    assert [[scene.events[cid] for cid in range(3)] for scene in original] == [
           [scene.events[cid] for cid in range(3)] for scene in bank.original]


def test_untouched_trip_worlds_reuse_same_objects_and_conditioned_events_obey_branch():
    belief = EvolvingTripBelief(network(), public_occurrences=False)
    bank = FirstBranchBank(belief, PARAMETERS, np.random.default_rng(81), 4)
    a = bank.for_action(0, 600, 800, n_catch=2)
    b = bank.for_action(2, 900, 1200, n_catch=2)
    for index, (_, scene) in enumerate(a):
        assert scene.events[2] is bank.original[index].events[2]
        assert (not scene.events[0].canceled and 600 <= scene.events[0].departure <= 800) == (index < 2)
    for index, (_, scene) in enumerate(b):
        assert scene.events[0] is bank.original[index].events[0]
        assert (not scene.events[2].canceled and 900 <= scene.events[2].departure <= 1200) == (index < 2)
    assert np.isclose(sum(p for p, _ in a), 1.)
    assert np.isclose(sum(p for p, _ in b), 1.)


def test_constant_branch_payoff_integrates_exactly_and_draws_use_original_tape():
    belief = EvolvingTripBelief(network(), public_occurrences=False)
    bank = FirstBranchBank(belief, PARAMETERS, np.random.default_rng(8), 4)
    sampler = bank.sampler(0, 600, 800)
    conditioned = sampler.conditional_law(True)
    with patch.object(conditioned, 'draw_uniforms', wraps=conditioned.draw_uniforms) as draw:
        scenes = bank.for_action(0, 600, 800)
        _ = [scene.events[0] for _, scene in scenes]
        count = sum(not scene.events[0].canceled and 600 <= scene.events[0].departure <= 800
                    for _, scene in scenes)
        assert draw.call_count == count
        for index, call in enumerate(draw.call_args_list):
            np.testing.assert_array_equal(call.args[0], bank.uniforms[sampler.trip][index])
    value = sum(weight * (10 if not scene.events[0].canceled and 600 <= scene.events[0].departure <= 800
                          else 100) for weight, scene in scenes)
    assert np.isclose(value, sampler.probability(True) * 10 + sampler.probability(False) * 100)


def test_zero_catch_mass_uses_every_original_world_index():
    belief = EvolvingTripBelief(network(), public_occurrences=False)
    bank = FirstBranchBank(belief, PARAMETERS, np.random.default_rng(10), 4)
    scenes = bank.for_action(0, 800, 800)
    assert len(scenes) == 4
    assert [weight for weight, _ in scenes] == [.25] * 4
    assert all(scene.events[2] is bank.original[i].events[2] for i, (_, scene) in enumerate(scenes))
