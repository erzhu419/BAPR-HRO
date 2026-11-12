from itertools import product
import numpy as np
import pytest

from experiments.reconstruction_v24.pair_decision import PairDecision
from src.full_menu_opportunity import FullMenuOpportunity


class Scores:
    def __init__(self, laws, costs):
        self.laws, self.costs = laws, costs

    def departure_distribution(self, cid):
        times, mass, cancel = self.laws[cid]
        return np.array(times), np.array(mass), cancel

    def clock_nominal_score(self, cid, clock):
        return self.costs[cid]

    def board_score(self, cid, clock):
        raise AssertionError('The full-menu nominal policy must not compile exact scores')


@pytest.mark.parametrize('current_cost', [50., 80.])
def test_two_candidates_match_pair_control_and_preserve_unobserved_past_mass(current_cost):
    compiler = Scores({0: ([10], [1.], 0.), 1: ([0, 20], [.5, .5], 0.)}, {0: current_cost, 1: 30.})
    ranks = {0: (40, 0), 1: (10, 1)}
    full = FullMenuOpportunity(compiler, ranks, 5, 5, 100)
    pair = PairDecision(compiler, ranks, 5, 5, 100)
    assert full.wait_score(10, {0}) == pair.wait_score(1, 10, 'clock_nominal') == 65.
    pair_choice = pair.choose((0,), 1, 10, 'clock_nominal')[0]
    assert full.choose((0,), (), 10)[0] == (0 if pair_choice == 0 else None)


def test_three_candidates_capture_complementary_rescue_and_reconsider_public_offers():
    compiler = Scores({0: ([10], [1.], 0.), 1: ([20], [.5], .5), 2: ([30], [.5], .5)},
                      {0: 60., 1: 30., 2: 50.})
    full = FullMenuOpportunity(compiler, {i: (i, i) for i in range(3)}, 0, 0, 100)
    assert full.wait_score(10, {0}) == 52.5
    assert full.choose((0,), (), 10)[0] is None
    # The earlier refusal did not commit to B: C can be accepted after B's public cancellation.
    assert full.choose((2,), {0, 1}, 30)[0] == 2
    # A cancellation already known at the first offer changes its legal wait reference.
    assert full.choose((0,), {1}, 10)[0] == 0


def test_wait_reference_uses_declared_first_opportunity_tie_rank():
    compiler = Scores({0: ([10], [1.], 0.), 1: ([20], [1.], 0.), 2: ([20], [1.], 0.)},
                      {0: 60., 1: 90., 2: 25.})
    ranks = {0: (0, 0), 1: (1, 1), 2: (2, 2)}
    full = FullMenuOpportunity(compiler, ranks, 0, 0, 100)
    assert full.wait_score(10, {0}) == 90.
    assert full.choose((0,), (), 10)[0] == 0
    assert full.choose((1, 2), {0}, 20, 'first_offer')[0] == 1
    assert full.choose((1, 2), {0}, 20)[0] == 2


def test_wait_product_matches_independent_world_enumeration_with_past_ties_and_cancellation():
    laws = {0: ([10], [1.], 0.), 1: ([0, 20, 30], [.2, .3, .2], .3),
            2: ([20, 40], [.4, .4], .2), 3: ([25, 30], [.5, .25], .25)}
    costs = {0: 70., 1: 35., 2: 60., 3: 50.}
    ranks = {i: (i, i) for i in laws}
    full = FullMenuOpportunity(Scores(laws, costs), ranks, 5, 5, 100)
    # Condition only on there having been no departure from entry5 through now10.
    atoms = [tuple(zip(laws[cid][0], laws[cid][1]))+((None, laws[cid][2]),) for cid in (1, 2, 3)]
    total, mass = 0., 0.
    for world in product(*atoms):
        if any(t is not None and 5 <= t <= 10 for t, _ in world):
            continue
        probability = np.prod([p for _, p in world])
        offers = [(t, ranks[cid], cid) for cid, (t, _) in zip((1, 2, 3), world)
                  if t is not None and 10 < t < 100]
        total += probability*(costs[min(offers)[2]] if offers else 100)
        mass += probability
    assert full.wait_score(10, {0}) == pytest.approx(total/mass)
