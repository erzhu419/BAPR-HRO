from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v24.pair_decision import PairDecision


class Scores:
    def __init__(self, distributions, values):
        self.distributions, self.values = distributions, values

    def departure_distribution(self, cid):
        t, p, c = self.distributions[cid]
        return np.array(t), np.array(p), c

    def board_score(self, cid, clock):
        return SimpleNamespace(exact=self.values[cid], mean_arrival=self.values[cid])

    def clock_nominal_score(self, cid, clock):
        return self.values[cid]


def test_unseen_pre_entry_departure_is_not_future_arrival():
    scores = Scores({0: ([10], [1.], 0), 1: ([0, 20], [.5, .5], 0)}, {0: 50, 1: 30})
    rule = PairDecision(scores, {0: (40, 0), 1: (10, 1)}, 5, 5, 100)
    assert rule.wait_score(1, 10, 'exact') == 65
    assert rule.choose((0,), 1, 10, 'exact')[0] == 0
    result = rule.expectation()
    assert result['expected_arrival']['exact'] == 50
    assert result['history_mass'] == 1


def test_observed_before_readiness_and_simultaneous_offers_are_partitioned():
    scores = Scores({0: ([10, 20], [.5, .5], 0),
                     1: ([3, 10, 30], [.2, .3, .4], .1)}, {0: 60, 1: 40})
    rule = PairDecision(scores, {0: (10, 0), 1: (20, 1)}, 0, 5, 100)
    result = rule.expectation()
    # B before readiness: .2*60. B at10: .3*40. B future/cancel:
    # unseen=.5, waiting E=.8*40+.2*100=52 instead of60.
    assert result['expected_arrival']['exact'] == pytest.approx(50.)
    assert result['expected_arrival']['first_offer'] == pytest.approx(57.)
    assert result['history_mass'] == pytest.approx(1.)
