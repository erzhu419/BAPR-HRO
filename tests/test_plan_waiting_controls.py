import numpy as np
import pytest

from src.full_menu_opportunity import FullMenuOpportunity
from src.plan_waiting_controls import (CombinedWaitingOpportunity, SchedulePriority,
                                      SingleTargetOpportunity, TwoTargetOpportunity)


class Scores:
    def __init__(self, laws, values):
        self.laws, self.values = laws, values

    def departure_distribution(self, cid):
        times, mass, cancel = self.laws[cid]
        return np.array(times), np.array(mass, dtype=float), cancel

    def clock_nominal_score(self, cid, clock):
        return self.values[cid]

    def board_score(self, cid, clock):
        raise AssertionError('Cheap waiting controls must not prepare exact scores')


def make(laws, values, entry=0, ready=0, deadline=100):
    reference = FullMenuOpportunity(Scores(laws, values), {cid: (cid, cid) for cid in laws},
                                     entry, ready, deadline)
    return SingleTargetOpportunity(reference)


def test_single_target_suffix_matches_enumerated_conditional_worlds():
    laws = {0: ([10], [1.], 0.), 1: ([0, 8, 20, 120], [.2, .2, .3, .1], .2),
            2: ([0, 30], [.4, .5], .1)}
    values = {0: 80., 1: 40., 2: 50.}
    rule = make(laws, values, entry=5, ready=5)
    candidates = {}
    for cid in (1, 2):
        times, mass, canceled = laws[cid]
        worlds = [*zip(times, mass), (None, canceled)]
        surviving = [(time, p) for time, p in worlds if time is None or time < 5 or time > 10]
        candidates[cid] = sum(p*(values[cid] if time is not None and 10 < time < 100 else 100)
                              for time, p in surviving)/sum(p for _, p in surviving)
    assert rule.wait_score(10, {0}) == pytest.approx(min(candidates.values()))
    assert rule.choose((0,), (), 10)[1]['target'] == min(candidates, key=candidates.get)


def test_unseen_past_cancel_after_deadline_and_zero_survival_are_not_future_certainty():
    rule = make({0: ([10], [1.], 0.), 1: ([0, 20, 120], [.25, .25, .25], .25),
                 2: ([8], [1.], 0.)}, {0: 70., 1: 40., 2: 20.}, entry=5, ready=5)
    assert rule.wait_score(10, {0}) == 85.
    choice, receipt = rule.choose((0,), (), 10)
    assert choice == 0 and receipt['unsupported_candidates'] == [2]
    assert rule.wait_score(20, {0, 2}) == 100.  # The atom at the public clock is not future.


def test_refusal_retargets_from_new_public_offer_without_locking_old_target():
    rule = make({0: ([10], [1.], 0.), 1: ([20], [1.], 0.), 2: ([30], [1.], 0.)},
                {0: 80., 1: 50., 2: 40.})
    choice, receipt = rule.choose((0,), (), 10)
    assert choice is None and receipt['target'] == 2
    assert rule.choose((1,), {0}, 20)[0] is None
    assert rule.choose((2,), {0, 1}, 30)[0] == 2
    # A public cancellation removes old target2; the next visible vehicle is accepted.
    choice, receipt = rule.choose((1,), {0, 2}, 20)
    assert choice == 1 and receipt['target'] is None


def test_schedule_priority_uses_receipts_not_the_passage_of_public_clock():
    rule = SchedulePriority({0: 30, 1: 40, 2: 40})
    assert rule.choose((1,), (), 10)[0] is None
    assert rule.choose((1,), (), 100)[0] is None  # Clock passing the schedule is no departure receipt.
    assert rule.choose((1, 2), {0, 1, 2}, 100)[0] == 1
    assert rule.choose((2,), {0, 1, 2}, 100)[0] == 2


def test_off_grid_strict_future_and_boarding_ties_use_the_reference_rank():
    rule = make({0: ([10], [1.], 0.), 1: ([20], [1.], 0.), 2: ([20], [1.], 0.)},
                {0: 40., 1: 40., 2: 40.})
    assert rule.wait_score(11, {0}) == rule.wait_score(19, {0}) == 40.
    assert rule.choose((0,), (), 11)[0] == 0  # Equal wait value prefers current boarding.
    assert rule.choose((1, 2), {0}, 20)[0] == 1


def test_combined_waiting_keeps_complementary_rescue_and_single_target_advantages():
    # Neither uncertain single target justifies waiting, but their first-offer rescue does.
    single = make({0: ([10], [1.], 0.), 1: ([20], [.5], .5), 2: ([30], [.5], .5)},
                  {0: 60., 1: 30., 2: 50.})
    combined = CombinedWaitingOpportunity(single)
    assert single.choose((0,), (), 10)[0] == 0
    choice, receipt = combined.choose((0,), (), 10)
    assert choice is None
    assert receipt['single_wait'] == 65. and receipt['first_offer_wait'] == 52.5
    # Conversely, looking past a poor next offer favors the single-target reference.
    single = make({0: ([10], [1.], 0.), 1: ([20], [1.], 0.), 2: ([30], [1.], 0.)},
                  {0: 70., 1: 90., 2: 40.})
    combined = CombinedWaitingOpportunity(single)
    assert single.reference.choose((0,), (), 10)[0] == 0
    choice, receipt = combined.choose((0,), (), 10)
    assert choice is None
    assert receipt['single_wait'] == 40. and receipt['first_offer_wait'] == 90.


def test_two_target_refusals_improve_combined_wait_and_retarget_after_public_receipts():
    single = make({0: ([5], [1.], 0.), 1: ([20], [.5], .5),
                   2: ([10, 30], [.5, .5], 0.)}, {0: 55., 1: 20., 2: 70.})
    combined = CombinedWaitingOpportunity(single)
    pair = TwoTargetOpportunity(combined)
    assert combined.choose((0,), (), 5)[0] == 0
    action, receipt = pair.choose((0,), (), 5)
    # If vehicle2 appears at10, decline it and receive mean60 from vehicle1.
    # Otherwise take vehicle1 at20 if present, then use vehicle2 at30.
    assert receipt['wait'] == .5*60.+.5*(.5*20.+.5*70.) == 52.5
    assert receipt['combined_wait'] == 57.5
    assert receipt['best_pair'] == (1, 2) and action is None
    assert pair.choose((2,), {0}, 10)[0] is None
    assert pair.choose((1,), {0, 2}, 20)[0] == 1
    # A public cancellation of1 makes the currently visible2 acceptable.
    assert pair.choose((2,), {0, 1}, 10)[0] == 2


def test_two_target_subset_keeps_unobserved_past_and_shared_board_preparation():
    single = make({0: ([10], [1.], 0.), 1: ([0, 20], [.5, .5], 0.),
                   2: ([30], [.5], .5)}, {0: 50., 1: 30., 2: 60.}, entry=5, ready=5)
    pair = TwoTargetOpportunity(CombinedWaitingOpportunity(single))
    # Half the worlds retain unseen past1;2 is then an optional fallback.
    # The other half take1. Unknown past is not evidence of a future arrival.
    assert pair.wait_score(10, {0}) == .5*30.+.5*(.5*60.+.5*100.) == 55.
    assert pair.choose((0,), (), 10)[0] == 0
    assert all(dp.reference.marks is single.reference.marks for _, _, dp in pair.pairs)
    assert pair.wait_score(35, {0, 1, 2}) == 100.
