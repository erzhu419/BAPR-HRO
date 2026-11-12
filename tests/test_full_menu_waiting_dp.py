from itertools import product

import numpy as np
import pytest

from src.full_menu_opportunity import FullMenuOpportunity
from src.full_menu_waiting_dp import FullMenuWaitingDP


class Scores:
    def __init__(self, laws, values):
        self.laws, self.values = laws, values

    def departure_distribution(self, cid):
        times, mass, cancel = self.laws[cid]
        return np.array(times), np.array(mass, dtype=float), cancel

    def clock_nominal_score(self, cid, clock):
        return self.values[cid]

    def board_score(self, cid, clock):
        raise AssertionError('Waiting DP must not prepare exact board scores')


def controller(laws, values, entry=0, ready=0, deadline=100):
    reference = FullMenuOpportunity(Scores(laws, values), {cid: (cid, cid) for cid in laws},
                                     entry, ready, deadline)
    return reference, FullMenuWaitingDP(reference)


def brute_value(worlds, remaining, now, ready, deadline, values):
    """Independent oracle: split whole worlds only by their next public event."""
    total_mass = sum(weight for _, weight in worlds)
    histories = {}
    no_offer = 0.
    for world, weight in worlds:
        clocks = [world[cid] for cid in remaining if world[cid] is not None and now < world[cid] < deadline]
        if not clocks:
            no_offer += weight
            continue
        clock = min(clocks)
        visible = tuple(cid for cid in remaining if world[cid] == clock)
        histories.setdefault((clock, visible), []).append((world, weight))
    result = no_offer*deadline
    for (clock, visible), group in histories.items():
        future = brute_value(group, tuple(cid for cid in remaining if cid not in visible),
                             clock, ready, deadline, values)
        board = min(values[cid] for cid in visible) if clock >= ready else deadline
        result += sum(weight for _, weight in group)*min(board, future)
    return result/total_mass


def test_multiple_refusals_improve_over_the_one_step_wait_reference():
    laws = {0: ([10], [1.], 0.), 1: ([20], [1.], 0.), 2: ([30], [1.], 0.)}
    reference, dp = controller(laws, {0: 70., 1: 90., 2: 40.})
    assert reference.choose((0,), (), 10)[0] == 0  # Next-offer-only value is90.
    assert dp.choose((0,), (), 10)[0] is None
    assert dp.choose((1,), {0}, 20)[0] is None
    assert dp.choose((2,), {0, 1}, 30)[0] == 2
    assert dp.values[0, dp.full_mask] == 40.
    # Simultaneous legal offers still choose the minimum board cost.
    _, tied = controller({0: ([30], [1.], 0.), 1: ([30], [1.], 0.)}, {0: 90., 1: 40.})
    assert tied.choose((0, 1), (), 30)[0] == 1


def test_exact_dp_matches_causal_world_enumeration_with_early_receipts_ties_and_cancellation():
    laws = {0: ([0, 10, 20], [.2, .4, .3], .1),
            1: ([8, 20, 35], [.2, .5, .2], .1), 2: ([20, 30], [.6, .3], .1)}
    values = {0: 40., 1: 45., 2: 40.}
    _, dp = controller(laws, values, entry=5, ready=15, deadline=50)
    atoms = [tuple(zip(times, mass))+((None, cancel),) for times, mass, cancel in laws.values()]
    worlds = [(dict(zip(laws, (clock for clock, _ in row))), np.prod([p for _, p in row]))
              for row in product(*atoms)]
    exact = brute_value(worlds, tuple(laws), 4, 15, 50, values)
    assert dp.values[0, dp.full_mask] == pytest.approx(exact)
    assert list(dp.times[:2]) == [8, 10]  # These remove vehicles but are too early to board.


def test_before_entry_cancellation_and_after_deadline_mass_remain_in_survival():
    _, dp = controller({0: ([10], [1.], 0.), 1: ([0, 20, 70], [.25, .25, .25], .25)},
                       {0: 48., 1: 30.}, entry=5, ready=5, deadline=50)
    assert dp.wait_score(10, {0}) == 45.
    assert dp.wait_score(25, {0}) == 50.
    assert dp.supported[-1, dp.bit_by_cid[1]]


def test_off_grid_queries_use_strictly_next_event_without_interpolation():
    _, dp = controller({0: ([10], [1.], 0.), 1: ([20], [1.], 0.), 2: ([30], [1.], 0.)},
                       {0: 80., 1: 60., 2: 40.})
    assert dp.wait_score(11, {0}) == dp.wait_score(19, {0}) == 40.
    action, receipt = dp.choose((0,), (), 13)
    assert action is None and receipt['clock_mode'] == 'off_grid'
    assert receipt['next_event_index'] == 1
    assert not receipt['unsupported_history']


def test_zero_survival_history_is_explicitly_unsupported_and_uses_reference_fallback():
    reference, dp = controller({0: ([10], [1.], 0.), 1: ([15], [1.], 0.)}, {0: 30., 1: 40.})
    choice, receipt = dp.choose((1,), (), 15)  # Candidate0 should already have been observed.
    assert choice == 1 and receipt['unsupported_history']
    assert receipt['wait'] == reference.wait_score(15, {1}) == 100.
    assert not dp.supported[-1, dp.bit_by_cid[0]]
    assert np.isnan(dp.values[-1, dp.bit_by_cid[0]])
    assert dp.counters['unsupported_queries'] == 1
