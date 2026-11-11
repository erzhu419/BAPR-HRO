from itertools import product
from math import log

import pytest

from src.continuation_confidence import continuation_log_evalue, continuation_lower
from src.temporal_scan import bounded_mean_lower


def test_mixture_inverts_conservatively_and_improves_v9_for_concentrated_losses():
    losses = [330, 390, 450, 510] * 8
    lower = continuation_lower(losses, 3200, .05)
    assert bounded_mean_lower(losses, 3200, .05)['lower'] < lower < 420
    assert continuation_log_evalue(losses, lower) >= -log(.05)
    assert continuation_log_evalue(losses, lower + 1e-6) < -log(.05)
    assert continuation_lower(losses, 3200, .05, floor=300) > lower


def test_zero_loss_world_does_not_kill_smaller_bets():
    losses = [0] + [500] * 31
    assert continuation_lower(losses, 3000, .05) > 300
    assert continuation_lower([80] * 8, 3000, .05, floor=80) == 80
    with pytest.raises(ValueError, match='bounded oracle losses'):
        continuation_lower(losses, 3000, .05, floor=80)


def test_any_prefix_false_rejection_probability_on_binary_oracle():
    alpha, p = .2, .4
    n = 8
    failure_mass = 0.
    for worlds in product((0., 100.), repeat=n):
        if any(continuation_log_evalue(worlds[:i], 100. * p) >= -log(alpha) for i in range(1, n + 1)):
            successes = sum(worlds) / 100.
            failure_mass += p ** successes * (1. - p) ** (n - successes)
    assert failure_mass <= alpha
