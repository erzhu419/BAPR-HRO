from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v32.conditional import condition_platform, departure_evidence, UnsupportedHistory
from src.evolving_trip_belief import _ChainLaw


def law():
    times = [np.array([0, 10, 20, 0]), np.array([30, 40, 50, 30])]
    canceled = [np.array([False, False, False, True]) for _ in times]
    initial = np.array([.2, .1, .5, .2])
    with np.errstate(divide='ignore'):
        transition = np.log(np.eye(4))
        result = _ChainLaw(times, np.log(initial), lambda i: transition,
                           [np.zeros(4), np.zeros(4)], canceled, (True,))
    return result


def test_local_nonoccurrence_retains_unseen_past_and_cancellation_and_updates_shared_suffix():
    original = law()
    changed, receipt = departure_evidence(original, 0, 5, 15)
    expected = np.array([.2, 0., .5, .2])/.9
    assert changed.marginal(0) == pytest.approx(expected)
    assert changed.marginal(1) == pytest.approx(expected)
    assert receipt['prior_mass'] == pytest.approx(.9)
    assert changed.times is original.times and changed._transition(0) is original._transition(0)
    assert original.marginal(0) == pytest.approx([.2, .1, .5, .2])


def test_offgrid_observation_uses_desired_posterior_weights_instead_of_prior_weighted_blend():
    changed, receipt = departure_evidence(law(), 0, 5, 15, observed=15)
    assert changed.marginal(0) == pytest.approx([0., .5, .5, 0.])
    assert changed.marginal(1) == pytest.approx([0., .5, .5, 0.])
    assert receipt['mode'] == 'adjacent_active_projection'
    exact, _ = departure_evidence(law(), 0, 5, 10, observed=10)
    assert exact.marginal(1) == pytest.approx([0., 1., 0., 0.])


def test_wrapper_reuses_one_conditioned_trip_for_all_its_downstream_occurrences():
    original, independent = law(), law()
    belief = SimpleNamespace(_positions={(1, 'departure'): ('shared', 0)},
        _law=lambda trip, params: original if trip == 'shared' else independent)
    conditioned, receipts = condition_platform(belief, {}, (1,), 5, 15, {1: 15})
    assert conditioned._law('shared', {}).marginal(1) == pytest.approx([0., .5, .5, 0.])
    assert conditioned._law('shared', {}) is conditioned._law('shared', {})
    assert conditioned._law('independent', {}) is independent
    assert belief._law('shared', {}) is original
    assert receipts[0]['clock'] == 15


def test_no_future_receipt_or_support_expansion_is_allowed():
    with pytest.raises(ValueError, match='public visit prefix'):
        departure_evidence(law(), 0, 5, 15, observed=20)
    with pytest.raises(UnsupportedHistory, match='outside frozen'):
        departure_evidence(law(), 0, 5, 25, observed=25)
