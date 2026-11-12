from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from src.evolving_trip_belief import _ChainLaw, _Event
from src.public_trip_conditionals import condition_on_departure


def finite_law(*, last_evidence=(1., .6, .8)):
    slots = (_Event(0, 'departure', 0), _Event(0, 'arrival', 3),
             _Event(1, 'departure', 6), _Event(1, 'arrival', 9))
    clocks = [np.array([slot.scheduled, slot.scheduled+2, slot.scheduled+2]) for slot in slots]
    canceled = [np.array([False, False, True]) for _ in slots]
    initial = np.array([.2, .6, .2])
    transition = (
        np.array([[.75, .25, 0.], [.2, .8, 0.], [0., 0., 1.]]),
        np.array([[.4, .5, .1], [.1, .65, .25], [.1, .1, .8]]),
        np.array([[.6, .4, 0.], [.25, .75, 0.], [0., 0., 1.]]),
    )
    evidence = [np.ones(3), np.ones(3), np.ones(3), np.asarray(last_evidence)]
    with np.errstate(divide='ignore'):
        logs = [np.log(p) for p in transition]
        law = _ChainLaw(clocks, np.log(initial), logs.__getitem__, [np.log(e) for e in evidence],
                        canceled, (True, True, True))
    belief = SimpleNamespace(_positions={(slot.cid, slot.kind): ('trip', i)
                                         for i, slot in enumerate(slots)},
                             _trips={'trip': slots}, _law=lambda trip, parameters: law)
    return belief, law, initial, transition, evidence


def test_departure_conditioned_marginals_match_full_path_enumeration():
    belief, law, initial, transitions, evidence = finite_law()
    result = condition_on_departure(belief, {}, 0, 2)
    exact = np.zeros((4, 3))
    for path in product(range(3), repeat=4):
        if path[0] != 1:  # the canceled state has the same clock but is not a departure
            continue
        weight = initial[path[0]] * np.prod([e[s] for e, s in zip(evidence, path)])
        weight *= np.prod([kernel[a, b] for kernel, a, b in zip(transitions, path, path[1:])])
        for index, state in enumerate(path):
            exact[index, state] += weight
    exact /= exact.sum(axis=1, keepdims=True)
    for index, (cid, kind) in enumerate(belief._positions):
        assert result.law.marginal(index) == pytest.approx(exact[index])
        marginal = result.marginal(cid, kind)
        assert marginal.cancel_probability == pytest.approx(exact[index, 2])
        assert marginal.operating_probability == pytest.approx(exact[index, :2].sum())
        assert marginal.mean_operating_clock == pytest.approx(
            law.times[index][:2] @ exact[index, :2] / exact[index, :2].sum())
    assert result.law.times is law.times
    assert result.law._transition(1) is law._transition(1)
    assert np.array_equal(law.emissions[0], np.zeros(3))


def test_conditioned_sampling_retains_service_dependence_and_suffix_boundary():
    belief, _, _, _, _ = finite_law()
    anchor = condition_on_departure(belief, {}, 1, 8)
    assert anchor.law.marginal(2) == pytest.approx([0., 1., 0.])
    with pytest.raises(ValueError, match='outside'):
        anchor.marginal(0)
    draws = anchor.sample_suffix(np.random.default_rng(31), 3000)
    assert all(set(row) == {1} and row[1].departure == 8 and not row[1].canceled for row in draws)
    assert np.mean([row[1].arrival for row in draws]) == pytest.approx(
        anchor.marginal(1).mean_operating_clock, abs=.05)

    first = condition_on_departure(belief, {}, 0, 2)
    draws = first.sample_suffix(np.random.default_rng(31), 3000)
    assert np.mean([row[1].canceled for row in draws]) == pytest.approx(
        first.marginal(1).cancel_probability, abs=.035)


def test_departure_outside_original_support_does_not_regrid():
    belief, law, _, _, _ = finite_law(last_evidence=(0., 0., 1.))
    assert condition_on_departure(belief, {}, 0, 1) is None
    assert condition_on_departure(belief, {}, 1, 8) is None
    assert tuple(law.times[0]) == (0, 2, 2)
