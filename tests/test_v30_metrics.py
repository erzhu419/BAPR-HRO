import pytest

from experiments.reconstruction_v30.metrics import distribution_score


def test_crps_and_failure_mass_against_explicit_double_sum():
    d = dict(values=[10, 20], probabilities=[.2, .3], failure_mass=.5)
    scored = distribution_score(d, 18, deadline=30)
    atoms = [(10, .2), (20, .3), (30, .5)]
    expected = sum(p*abs(x-18) for x, p in atoms)-.5*sum(
        p*q*abs(x-y) for x, p in atoms for y, q in atoms)
    assert scored['crps'] == pytest.approx(expected)
    assert scored['mean'] == pytest.approx(23)
    assert distribution_score(d, 18)['mean'] == pytest.approx(16)
    assert distribution_score(d, None, deadline=30) is None
