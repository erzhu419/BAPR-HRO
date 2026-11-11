"""Model-only lower confidence bounds for bounded temporal oracle losses."""

from math import log

import numpy as np


# Keep most of the weight on the strongest bet for concentrated travel times.
# The smaller bets survive a low or zero world and use no separate alpha split.
_BET_FRACTIONS = np.array([1., .5, .25])
_LOG_WEIGHTS = np.log([.90, .075, .025])


def _log_evalue(shifted, candidate):
    if candidate == 0.:
        return float('inf') if np.any(shifted > 0.) else 0.
    ratios = shifted / candidate
    factors = (1. - _BET_FRACTIONS[:, None]) + _BET_FRACTIONS[:, None] * ratios
    with np.errstate(divide='ignore'):
        wealth = np.log(factors).sum(axis=1) + _LOG_WEIGHTS
    return float(np.logaddexp.reduce(wealth))


def continuation_log_evalue(losses, mean, *, floor=0.):
    """Log wealth for testing whether the frozen model mean is at most `mean`."""
    shifted = np.asarray(losses, dtype=float) - floor
    if shifted.ndim != 1 or shifted.size == 0 or np.any(~np.isfinite(shifted) | (shifted < 0.)):
        raise ValueError('Oracle losses must be a nonempty vector above the certified floor')
    if not np.isfinite(mean) or mean < floor:
        raise ValueError('Candidate mean must be at least the certified floor')
    return _log_evalue(shifted, mean - floor)


def continuation_lower(losses, cap, alpha, *, floor=0.):
    """Invert a one-sided e-test for the mean of an oracle loss.

    With a deterministic pathwise floor b, test each m > b by mixing the
    products of (1-f) + f*(Y_i-b)/(m-b) for prespecified fractions f. Under
    E[Y_i | prior draws] <= m each factor is nonnegative with conditional
    expectation <= 1. Ville's inequality bounds false rejection by alpha
    at any prefix of the same tape. Wealth decreases in m, so inversion
    gives a lower confidence bound. A fresh decision history needs fresh
    model draws and its own error allocation when making a joint claim.
    """
    values = np.asarray(losses, dtype=float)
    if (not 0. < alpha < 1. or not 0. <= floor <= cap < float('inf')
            or values.ndim != 1 or values.size == 0
            or np.any(~np.isfinite(values) | (values < floor) | (values > cap))):
        raise ValueError('Use 0 < alpha < 1 and bounded oracle losses in [floor, cap]')
    shifted = values - floor
    # At the sample mean, every factor product is <= 1 by AM-GM.
    low, high = 0., float(shifted.mean())
    target = -log(alpha)
    while high - low > 1e-6:
        middle = (low + high) / 2
        if _log_evalue(shifted, middle) >= target:
            low = middle
        else:
            high = middle
    return floor + low
