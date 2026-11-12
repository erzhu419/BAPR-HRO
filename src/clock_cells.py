"""Normal probability masses for integer event-clock grids."""

import numpy as np
from scipy.special import log_ndtr


def log_normal_interval(lower, upper, mean=0., sd=1.):
    """Stable log normal probability for finite or infinite interval endpoints."""
    low, high = np.broadcast_arrays((np.asarray(lower) - mean) / sd,
                                    (np.asarray(upper) - mean) / sd)
    # Subtract survival probabilities in the positive tail and CDFs otherwise.
    positive = low >= 0
    large = np.where(positive, log_ndtr(-low), log_ndtr(high))
    small = np.where(positive, log_ndtr(-high), log_ndtr(low))
    result = np.full(low.shape, -np.inf)
    valid = (high > low) & np.isfinite(large)
    with np.errstate(divide='ignore', invalid='ignore'):
        difference = small[valid] - large[valid]
        result[valid] = large[valid] + np.log(-np.expm1(np.minimum(0., difference)))
    return result


def bounded_clock_log_mass(clocks, mean, sd):
    """Cell masses of a normal restricted to the two endpoint clocks."""
    if len(clocks) == 1:
        return np.zeros(1)
    middle = (clocks[:-1].astype(float) + clocks[1:]) / 2
    bounds = np.r_[clocks[0], middle, clocks[-1]]
    return log_normal_interval(bounds[:-1], bounds[1:], mean, sd)


def bounded_elapsed_log_mass(previous, following, mean, sd):
    """Unnormalized masses for ordered, bounded elapsed normal cells.

    The first legal cell starts at the preceding clock if that clock lies
    inside the interval. A predecessor at the only possible endpoint has the
    limiting conditional point mass there.
    """
    middle = (following[:-1].astype(float) + following[1:]) / 2
    bounds = np.r_[following[0], middle, following[-1]]
    first = np.searchsorted(following, previous, side='left')
    columns = np.arange(len(following))[None, :]
    offset = previous[:, None]
    lower = np.where(columns == first[:, None], np.maximum(following[0], offset), bounds[:-1])
    mass = log_normal_interval(lower - offset, bounds[1:] - offset, mean, sd)
    mass[columns < first[:, None]] = -np.inf
    mass[previous == following[-1], -1] = 0.
    return mass
