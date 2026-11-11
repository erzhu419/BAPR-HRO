"""Past-only location/scale fits for nonnegative ordered elapsed durations.

The continuous fit is a truncated-normal likelihood. The grid fit instead
maximizes expected categorical log likelihood after independently rounding an
elapsed duration to adjacent grid points with probabilities preserving its
mean. Neither is the likelihood of a complete adaptive-grid trip HMM.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp


def continuous_mean(scheduled, location, scale):
    mean = np.asarray(scheduled, dtype=float) + location
    alpha = mean / scale
    mills = np.exp(-alpha * alpha / 2 - np.log(2 * np.pi) / 2 - log_ndtr(alpha))
    return mean + scale * mills


def continuous_objective(theta, scheduled, elapsed):
    """Mean negative log density and gradient in (seconds, log seconds)."""
    location, log_scale = theta
    scale = np.exp(log_scale)
    z = (elapsed - scheduled - location) / scale
    alpha = (scheduled + location) / scale
    mills = np.exp(-alpha * alpha / 2 - np.log(2 * np.pi) / 2 - log_ndtr(alpha))
    value = np.mean(z * z / 2 + log_scale + log_ndtr(alpha) + np.log(2 * np.pi) / 2)
    gradient = [np.mean((-z + mills) / scale), np.mean(1 - z * z - alpha * mills)]
    return float(value), np.asarray(gradient)


def duration_probabilities(scheduled, location, scale, step=60., upper=None):
    """Normalized nonnegative grid duration law, with negligible upper tail.

    The upper integration limit is at least twelve scale units beyond the
    unconstrained location, and includes ``upper`` when supplied. The finite
    trip delay support and observation anchors are not included in this law.
    """
    stop = max(step, scheduled + location + 12 * scale, upper or 0.)
    grid = np.arange(int(np.ceil(stop / step)) + 1) * step
    log_weights = -.5 * ((grid - scheduled - location) / scale) ** 2
    return grid, np.exp(log_weights - logsumexp(log_weights))


class GridDurationObjective:
    """Expected log likelihood of a predeclared randomized coarsening.

    An observed y between k*h and (k+1)*h contributes weights 1-f and f to
    those two categorical outcomes, f=(y-k*h)/h. It is a training objective
    for grid predictions, not an exact density likelihood for second clocks.
    """
    def __init__(self, scheduled, elapsed, step=60.):
        self.step = step
        self.scheduled, counts = np.unique(scheduled, return_counts=True)
        self.weights = counts / counts.sum()
        lower = np.floor(elapsed / step) * step
        fraction = (elapsed - lower) / step
        self.mean_y = np.mean(elapsed)
        self.mean_y2 = np.mean((1 - fraction) * lower ** 2 + fraction * (lower + step) ** 2)
        self.mean_d = np.mean(scheduled)
        self.mean_d2 = np.mean(scheduled ** 2)
        self.mean_dy = np.mean(scheduled * elapsed)
        self.upper_observed = float(np.max(elapsed) + step)

    def __call__(self, theta):
        location, log_scale = theta
        scale = np.exp(log_scale)
        upper = max(self.upper_observed, float(np.max(self.scheduled)) + location + 12 * scale)
        grid = np.arange(int(np.ceil(upper / self.step)) + 1) * self.step
        residual = grid[None, :] - self.scheduled[:, None] - location
        logits = -.5 * (residual / scale) ** 2
        normalizer = logsumexp(logits, axis=1)
        probabilities = np.exp(logits - normalizer[:, None])
        observed_square = (self.mean_y2 + self.mean_d2 + location ** 2 - 2 * self.mean_dy
                           - 2 * location * (self.mean_y - self.mean_d)) / scale ** 2
        value = observed_square / 2 + self.weights @ normalizer
        gradient = [(self.mean_d + location - self.mean_y
                     + self.weights @ np.sum(probabilities * residual, axis=1)) / scale ** 2,
                    -observed_square + self.weights @ np.sum(probabilities * residual ** 2, axis=1) / scale ** 2]
        return float(value), np.asarray(gradient)


def fit_ordered_duration(scheduled, elapsed, *, objective='continuous', step=60., resolution=1.):
    """Fit one kind using only declared training transitions.

    Scale's lower bound is the timestamp resolution. Three fixed starts
    diagnose optimization dependence; no held-out clock enters selection.
    """
    scheduled, elapsed = np.asarray(scheduled, dtype=float), np.asarray(elapsed, dtype=float)
    if scheduled.shape != elapsed.shape or not len(elapsed) or np.any(elapsed < 0):
        raise ValueError('Require paired training durations with nonnegative actual elapsed time')
    residual = elapsed - scheduled
    raw_location, raw_scale = float(np.mean(residual)), max(resolution, float(np.std(residual)))
    if objective == 'continuous':
        function = lambda theta: continuous_objective(theta, scheduled, elapsed)
    elif objective == 'grid':
        function = GridDurationObjective(scheduled, elapsed, step)
    else:
        raise ValueError(objective)
    starts = [(raw_location, raw_scale), (-100., 60.), (-1000., 120.)]
    attempts = []
    for location, scale in starts:
        result = minimize(function, [location, np.log(max(scale, resolution))], jac=True,
                          method='L-BFGS-B', bounds=[(None, None), (np.log(resolution), None)],
                          options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-7})
        attempts.append({'location_seconds': float(result.x[0]), 'scale_seconds': float(np.exp(result.x[1])),
                         'mean_negative_log_likelihood': float(result.fun), 'converged': bool(result.success),
                         'gradient': result.jac.tolist(), 'message': str(result.message)})
    best = min((r for r in attempts if r['converged']), key=lambda r: r['mean_negative_log_likelihood'])
    return dict(best, objective=objective, timestamp_resolution_seconds=resolution,
                n=len(elapsed), actual_zero_count=int(np.sum(elapsed == 0)), starts=attempts)
