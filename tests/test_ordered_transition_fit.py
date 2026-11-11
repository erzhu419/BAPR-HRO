import numpy as np

from src.ordered_transition_fit import (
    GridDurationObjective, continuous_objective, duration_probabilities, fit_ordered_duration)


def test_normalized_likelihood_gradients():
    scheduled = np.array([0., 0., 60., 60., 120.])
    elapsed = np.array([0., 13., 17., 75., 105.])
    theta = np.array([-30., np.log(45.)])
    for objective in (lambda t: continuous_objective(t, scheduled, elapsed),
                      GridDurationObjective(scheduled, elapsed)):
        value, gradient = objective(theta)
        offsets = np.eye(2) * 1e-5
        numerical = np.array([(objective(theta + h)[0] - objective(theta - h)[0]) / 2e-5 for h in offsets])
        assert np.isfinite(value)
        np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-6)


def test_grid_objective_is_coarsened_categorical_log_likelihood():
    scheduled, elapsed = np.array([0., 60.]), np.array([13., 95.])
    location, scale = -25., 50.
    objective = GridDurationObjective(scheduled, elapsed)
    expected = []
    for d, y in zip(scheduled, elapsed):
        grid, probability = duration_probabilities(d, location, scale, upper=155.)
        index, fraction = int(y // 60), y % 60 / 60
        expected.append(-(1 - fraction) * np.log(probability[index]) - fraction * np.log(probability[index + 1]))
        assert abs(probability.sum() - 1) < 1e-14
    assert abs(objective([location, np.log(scale)])[0] - np.mean(expected)) < 1e-12


def test_normalized_grid_fit_recovers_categorical_generator():
    rng = np.random.default_rng(8)
    scheduled = np.repeat([0., 60., 120.], 10000)
    elapsed = []
    for d in (0., 60., 120.):
        grid, probability = duration_probabilities(d, -25., 50.)
        elapsed.extend(rng.choice(grid, p=probability, size=10000))
    fitted = fit_ordered_duration(scheduled, elapsed, objective='grid')
    assert abs(fitted['location_seconds'] + 25) < 1.5
    assert abs(fitted['scale_seconds'] - 50) < 1.5
