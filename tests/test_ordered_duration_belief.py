import numpy as np
import pytest
from scipy.special import ndtr

from src.ordered_duration_belief import (
    OrderedDurationBelief, elapsed_clock_log_mass, initial_clock_log_mass,
    log_normal_interval)
from src.transit_graph import TransitGraph, Stop, Connection


PARAMETERS = {'A': {'mean': 0., 'std': .5, 'cancel_rate': .2}}


def graph(rows=None):
    network = TransitGraph()
    rows = rows or [(0, 1, 300, 300, 'trip'), (1, 2, 300, 360, 'trip'),
                    (2, 3, 360, 420, 'trip')]
    for sid in {s for row in rows for s in row[:2]}:
        network.add_stop(Stop(sid, str(sid), min_transfer_time=120))
    for cid, (source, target, dep, arr, trip) in enumerate(rows):
        network.add_connection(Connection(cid, 'A', trip, source, target, dep, arr))
    return network


def test_initial_cells_include_both_tails_and_nonuniform_midpoints():
    clocks = np.array([0, 1, 20, 60])
    mass = np.exp(initial_clock_log_mass(clocks, 10., 15.))
    edges = np.r_[-np.inf, .5, 10.5, 40., np.inf]
    expected = np.diff(ndtr((edges - 10.) / 15.))
    np.testing.assert_allclose(mass, expected, rtol=1e-14, atol=1e-15)
    assert mass.sum() == pytest.approx(1.)


def test_log_cell_probabilities_are_stable_in_both_far_tails():
    positive = log_normal_interval(40., 40.1)
    negative = log_normal_interval(-40.1, -40.)
    assert np.isfinite(positive)
    assert positive == pytest.approx(float(negative), abs=1e-12)
    assert float(log_normal_interval(-np.inf, np.inf)) == 0.
    assert float(log_normal_interval(1., 1.)) == -np.inf
    mass = elapsed_clock_log_mass(np.array([7]), np.array([7, 8, 27]), -100., 1.)
    assert np.isfinite(mass).all()
    assert np.exp(mass).sum() == pytest.approx(1.)


def test_first_allowed_cell_starts_at_predecessor_not_its_grid_midpoint():
    mass = np.exp(elapsed_clock_log_mass(np.array([7, 30, 50]), np.array([0, 20, 40]), 10., 5.))
    expected_first = (ndtr((23 - 10.) / 5.) - ndtr(-10. / 5.)) / ndtr(10. / 5.)
    assert mass[0, 0] == 0
    assert mass[0, 1] == pytest.approx(expected_first)
    assert mass[0].sum() == pytest.approx(1.)
    np.testing.assert_array_equal(mass[1], [0., 0., 1.])
    assert mass[2].sum() == 0  # no ordered successor, not a fabricated row


def test_twenty_second_core_and_old_tails_share_no_base_density_cache():
    network = graph([(0, 1, 300, 360, 'one'), (0, 1, 900, 960, 'two')])
    params = {'A': dict(PARAMETERS['A'], cancel_rate=0.)}
    belief = OrderedDurationBelief(network, public_occurrences=False)
    first = belief._law('one', params)
    second = belief._law('two', params)
    assert len(first.times[0]) == 90
    assert {0, 20, 40, 300, 360, 4140} <= set(first.times[0])
    assert first._transition(0) is second._transition(0)
    assert not first._transition(0).flags.writeable
    assert belief._law('one', params) is first
    assert belief._transition_cache == {}
    assert len(belief._duration_transition_cache) > 0


def test_zero_duration_and_one_second_exact_receipts_remain_exact_in_samples():
    network = graph([(0, 1, 100, 100, 'trip'), (1, 2, 100, 101, 'trip')])
    belief = OrderedDurationBelief(network, public_occurrences=False)
    for cid, kind in [(0, 'departure'), (0, 'arrival'), (1, 'departure'), (1, 'arrival')]:
        belief.observe_connection_event(cid, kind, 7)
    for scene in belief.sample(PARAMETERS, np.random.default_rng(13), 20):
        assert (scene.events[0].departure, scene.events[0].arrival) == (107, 107)
        assert (scene.events[1].departure, scene.events[1].arrival) == (107, 108)
    nominal = belief.mean_events(PARAMETERS)
    assert nominal[0].arrival == nominal[1].departure == 107
    assert nominal[1].arrival == 108


def test_service_cancellation_recovers_and_q0_is_eliminated():
    belief = OrderedDurationBelief(graph(), public_occurrences=False)
    belief.observe_connection_event(0, 'departure', 0)
    belief.observe_connection_event(0, 'arrival', 0)
    belief.observe_connection_event(1, 'cancel', 0)
    belief.observe_connection_event(2, 'departure', 0)
    belief.observe_connection_event(2, 'arrival', 0)
    for scene in belief.sample(PARAMETERS, np.random.default_rng(14), 12):
        assert not scene.events[0].canceled
        assert scene.events[1].canceled
        assert not scene.events[2].canceled
    plain = OrderedDurationBelief(graph(), public_occurrences=False)
    q0 = {'A': dict(PARAMETERS['A'], cancel_rate=0.)}
    assert all(not states.any() for states in plain._law('trip', q0).canceled)


def test_partial_forecast_posterior_matches_brute_force_joint_probability():
    network = graph([(0, 1, 300, 300, 'trip')])
    belief = OrderedDurationBelief(network, public_occurrences=False)
    belief.observe_forecast(0, 'arrival', 317, 10)
    law = belief._law('trip', PARAMETERS)
    joint = (np.exp(law.initial + law.emissions[0])[:, None]
             * np.exp(law._transition(0)) * np.exp(law.emissions[1])[None, :])
    joint /= joint.sum()
    np.testing.assert_allclose(law.marginal(0), joint.sum(axis=1), atol=2e-15)
    np.testing.assert_allclose(law.marginal(1), joint.sum(axis=0), atol=2e-15)
    assert np.allclose(np.exp(law._transition(0)).sum(axis=1), 1.)


def test_mean_and_sample_use_the_same_ordered_clock_law():
    network = graph([(0, 1, 300, 300, 'trip')])
    belief = OrderedDurationBelief(network, public_occurrences=False, travel_sd=10/60, travel_drift=15/60)
    belief.observe_connection_event(0, 'departure', 0)
    params = {'A': dict(PARAMETERS['A'], cancel_rate=0.)}
    expected = belief.mean_events(params)[0].arrival - 300
    elapsed = np.array([scene.events[0].arrival - scene.events[0].departure
                        for scene in belief.sample(params, np.random.default_rng(35), 4000)])
    assert elapsed.min() >= 0
    assert elapsed.mean() == pytest.approx(expected, abs=.8)


def test_copy_keeps_other_trip_evidence_private_and_rebuilds_anchored_law():
    network = graph([(0, 1, 300, 360, 'one'), (0, 1, 900, 960, 'two')])
    belief = OrderedDurationBelief(network, public_occurrences=False)
    before = belief.mean_events(PARAMETERS)[1]
    original = belief._law('one', PARAMETERS)
    branch = belief.copy()
    branch.observe_connection_event(0, 'arrival', 7)
    assert branch.mean_events(PARAMETERS)[0].arrival == 367
    assert branch._law('one', PARAMETERS) is not original
    assert belief.mean_events(PARAMETERS)[1] == before
    assert not belief._known_times
    assert belief._transition_cache == branch._transition_cache == {}


def test_second_clock_contract_is_explicit():
    with pytest.raises(ValueError, match='minute must equal60'):
        OrderedDurationBelief(graph(), minute=1)
