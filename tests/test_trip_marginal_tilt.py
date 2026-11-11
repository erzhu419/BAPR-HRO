import numpy as np
import pytest

from src.conditional_risk import StudentClockLaw
from src.evolving_trip_belief import EvolvingTripBelief, _ChainLaw
from src.transit_graph import TransitGraph, Stop, Connection
from src.trip_marginal_tilt import TripMarginalTilt


def setup(cancel=0.):
    graph = TransitGraph()
    for i in range(3):
        graph.add_stop(Stop(i, str(i)))
    graph.add_connection(Connection(0, 'R', 'trip', 0, 1, 600, 660))
    graph.add_connection(Connection(1, 'R', 'trip', 1, 2, 1200, 1260))
    base = EvolvingTripBelief(graph, minute=60, public_occurrences=False)
    pars = {'R': {'mean': 1., 'std': 1., 'cancel_rate': cancel}}
    return base, pars, TripMarginalTilt(base, pars, 0, StudentClockLaw(720, 35, .99))


def test_tilt_preserves_cancel_mass_and_samples_its_own_joint_mean():
    base, pars, changed = setup(.2)
    root = base._law('trip', pars)
    law = changed._law('trip', pars)
    mask = ~law.canceled[0]
    np.testing.assert_allclose(law.marginal(0), changed.target_marginal, atol=1e-13)
    assert np.isclose(law.marginal(0)[~mask].sum(), root.marginal(0)[~mask].sum())
    scenes = changed.sample(pars, np.random.default_rng(4), 3000)
    active = [scene.events[0] for scene in scenes if not scene.events[0].canceled]
    conditional_mean = changed.mean_events(pars)[0].departure
    assert abs(np.mean([event.departure for event in active]) - conditional_mean) < 5
    assert base._known_times == {}


def test_new_grid_anchor_does_not_change_the_frozen_prior():
    base, pars, changed = setup()
    original = changed._law('trip', pars)
    branch = changed.copy()
    branch.observe_connection_event(0, 'departure', 0)
    after = branch._law('trip', pars)
    for a, b in zip(after.times, original.times):
        np.testing.assert_array_equal(a, b)
    # Knowing departure makes its persistent likelihood ratio constant;
    # the future equals the original root prior conditioned on departure600.
    root = base._law('trip', pars)
    emission = list(root.emissions)
    emission[0] = np.where(root.times[0] == 600, 0., -np.inf)
    expected = _ChainLaw(root.times, root.initial, root._transition, emission, root.canceled, root.stochastic)
    np.testing.assert_allclose(after.marginal(3), expected.marginal(3), atol=1e-13)
    np.testing.assert_allclose(changed._law('trip', pars).marginal(0), original.marginal(0))
    assert changed.base._known_times == {}


def test_off_grid_evidence_requests_a_new_context_and_wait_target_is_not_overwritten():
    base, pars, changed = setup()
    branch = changed.copy()
    branch.observe_connection_event(0, 'departure', 1)
    with pytest.raises(ValueError, match='frozen planning grid'):
        branch._law('trip', pars)
    base.observe_wait(0, 600, 650)
    preserved = TripMarginalTilt(base, pars, 0, StudentClockLaw(720, 35, .99))
    assert not preserved.applied
    np.testing.assert_allclose(preserved._law('trip', pars).marginal(0), base._law('trip', pars).marginal(0))
    with pytest.raises(ValueError, match='fixes route parameters'):
        preserved.mean_events({'R': dict(pars['R'], mean=10.)})


def two_trip_setup():
    graph = TransitGraph()
    for i in range(3):
        graph.add_stop(Stop(i, str(i)))
    graph.add_connection(Connection(0, 'R', 'A', 0, 1, 600, 660))
    graph.add_connection(Connection(1, 'R', 'B', 0, 2, 1200, 1260))
    base = EvolvingTripBelief(graph, minute=60, public_occurrences=False)
    pars = {'R': {'mean': 1., 'std': 1., 'cancel_rate': .2}}
    return base, pars


def test_mapping_defines_one_common_posterior_and_cross_trip_conditioning_is_private():
    base, pars = two_trip_setup()
    targets = {0: StudentClockLaw(720, 35, 0), 1: StudentClockLaw(1320, 40, 1)}
    common = TripMarginalTilt(base, pars, targets)
    assert common.applied_cids == [0, 1] and not common.skipped_cids
    for cid in targets:
        trip, index = common._positions[cid, 'departure']
        np.testing.assert_allclose(common._law(trip, pars).marginal(index), common.projected_marginals[cid])
    unchanged = common.event_distribution(1, 'departure', pars)[1].copy()
    branch = common.copy()
    branch.observe_connection_event(0, 'departure', 0)
    np.testing.assert_array_equal(branch.event_distribution(1, 'departure', pars)[1], unchanged)
    assert not common.base._known_times and not base._known_times


def test_two_marginals_on_the_same_trip_are_not_claimed_compatible():
    base, pars, _ = setup()
    with pytest.raises(ValueError, match='one marginal target per trip'):
        TripMarginalTilt(base, pars, {0: StudentClockLaw(720, 35, 0), 1: StudentClockLaw(1320, 40, 0)})


def test_known_facts_and_zero_operating_support_have_explicit_skip_records():
    base, pars = two_trip_setup()
    base.observe_connection_event(0, 'departure', 0)
    base.observe_connection_event(1, 'cancel', 0)
    common = TripMarginalTilt(base, pars, {0: StudentClockLaw(720, 35, 0), 1: StudentClockLaw(1320, 40, 0)})
    assert common.skipped_cids == {0: 'known_departure', 1: 'known_cancellation'}
    all_cancel = {'R': dict(pars['R'], cancel_rate=1.)}
    empty = TripMarginalTilt(EvolvingTripBelief(base.graph, public_occurrences=False), all_cancel,
                              {0: StudentClockLaw(720, 35, 0)})
    assert empty.skipped_cids == {0: 'zero_operating_support'}


def test_empty_mapping_preserves_original_model_and_random_tape_draws():
    base, pars = two_trip_setup()
    unchanged = TripMarginalTilt(base, pars, {})
    a = unchanged.sample(pars, np.random.default_rng(0), 8)
    b = base.sample(pars, np.random.default_rng(0), 8)
    assert [dict(scene.events) for scene in a] == [dict(scene.events) for scene in b]


def test_projected_catch_probability_is_reported_separately_from_continuous_cdf():
    _, pars, changed = setup(.2)
    report = changed.projection_summary({0: 710}, deadline=2000)[0]
    law = changed._law('trip', pars)
    expected = law.marginal(0)[(~law.canceled[0]) & (law.times[0] >= 710)
                              & (law.times[0] <= 1430)].sum()
    assert np.isclose(report['projected_catch'], expected)
    assert abs(report['catch_difference']) > 1e-4
    assert changed.projection_summary({0: 710}, deadline=710)[0]['projected_catch'] == 0
