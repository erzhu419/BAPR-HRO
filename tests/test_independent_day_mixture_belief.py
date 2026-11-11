"""Same marginal law, different offset coupling and cross-trip learning."""
import numpy as np

from src.independent_day_mixture_belief import IndependentDayMixtureBelief
from src.joint_day_belief import JointDayBelief
from src.transit_graph import Connection
from tests.test_joint_day_belief import graph, PARAMETERS


def test_initial_marginals_match_shared_and_observation_does_not_change_other_trip():
    shared = JointDayBelief(graph(), public_occurrences=False)
    separate = IndependentDayMixtureBelief(graph(), public_occurrences=False)
    for cid in (0, 1):
        for kind in ('departure', 'arrival'):
            a, b = shared.event_distribution(cid, kind, PARAMETERS), separate.event_distribution(cid, kind, PARAMETERS)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_allclose(a[1], b[1], atol=1e-15)
            assert a[2] == b[2]
    prior_b = separate.event_distribution(1, 'departure', PARAMETERS)
    separate.observe_connection_event(0, 'arrival', 180)
    after_b = separate.event_distribution(1, 'departure', PARAMETERS)
    np.testing.assert_array_equal(prior_b[1], after_b[1])
    shared.observe_connection_event(0, 'arrival', 180)
    np.testing.assert_allclose(separate.trip_posterior(0, PARAMETERS)[1], shared.day_posterior(PARAMETERS)[1])
    assert not np.allclose(separate.trip_posterior(0, PARAMETERS)[1], separate.prior_weights)
    np.testing.assert_allclose(separate.trip_posterior(1, PARAMETERS)[1], separate.prior_weights)


def test_trip_offset_is_persistent_and_independent_between_trips():
    network = graph()
    network.add_connection(Connection(2, 'R', 'A', 1, 0, 780, 900))
    model = IndependentDayMixtureBelief(network, public_occurrences=False, travel_sd=.05, dwell_sd=.05,
                                      deltas_seconds=[-180, 180], prior_weights=[.5, .5])
    worlds = model.sample(PARAMETERS, np.random.default_rng(23), 1500)
    values = np.array([[world.events[cid].delay for cid in (0, 1, 2)] for world in worlds])
    assert abs(np.corrcoef(values[:, :2].T)[0, 1]) < .1
    assert np.corrcoef(values[:, [0, 2]].T)[0, 1] > .8
    assert all(world.events[0].arrival <= world.events[2].departure <= world.events[2].arrival for world in worlds)


def test_copy_is_private_and_wait_with_rounded_cancel_is_finite():
    model = IndependentDayMixtureBelief(graph(), public_occurrences=False)
    branch = model.copy()
    branch.observe_connection_event(0, 'arrival', 180)
    np.testing.assert_allclose(model.trip_posterior(0, PARAMETERS)[1], model.prior_weights)
    branch.observe_wait(1, 600, 1320)
    parameters = {'R': dict(PARAMETERS['R'], cancel_rate=.1)}
    _, mass, _ = branch.event_distribution(1, 'departure', parameters)
    assert np.isfinite(mass).all()
