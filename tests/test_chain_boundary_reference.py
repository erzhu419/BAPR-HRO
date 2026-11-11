import numpy as np

from experiments.reconstruction_v5.chain_boundary_reference import (
    BoundaryChain, full_marginals, ordered_service_feasible)


def chain(length=12):
    initial = np.log(np.array([.6, .3, .1]))
    kernel = np.log(np.array([[.8, .15, .05], [.1, .8, .1], [.05, .15, .8]]))
    return initial, [kernel] * (length - 1), [np.zeros(3) for _ in range(length)]


def test_exact_state_separator_and_neutral_future_match_full_smoothing():
    initial, kernels, emissions = chain()
    emissions[6] = np.array([-np.inf, 0., -np.inf])
    query = BoundaryChain(initial, kernels, emissions).marginal(9)
    np.testing.assert_allclose(query.probabilities, full_marginals(initial, kernels, emissions)[9], atol=1e-14)
    assert (query.left_separator, query.forward_steps, query.backward_steps) == (6, 3, 0)


def test_future_informative_offpath_emission_cannot_be_discarded():
    initial, kernels, emissions = chain()
    emissions[-1] = np.log(np.array([.001, .01, 1.]))
    query = BoundaryChain(initial, kernels, emissions).marginal(4)
    np.testing.assert_allclose(query.probabilities, full_marginals(initial, kernels, emissions)[4], atol=1e-14)
    no_evidence = BoundaryChain(initial, kernels, [np.zeros(3)] * len(emissions)).marginal(4)
    assert np.max(abs(query.probabilities - no_evidence.probabilities)) > .01
    assert query.backward_steps == 7


def test_service_status_alone_is_not_a_full_separator():
    initial, kernels, emissions = chain()
    emissions[6] = np.array([0., 0., -np.inf])
    query = BoundaryChain(initial, kernels, emissions).marginal(9)
    assert query.left_separator is None
    np.testing.assert_allclose(query.probabilities, full_marginals(initial, kernels, emissions)[9], atol=1e-14)


def test_substochastic_future_cannot_use_zero_backward_shortcut():
    initial, kernels, emissions = chain()
    with np.errstate(divide='ignore'):
        kernels[-1] = np.log(np.array([[.8, .15, .05], [.1, .8, .1], [0., 0., 0.]]))
    query = BoundaryChain(initial, kernels, emissions).marginal(8)
    assert query.backward_steps == 3
    np.testing.assert_allclose(query.probabilities, full_marginals(initial, kernels, emissions)[8], atol=1e-14)


def test_linear_feasibility_detects_clock_order_and_service_support():
    clocks = [np.array([0, 1, 0, 1]), np.array([1, 2, 1, 2]), np.array([2, 3, 2, 3])]
    states = [np.array([False, False, True, True])] * 3
    emissions = [np.array([0., 0., -np.inf, -np.inf]), np.zeros(4),
                 np.array([-np.inf, -np.inf, 0., 0.])]
    identity = np.eye(2, dtype=bool)
    assert ordered_service_feasible(clocks, states, emissions, [True, False], [identity, identity])[0] is False
    assert ordered_service_feasible(clocks, states, emissions, [True, False], [identity, np.ones((2,2),bool)]) == (True, 12)
    bad_order = [np.array([5, 6, 5, 6]), *clocks[1:]]
    assert ordered_service_feasible(bad_order, states, emissions, [True, False], [identity, identity])[0] is False


def test_adaptive_past_anchor_metadata_remains_needed_after_an_exact_separator():
    from src.evolving_trip_belief import EvolvingTripBelief
    from src.transit_graph import Connection, Stop, TransitGraph
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop)))
    for cid in range(3):
        graph.add_connection(Connection(cid, 'R', 'T', cid, cid + 1, cid * 240, cid * 240 + 120))
    parameters = {'R': {'mean': 0., 'std': 2., 'cancel_rate': 0.}}
    first = EvolvingTripBelief(graph, minute=60, public_occurrences=False)
    first.observe_connection_event(1, 'arrival', 0)  # exact separator at360
    second = first.copy()
    second.observe_connection_event(0, 'departure', 1)  # earlier off-grid anchor
    law1, law2 = first._law('T', parameters), second._law('T', parameters)
    index = first._positions[2, 'departure'][1]
    assert set(law1.times[index]) != set(law2.times[index])
    mean1 = law1.times[index] @ law1.marginals[index]
    mean2 = law2.times[index] @ law2.marginals[index]
    assert abs(mean1 - mean2) > 1e-6


def test_two_clock_feasibility_matches_dense_support_propagation():
    rng = np.random.default_rng(119)
    for repetition in range(80):
        times = [np.tile(np.arange(i, i + 4), 2) for i in range(5)]
        states = [np.repeat([False, True], 4)] * 5
        emissions = [np.where(rng.random(8) < .3, -np.inf, 0.) for _ in range(5)]
        steps = [rng.random((2, 2)) > .3 for _ in range(4)]
        initial = rng.random(2) > .3
        reachable = np.array([initial[int(s)] for s in states[0]]) & np.isfinite(emissions[0])
        for i in range(1, 5):
            kernel_support = ((times[i - 1][:, None] <= times[i][None, :]) &
                              steps[i - 1][states[i - 1].astype(int)[:, None],
                                           states[i].astype(int)[None, :]])
            reachable = (reachable[:, None] & kernel_support).any(axis=0) & np.isfinite(emissions[i])
        assert ordered_service_feasible(times, states, emissions, initial, steps)[0] == bool(reachable.any())
