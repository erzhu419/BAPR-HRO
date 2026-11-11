"""Parity and deferred-inference regressions for the city belief runtime."""
import numpy as np
import pytest

from experiments.reconstruction_v5.runtime_reference_evolving import EvolvingTripBelief as ReferenceBelief
from src.evolving_trip_belief import EvolvingTripBelief
from src.transit_graph import Connection, Stop, TransitGraph


def network():
    graph = TransitGraph()
    for sid in range(4):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=120))
    for cid in range(3):
        graph.add_connection(Connection(cid, 'A', 'A', cid, cid + 1,
                                        600 + cid * 300, 840 + cid * 300))
    graph.add_connection(Connection(3, 'B', 'B', 0, 3, 650, 1400))
    return graph


def parameters(cancel):
    return {route: {'mean': 1., 'std': 2., 'cancel_rate': cancel} for route in ('A', 'B')}


def long_network():
    graph = TransitGraph()
    for sid in range(31):
        graph.add_stop(Stop(sid, str(sid)))
    for cid in range(30):
        graph.add_connection(Connection(cid, 'A', 'A', cid, cid + 1,
                                        600 + cid * 300, 840 + cid * 300))
    return graph


@pytest.mark.parametrize('cancel', (0., .1, 1.))
@pytest.mark.parametrize('observed', (False, True))
def test_deferred_bank_matches_eager_reference_and_rng_state(cancel, observed):
    params = parameters(cancel)
    models = [cls(network(), public_occurrences=False) for cls in (ReferenceBelief, EvolvingTripBelief)]
    if observed:
        for model in models:
            if cancel == 1:
                model.observe_connection_event(0, 'cancel', 0)
            else:
                model.observe_connection_event(0, 'departure', 37)
                model.observe_connection_event(0, 'arrival', 102)
                model.observe_wait(1, 1000, 1100)
            model.observe_forecast(2, 'arrival', 1540, 75)
    for cid in range(4):
        for kind in ('departure', 'arrival'):
            old, new = [m.event_distribution(cid, kind, params) for m in models]
            np.testing.assert_array_equal(new[0], old[0])
            np.testing.assert_allclose(new[1], old[1], rtol=1e-12, atol=1e-14)
            assert new[2] == pytest.approx(old[2], abs=1e-14)
    for seed in (0, 83, 20230726):
        old_rng, new_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        old = models[0].sample(params, old_rng, 8)
        new = models[1].sample(params, new_rng, 8)
        # Reversing first access exercises fixed worlds independent of evaluation order.
        for i in range(7, -1, -1):
            assert dict(new[i].events) == dict(old[i].events)
        np.testing.assert_array_equal(new_rng.random(10), old_rng.random(10))


def test_only_requested_trips_are_realized_and_public_history_is_frozen():
    model = EvolvingTripBelief(network(), public_occurrences=False)
    params = parameters(0.)
    bank = model.sample(params, np.random.default_rng(7), 4)
    nominal = model.mean_events(params)
    assert len(bank[0].events) == len(nominal) == 4
    assert all(memo.law is None for memo in model._cache.values())
    initial = nominal[0]
    assert model._cache['A'].law is not None
    assert model._cache['B'].law is None
    first_event = bank[0].events[0]
    branch = model.copy()
    branch.observe_connection_event(0, 'arrival', 900)
    assert branch.mean_events(params)[0].arrival == 1740
    assert not model._known_times
    params['A']['mean'] = 20
    model.observe_connection_event(0, 'arrival', 800)
    assert nominal[0] is initial
    assert nominal[1].departure < 1640
    assert bank[0].events[0] == first_event
    assert model._cache['B'].law is None
    with pytest.raises(TypeError):
        bank[0].events[0] = first_event


def test_unrealized_trip_keeps_its_original_evidence_and_parameters():
    params = parameters(.1)
    model = EvolvingTripBelief(network(), public_occurrences=False)
    reference = ReferenceBelief(network(), public_occurrences=False)
    expected = reference.sample(params, np.random.default_rng(19), 4)
    bank = model.sample(params, np.random.default_rng(19), 4)
    model.observe_connection_event(3, 'departure', 900)
    params['B']['mean'] = 30
    model.mean_events(params)[3]
    for i in range(4):
        assert bank[i].events[3] == expected[i].events[3]
        assert dict(bank[i].events) == dict(expected[i].events)


def test_parameter_change_and_branch_revision_do_not_reuse_a_different_law():
    model = EvolvingTripBelief(network(), public_occurrences=False)
    no_cancel = parameters(0.)
    original_view = model.mean_events(no_cancel)
    first = original_view[0]
    assert len(model._law('A', no_cancel).times[0]) == 70
    with_cancel = parameters(.2)
    second = model.mean_events(with_cancel)[0]
    assert len(model._law('A', with_cancel).times[0]) == 140
    assert second.cancel_probability > 0
    expected = ReferenceBelief(network(), public_occurrences=False).mean_events(no_cancel)[2]
    assert original_view[2].cancel_probability == 0
    assert original_view[2].departure == pytest.approx(expected.departure)
    assert model.mean_events(no_cancel)[0] == first
    branch = model.copy()
    branch.observe_wait(0, 600, 720)
    assert branch._cache['A'] is not model._cache['A']
    assert branch._cache['B'] is model._cache['B']


def test_flat_nominal_cache_is_shared_only_for_the_same_public_history_and_parameters():
    model = EvolvingTripBelief(network(), public_occurrences=False)
    params = parameters(.1)
    original = model.mean_events(params)
    base_a, base_b = original[0], original[3]
    assert model.mean_events(params)[0] is base_a
    candidate = model.copy()
    candidate.observe_connection_event(0, 'arrival', 200)
    changed = candidate.mean_events(params)
    assert changed[0].arrival == 1040
    assert changed[3] is base_b
    assert original[0] is base_a
    other = model.copy()
    other.observe_connection_event(0, 'arrival', 300)
    assert other.mean_events(params)[0].arrival == 1140
    assert changed[0].arrival == 1040
    different_parameters = parameters(.1)
    different_parameters['A']['mean'] = 20
    updated = model.mean_events(different_parameters)
    assert updated[0].departure > base_a.departure
    assert updated[3] is base_b
    assert original[1].departure < updated[1].departure


@pytest.mark.parametrize('operation', ('sample', 'mean_events'))
def test_zero_support_public_history_still_fails_before_returning_a_lazy_view(operation):
    model = EvolvingTripBelief(network(), public_occurrences=False)
    model.observe_connection_event(0, 'cancel', 0)
    with pytest.raises(ValueError, match='no ordered trip event path'):
        if operation == 'sample':
            model.sample(parameters(0.), np.random.default_rng(0), 4)
        else:
            model.mean_events(parameters(0.))


def test_hard_support_contains_off_grid_anchor_and_ordered_conditional_support():
    model = EvolvingTripBelief(network(), public_occurrences=False)
    params = parameters(.1)
    model.observe_connection_event(0, 'arrival', 7000)
    model.observe_forecast(2, 'departure', -10000, 10000)
    for cid in range(3):
        for kind in ('departure', 'arrival'):
            bounds = model.hard_support(cid, kind, params)
            clocks, weights, _ = model.event_distribution(cid, kind, params)
            positive = clocks[weights > 0]
            assert bounds is not None
            assert min(positive) >= bounds[0] and max(positive) <= bounds[1]
    assert model.hard_support(1, 'departure', params)[1] > network().connections[1].dep_time + 64 * 60
    assert model.hard_support(1, 'departure', params)[0] >= 7840
    canceled = model.copy()
    canceled.observe_connection_event(1, 'cancel', 0)
    assert canceled.hard_support(1, 'departure', params) is None


def test_cold_query_builds_only_its_forward_prefix_without_backward_messages():
    model = EvolvingTripBelief(long_network(), public_occurrences=False)
    model.event_distribution(10, 'departure', parameters(.1))
    law = model._law('A', parameters(.1))
    assert len(law._forward) == 21
    assert sum(matrix is not None for matrix in law._transitions) == 20
    assert all(len(values) == 1 for values in law._backward.values())


def test_joint_separator_queries_past_and_sampling_match_global_reference():
    params = parameters(.1)
    reference = ReferenceBelief(long_network(), public_occurrences=False)
    model = EvolvingTripBelief(long_network(), public_occurrences=False)
    for belief in (reference, model):
        belief.observe_connection_event(27, 'arrival', 37)
    model.event_distribution(28, 'departure', params)
    law = model._law('A', params)
    assert len(law._forward) == 1
    assert len(law._conditional_forward[55]) == 2
    assert sum(matrix is not None for matrix in law._transitions) == 1
    for cid, kind in ((28, 'arrival'), (4, 'arrival'), (27, 'arrival'), (29, 'arrival')):
        old = reference.event_distribution(cid, kind, params)
        new = model.event_distribution(cid, kind, params)
        np.testing.assert_array_equal(new[0], old[0])
        np.testing.assert_allclose(new[1], old[1], atol=2e-13, rtol=2e-12)
        assert new[2] == pytest.approx(old[2], abs=2e-13)
    # Local rescaling must not change the global FFBS messages or random tape.
    original = reference.sample(params, np.random.default_rng(731), 4)
    optimized = model.sample(params, np.random.default_rng(731), 4)
    assert [dict(s.events) for s in optimized] == [dict(s.events) for s in original]


def test_future_soft_forecast_is_retained_in_the_query_backward_boundary():
    params = parameters(.1)
    reference = ReferenceBelief(long_network(), public_occurrences=False)
    model = EvolvingTripBelief(long_network(), public_occurrences=False)
    for belief in (reference, model):
        belief.observe_forecast(29, 'arrival', 11000, 30)
    old = reference.event_distribution(10, 'departure', params)
    new = model.event_distribution(10, 'departure', params)
    np.testing.assert_allclose(new[1], old[1], atol=2e-13, rtol=2e-12)
    law = model._law('A', params)
    assert not law._exact
    assert len(law._backward[59]) == 40


def test_dead_future_transition_rows_are_not_treated_as_a_neutral_suffix():
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid)))
    graph.add_connection(Connection(0, 'A', 'A', 0, 1, 600, 900))
    graph.add_connection(Connection(1, 'A', 'A', 1, 2, 850, 1000))
    params = {'A': {'mean': 63., 'std': 3., 'cancel_rate': .1}}
    reference = ReferenceBelief(graph, public_occurrences=False)
    model = EvolvingTripBelief(graph, public_occurrences=False)
    old = reference.event_distribution(0, 'departure', params)
    new = model.event_distribution(0, 'departure', params)
    np.testing.assert_allclose(new[1], old[1], atol=2e-13, rtol=2e-12)
    law = model._law('A', params)
    assert not law.stochastic[1]
    assert len(law._backward[2]) == 3


def test_impossible_ordered_evidence_fails_before_allocating_dense_kernels():
    model = EvolvingTripBelief(network(), public_occurrences=False)
    model.observe_connection_event(0, 'arrival', 102)
    model.observe_connection_event(1, 'departure', 0)
    with pytest.raises(ValueError, match='no ordered trip event path'):
        model.mean_events(parameters(.1))
    assert not model._transition_cache


def test_known_operating_service_without_an_exact_clock_is_not_a_point_separator():
    model = EvolvingTripBelief(network(), public_occurrences=False)
    model.observe_attempt(0, 'missed', 1200)
    model.event_distribution(1, 'departure', parameters(.1))
    assert not model._law('A', parameters(.1))._exact


def test_linear_production_guard_matches_dense_two_service_reachability():
    from src.evolving_trip_belief import _ordered_path_exists
    rng = np.random.default_rng(119)
    for _ in range(80):
        times = [np.tile(np.arange(i, i + 4), 2) for i in range(5)]
        states = [np.repeat([False, True], 4)] * 5
        emissions = [np.where(rng.random(8) < .3, -np.inf, 0.) for _ in range(5)]
        steps = [rng.random((2, 2)) > .3 for _ in range(4)]
        initial_states = rng.random(2) > .3
        initial = np.where(initial_states[states[0].astype(int)], 0., -np.inf)
        reachable = np.isfinite(initial + emissions[0])
        for i in range(1, 5):
            support = ((times[i - 1][:, None] <= times[i][None, :]) &
                       steps[i - 1][states[i - 1].astype(int)[:, None], states[i].astype(int)[None, :]])
            reachable = (reachable[:, None] & support).any(axis=0) & np.isfinite(emissions[i])
        assert _ordered_path_exists(times, states, emissions, initial, steps) == bool(reachable.any())


def test_lazy_kernel_factory_does_not_retain_the_belief_or_its_public_history():
    import gc
    import weakref
    model = EvolvingTripBelief(network(), public_occurrences=False)
    model.observe_connection_event(0, 'arrival', 37)
    law = model._law('A', parameters(.1))
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None
    assert np.isfinite(law.marginal(3)).all()
