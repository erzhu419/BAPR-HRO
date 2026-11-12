import numpy as np
import pytest

from src.evolving_trip_belief import EvolvingTripBelief
from src.temporal_scan import sample_clocks, template_trip_map
from src.transit_graph import Connection, Stop, TransitGraph


PARAMETERS = {'route': {'mean': 1., 'std': 2., 'cancel_rate': .2}}


def network():
    graph = TransitGraph()
    for i in range(3):
        graph.add_stop(Stop(i, str(i), min_transfer_time=120))
    for trip, clock in (('one', 600), ('two', 1200)):
        graph.add_connection(Connection(0, 'route', trip, 0, 1, clock, clock + 240))
        graph.add_connection(Connection(0, 'route', trip, 1, 2, clock + 240, clock + 480))
    return graph


@pytest.mark.parametrize('step', [60, 20, 10])
def test_bounded_seconds_grid_retains_support_and_normalized_service_chain(step):
    belief = EvolvingTripBelief(network(), grid_step_seconds=step, travel_drift=.25)
    law = belief._law('one', PARAMETERS)
    assert belief.minute == 60
    assert belief.hard_support(0, 'departure', PARAMETERS) == (300, 4440)
    times, mass, canceled = belief.event_distribution(0, 'departure', PARAMETERS)
    assert np.array_equal(times, np.arange(300, 4441, step))
    assert mass.sum() == pytest.approx(1.)
    assert canceled == pytest.approx(.2)
    for i, log_transition in enumerate(law.transitions):
        probabilities = np.exp(log_transition)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
        assert not probabilities[law.times[i + 1][None, :] < law.times[i][:, None]].any()
    same_segment = np.exp(law.transitions[0])
    assert not same_segment[~law.canceled[0]][:, law.canceled[1]].any()
    dwell = np.exp(law.transitions[1])
    np.testing.assert_allclose(dwell[~law.canceled[1]][:, law.canceled[2]].sum(axis=1), .02)


def test_refining_only_one_trip_preserves_evidence_and_unselected_caches():
    base = EvolvingTripBelief(network(), public_occurrences=False)
    base.observe_connection_event(0, 'departure', 17)
    base.observe_wait(1, 900, 980)
    before = base._law('two', PARAMETERS)
    refined = base.with_grid_step(20, trips={'one'})
    assert refined._law('two', PARAMETERS) is before
    assert base.grid_step_for_trip('one') is None
    assert refined.grid_step_for_trip('one') == 20
    assert refined.grid_step_for_trip('two') is None
    clock, mass, canceled = refined.event_distribution(0, 'departure', PARAMETERS)
    assert clock[mass > 0].tolist() == [617]
    assert canceled == 0.
    clock, mass, _ = refined.event_distribution(1, 'departure', PARAMETERS)
    assert mass[(900 <= clock) & (clock <= 980)].sum() == 0.
    fine_draws = refined.sample(PARAMETERS, np.random.default_rng(19), 12)
    assert all(scene.events[0].departure == 617 for scene in fine_draws)


def test_template_grouping_respects_mixed_steps_without_moving_the_random_tape():
    base = EvolvingTripBelief(network())
    mixed = base.with_grid_step(20, trips={'one'})
    coarse_templates = template_trip_map(base, PARAMETERS)
    mixed_templates = template_trip_map(mixed, PARAMETERS)
    assert coarse_templates['two'] == ('one', 600)
    assert mixed_templates['two'] == ('two', 0)
    original = sample_clocks(base, PARAMETERS, np.random.default_rng(28), 40, templates=coarse_templates)
    refined = sample_clocks(mixed, PARAMETERS, np.random.default_rng(28), 40, templates=mixed_templates)
    for old, new in zip(original, refined, strict=True):
        np.testing.assert_array_equal(old[:, 2:], new[:, 2:])
    assert any(t % 60 for t in refined[0][:, 0])


def test_default_and_reverted_grid_use_the_original_discrete_weights():
    base = EvolvingTripBelief(network())
    integrated = base.with_grid_step(60)
    reverted = integrated.with_grid_step(None)
    times, mass, canceled = base.event_distribution(0, 'departure', PARAMETERS)
    np.testing.assert_array_equal(times, integrated.event_distribution(0, 'departure', PARAMETERS)[0])
    expected = np.exp(-.5 * (((times - 600) / 60 - 1.) / 2.) ** 2)
    np.testing.assert_allclose(mass, expected / expected.sum())
    assert canceled == pytest.approx(.2)
    assert not np.array_equal(mass, integrated.event_distribution(0, 'departure', PARAMETERS)[1])
    np.testing.assert_array_equal(mass, reverted.event_distribution(0, 'departure', PARAMETERS)[1])
    assert integrated._law('one', PARAMETERS)._transition(0) is not base._law('one', PARAMETERS)._transition(0)
