from itertools import product

import numpy as np
import pytest

from src.service_state_model import (
    BetaRate, RouteCompletion, ServiceFact, ServiceStateModel, ServiceTrip,
    completion_label, fit_completion_parameters,
)


def parameters():
    return {'R': RouteCompletion(BetaRate(1, 19), BetaRate(1, 9), BetaRate(2, 3))}


def model():
    return ServiceStateModel((ServiceTrip('R', (0, 1)), ServiceTrip('R', (2, 3))), parameters())


def record(cid, trip, sequence, dep='REAL', arr='REAL', identity='same'):
    return dict(id=cid, route_id='R', trip_id=trip, dep_sequence=sequence, arr_sequence=sequence + 1,
        dep_stop=str(sequence), arr_stop=str(sequence + 1),
        departure={'status': dep, 'official_journey': trip},
        arrival={'status': arr, 'official_journey': trip if identity == 'same' else 'different'})


def test_integrated_joint_law_normalizes_and_retains_all_service_paths():
    law = model()
    masses = [law.path_probability(dict(zip(law.connections, path)))
              for path in product((True, False), repeat=4)]
    assert sum(masses) == pytest.approx(1.)
    assert all(mass > 0 for mass in masses)


def test_same_route_failure_updates_other_trip_without_resetting_the_joint_law():
    root = model().exact().condition({0: True, 2: True})
    assert root.completion_probability(3) == pytest.approx(.9)
    branch = root.condition({1: False})
    assert branch.model is root.model
    assert branch.worlds is root.worlds
    assert branch.completion_probability(3) == pytest.approx(9 / 11)
    assert root.completion_probability(3) == pytest.approx(.9)
    together = model().exact().condition({0: True, 2: True, 1: False})
    np.testing.assert_allclose(together.weights, branch.weights)
    np.testing.assert_allclose(branch.condition({1: False}).weights, branch.weights)


def test_recovery_after_a_canceled_segment_has_positive_mass():
    law = ServiceStateModel((ServiceTrip('R', (0, 1, 2)),), parameters())
    posterior = law.exact().condition({0: True, 1: False})
    assert posterior.completion_probability(2) == pytest.approx(.4)
    assert law.path_probability({0: True, 1: False, 2: True}) > 0


def test_unobserved_prefix_is_integrated_instead_of_rerooting_the_service_prior():
    law = ServiceStateModel((ServiceTrip('R', (0, 1, 2)),), parameters())
    posterior = law.exact().condition({1: True})
    assert 1. - posterior.completion_probability(2) < parameters()['R'].stop.mean
    direct = {path: weight for path, weight in zip(law.exact().worlds, law.exact().weights) if path[1]}
    expected = sum(weight for path, weight in direct.items() if not path[2]) / sum(direct.values())
    assert 1. - posterior.completion_probability(2) == pytest.approx(expected)


def test_future_service_receipt_cannot_condition_an_earlier_history():
    root = model().exact()
    facts = (ServiceFact(0, True, 10), ServiceFact(1, False, 30))
    before = root.condition_received(facts, 10)
    assert dict(before.evidence) == {0: True}
    after = before.condition_received(facts, 30)
    assert dict(after.evidence) == {0: True, 1: False}
    with pytest.raises(ValueError, match='Contradictory'):
        after.condition({1: True})


def test_linear_prior_sampler_matches_the_integrated_cross_trip_joint():
    law = model()
    posterior = law.exact()
    expected = sum(weight for path, weight in zip(posterior.worlds, posterior.weights)
                   if path[0] and not path[1] and path[2] and not path[3])
    scenes = law.sample(np.random.default_rng(41), 12000)
    actual = np.mean([row[0] and not row[1] and row[2] and not row[3] for row in scenes])
    assert abs(actual - expected) < .006
    with pytest.raises(TypeError):
        scenes[0][0] = False


def test_arrival_only_cancellation_is_completion_failure_without_invented_time():
    item = record(0, 'actual-trip', 1, dep='PROGNOSE', arr='CANCELED')
    label = completion_label(item)
    assert label.can_complete is False
    assert label.journey == 'actual-trip'
    assert not hasattr(label, 'failure_time') and not hasattr(label, 'received_at')
    assert completion_label(record(1, 'A', 1, arr='CANCELED', identity='different')).can_complete is None


def test_training_separates_initial_stop_and_restore_and_retains_unknowns():
    rows = [record(0, 'A', 1), record(1, 'A', 2), record(2, 'A', 3, arr='MISSING'),
            record(3, 'B', 1, dep='CANCELED', arr='CANCELED'), record(4, 'B', 2)]
    fitted, audit = fit_completion_parameters({'training-day': rows})
    row = audit['by_route']['R']
    assert row == {'initial_success': 1, 'operating_operating': 1, 'transition_unknown': 1,
                   'initial_failure': 1, 'failed_operating': 1}
    assert fitted['R'].initial_failure.mean == pytest.approx(.5)
    assert fitted['R'].stop.mean == pytest.approx(.25)
    assert fitted['R'].restore.mean == pytest.approx(.75)
