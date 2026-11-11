from types import SimpleNamespace

import numpy as np

from src.conditional_risk import (ConditionalStudentModel, StudentClockLaw,
                                  observable_group, retarget_event_chain, simultaneous_group_envelope)
from src.evolving_trip_belief import _ChainLaw


def test_conditional_student_recovers_a_heteroskedastic_training_law():
    from scipy.special import stdtrit
    class Marginal:
        def prior(self, c, kind):
            return SimpleNamespace(mean=c[f'scheduled_{kind}_seconds'], cancel_probability=.1)
    pairs = []
    for x in np.linspace(-200, 600, 17):
        for q in np.arange(1, 50) / 50:
            y = .3 * x + (30 + .15 * abs(x)) * stdtrit(3, q)
            before = {'arrival': {'status': 'REAL', 'time_seconds': 600 + x},
                      'scheduled_arrival_seconds': 600}
            after = {'departure': {'status': 'REAL', 'time_seconds': 900 + y},
                     'scheduled_departure_seconds': 900}
            pairs.append((before, after))
    model = ConditionalStudentModel(Marginal(), {'training': pairs})
    assert abs(model.beta - .3) < .01
    assert abs(model.base_scale - 30) < 4
    assert abs(model.scale_slope - .15) < .025
    incoming, outgoing = pairs[-1]
    law = model.predict(incoming, outgoing, 1100)
    other = dict(outgoing, departure={'status': 'REAL', 'time_seconds': 99999})
    assert model.predict(incoming, other, 1100) == law
    assert np.isclose(law.cdf(law.quantile(.05)), .05)
    assert law.board_probability(800) <= .9


def test_group_rank_score_uses_joint_date_max_and_exact_finite_sample_rank():
    def row(delay, actual):
        return {'incoming_delay': delay, 'actual': actual,
                'methods': {'m': {'lower': 0., 'upper': 2.}}}
    days = {str(d): [row(0, 1)] * 9 + [row(0, 100)] + [row(200, 2 + d)] for d in range(1, 10)}
    factor, detail = simultaneous_group_envelope(days, 'm')
    assert factor == 9
    assert detail['groups_by_date']['9']['late']['n'] == 1
    assert detail['groups_by_date']['9']['ordinary']['score'] == -1
    assert observable_group({'incoming_delay': 180}) == 'ordinary'


def test_marginal_replacement_changes_the_complete_joint_not_only_catch_p():
    transitions = np.log(np.array([[.9, .1], [.2, .8]]))
    root = _ChainLaw([np.array([0, 1]), np.array([1, 2])], np.log([.8, .2]),
                     lambda _: transitions, [np.zeros(2), np.zeros(2)],
                     [np.zeros(2, bool), np.zeros(2, bool)], (True,))
    changed, target = retarget_event_chain(root, 0, StudentClockLaw(1, .2, .99))
    np.testing.assert_allclose(changed.marginal(0), target, atol=1e-14)
    # Terminal cost10 at clock2 is evaluated from the retargeted joint.
    expected = target @ np.exp(transitions)[:, 1] * 10
    assert np.isclose(changed.marginal(1)[1] * 10, expected)
    assert not np.isclose(expected, root.marginal(1)[1] * 10)
    np.testing.assert_array_equal(root.emissions[0], np.zeros(2))


def test_clock_projection_preserves_cancel_mass_with_operating_support_holes():
    with np.errstate(divide='ignore'):
        initial = np.log([.4, 0, .3, .3])
    root = _ChainLaw([np.array([0, 1, 0, 1])], initial, lambda _: None,
                     [np.zeros(4)], [np.array([False, False, True, True])], ())
    changed, target = retarget_event_chain(root, 0, StudentClockLaw(1, .2, .01))
    np.testing.assert_allclose(changed.marginal(0), [.4, 0, .3, .3])
    np.testing.assert_allclose(target, [.4, 0, .3, .3])
