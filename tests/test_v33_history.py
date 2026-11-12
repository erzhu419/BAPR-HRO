from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v33.history import ConditionalEngine
from src.evolving_trip_belief import _ChainLaw


def engine():
    base = np.array([0., 10., 20., 30., 40., 50., 0.])
    times = [base, base+60, base+80]
    canceled = [np.array([False]*6+[True]) for _ in times]
    with np.errstate(divide='ignore'):
        transition = np.log(np.eye(7))
        law = _ChainLaw(times, np.log(np.ones(7)/7), lambda i: transition,
                        [np.zeros(7) for _ in times], canceled, (True, True))
    belief = SimpleNamespace(_positions={(1,'departure'):('a',0), (1,'arrival'):('a',1),
        (2,'departure'):('a',2), (3,'departure'):('b',0)}, _law=lambda trip, params:law)
    return ConditionalEngine(belief, {}), law


def test_evidence_survives_onboard_motion_and_updates_the_same_downstream_vehicle():
    history, base = engine()
    history.observe(1, 'arrival', 75)
    assert history.law('a').marginal(2) == pytest.approx([0., .5, .5, 0., 0., 0., 0.])
    # At a later platform, no departure2 through95 excludes its90 atom.
    history.observe_platform((2,), 85, 95, {})
    assert history.law('a').marginal(2) == pytest.approx([0., 0., 1., 0., 0., 0., 0.])
    assert history.law('a').times is base.times
    assert history.law('a')._transition(0) is base._transition(0)
    assert len(history.public_state()) == 2


def test_leaving_a_platform_does_not_extend_its_nonoccurrence_evidence():
    history, _ = engine()
    history.observe_platform((1,), 10, 20, {})
    history.observe_platform((3,), 100, 110, {})  # Another visit to another vehicle.
    assert history.law('a').marginal(0)[3] > 0  # Departure30 remained unobserved.
    history.observe_platform((1,), 50, 50, {})
    assert history.public_state()[0]['windows'] == ((10., 20.), (50., 50.))
    assert history.law('a').marginal(0) == pytest.approx([.25, 0., 0., .25, .25, 0., .25])


def test_receipt_replaces_same_slot_nonoccurrence_and_is_idempotent():
    history, _ = engine()
    history.observe_platform((1,), 0, 10, {})
    history.observe_platform((1,), 0, 12, {1:12})
    # Restore the adjacent10 atom for the declared off-grid projection.
    assert history.law('a').marginal(0) == pytest.approx([0., .8, .2, 0., 0., 0., 0.])
    signature, cached = history.signature(), history.law('a')
    assert not history.observe(1, 'departure', 12)['changed']
    history.observe_platform((1,), 50, 60, {})
    assert history.signature() == signature and history.law('a') is cached


def test_clones_preserve_other_slots_and_scope_message_reuse_by_relevant_trip():
    history, base = engine()
    history.observe(1, 'departure', 10)
    a = history.law('a')
    branch = history.clone()
    branch.observe(3, 'departure', 20)
    assert branch.signature(('a',)) == history.signature(('a',))
    assert branch.signature() != history.signature()
    assert branch.law('a') is a
    branch.observe(1, 'arrival', 70)
    assert len(branch.public_state()) == 3 and len(history.public_state()) == 1
    assert branch.as_belief()._law('a', {}) is branch.law('a')
    assert history.law('a') is a and history.belief._law('a', {}) is base


def test_platform_receipt_cannot_read_the_future_or_before_entry():
    history, _ = engine()
    for observed in (5, 30):
        with pytest.raises(ValueError, match='public visit prefix'):
            history.observe_platform((1,), 10, 20, {1:observed})
