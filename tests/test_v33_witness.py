"""Checks for the archived witness's public-history and physical scope."""

import numpy as np
import pytest

from experiments.reconstruction_v33.witness import observed_origin_history, seen_interval_evidence
from experiments.reconstruction_v32.conditional import UnsupportedHistory
from src.evolving_trip_belief import _ChainLaw


def single_slot_law():
    times = [np.array([0, 10, 20, 30, 0, 10, 20, 30])]
    canceled = [np.array([False]*4+[True]*4)]
    mass = np.array([.05, .15, .25, .35, .05, .05, .05, .05])
    return _ChainLaw(times, np.log(mass), None, [np.zeros(8)], canceled, ())


def test_seen_only_receipt_keeps_interval_instead_of_recovering_hidden_clock():
    trace = [dict(clock=29451, visible=[9450], seen=[9450, 9366]),
             dict(clock=29535, visible=[10241], seen=[10241, 9450, 9366])]
    history = {r['cid']: r for r in observed_origin_history(trace, [9450, 9366, 10241, 11133], 29400)}
    assert history[9450]['observed'] == 29451
    assert history[10241]['observed'] == 29535
    assert history[9366] == dict(cid=9366, kind='seen_interval', entry=29400, through=29451)
    assert history[11133] == dict(cid=11133, kind='nonoccurrence', entry=29400, through=29535)
    assert 'observed' not in history[9366]


def test_seen_interval_conditions_only_operating_occurrence_and_preserves_relative_mass():
    law = single_slot_law()
    conditioned, mass = seen_interval_evidence(law, 0, 10, 20)
    assert mass == pytest.approx(.4)
    assert conditioned.marginal(0) == pytest.approx([0, .375, .625, 0, 0, 0, 0, 0])
    assert law.marginal(0) == pytest.approx([.05, .15, .25, .35, .05, .05, .05, .05])


def test_unsupported_seen_interval_is_not_treated_as_an_absent_vehicle():
    with pytest.raises(UnsupportedHistory):
        seen_interval_evidence(single_slot_law(), 0, 11, 19)


def test_origin_history_does_not_extend_to_later_onboard_clocks():
    trace = [dict(clock=29535, visible=[10241], seen=[10241])]
    history = observed_origin_history(trace, [10241, 11133], 29400)
    # No onboard time parameter exists: a later arrival cannot accidentally
    # become additional evidence of nonoccurrence at the abandoned origin.
    unseen = next(row for row in history if row['cid'] == 11133)
    assert unseen['through'] == 29535
    assert unseen['through'] != 29720
