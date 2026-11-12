from types import SimpleNamespace

import pytest

from src.conditional_rescue_policy import ConditionalItineraries, DepartureScore
from src.interruptible_recourse import RecourseCursor
from src.platform_transit import PlatformState
from src.tail_rescue import TailRescuePlan
from src.through_wait_value import ThroughWaitValue


def score_table(rows):
    scores = ConditionalItineraries.__new__(ConditionalItineraries)
    scores.graph = SimpleNamespace(connections=[SimpleNamespace(trip_id=i) for i in range(4)])
    scores.deadline, scores.ready = 100, 5
    scores.rows = {cid: {t: DepartureScore(t, mass, value, 1., 1) for t, mass, value in events}
                   for cid, events in enumerate(rows)}
    return scores


def test_waiting_has_no_advance_knowledge_of_the_later_departure():
    scores = score_table([(), (), [(20, 1., 60)], [(30, .5, 35)]])
    state = PlatformState(0, 10)
    cursor = RecourseCursor(TailRescuePlan(()), 10)
    # Declining the first offer gives 67.5 in expectation, despite a chance
    # of a much better future arrival. Clairvoyance would give 47.5 instead.
    assert ThroughWaitValue(scores, (2, 3), margin=0).waiting_score(state, cursor) == 60
    assert ThroughWaitValue(scores, (2, 3), margin=7.5).waiting_score(state, cursor) == 67.5


def test_simultaneous_through_and_ab_events_use_the_policy_priority_and_margin():
    scores = score_table([[(20, .5, 80)], [(20, 1., 60)], [(20, .5, 50)], [(20, 1., 65)]])
    state = PlatformState(0, 10)
    cursor = RecourseCursor(TailRescuePlan((0,), 0, (1,)), 10, ready=10, expiry=20)
    value = ThroughWaitValue(scores, (2, 3), margin=10)
    # B cannot beat simultaneous A. The 50 offer ties B after the margin,
    # while both through offers clear A. Four equally likely outcomes.
    assert value.waiting_score(state, cursor) == (50 + 65 + 60 + 60) / 4
    assert ThroughWaitValue(scores, ()).waiting_score(state, cursor) == scores.waiting_score(state, cursor) == 70


def test_wait_uses_survival_and_removes_publicly_failed_or_visible_trips():
    scores = score_table([[(35, 1., 70)], (), [(10, .25, 50), (20, .25, 55)], [(15, 1., 30)]])
    cursor = RecourseCursor(TailRescuePlan((0,)), 0, ready=5, expiry=35)
    state = PlatformState(0, 10, public_log=((10, 3, 'published_cancel'),))
    value = ThroughWaitValue(scores, (2, 3), margin=0)
    assert value.waiting_score(state, cursor) == pytest.approx((55 + 2 * 70) / 3)
    seen = PlatformState(0, 10, public_log=((10, 2, 'departure'), (10, 3, 'published_cancel')))
    assert value.waiting_score(seen, cursor) == 70
