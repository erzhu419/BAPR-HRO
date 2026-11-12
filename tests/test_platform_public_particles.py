from types import SimpleNamespace

import numpy as np
import pytest

from src.evolving_trip_belief import _ChainLaw, _Event
from src.platform_public_particles import PlatformPublicParticles
from src.platform_transit import PlatformState


def fixture():
    slots = (_Event(0, 'departure', 0), _Event(0, 'arrival', 3),
             _Event(1, 'departure', 6), _Event(1, 'arrival', 9))
    clocks = [np.array([slot.scheduled, slot.scheduled+2, slot.scheduled+2]) for slot in slots]
    canceled = [np.array([False, False, True]) for _ in slots]
    initial = np.array([.2, .6, .2])
    transition = (
        np.array([[.75, .25, 0.], [.2, .8, 0.], [0., 0., 1.]]),
        np.array([[.4, .5, .1], [.1, .65, .25], [.1, .1, .8]]),
        np.array([[.6, .4, 0.], [.25, .75, 0.], [0., 0., 1.]]),
    )
    with np.errstate(divide='ignore'):
        steps = [np.log(p) for p in transition]
        law = _ChainLaw(clocks, np.log(initial), steps.__getitem__, [np.zeros(3) for _ in slots],
                        canceled, (True, True, True))
    graph = SimpleNamespace(connections=[SimpleNamespace(dep_stop=0, arr_stop=1),
                                        SimpleNamespace(dep_stop=1, arr_stop=2)])
    belief = SimpleNamespace(graph=graph, _positions={(slot.cid, slot.kind): ('trip', i)
                                                     for i, slot in enumerate(slots)},
                             _trips={'trip': slots}, _law=lambda trip, parameters: law)
    return PlatformPublicParticles(belief, {})


def test_silence_at_platform_excludes_only_operating_events():
    compiler = fixture()
    history = compiler.bind()
    state = PlatformState(0, 1, initialized=True, local_since=0)
    dep, arr, canceled = history.sample(state, 4000, np.random.default_rng(4), (0,))
    assert (dep == 2).all()  # The operating departure at 0 was absent.
    assert canceled.mean() == pytest.approx(.25, abs=.025)
    assert set(arr[:, 0]) == {3, 5}
    # The compiled inference object is shared, but execution histories are not.
    elsewhere = PlatformState(2, 1, initialized=True, local_since=0)
    other, _, _ = compiler.bind().sample(elsewhere, 4000, np.random.default_rng(4), (0,))
    assert (other == 0).mean() == pytest.approx(.2, abs=.025)


def test_future_draw_preserves_same_trip_evidence_and_current_stop_absence():
    history = fixture().bind()
    log = ((2, 0, 'departure'),)
    history.observe(PlatformState(0, 2, public_log=log, initialized=True, local_since=0))
    log += ((5, 0, 'arrival'),)
    state = PlatformState(1, 6, public_log=log, initialized=True, local_since=5)
    dep, arr, canceled = history.sample(state, 4000, np.random.default_rng(18), (1, 0))
    assert (dep[:, 1] == 2).all() and (arr[:, 1] == 5).all()
    assert (dep[:, 0] == 8).all()  # A departure at 6 would already be public.
    assert canceled[:, 0].mean() == pytest.approx(.25/(.65+.25), abs=.025)
    assert set(arr[:, 0]) == {9, 11}  # The anchored public clock leaves a nondegenerate future.
