"""Joint tail G must integrate actual later adaptation, with separate streams."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v33.history import ConditionalEngine
from experiments.reconstruction_v33.joint import JointTailPlanner
from test_v33_planner import example


def setup():
    graph, model, base, source, second = example()
    model._law = lambda trip, parameters: base.laws[trip]
    model._trips = {}
    for (cid, kind), (trip, slot) in model._positions.items():
        model._trips.setdefault(trip, {})[slot] = SimpleNamespace(cid=cid, kind=kind)
    model._trips = {trip:[by_slot[i] for i in sorted(by_slot)] for trip,by_slot in model._trips.items()}
    protocol = dict(joint_tail_quadrature_worlds=32, joint_tail_seed=33001)
    return graph, model, source, second, protocol


def test_root_G_executes_conditional_later_vehicle_adaptation():
    graph, model, source, second, protocol = setup()
    store = dict(values={}, receipts=[])
    planner = JointTailPlanner(graph, model, {}, 100, source, 'optimal', store, protocol, 0)
    compiled = planner.compile_ride(source, ConditionalEngine(model, {}))
    platform = source.exits[0]
    paths = [p for ride in platform.rides.values() for p in ride.paths()]
    bank = TripWorlds(model, {}, paths, 32, 33001)
    # This decision observes second's arrival: early takes fast, late keeps
    # the second vehicle to the later hub. Locking the first exit would cost60.
    outcomes = np.where(bank.events[1, 'arrival'][0] == 10, 15., 40.)
    assert compiled.score(0) == pytest.approx(outcomes.mean())
    assert store['receipts'][0]['means'][0] == pytest.approx(outcomes.mean())
    assert store['receipts'][0]['standard_errors'][0] == pytest.approx(outcomes.std(ddof=1)/np.sqrt(32))


def test_root_controls_share_one_tail_integral_without_access_to_eval_world():
    graph, model, source, _, protocol = setup()
    store = dict(values={}, receipts=[])
    values = []
    for method in ('rolling_fixed', 'one_step_rollout', 'optimal'):
        planner = JointTailPlanner(graph, model, {}, 100, source, method, store, protocol, 0)
        values.append(planner.compile_ride(source, ConditionalEngine(model, {})).score(0))
    assert values == pytest.approx([values[0]]*3)
    assert len(store['receipts']) == 1
    assert store['receipts'][0]['seed'] == 33001
