"""Contract tests for shared-work causal planning baselines."""

import numpy as np
import pytest

from src.causal_planning_router import BudgetedPolicySearch, CausalPlanningRouter
from src.committed_transit import WAIT
from src.information_planner import ExactPlanner, FiniteScenarioModel, Outcome, Scenario
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import JourneyScenario, RegimeSchedule, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def test_balanced_rollout_uses_complete_common_rounds_and_respects_budget():
    risky, safe = ('select', 0), ('select', 1)
    actions = (WAIT, risky, safe)
    bank = (Scenario('early', .5, {'risky': 0}), Scenario('late', .5, {'risky': 10}))

    def transition(world, state, action):
        loss = {WAIT: 20, risky: world.factors['risky'], safe: 3}[action]
        return Outcome('done', (), loss, True)

    model = FiniteScenarioModel(bank, 'start', lambda state: actions if state == 'start' else (),
                                transition, 1)

    def plan(budget):
        return BudgetedPolicySearch(model, {}, lambda *_: WAIT, lambda *_: None,
                                    np.random.default_rng(0), budget, 20).plan('rollout')

    partial = plan(2)
    assert partial.transitions == 2
    assert partial.action == WAIT
    assert partial.value is None
    assert partial.common_rounds == 0
    complete = plan(6)
    assert complete.transitions == 6
    assert complete.common_rounds == 2
    assert complete.action == ExactPlanner(model).solve().action == safe
    assert complete.value == 3


def test_uct_shares_public_search_node_across_hidden_worlds():
    boarding = ('select', 0)
    worlds = (Scenario('a', .5, {'cost': 2}), Scenario('b', .5, {'cost': 4}))

    def actions(state):
        return (WAIT,) if state == 'start' else (WAIT, boarding)

    def transition(world, state, action):
        if state == 'start':
            return Outcome('same history', (), 1)
        return Outcome('done', (), 10 if action == WAIT else world.factors['cost'], True)

    model = FiniteScenarioModel(worlds, 'start', actions, transition, 2)
    planner = BudgetedPolicySearch(model, {}, lambda *_: WAIT, lambda *_: None,
                                   np.random.default_rng(0), 64, 20)
    result = planner.plan('uct')
    assert result.transitions == 64
    assert result.tree_nodes == 2
    assert set(planner.tree['same history']) == {WAIT, boarding}
    assert result.action == WAIT


def graph_with_rows(rows):
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=2))
    for row in rows:
        graph.add_connection(Connection(-1, *row))
    return graph


PRIORS = {'A': {'mean': 0., 'std': .5, 'cancel_rate': 0.}}


def test_full_menu_keeps_legal_early_vehicle_outside_initial_horizon_scan():
    rows = [('A', 'regular', 0, 2, 1, 12), ('A', 'early', 0, 2, 31, 32)]
    full = CausalPlanningRouter(graph_with_rows(rows), PRIORS, menu='full', max_time=30)
    fixed = CausalPlanningRouter(graph_with_rows(rows), PRIORS, menu='fixed', max_time=30)
    full.route(0, 2, 0)
    fixed.route(0, 2, 0)
    assert full.menu == {0, 1}
    assert fixed.menu == {0}


def test_material_menu_freezes_five_labels_and_keeps_their_trip_segments():
    rows = [('A', f'trip-{i}', 0, 2, 1, 8 + i) for i in range(7)]
    rows.append(('A', 'trip-0', 2, 1, 31, 32))
    router = CausalPlanningRouter(graph_with_rows(rows), PRIORS, menu='fixed_top5', max_time=30)
    router.route(0, 2, 0)
    assert router.menu == {0, 1, 2, 3, 4, 7}


@pytest.mark.parametrize('algorithm', ['rollout', 'uct'])
def test_full_router_uses_committed_multi_leg_execution_without_top_five(algorithm):
    rows = [('A', f'trip-{i}', 0, 1, 0, 5) for i in range(6)]
    rows.append(('A', 'connection', 1, 2, 7, 12))
    graph = graph_with_rows(rows)
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                             for c in graph.connections})
    router = CausalPlanningRouter(graph, PRIORS, algorithm=algorithm, max_time=30,
                                 samples=2, work_budget=96)
    result = simulate_bandit_journey(graph, router, 0, 2, 0, RegimeSchedule([(0, 'fixed')]),
                                    np.random.default_rng(0), max_time=30, scenario=scene)
    assert result.arrival_time == 12
    assert router.predictions[0]['root_actions'] == 7  # all six boardings plus WAIT
    assert all(p['work'] <= 96 for p in router.predictions)
