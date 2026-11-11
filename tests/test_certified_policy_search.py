from fractions import Fraction

import numpy as np
import pytest

from src.causal_planning_router import BudgetedPolicySearch
from src.certified_policy_search import CertifiedPolicySearch
from src.committed_transit import WAIT
from src.information_planner import ExactPlanner, FiniteScenarioModel, Outcome, Scenario


def one_step_model():
    risky, safe, dominated = ('select', 0), ('select', 1), ('select', 2)
    actions = (WAIT, risky, safe, dominated)
    worlds = (Scenario('a', .75, {'risky': 0}), Scenario('b', .25, {'risky': 12}))

    def transition(world, state, action):
        loss = {WAIT: 20, risky: world.factors['risky'], safe: 2, dominated: 18}[action]
        return Outcome('done', (), loss, True)

    model = FiniteScenarioModel(worlds, 'start', lambda s: actions if s == 'start' else (),
                                transition, 1)
    return model, {WAIT: 0, risky: 0, safe: 2, dominated: 18}


@pytest.mark.parametrize('budget', [1, 2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize('witness_first', [False, True])
def test_partial_world_upper_bounds_and_pruning_cover_exact_values(budget, witness_first):
    model, bounds = one_step_model()
    planner = CertifiedPolicySearch(model, {}, lambda *_: WAIT, lambda *_: None,
                                     np.random.default_rng(0), budget, 20)
    def after_witness(upper):
        assert planner.work > 0
        assert upper == 20  # this default policy has a constant deadline loss
        return bounds

    result = planner.plan_certified(after_witness if witness_first else bounds, remaining_cap=20)
    exact = ExactPlanner(model).solve()
    assert result.lower <= exact.value <= result.upper
    assert exact.action_values[result.planning.action] <= result.upper
    assert result.planning.transitions <= budget
    assert all(exact.action_values[a] > exact.value for a in result.removed)
    assert all(result.action_upper[a] >= exact.action_values[a] for a in bounds)
    if budget >= 6:
        assert result.planning.action == exact.action == ('select', 1)
        assert result.upper == 2
        assert result.lower == 0 and result.gap == 2
        assert ('select', 2) in result.removed


def test_partial_budget_improves_witness_without_a_complete_common_round():
    model, lower = one_step_model()
    args = (model, {}, lambda *_: WAIT, lambda *_: None, np.random.default_rng(0), 3, 20)
    baseline = BudgetedPolicySearch(*args).plan('rollout')
    candidate = CertifiedPolicySearch(*args).plan_certified(lower, remaining_cap=20)
    assert baseline.action == WAIT and baseline.common_rounds == 0
    assert candidate.planning.action == ('select', 0)
    # Only world a of this candidate is complete. World's b worst-case loss
    # has probability 1/4, so U=5, whereas its fully evaluated loss is 3.
    assert candidate.upper == 5
    assert candidate.evaluated_worlds[('select', 0)] == 1


def test_tight_lower_bounds_can_certify_without_simulating_every_action():
    model, _ = one_step_model()
    exact = ExactPlanner(model).solve()
    search = CertifiedPolicySearch(model, {}, lambda *_: ('select', 1), lambda *_: None,
                                    np.random.default_rng(0), 8, 20)
    result = search.plan_certified(exact.action_values, 20)
    assert result.lower == result.upper == 2
    assert result.planning.transitions == 2
    assert set(result.removed) == {WAIT, ('select', 0), ('select', 2)}


def test_tail_actions_share_public_history_across_worlds():
    a, b = ('select', 0), ('select', 1)
    worlds = (Scenario('a', .5, {'best': a}), Scenario('b', .5, {'best': b}))
    requests = []

    def tail(state, belief, actions):
        requests.append(state)
        return a

    def transition(world, state, action):
        if state == 'root':
            return Outcome('shared', (), 1)
        return Outcome('done', (), 0 if action == world.factors['best'] else 8, True)

    model = FiniteScenarioModel(worlds, 'root',
                                lambda s: (WAIT,) if s == 'root' else ((a, b) if s == 'shared' else ()),
                                transition, 2)
    search = CertifiedPolicySearch(model, {}, tail, lambda *_: None, np.random.default_rng(0), 8, 10)
    result = search.plan_certified({WAIT: 1}, 10)
    assert requests.count('shared') == 1
    assert result.upper == 5
    assert result.lower == 1  # clairvoyance cannot be called a causal value
    assert ExactPlanner(model).solve().value == 5


def test_terminal_root_bound_uses_terminal_cost():
    model = FiniteScenarioModel((Scenario('a', 1., {}),), 'done', lambda _: (),
                                lambda *_: None, 1, lambda *_: 0)
    search = CertifiedPolicySearch(model, {}, lambda *_: WAIT, lambda *_: None,
                                    np.random.default_rng(0), 8, 10)
    result = search.plan_certified({}, 10)
    assert result.lower == result.upper == Fraction(0)


def test_certified_router_runs_same_trip_and_transfer_physics():
    from src.certified_router import CertifiedPlanningRouter
    from src.simulate_bandit import simulate_bandit_journey
    from src.simulator import JourneyScenario, RegimeSchedule, VehicleEvent
    from src.transit_graph import Connection, Stop, TransitGraph

    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=2))
    for row in [('A', 'feeder', 0, 1, 1, 4), ('A', 'feeder', 1, 2, 4, 7),
                ('A', 'other', 1, 2, 5, 6), ('A', 'slow', 0, 2, 2, 15)]:
        graph.add_connection(Connection(-1, *row))
    world = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                             for c in graph.connections})
    router = CertifiedPlanningRouter(graph, {'A': {'mean': 0., 'std': .25, 'cancel_rate': 0.}},
                                      samples=2, work_budget=64, max_time=20)
    result = simulate_bandit_journey(graph, router, 0, 2, 0, RegimeSchedule([(0, 'fixed')]),
                                     np.random.default_rng(0), max_time=20, scenario=world)
    assert result.arrival_time == 7
    assert all(p['work'] <= 64 for p in router.predictions)
    assert all(p['finite_bank_lower'] <= p['finite_bank_upper'] for p in router.predictions)
    assert router.predictions[0]['root_actions'] == 3
