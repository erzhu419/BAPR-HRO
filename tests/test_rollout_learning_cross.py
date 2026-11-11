from copy import deepcopy

import numpy as np
import pytest

from experiments.reconstruction_v3.run_rollout_learning_cross import LearningRouter
from src.transit_graph import Connection, Stop, TransitGraph


def network():
    graph = TransitGraph()
    for sid in (0, 1):
        graph.add_stop(Stop(sid, str(sid)))
    graph.add_connection(Connection(0, 'A', 'known', 0, 1, 3, 5))
    graph.add_connection(Connection(1, 'A', 'unknown', 0, 1, 8, 11))
    return graph


@pytest.mark.parametrize('update_mean', (False, True))
@pytest.mark.parametrize('update_cancel', (False, True))
def test_toggles_change_unknown_trip_weights_and_preserve_public_facts(update_mean, update_cancel):
    graph = network()
    prior = {'A': {'mean': 0., 'std': 1., 'cancel_rate': .1}}
    router = LearningRouter(graph, prior, online_mean=update_mean, online_cancel=update_cancel,
                            samples=8, work_budget=512, max_time=30)
    router.route(0, 1, 0)
    router.set_information_context(0, 0, None, 0, set(), set(), set())
    baseline = router.baseline_parameters['A']
    router.observe_delay('A', 8)
    router.observe_cancel('A')
    router.observe_vehicle_event(0, False, 3)
    parameters = router._parameters()
    current = parameters['A']
    assert (current['mean'] > baseline['mean']) == update_mean
    assert (current['cancel_rate'] > baseline['cancel_rate']) == update_cancel
    assert current['std'] == baseline['std']
    outcomes, weights = router.trip_belief.posterior(0, parameters)
    assert outcomes.tolist() == [3]
    assert weights.tolist() == [1.]
    outcomes, weights = router.trip_belief.posterior(1, parameters)
    assert weights[outcomes == 999].sum() == pytest.approx(current['cancel_rate'])
    operating = outcomes != 999
    conditional_mean = np.dot(outcomes[operating], weights[operating]) / weights[operating].sum()
    assert (conditional_mean > 1) == update_mean
    assert router.work_budget == 512
    assert router.samples == 8


def test_final_planner_bypasses_beta_penalty_and_retains_predictive_std():
    graph = network()
    prior = {'A': {'mean': 0., 'std': 1., 'cancel_rate': .1}}
    results = []
    for beta in (0., 1e12):
        router = LearningRouter(deepcopy(graph), prior, online_mean=True, online_cancel=True,
                                samples=8, work_budget=512, max_time=30, seed=7)
        router.beta = beta
        router._score = lambda *_: pytest.fail('Final planner called marginal penalty score')
        router.route(0, 1, 0)
        router.set_information_context(0, 0, None, 0, set(), set(), set())
        router.select_connection(0, 0, np.random.default_rng(0))
        results.append(router.predictions[-1])
    for field in ('connection', 'value', 'work', 'complete_common_rounds'):
        assert results[0][field] == results[1][field]
    parameters = router._parameters()
    _, narrow = router.trip_belief.posterior(1, parameters)
    parameters['A'] = parameters['A'] | {'std': 4.}
    _, wide = router.trip_belief.posterior(1, parameters)
    assert not np.allclose(narrow, wide)
