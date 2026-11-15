"""Strong V53 scheduling remains active around V55 exact reductions."""
import pytest

from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v52.reference import CausalReference
from experiments.reconstruction_v53.priority import ImpactReference
from experiments.reconstruction_v53.reference import CompressedReference
from experiments.reconstruction_v55.priority import ImpactStructuralReference
from experiments.reconstruction_v55.reference import StructuralReference
from test_v52_reference import information_case
from test_v55_reference import same_values
from test_v55_semantic import barrier_case


def priority(base, *, cutoff=0., limit=None, history=None):
    solver = ImpactStructuralReference(base.graph, base.model, base.parameters,
        base.deadline, base.snapshot if history is None else history, base.observations,
        max_states=limit, max_process_cpu=None, impact_cutoff=cutoff, keep_decisions=True)
    solver.root_law.roots.update(base.root_law.roots)
    return solver


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('observed', [False,True])
def test_zero_cutoff_preserves_all_five_complete_policy_targets(mode, observed):
    base, platform, _ = information_case(observed=observed)
    solver = priority(base)
    row = dict(kind='platform_start', clock=0., entry=0.)
    expected = base.solve(platform, row, mode=mode)
    actual = solver.solve(platform, row, mode=mode)
    same_values(actual, expected)
    assert actual['implementation'] == 'structural_impact_conditional'
    assert actual['impact_cutoff'] == 0.
    assert actual['counters']['impact_pruned_branches'] == 0
    assert solver.reach == 1.
    if observed:
        assert any(key[0] == 'impact_event' for key in solver.memo)
    else:
        assert actual['counters']['terminal_dp_admitted'] > 0


@pytest.mark.parametrize('mode', CausalReference.MODES)
def test_positive_cutoff_actually_prunes_and_keeps_full_target_bounds(mode):
    base, platform, _ = information_case(observed=True)
    solver = priority(base, cutoff=30.)
    row = dict(kind='platform_start', clock=0., entry=0.)
    expected = base.solve(platform, row, mode=mode)['value']['mean']
    result = solver.solve(platform, row, mode=mode)
    assert result['value']['lower']-1e-8 <= expected <= result['value']['upper']+1e-8
    assert result['counters']['impact_pruned_branches'] > 0
    assert result['reasons']['impact_cutoff'] > 0
    assert solver.settled_reach and solver.reach == 1.


def test_cooperative_mro_keeps_impact_node_expectation_and_settled_cache():
    assert ImpactStructuralReference._node is ImpactReference._node
    assert ImpactStructuralReference._expect is ImpactReference._expect
    assert ImpactStructuralReference._platform is StructuralReference._platform
    base, platform, _ = information_case(observed=True)
    solver = priority(base, cutoff=30.)
    solver._prepare(platform)
    first = solver._platform(platform, 0., solver.snapshot, depth=-1)
    states = solver.stats['states']
    second = solver._platform(platform, 0., solver.snapshot, depth=-1)
    assert second == first
    assert solver.stats['states'] == states
    assert solver.stats['settled_cache_hits'] > 0
    assert solver.stats['impact_pruned_branches'] > 0


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('limit', [None,0,8])
def test_onboard_both_arms_root_choice_and_cumulative_allowance(monkeypatch, mode, limit):
    model, history, platform, _, planner, _ = barrier_case(monkeypatch, forecast=43.)
    history.observe(0, 'departure', 10.)
    history.observe(0, 'arrival', 12.)
    base = CompressedReference(planner.graph, model, planner.parameters, 60., history,
        ObservationContract.from_menu(platform), max_states=None, max_process_cpu=None)
    solver = priority(base, cutoff=0. if limit is None else .01, limit=limit)
    ride = platform.rides[0]
    row = dict(kind='onboard', cid=0, clock=12.)
    expected = base.solve(ride, row, mode=mode)
    actual = solver.solve(ride, row, mode=mode)
    assert actual['cheap_action'] == expected['cheap_action']
    for action, truth in expected['actions'].items():
        bound = actual['actions'][action]
        assert bound['lower']-1e-8 <= truth['mean'] <= bound['upper']+1e-8
        if not bound['complete']:
            assert bound['mean'] is None
    if limit is None:
        same_values(actual, expected)
        assert actual['semantic']['goal_barriers'] > 0
        assert actual['counters']['terminal_dp_admitted'] > 0
    else:
        assert actual['counters']['states'] <= limit
    assert solver.reach == 1.


def test_current_working_board_choice_is_unchanged_with_impact_scheduler():
    base, platform, _ = information_case(observed=False)
    history = base.snapshot.clone()
    history.observe_platform((0,2), 0., 10., {0:10.})
    old = ImpactReference(base.graph, base.model, base.parameters, 60., history,
        base.observations, max_states=None, max_process_cpu=None, impact_cutoff=0.)
    old.root_law.roots.update(base.root_law.roots)
    solver = priority(base, history=history)
    row = dict(kind='platform', clock=10., entry=0., visible=[0], observed=[0])
    actual = solver.solve(platform, row, mode='cheap_working')
    expected = old.solve(platform, row, mode='cheap_working')
    same_values(actual, expected)
    assert actual['cheap_action'] == 0
