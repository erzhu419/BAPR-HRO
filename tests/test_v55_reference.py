"""Full-policy equivalence of semantic barriers and terminal integration."""
import pytest

from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v52.reference import CausalReference
from experiments.reconstruction_v53.reference import CompressedReference
from experiments.reconstruction_v55.reference import StructuralReference
from experiments.reconstruction_v55.terminal import terminal_platform
from test_v52_reference import information_case
from test_v53_relevance import reference_with_nuisance
from test_v55_semantic import barrier_case


def convert(base, cls=StructuralReference, *, history=None, limit=None):
    solver = cls(base.graph, base.model, base.parameters, base.deadline,
        base.snapshot if history is None else history, base.observations,
        max_states=limit, max_process_cpu=None, keep_decisions=True)
    solver.root_law.roots.update(base.root_law.roots)
    return solver


def same_values(actual, expected):
    assert actual['value']['complete'] and expected['value']['complete']
    assert actual['value']['mean'] == pytest.approx(expected['value']['mean'], abs=1e-8)
    assert actual['cheap_action'] == expected['cheap_action']
    assert actual['observation_contract'] == expected['observation_contract']
    assert actual['future_public_frames'] == expected['future_public_frames'] == 0
    assert actual['actions'].keys() == expected['actions'].keys()
    for action, reference in expected['actions'].items():
        assert actual['actions'][action]['complete']
        assert actual['actions'][action]['mean'] == pytest.approx(reference['mean'], abs=1e-8)


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('case', ['terminal','informative_nuisance','not_ready'])
def test_five_targets_match_causal_and_compressed_references(mode, case):
    if case == 'informative_nuisance':
        base, platform = reference_with_nuisance(False)
    else:
        base, platform, _ = information_case(observed=False, mct=11 if case == 'not_ready' else 0)
    compressed = convert(base, CompressedReference)
    new = convert(base)
    row = dict(kind='platform_start', clock=0., entry=0.)
    expected = base.solve(platform, row, mode=mode)
    ordinary = compressed.solve(platform, row, mode=mode)
    actual = new.solve(platform, row, mode=mode)
    same_values(actual, expected)
    same_values(actual, ordinary)
    if case != 'informative_nuisance':
        assert actual['counters'].get('terminal_platform_calls', 0) > 0
    else:
        # An informative non-action B departure forbids the independent
        # terminal subset shortcut; ordinary causal traversal retains it.
        assert terminal_platform(new, platform, 0., new.snapshot, -1) is None
        assert actual['counters']['projected_event_states'] > 0


@pytest.mark.parametrize('mode', CausalReference.MODES)
def test_onboard_both_arms_and_working_root_choice_survive_barrier(monkeypatch, mode):
    model, history, platform, _, baseline, _ = barrier_case(monkeypatch, forecast=43.)
    history.observe(0, 'departure', 10.)
    history.observe(0, 'arrival', 12.)
    ride = platform.rides[0]
    contract = ObservationContract.from_menu(platform)
    base = CompressedReference(baseline.graph, model, baseline.parameters, 60., history,
        contract, max_states=None, max_process_cpu=None)
    new = convert(base)
    row = dict(kind='onboard', cid=0, clock=12.)
    expected = base.solve(ride, row, mode=mode)
    actual = new.solve(ride, row, mode=mode)
    same_values(actual, expected)
    assert set(actual['actions']) == {'alight','continue'}
    assert actual['difference_interval'] == pytest.approx(expected['difference_interval'], abs=1e-8)
    assert actual['semantic']['goal_barriers'] > 0
    assert actual['counters'].get('terminal_platform_calls', 0) > 0


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('limit', [0,5])
def test_original_state_allowance_keeps_full_policy_in_interval(mode, limit):
    base, platform, _ = information_case(observed=False)
    new = convert(base, limit=limit)
    row = dict(kind='platform_start', clock=0., entry=0.)
    expected = base.solve(platform, row, mode=mode)['value']['mean']
    result = new.solve(platform, row, mode=mode)
    assert result['value']['lower']-1e-8 <= expected <= result['value']['upper']+1e-8
    assert result['counters']['states'] <= limit
    if not result['value']['complete']:
        assert result['value']['mean'] is None


@pytest.mark.parametrize('mode', CausalReference.MODES)
def test_goal_barrier_admits_physical_terminal_platform_without_rewriting_menu(monkeypatch, mode):
    model, history, platform, _, baseline, _ = barrier_case(monkeypatch, cancel=.05, forecast=43.)
    # Remove the actual pre-goal transfer; retain the original post-goal
    # scoring tail. Physical rides are terminal but old syntax reads C.
    platform.rides[0].exits.pop(0)
    base = CompressedReference(baseline.graph, model, baseline.parameters, 60., history,
        ObservationContract.from_menu(platform), max_states=None, max_process_cpu=None)
    new = convert(base)
    row = dict(kind='platform_start', clock=0., entry=0.)
    expected = base.solve(platform, row, mode=mode)
    actual = new.solve(platform, row, mode=mode)
    same_values(actual, expected)
    assert actual['semantic']['goal_barriers'] > 0
    assert actual['counters'].get('terminal_platform_calls', 0) > 0
    assert new.reads.platform_trips(platform) == {'A','D'}
    assert platform.rides[0].path == (0,1,2)
    assert 'C' not in new.root_law.roots


def test_mixed_platform_rejects_terminal_shortcut_and_preserves_root_board_action(monkeypatch):
    model, history, platform, _, baseline, _ = barrier_case(monkeypatch, forecast=43.)
    history.observe_platform((0,5), 0., 10., {0:10.})
    base = CompressedReference(baseline.graph, model, baseline.parameters, 60., history,
        ObservationContract.from_menu(platform), max_states=None, max_process_cpu=None)
    new = convert(base)
    new._prepare(platform)
    assert terminal_platform(new, platform, 0., new.snapshot, 0) is None
    row = dict(kind='platform', clock=10., entry=0., visible=[0], observed=[0])
    expected = base.solve(platform, row, mode='cheap_working')
    actual = new.solve(platform, row, mode='cheap_working')
    same_values(actual, expected)
    assert actual['cheap_action'] in (0,'wait')
    assert all(record['clock'] > 10 for record in actual['decisions'])
