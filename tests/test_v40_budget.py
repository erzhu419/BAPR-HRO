from copy import copy
from types import SimpleNamespace

import pytest

from experiments.reconstruction_v39.evaluator import evaluate_actions as fixed_evaluate
from experiments.reconstruction_v40 import budget
from experiments.reconstruction_v38.planner import FastPlanner
from test_v38_evaluator import fixture
from test_v39_state import fixture as state_fixture


@pytest.mark.parametrize('variant', ['v38', 'currentmemo', 'frontier'])
def test_original_128_worlds_complete_traces_missing_bounds_and_gate_are_unchanged(variant):
    graph, model, parameters, node, history = fixture()
    args = (graph, model, parameters, 100, node, dict(kind='onboard', cid=0, clock=10),
            history, 'continue', 'alight')
    original = fixed_evaluate(*args, count=128, seed=37050, variant='v38')
    result = budget.evaluate_actions(*args, max_count=128, seed=37050, variant=variant)
    assert result['paired'] == original['paired']
    assert result['selected_action'] == original['selected_action']
    for label in result['rows']:
        fields = ('loss', 'status', 'trace', 'decisions', 'world_index')
        assert [{k: r[k] for k in fields} for r in result['rows'][label]] == [
            {k: r[k] for k in fields} for r in original['rows'][label]]
    assert result['committed_count'] == result['executed_count'] == 128
    assert result['blocks'][0]['seed'] == 37050
    assert len(result['prefixes']) == 128 and all(p['committed'] for p in result['prefixes'])
    assert not result['budget_exceeded'] and not result['fallback']


def harness(monkeypatch, *, sampling=.1, action=.1, losses=None, start=0., summary=0.):
    clock = SimpleNamespace(now=start)
    calls = []
    monkeypatch.setattr(budget, 'process_time', lambda: clock.now)
    monkeypatch.setattr(budget, 'perf_counter', lambda: clock.now)

    class History:
        def public_state(self):
            return []

        def clone(self):
            return copy(self)

        def as_belief(self):
            return self

    def worlds(belief, parameters, paths, count, seed):
        clock.now += sampling
        calls.append(('sampling', seed, count, paths))
        return SimpleNamespace(belief=belief)

    class Controller:
        def __init__(self, *args, **kwargs):
            pass

        def forced(self, node, row, choice, history, world):
            clock.now += action
            calls.append((choice, world.index))
            loss = (losses(world.index, choice) if losses else
                    1. if choice == 'continue' else 10.)
            return dict(loss=loss, status='unresolved' if loss is None else 'arrived',
                trace=[dict(choice=choice, index=world.index)], decisions=dict(onboard=1),
                total_process_cpu=action, total_wall=action, compute_wall=action)

    monkeypatch.setattr(budget, 'TripWorlds', worlds)
    monkeypatch.setattr(budget, 'FastPlanner', lambda *args: SimpleNamespace(stats={}))
    monkeypatch.setattr(budget, 'SnapshotExecutor', Controller)
    monkeypatch.setattr(budget, 'OriginalSnapshotExecutor', Controller)
    original_summary = budget.summarize_pairs

    def summarize(*args):
        clock.now += summary
        return original_summary(*args)

    monkeypatch.setattr(budget, 'summarize_pairs', summarize)
    args = (None, None, {}, 100, SimpleNamespace(path=(1,), paths=lambda: [(1,)]),
            dict(kind='onboard', cid=1, clock=0), History(), 'continue', 'alight')
    return args, clock, calls


def test_late_pair_is_complete_charged_and_excluded_from_the_incumbent(monkeypatch):
    losses = lambda index, choice: ((1., 100., 0.)[index] if choice == 'continue'
                                    else (10., 0., 100.)[index])
    args, clock, calls = harness(monkeypatch, losses=losses, summary=.03)
    result = budget.evaluate_actions(*args, seed=7, max_count=128, cpu_budget=.55, variant='v38')
    assert result['executed_count'] == 3 and result['committed_count'] == 2
    assert result['selected_action'] == 'alight'
    assert result['prefixes'][-1]['selected_action'] == 'continue'
    assert not result['prefixes'][-1]['committed']
    assert result['stop_reason'] == 'late_pair' and result['late_pair_count'] == 1
    assert result['paired']['full_bank_mean_gain_interval'] == [45.5, 45.5]
    assert [row['committed'] for row in result['rows']['nominal']] == [True, True, False]
    assert calls[1:] == [('continue', 0), ('alight', 0), ('continue', 1),
                        ('alight', 1), ('continue', 2), ('alight', 2)]
    assert result['cost']['total_process_cpu'] == pytest.approx(.73)
    assert result['cost']['cpu_overrun'] == pytest.approx(.18)
    assert result['answer_available_process_cpu'] == pytest.approx(.5)


def test_cold_external_cost_and_sampling_can_consume_the_allowance_without_a_pair(monkeypatch):
    args, clock, calls = harness(monkeypatch, start=.2)
    result = budget.evaluate_actions(*args, seed=7, max_count=128, cpu_budget=.25,
                                    started_cpu=0., variant='v38')
    assert result['stop_reason'] == 'sampling_budget_exhausted'
    assert result['committed_count'] == result['executed_count'] == 0
    assert result['selected_action'] == 'alight' and result['fallback']
    assert result['paired']['full_bank_mean_gain_interval'] is None
    assert result['cost']['pre_evaluation_process_cpu'] == pytest.approx(.2)
    assert result['cost']['total_process_cpu'] == pytest.approx(.3)
    assert len(calls) == 1 and result['budget_exceeded']


def test_late_first_pair_never_becomes_a_one_arm_or_late_action_improvement(monkeypatch):
    args, _, _ = harness(monkeypatch)
    result = budget.evaluate_actions(*args, seed=7, max_count=128, cpu_budget=.2, variant='v38')
    assert result['executed_count'] == 1 and result['committed_count'] == 0
    assert len(result['rows']['nominal']) == len(result['rows']['combined']) == 1
    assert result['selected_action'] == 'alight' and result['fallback']


def test_later_block_sampling_preserves_the_previous_timely_incumbent(monkeypatch):
    args, _, calls = harness(monkeypatch, action=.0005)
    result = budget.evaluate_actions(*args, seed=7, max_count=512, cpu_budget=.25, variant='frontier')
    assert result['committed_count'] == result['executed_count'] == 128
    assert result['selected_action'] == 'continue' and not result['fallback']
    assert result['stop_reason'] == 'sampling_budget_exhausted'
    assert [b['seed'] for b in result['blocks']] == [7, budget.block_seed(7, 1)]
    assert result['cost']['sampling_process_cpu'] == pytest.approx(.2)


def test_unlimited_512_uses_four_nested_blocks_with_global_world_indices(monkeypatch):
    args, _, calls = harness(monkeypatch, action=.0005)
    result = budget.evaluate_actions(*args, seed=7, max_count=512, variant='currentmemo')
    assert result['committed_count'] == result['executed_count'] == 512
    assert [b['seed'] for b in result['blocks']] == [budget.block_seed(7, i) for i in range(4)]
    assert [r['world_index'] for r in result['rows']['nominal']] == list(range(512))
    assert result['rows']['nominal'][128]['index_in_block'] == 0
    assert result['rows']['nominal'][-1]['block_index'] == 3
    assert all(item[2] == 128 for item in calls if item[0] == 'sampling')
    assert result['cost']['total_process_cpu'] == pytest.approx(.912)


def test_unresolved_pair_mass_is_retained_in_each_prefix(monkeypatch):
    args, _, _ = harness(monkeypatch, losses=lambda index, choice:
        None if index == 0 and choice == 'continue' else 10.)
    result = budget.evaluate_actions(*args, seed=7, max_count=128, cpu_budget=.35, variant='v38')
    assert result['committed_count'] == 1
    assert result['prefixes'][0]['full_bank_mean_gain_interval'] == [-10., 90.]
    assert result['paired']['unresolved_pairs'] == 1 and result['selected_action'] == 'alight'


def test_currentmemo_can_ignore_a_currently_unselected_prediction_without_erasing_it():
    graph, model, parameters, history, ride = state_fixture()
    engine = budget.CurrentDecisionEngine(FastPlanner(graph, model, parameters, 100))
    first = engine.onboard(ride, 1, 35, history)
    changed = history.clone()
    changed.apply_frame(dict(receipt_seconds=1, trip_statuses=[],
        endpoint_updates=[[4, 'arrival', 'FORECAST', 75.]]), 35)
    assert engine.onboard(ride, 1, 35, changed) == first
    assert engine.stats['onboard_hits'] == 1
    assert changed._forecasts[4, 'arrival'][0] == 75.
    old_key = engine._history('onboard', ride, changed)
    changed.observe(4, 'departure', 45)
    assert changed.selected_forecast('C')[:3] == (4, 'arrival', 75.)
    assert engine._history('onboard', ride, changed) != old_key
    assert not engine._suffixes and not engine._platform_views
