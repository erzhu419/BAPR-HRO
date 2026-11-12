from types import SimpleNamespace

import pytest

from experiments.reconstruction_v33.nodes import Ride
from experiments.reconstruction_v34 import gated_rollout as module
from experiments.reconstruction_v34.forecast import MessageHistory


@pytest.mark.parametrize('bounds,expected', [([-3., -1.], 'nominal'), ([-1., 0.], 'combined'),
                                           ([0., 0.], 'combined'), ([-5., 2.], 'combined')])
def test_only_strictly_negative_full_bank_upper_endpoint_changes_the_action(bounds, expected):
    assert module.select_action('nominal', 'combined', {'full_bank_mean_gain_interval': bounds}) == expected


def test_no_disagreement_means_no_sampling_and_seeds_count_only_actual_gates(monkeypatch):
    seen = []

    def evaluate(graph, model, parameters, deadline, node, row, history, nominal, combined, *, count, seed):
        seen.append((nominal, combined, count, seed))
        return dict(selected_action=nominal, paired={'full_bank_mean_gain_interval': [-1., -1.]}, cost={})

    monkeypatch.setattr(module, 'conditional_action_check', evaluate)
    executor = module.GatedExecutor(None, SimpleNamespace(model=None, parameters={}), 100, root_index=2)
    row = dict(kind='onboard', cid=1, clock=10)
    assert executor.gate(None, row, None, 'continue', 'continue', {})[0] == 'continue'
    assert seen == [] and executor.gates == []
    assert executor.gate(None, row, None, 'continue', 'alight', {})[0] == 'continue'
    executor.gate(None, row, None, 'alight', 'alight', {})
    executor.gate(None, row, None, 'continue', 'alight', {})
    assert seen == [('continue', 'alight', 128, 36050), ('continue', 'alight', 128, 36051)]
    assert [g['gate_call_index'] for g in executor.gates] == [0, 1]


def test_conditional_evaluator_drops_future_frames_and_pairs_the_same_world_indices(monkeypatch):
    model = SimpleNamespace(graph=SimpleNamespace(connections=[]), _positions={})
    history = MessageHistory(model, {}, frames=[dict(receipt_seconds=99, trip_statuses=[], endpoint_updates=[])])
    calls = []

    def bank(belief, parameters, paths, count, seed):
        calls.append(('draw', tuple(paths), count, seed))
        return SimpleNamespace(belief=belief)

    class Continuation:
        def __init__(self, graph, planner, deadline, **options):
            assert options == dict(method='rolling_fixed', waiting='combined')

        def forced(self, node, row, action, snapshot, world):
            assert snapshot.frames == ()
            assert snapshot.advance(200)['frames_consumed'] == 0
            calls.append((action, world.index))
            return dict(loss=10.+world.index+(5. if action == 'alight' else 0.), status='arrived')

    monkeypatch.setattr(module, 'TripWorlds', bank)
    monkeypatch.setattr(module, 'SnapshotExecutor', Continuation)
    monkeypatch.setattr(module, 'ForecastPlanner', lambda *args: SimpleNamespace(stats={}))
    result = module.conditional_action_check(model.graph, model, {}, 100,
        Ride(0, (0,), {0: None}), dict(kind='onboard', cid=0, clock=10), history,
        'continue', 'alight', count=3, seed=7)
    assert calls == [('draw', ((0,),), 3, 7), ('continue', 0), ('continue', 1), ('continue', 2),
                     ('alight', 0), ('alight', 1), ('alight', 2)]
    assert result['selected_action'] == 'continue'
    assert result['paired']['full_bank_mean_gain_interval'] == [-5., -5.]
    assert history.frame_index == 0 and len(history.frames) == 1


def test_original_selection_hooks_preserve_method_and_waiting_contract():
    from experiments.reconstruction_v34.execute import Executor

    calls = []
    compiled = SimpleNamespace(decision=lambda cid, clock: (True, {'cid': cid, 'clock': clock}))
    planner = SimpleNamespace(
        compile_ride=lambda ride, history, method: calls.append(('ride', method)) or compiled,
        platform_reference=lambda platform, history, entry, ready, method: calls.append(('platform', method)) or 'reference')
    executor = Executor(None, planner, 100, method='one_step_rollout', waiting='pair')
    executor.choose = lambda reference, visible, observed, clock: (visible[0], {'clock': clock})
    assert executor.choose_onboard(None, 3, 20, None) == (True, {'cid': 3, 'clock': 20})
    assert executor.choose_platform(None, 1, 2, None, (3,), {3: 20}, 20) == (3, {'clock': 20})
    assert calls == [('ride', 'one_step_rollout'), ('platform', 'one_step_rollout')]
