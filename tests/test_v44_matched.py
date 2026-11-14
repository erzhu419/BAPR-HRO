import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v43.reference import evaluate_case
from experiments.reconstruction_v44 import matched
from test_v41_physical_posterior import fixture


def case(monkeypatch):
    graph, model, parameters, history = fixture(monkeypatch, cancel=0.)
    history.observe(0, 'departure', 0)
    history.observe(0, 'arrival', 60)
    platform = Platform(1, {1: Ride(1, (1,), {1: None})})
    node = Ride(0, (0, 1), {0: platform, 1: None})
    return graph, model, parameters, 240, node, dict(kind='onboard', cid=0, clock=60), history


def test_online_evaluator_reproduces_every_v43_updated_prefix_and_value(monkeypatch):
    args = case(monkeypatch)
    reference = evaluate_case(*args, count=128, seed=44007)
    result = matched.evaluate(*args, count=128, seed=44007)
    assert result['count'] == result['generated_count'] == 128
    assert result['seed'] == reference['seed'] and result['future_messages'] == 0
    assert result['public_history'] == reference['public_history']
    for arm in ('alight', 'continue'):
        for actual, expected in zip(result['rows'][arm], reference['rows'][arm], strict=True):
            assert actual['world_index'] == expected['world_index']
            assert actual['prefix'] == expected['prefix']
            assert actual['value'] == expected['updated']
    assert result['paired']['mean_gain'] == reference['summary']['updated']['difference']['mean']
    assert result['paired']['se'] == reference['summary']['updated']['difference']['se']
    assert result['paired']['full_bank_mean_gain_interval'] == [
        reference['summary']['updated']['difference'][bound] for bound in ('lower', 'upper')]


def test_one_local_solve_per_distinct_prefix_context_without_diagnostic_duplicates(monkeypatch):
    args = case(monkeypatch)
    solves = []
    original = matched.DualMeasureCombinedReference

    def counted(reference, probabilities):
        solves.append((reference.entry, tuple(reference.cids)))
        return original(reference, probabilities)

    monkeypatch.setattr(matched, 'DualMeasureCombinedReference', counted)
    result = matched.evaluate(*args, seed=44008)
    assert len(solves) == result['cache_stats']['compiles'] == 1
    assert result['cache_stats']['queries'] == 128
    assert result['cache_stats']['hits'] == 127
    cost = result['cost']
    assert cost['total_process_cpu'] >= (cost['preparation']['process_cpu']
        +cost['planner_setup']['process_cpu']+cost['prefix_process_cpu']+cost['reference_process_cpu'])


def test_unresolved_local_mass_keeps_bounds_and_cheap_fallback(monkeypatch):
    args = case(monkeypatch)

    class Unresolved:
        def __init__(self, reference, probabilities):
            self.reference = reference

        def receipt(self):
            return dict(lower=self.reference.entry, upper=self.reference.deadline,
                        mean=None, unresolved_probability=1., counters={})

    monkeypatch.setattr(matched, 'DualMeasureCombinedReference', Unresolved)
    result = matched.evaluate(*args, seed=44009)
    assert result['paired']['unresolved_pairs'] == 128
    assert result['paired']['mean_gain'] is None and result['paired']['se'] is None
    low, high = result['paired']['full_bank_mean_gain_interval']
    assert low < 0 < high
    assert matched.choose(result, 'alight') == 'alight'
    assert matched.choose(result, 'continue') == 'continue'
    assert len(result['rows']['alight']) == len(result['rows']['continue']) == 128


@pytest.mark.parametrize('interval,base,expected', [
    ([-3., -1.], 'continue', 'alight'), ([1., 3.], 'alight', 'continue'),
    ([-1., 1.], 'alight', 'alight'), ([0., 0.], 'continue', 'continue'),
    ([-1., 0.], 'continue', 'continue'), ([0., 1.], 'alight', 'alight'),
])
def test_choice_uses_strict_interval_sign_only(interval, base, expected):
    assert matched.choose({'paired': {'full_bank_mean_gain_interval': interval}}, base) == expected


@pytest.mark.parametrize('residual', [7.560e-12, 3.297e-12, -7.560e-12])
def test_observed_roundoff_ties_keep_the_current_action(residual):
    result = {'paired': {'full_bank_mean_gain_interval': [residual, residual]}}
    for base in ('alight', 'continue'):
        assert matched.choose(result, base) == base


@pytest.mark.parametrize('gap,base,expected', [
    (1e-9, 'alight', 'alight'), (-1e-9, 'continue', 'continue'),
    (1.001e-9, 'alight', 'continue'), (-1.001e-9, 'continue', 'alight'),
])
def test_planner_tie_precision_has_a_strict_boundary(gap, base, expected):
    result = {'paired': {'full_bank_mean_gain_interval': [gap, gap]}}
    assert matched.choose(result, base) == expected
