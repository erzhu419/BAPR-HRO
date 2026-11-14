"""Detect stale bulk values and changed paid work, not only final mean drift."""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v46.oracle import point
from experiments.reconstruction_v47.bulk import BulkDPNode
from experiments.reconstruction_v47.incremental import ValueNode
from experiments.reconstruction_v48.lazy_frontier import LazyFrontierOracle
from experiments.reconstruction_v49.bulk import (
    DependencyMemo, IncrementalBulkOracle, IncrementalBulkReducer)
from test_v48_frontier import CASES, case, shared_case, forced_case


@pytest.mark.parametrize('name,options', CASES)
def test_bulk_updates_preserve_every_checkpoint_and_paid_task(monkeypatch, name, options):
    if name.startswith('mixed'):
        graph, model, params, history, platform = case(monkeypatch, **options)
        entry = 10
    else:
        graph, model, params, history, platform, entry = shared_case(monkeypatch, **options)
    old = LazyFrontierOracle(graph, model, params, 60, history,
        max_compiles=10000, verify_frontier=True, trace=True)
    new = IncrementalBulkOracle(graph, model, params, 60, history,
        max_compiles=10000, verify_frontier=True, trace=True)
    checkpoints = tuple(range(1, 501))
    expected = old.platform(platform, entry, checkpoints=checkpoints)
    actual = new.platform(platform, entry, checkpoints=checkpoints)
    assert actual['actions'] == expected['actions']
    assert new.selection_trace == old.selection_trace
    assert len(actual['checkpoints']) == len(expected['checkpoints'])
    for a, b in zip(actual['checkpoints'], expected['checkpoints'], strict=True):
        assert a['actions'] == b['actions']
        assert a['counters']['charged_compiles'] == b['counters']['charged_compiles']
    stats = actual['counters']
    assert stats['bulk_recomputed_rows'] <= stats['bulk_full_rows_equivalent']
    if stats['bulk_platforms']:
        assert stats['bulk_recomputed_rows'] < stats['bulk_full_rows_equivalent']
    assert stats['lazy_width_increases'] == 0


def test_two_arm_actual_certificate_stop_preserves_original_trace(monkeypatch):
    graph, model, params, history, node, row = forced_case(monkeypatch)
    oracles, results = [], []
    for cls in (LazyFrontierOracle, IncrementalBulkOracle):
        oracle = cls(graph, model, params, 100, history, max_compiles=10000,
            verify_frontier=True, trace=True)
        results.append(oracle.forced(node, row, stop_on_certificate=True, certificate_every=1))
        oracles.append(oracle)
    old, new = results
    assert old['stop_reason'] == new['stop_reason'] == 'certificate'
    assert old['actions'] == new['actions']
    assert old['first_certificate']['charged_compiles'] == new['first_certificate']['charged_compiles']
    assert oracles[0].selection_trace == oracles[1].selection_trace


def test_memo_invalidates_late_descendant_and_zero_mass_dependencies():
    stats = defaultdict(int)
    memo = DependencyMemo(stats)
    leaf = ValueNode(0., 10., build=lambda: (point(3.), []))
    late = ValueNode(0., 10., expanded=True, children=[(1., leaf)])
    block = BulkDPNode(0., 10., lambda cache: late.value(cache).copy(), [late])
    parent = ValueNode(0., 10., expanded=True, children=[(.5, block), (.5, block)])
    assert np.array_equal(parent.value(memo), [0., 10., 1.])
    memo.invalidate(leaf)
    leaf.expanded, leaf.constant = True, point(3.)
    assert leaf not in memo and block not in memo and parent not in memo
    assert np.array_equal(parent.value(memo), [3., 3., 0.])
    assert stats['bulk_value_cache_invalidations'] == 4


def test_board_resolution_and_late_truth_change_update_only_affected_ancestors():
    stats = defaultdict(int)
    oracle = SimpleNamespace(deadline=30., stats=stats,
        _terminal_board_scores={}, _bulk_value_cache=DependencyMemo(stats))
    leaf = ValueNode(15., 30.)
    # One candidate at each of three clocks; all four event outcomes remain
    # in their original order, including zero-probability outcomes.
    rows = {}
    for j in range(3):
        rows[j, 0] = (np.array([1.]), np.array([0]), np.array([0]), np.array([30.]))
        rows[j, 1] = (np.array([.5, .5]), np.array([0, 1]), np.array([1, 0]), np.array([30., 25.]))
    truth = {(j, 0): leaf for j in range(3)}
    keys = {(j, 0): ('board', j) for j in range(3)}
    reducer = IncrementalBulkReducer(oracle, np.array([10., 12., 14.]), 10.,
        (1,), {1: 0}, [[], [0]], rows, truth, keys)

    def full_reference():
        values = np.tile(point(30.), (4, 2, 1))
        for j in range(2, -1, -1):
            for mask in (0, 1):
                pp, departed, remaining, waiting = rows[j, mask]
                local = values[j+1, remaining].copy()
                if mask:
                    board = oracle._terminal_board_scores.get(keys[j, 0])
                    if board is None:
                        local[1] = [reducer.times[j], 30., 1.]
                    elif board <= waiting[1]+1e-9:
                        local[1] = leaf.value({})
                values[j, mask] = pp @ local
        return values[0, 1]

    for j in (None, 0, 2, 1):
        if j is not None:
            oracle._terminal_board_scores[keys[j, 0]] = 20.
        assert np.array_equal(reducer({}), full_reference())
        rows_before = stats['bulk_recomputed_rows']
        assert np.array_equal(reducer({}), full_reference())
        assert stats['bulk_recomputed_rows'] == rows_before
    oracle._bulk_value_cache.invalidate(leaf)
    leaf.expanded, leaf.constant = True, point(18.)
    assert np.array_equal(reducer({}), full_reference())
    assert stats['bulk_changed_truth_inputs'] == 3
    assert stats['bulk_recomputed_rows'] < stats['bulk_full_rows_equivalent']


def test_resume_keeps_arithmetic_cache_and_unchanged_paid_work(monkeypatch):
    graph, model, params, history, platform = case(monkeypatch, forecasts=True)
    paused = IncrementalBulkOracle(graph, model, params, 60, history,
        max_compiles=10000, trace=True)
    paused._initialize({'policy': paused._platform_node(platform, 10., paused.snapshot)})
    for quota in (0, 1, 3, 7, 10000):
        paused._phase('policy', quota, None, 'pause', (), [])
        paused._values()
    once = IncrementalBulkOracle(graph, model, params, 60, history,
        max_compiles=10000, trace=True)
    result = once.platform(platform, 10.)
    assert paused._values() == result['actions']
    assert paused.selection_trace == once.selection_trace
