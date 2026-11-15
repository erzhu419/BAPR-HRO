"""Targeted scheduler audit plus independent full-policy small-world oracle."""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from time import process_time

import numpy as np
import pytest

from efficient_priority import (
    BulkDPNode, EfficientPriorityOracle, LD, OUT, REPO, RootFrontier)
from experiments.reconstruction_v47.incremental import ValueNode
from experiments.reconstruction_v46.oracle import point

sys.path.insert(0, str(REPO / 'tests'))
from test_v46_oracle import case
from test_v46_shared import shared_case
from experiments.reconstruction_v46.small_oracle import evaluate

REFERENCE = OUT.parent / 'theory_code' / 'priority_probe.py'
spec = importlib.util.spec_from_file_location('audit_naive_priority', REFERENCE)
reference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_module)

ROWS = []


class TracedReference(reference_module.PriorityOracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_trace = []

    def _expand(self, node):
        before = node.expanded
        super()._expand(node)
        if not before and node.expanded and node.kind is not None:
            self.selection_trace.append(next(repr(key) for key, value in self.nodes.items()
                                             if value is node))


CASES = [
    ('mixed_plain', dict(forecasts=False, mct=0, cancel=0.)),
    ('mixed_forecast_cancel', dict(forecasts=True, mct=0, cancel=.1)),
    ('mixed_ready', dict(forecasts=True, mct=3, cancel=0.)),
    ('shared_plain', dict(reverse=False, forecasts=False, cancel=0.)),
    ('shared_forecast_cancel', dict(reverse=False, forecasts=True, cancel=.05)),
    ('shared_reverse', dict(reverse=True, forecasts=True, cancel=0.)),
]


@pytest.mark.parametrize('name,options', CASES)
def test_independent_oracle_every_checkpoint_trace_and_pause(monkeypatch, name, options):
    if name.startswith('mixed'):
        graph, model, params, history, platform = case(monkeypatch, **options)
        entry = 10
    else:
        graph, model, params, history, platform, entry = shared_case(monkeypatch, **options)
    exact = evaluate(graph, model, params, 60, platform,
                     dict(kind='platform_start', entry=entry, clock=entry), history,
                     action='policy', max_evaluations=100000)
    assert exact['unresolved_probability'] == 0
    truth = exact['mean'] + entry
    fast = EfficientPriorityOracle(graph, model, params, 60, history,
                                   max_compiles=10000, verify_frontier=True, trace=True)
    result = fast.platform(platform, entry, checkpoints=tuple(range(1, 501)))
    assert result['mean'] is not None
    assert abs(result['mean'] - truth) < 1e-8
    previous = (-np.inf, np.inf)
    for snapshot in [*result['checkpoints'], result]:
        value = snapshot['actions']['policy']
        assert value['lower'] <= truth + 1e-8 <= value['upper'] + 2e-8
        assert value['lower'] >= previous[0] - 1e-8
        assert value['upper'] <= previous[1] + 1e-8
        previous = value['lower'], value['upper']
    naive = TracedReference(graph, model, params, 60, history, max_compiles=10000)
    naive_result = naive.platform(platform, entry)
    fast_trace = [row['key'] for row in fast.selection_trace]
    assert fast_trace == naive.selection_trace
    assert result['actions'] == naive_result['actions']

    paused = EfficientPriorityOracle(graph, model, params, 60, history,
                                     max_compiles=10000, verify_frontier=True, trace=True)
    root = paused._platform_node(platform, float(entry), paused.snapshot)
    paused._initialize({'policy': root})
    pause_values = []
    for limit in (0, 1, 3, 7, 10000):
        allocation = paused._phase('policy', limit, None, 'pause_test', (), [])
        snapshot = paused._snapshot('pause_test')
        value = snapshot['actions']['policy']
        assert snapshot['counters']['charged_compiles'] <= limit
        assert value['lower'] <= truth + 1e-8 <= value['upper'] + 2e-8
        pause_values.append(dict(limit=limit, charged=paused.stats['charged_compiles'],
                                 lower=value['lower'], upper=value['upper'],
                                 stop_reason=allocation['stop_reason']))
    assert paused._values() == result['actions']
    assert [row['key'] for row in paused.selection_trace] == fast_trace
    ROWS.append(dict(
        case=name, independent_truth=truth, efficient_value=result['mean'],
        checkpoints_checked=len(result['checkpoints']) + 1,
        compiled=result['counters']['charged_compiles'],
        full_probability_count=exact.get('evaluations'),
        exact_paid_trace_matches_naive=True,
        frontier_checks=fast.stats['priority_reference_checks'],
        max_relative_mass_error=fast.stats['priority_max_relative_mass_error'],
        pause_resume=pause_values))


class TinyOracle(EfficientPriorityOracle):
    """Synthetic finite DAGs isolate scheduler invariants from transit code."""

    def __init__(self):
        self.nodes = {}
        self.deadline = 10.
        self.max_process_cpu = None
        self.cpu_start = process_time()
        self._stopped_cpu = False
        self.stats = dict(charged_compiles=0, dag_expansions=0,
                          dag_zero_expansions=0, unit=0)
        self._setup_priority(verify_frontier=True)

    def node(self, name, *, children=None, build=None, kind='unit', lower=0., upper=10.):
        if children is not None:
            node = ValueNode(lower, upper, expanded=True, children=children)
        else:
            node = ValueNode(lower, upper, kind=kind,
                             build=build if build else lambda: (point(5.), []))
        self.nodes[name] = node
        return node

    def cost(self):
        return dict(process_cpu=process_time() - self.cpu_start)


def test_multi_parent_and_late_attachment_propagate_through_existing_subgraph():
    oracle = TinyOracle()
    final = oracle.node('final')
    middle = oracle.node('middle', build=lambda: (None, [(1., final)]))
    shared = oracle.node('shared', children=[(1., middle)])
    left = oracle.node('left', children=[(1., shared)])
    late = oracle.node('late', build=lambda: (None, [(1., shared)]), lower=9.)
    root = oracle.node('root', children=[(.4, left), (.6, late)])
    oracle._initialize({'alight': root, 'continue': late})
    assert oracle._next('alight') is middle
    oracle._expand(middle)
    assert oracle._next('alight') is final
    frontier = oracle._frontiers['alight']
    assert frontier.weights[final] == LD(.4)
    # C modifies an A-reachable node outside A's heap extraction sequence.
    assert oracle._next('continue') is late
    oracle._expand(late)
    assert oracle._next('alight') is final
    assert frontier.weights[shared] == LD(1.)
    assert frontier.weights[middle] == LD(1.)
    assert frontier.weights[final] == LD(1.)
    frontier.reference_check()


def test_duplicate_physical_edges_and_creation_order_tie():
    oracle = TinyOracle()
    first = oracle.node('first')
    second = oracle.node('second')
    root = oracle.node('root', children=[(.25, second), (.25, second), (.5, first)])
    oracle._initialize({'alight': root})
    assert oracle._next('alight') is first
    frontier = oracle._frontiers['alight']
    assert frontier.weights[second] == LD(.5)
    oracle._expand(first)
    assert oracle._next('alight') is second


def test_bulk_zero_dependencies_invalidate_width_but_never_carry_root_mass():
    oracle = TinyOracle()
    deep = oracle.node('deep')
    dependency = oracle.node('dependency', children=[(1., deep)])
    block = BulkDPNode(0., 10., lambda memo: dependency.value(memo), [dependency])
    oracle.nodes['block'] = block
    other = oracle.node('other')
    root = oracle.node('root', children=[(.6, block), (.4, other)])
    oracle._initialize({'alight': root, 'continue': dependency})
    assert oracle._next('alight') is deep
    frontier = oracle._frontiers['alight']
    assert deep not in frontier.seen
    assert dependency not in frontier.seen
    assert frontier.weights[block] == LD(.6)
    assert oracle._next('continue') is deep
    oracle._expand(deep)
    assert oracle._next('alight') is other
    assert frontier._widths[block] == 0
    frontier.reference_check()


def test_bulk_shared_child_updates_both_physical_priority_and_opaque_width():
    oracle = TinyOracle()
    shared = oracle.node('shared')
    block = BulkDPNode(0., 10., lambda memo: shared.value(memo), [shared])
    oracle.nodes['block'] = block
    other = oracle.node('other')
    root = oracle.node('root', children=[(.6, block), (.2, shared), (.2, other)])
    oracle._initialize({'alight': root})
    assert oracle._next('alight') is shared
    frontier = oracle._frontiers['alight']
    assert frontier.weights[shared] == LD(.2)
    oracle._expand(shared)
    assert oracle._next('alight') is other
    assert frontier._widths[block] == 0


def test_cpu_pause_keeps_unexposed_arithmetic_work_and_all_positive_mass():
    oracle = TinyOracle()
    leaf = oracle.node('leaf')
    arithmetic = oracle.node('arithmetic', kind=None, build=lambda: (None, [(1., leaf)]))
    root = oracle.node('root', kind=None, build=lambda: (None, [(1., arithmetic)]))
    oracle._initialize({'alight': root})
    frontier = oracle._frontiers['alight']
    calls = 0

    def short_cpu_window():
        nonlocal calls
        calls += 1
        return calls <= 1

    oracle._cpu_ok = short_cpu_window
    assert not frontier.sync()
    assert root.expanded and not arithmetic.expanded
    assert oracle.stats['charged_compiles'] == 0
    oracle._cpu_ok = lambda: True
    assert oracle._next('alight') is leaf
    assert root.expanded and arithmetic.expanded
    assert frontier.weights[leaf] == LD(1.)
    assert oracle.stats['dag_zero_expansions'] == 2
    frontier.reference_check()


def test_tiny_positive_paths_are_retained_for_ordering_and_value_bounds():
    oracle = TinyOracle()
    tiny = oracle.node('tiny')
    large = oracle.node('large')
    root = oracle.node('root', children=[(1e-250, tiny), (1., large)])
    oracle._initialize({'alight': root})
    assert oracle._next('alight') is large
    oracle._expand(large)
    assert oracle._next('alight') is tiny
    assert oracle._frontiers['alight'].weights[tiny] == LD(1e-250)


def test_write_small_check_evidence():
    assert len(ROWS) == len(CASES), 'Run the full validation file before recording evidence'
    path = OUT / 'small_checks.json'
    path.write_text(json.dumps(dict(all_passed=True, cases=ROWS,
                                   synthetic_scheduler_checks=6), indent=2) + '\n')
