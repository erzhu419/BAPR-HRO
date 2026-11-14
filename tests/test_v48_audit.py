import copy
from collections import defaultdict

import numpy as np

import pytest

from experiments.reconstruction_v47.incremental import ValueNode
from experiments.reconstruction_v47.bulk import BulkDPNode
from experiments.reconstruction_v48.frontier import RootFrontier
from experiments.reconstruction_v48.audit import (
    full_scan_weights, full_scan_priority, stopped_prefix, selected_action, values,
)


def test_scan_sums_shared_paths_and_late_parent_descendants():
    leaf = ValueNode(10., 20., build=lambda: None)
    common = ValueNode(0., 30., expanded=True, children=[(.5, leaf)])
    first = ValueNode(0., 30., expanded=True, children=[(.6, common)])
    late = ValueNode(0., 30., build=lambda: None)
    root = ValueNode(0., 30., expanded=True, children=[(.4, first), (.6, late)])
    weights = full_scan_weights(root, is_block=lambda node: False)
    assert weights[common] == pytest.approx(.24)
    assert weights[leaf] == pytest.approx(.12)
    late.expanded = True
    late.children = [(.8, common)]
    updated = full_scan_weights(root, is_block=lambda node: False)
    assert updated[common] == pytest.approx(.72)
    assert updated[leaf] == pytest.approx(.36)
    assert full_scan_priority(root, is_block=lambda node: False)[leaf] == pytest.approx(3.6)


def test_bulk_zero_weight_work_is_an_opaque_physical_block():
    task = ValueNode(0., 20., build=lambda: None)
    block = ValueNode(5., 15., build=lambda: None, children=[(0., task)])
    root = ValueNode(0., 20., expanded=True, children=[(.3, block)])
    priorities = full_scan_priority(root, is_block=lambda node: node is block)
    assert priorities[block] == pytest.approx(3.)
    assert task not in priorities
    assert block.children == [(0., task)]  # Work dependencies remain intact.


def test_tiny_positive_root_influence_survives_float64_underflow():
    leaf = ValueNode(0., 1., build=lambda: None)
    middle = ValueNode(0., 1., expanded=True, children=[(1e-250, leaf)])
    root = ValueNode(0., 1., expanded=True, children=[(1e-100, middle)])
    weight = full_scan_weights(root, is_block=lambda node: False)[leaf]
    assert weight > 0
    assert float(weight) == 0


def point(count, *, certified=False):
    a = dict(lower=20. if certified else 0., upper=30., mean=None, unresolved_probability=.5)
    c = dict(lower=10., upper=10., mean=10., unresolved_probability=0.)
    return dict(actions={'alight':a, 'continue':c},
                difference_interval=[a['lower']-10., 20.],
                certified_action='continue' if certified else None,
                counters=dict(charged_compiles=count))


def test_early_stop_uses_first_certificate_and_preserves_fallback():
    first, cert = point(0), point(250, certified=True)
    full = dict(clock=0., deadline=100., checkpoints=[first, cert, point(500, certified=True)])
    stop = dict(clock=0., deadline=100., checkpoints=[first, cert, copy.deepcopy(cert)], **cert)
    assert stopped_prefix(full, stop)['stopping_count'] == 250
    stop['selected_action'] = 'continue'
    assert selected_action(stop, 'alight') == 'continue'
    uncertain = dict(**first, selected_action='alight')
    assert selected_action(uncertain, 'alight') == 'alight'
    uncertain['selected_action'] = 'continue'
    with pytest.raises(ValueError, match='fallback'):
        selected_action(uncertain, 'alight')


def test_paid_work_after_saved_certificate_is_rejected():
    cert = point(250, certified=True)
    full = dict(clock=0., deadline=100., checkpoints=[point(0), cert, point(500, certified=True)])
    late = dict(clock=0., deadline=100., checkpoints=[point(0), cert, point(500, certified=True)],
                **point(500, certified=True))
    with pytest.raises(ValueError, match='after its certificate'):
        stopped_prefix(full, late)


def test_rounded_point_with_unexpanded_dependencies_stays_incomplete():
    row = point(250, certified=True)
    row['actions']['alight'].update(lower=30., upper=30., unresolved_probability=0.,
                                    pending_dependencies=2)
    row['difference_interval'] = [20., 20.]
    values(row, clock=0., deadline=100.)
    row['actions']['alight']['mean'] = 30.
    with pytest.raises(ValueError, match='Inexact'):
        values(row, clock=0., deadline=100.)
    row['actions']['alight'].update(mean=None, pending_dependencies=0)
    with pytest.raises(ValueError, match='retained unfinished'):
        values(row, clock=0., deadline=100.)


class TinyOracle:
    """Build arbitrary valid shared DAGs independently of the transit solver."""
    def __init__(self):
        self.stats = defaultdict(int)
        self._creation_order = {}
        self._expansion_events = []

    def register(self, node):
        self._creation_order[node] = len(self._creation_order)
        return node

    def node(self, build, kind='paid'):
        return self.register(ValueNode(0., 10., build=build, kind=kind))

    def _cpu_ok(self):
        return True

    def _expand(self, node):
        if node.expanded:
            return
        node.constant, node.children = node.build()
        node.build, node.expanded = None, True
        self._expansion_events.append(node)


def compare_independent_frontier(frontier):
    assert frontier.sync()
    opaque = lambda node: isinstance(node, BulkDPNode)
    weights = full_scan_weights(frontier.root, is_block=opaque)
    assert set(weights) == frontier.seen
    for node, weight in weights.items():
        assert abs(weight-frontier.weights[node]) <= np.longdouble('1e-15')*max(
            abs(weight), np.finfo(np.longdouble).tiny)
    priorities = full_scan_priority(frontier.root, is_block=opaque)
    winner = max(priorities, key=lambda node:(priorities[node], -frontier.oracle._creation_order[node])) if priorities else None
    assert frontier.peek() is winner


@pytest.mark.parametrize('external_first', [False, True])
def test_incremental_heap_matches_independent_scan_after_late_parent(external_first):
    oracle = TinyOracle()
    leaf = oracle.node(lambda:(np.array([3., 3., 0.]), []))
    other = oracle.node(lambda:(np.array([7., 7., 0.]), []))
    common = oracle.node(lambda:(None, [(1., leaf)]))
    first = oracle.node(lambda:(None, [(.8, common), (.2, other)]))
    late = oracle.node(lambda:(None, [(.5, common), (.5, other)]))
    root = oracle.node(lambda:(None, [(.6, first), (.4, late)]), kind=None)
    frontier = RootFrontier(oracle, root)
    compare_independent_frontier(frontier)
    if external_first:
        oracle._expand(common)
        compare_independent_frontier(frontier)
    oracle._expand(first)
    compare_independent_frontier(frontier)
    if not external_first:
        oracle._expand(common)
        compare_independent_frontier(frontier)
    assert float(frontier.weights[leaf]) == pytest.approx(.48)
    oracle._expand(late)
    compare_independent_frontier(frontier)
    assert float(frontier.weights[leaf]) == pytest.approx(.68)
    oracle._expand(leaf)
    compare_independent_frontier(frontier)
    oracle._expand(other)
    compare_independent_frontier(frontier)


def test_bulk_dirty_watchers_and_new_physical_parent_match_independent_scan():
    oracle = TinyOracle()
    leaf = oracle.node(lambda:(np.array([3., 3., 0.]), []))
    common = oracle.node(lambda:(None, [(1., leaf)]))
    block = oracle.register(BulkDPNode(0., 10., lambda memo:common.value(memo), [common]))
    ordinary = oracle.node(lambda:(None, [(1., common)]))
    root = oracle.node(lambda:(None, [(.5, block), (.5, ordinary)]), kind=None)
    frontier = RootFrontier(oracle, root)
    compare_independent_frontier(frontier)
    assert common not in frontier.seen  # Only a zero-weight block dependency yet.
    oracle._expand(common)
    compare_independent_frontier(frontier)
    oracle._expand(leaf)
    compare_independent_frontier(frontier)
    assert frontier.peek() is ordinary
    oracle._expand(ordinary)
    compare_independent_frontier(frontier)
    assert float(frontier.weights[leaf]) == pytest.approx(.5)
