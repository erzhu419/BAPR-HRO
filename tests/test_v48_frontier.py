"""Adapted v12 prototype checks, plus two-arm and stopping integration."""
from time import process_time

import numpy as np
import pytest

from experiments.reconstruction_v48.frontier import FrontierOracle, BulkDPNode, LD
from experiments.reconstruction_v47.incremental import ValueNode
from experiments.reconstruction_v46.oracle import point
from test_v46_oracle import case
from test_v46_shared import shared_case
from experiments.reconstruction_v46.small_oracle import evaluate


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
    fast = FrontierOracle(graph, model, params, 60, history,
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
    fast_trace = [row['key'] for row in fast.selection_trace]

    paused = FrontierOracle(graph, model, params, 60, history,
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



class TinyOracle(FrontierOracle):
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


def test_continue_uses_the_same_root_weight_rule():
    oracle = TinyOracle()
    narrow = oracle.node('high_probability_narrow', lower=9.)
    wide = oracle.node('lower_probability_wide')
    root = oracle.node('continue', children=[(.9, narrow), (.1, wide)])
    oracle._initialize({'continue':root})
    assert oracle._next('continue') is wide
    oracle._frontiers['continue'].reference_check()


def forced_case(monkeypatch):
    from experiments.reconstruction_v33.nodes import Platform, Ride
    from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
    from src import evolving_trip_belief as evolving
    from src.transit_graph import Connection
    graph, _, params, old_history, platform, _ = shared_case(monkeypatch, forecasts=True)
    for cid, trip, origin, target, departure, arrival in (
            (5,'H',2,0,1,9), (6,'H',0,1,10,50), (7,'H',1,2,51,80),
            (8,'Z',1,2,65,75)):
        graph.add_connection(Connection(cid,'R',trip,origin,target,departure,arrival))
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False,
        travel_sd=.1, dwell_sd=.1, cancel_persistence=.8)
    history = PhysicalHistory(model, params, fit=old_history.fit)
    history.observe(5,'departure',1)
    history.observe(5,'arrival',9)
    tail = Platform(1,{8:Ride(8,(8,),{8:None})})
    node = Ride(5,(5,6,7),{5:platform,6:tail,7:None})
    return graph, model, params, history, node, dict(kind='onboard',cid=5,clock=9.)


def test_both_arms_preserve_independent_values_and_stop_at_saved_certificate(monkeypatch):
    graph, model, params, history, node, row = forced_case(monkeypatch)
    full = FrontierOracle(graph, model, params, 100, history,
        max_compiles=10000, verify_frontier=True, trace=True)
    full_result = full.forced(node,row,certificate_every=1)
    stopped = FrontierOracle(graph, model, params, 100, history,
        max_compiles=10000, verify_frontier=True, trace=True)
    result = stopped.forced(node,row,stop_on_certificate=True,certificate_every=1)
    assert result['stop_reason'] == 'certificate'
    assert result['counters']['charged_compiles'] < full_result['counters']['charged_compiles']
    assert result['first_certificate']['actions'] == result['actions']
    assert result['first_certificate']['charged_compiles'] == result['counters']['charged_compiles']
    assert stopped.selection_trace == full.selection_trace[:len(stopped.selection_trace)]
    for arm in ('alight','continue'):
        independent = evaluate(graph,model,params,100,node,row,history,
            action=arm,max_evaluations=100000)
        truth = independent['mean']+row['clock']
        assert full_result['actions'][arm]['lower'] == pytest.approx(truth, abs=1e-8)
        for snapshot in result['checkpoints']:
            value = snapshot['actions'][arm]
            assert value['lower'] <= truth+1e-8 <= value['upper']+2e-8


def test_pending_zero_width_continue_returns_quota_without_claiming_exactness(monkeypatch):
    from types import SimpleNamespace
    import experiments.reconstruction_v48.frontier as module
    oracle = TinyOracle()
    oracle.max_compiles = 4
    oracle.snapshot = None
    last = oracle.node('A_last', build=lambda:(point(2.),[]))
    middle = oracle.node('A_middle', build=lambda:(None,[(1.,last)]))
    alight = oracle.node('A_first', build=lambda:(None,[(1.,middle)]))
    tiny = oracle.node('C_tiny')
    deep = oracle.node('C_deep', children=[(1e-250,tiny)])
    continuation = BulkDPNode(0.,10.,
        lambda memo:point(5.)+1e-250*deep.value(memo),[deep])
    oracle.nodes['C_bulk'] = continuation
    oracle._prepare = lambda node:None
    oracle._platform_node = lambda platform,clock,history:alight
    oracle._point = lambda key,value:continuation
    monkeypatch.setattr(module,'structure_gate',lambda model,node,row:dict(eligible=True))
    oracle.model = None
    node = SimpleNamespace(path=[0], exits={0:object()})
    result = oracle.forced(node,dict(clock=0.,cid=0),certificate_every=1)
    c = result['actions']['continue']
    assert c['lower'] == c['upper'] == 5.
    assert c['unresolved_probability'] == 0.
    assert c['pending_dependencies'] > 0 and c['mean'] is None
    assert result['complete'] is False
    assert result['recycling_eligibility_reason'] == 'continue_actionable_frontier_exhausted'
    assert result['recycled_compiles'] == 1
    assert result['actions']['alight']['mean'] == 2.
    assert not tiny.expanded
