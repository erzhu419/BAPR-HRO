"""Same-winner/paid-trace checks for deferred opaque-block reductions."""
import numpy as np
import pytest

from experiments.reconstruction_v46.oracle import point
from experiments.reconstruction_v48.frontier import FrontierOracle, BulkDPNode, LD
from experiments.reconstruction_v48.lazy_frontier import LazyFrontierOracle
from test_v48_frontier import CASES, TinyOracle, case, shared_case, forced_case


class TinyLazyOracle(TinyOracle, LazyFrontierOracle):
    pass


@pytest.mark.parametrize('name,options', CASES)
def test_lazy_matches_full_scan_winners_paid_sequence_and_checkpoint_values(monkeypatch,name,options):
    if name.startswith('mixed'):
        graph,model,params,history,platform = case(monkeypatch,**options)
        entry = 10
    else:
        graph,model,params,history,platform,entry = shared_case(monkeypatch,**options)
    eager = FrontierOracle(graph,model,params,60,history,max_compiles=10000,
        verify_frontier=True,trace=True)
    lazy = LazyFrontierOracle(graph,model,params,60,history,max_compiles=10000,
        verify_frontier=True,trace=True)
    checkpoints = tuple(range(1,501))
    original = eager.platform(platform,entry,checkpoints=checkpoints)
    result = lazy.platform(platform,entry,checkpoints=checkpoints)
    assert result['actions'] == original['actions']
    assert lazy.selection_trace == eager.selection_trace
    assert len(result['checkpoints']) == len(original['checkpoints'])
    for a,b in zip(result['checkpoints'],original['checkpoints'],strict=True):
        assert a['actions'] == b['actions']
        assert a['counters']['charged_compiles'] == b['counters']['charged_compiles']
    assert result['counters']['lazy_max_observed_width_increase'] == 0.


def test_dirty_low_priority_block_is_not_reduced_until_it_can_win():
    oracle = TinyLazyOracle()
    dependency = oracle.node('dependency')
    calls = []
    def reduce(memo):
        calls.append(1)
        return dependency.value(memo)
    block = BulkDPNode(0.,10.,reduce,[dependency])
    oracle.nodes['block'] = block
    high = oracle.node('high')
    root = oracle.node('root',children=[(.01,block),(.99,high)])
    oracle.verify_frontier = False
    oracle._initialize({'alight':root,'continue':dependency})
    assert oracle._next('alight') is high
    assert calls == []
    # Another arm changes the dependency; its dirty block remains deferred.
    oracle._expand(dependency)
    assert oracle._next('alight') is high
    assert calls == []
    oracle._expand(high)
    assert oracle._next('alight') is None
    assert len(calls) == 1
    assert oracle._frontiers['alight']._widths[block] == 0.


def test_late_parent_republishes_current_mass_times_cached_block_bound():
    oracle = TinyLazyOracle()
    final = oracle.node('final')
    block = BulkDPNode(0.,10.,lambda memo:final.value(memo),[final])
    oracle.nodes['block'] = block
    late = oracle.node('late',lower=9.,build=lambda:(None,[(1.,block)]))
    other = oracle.node('other')
    root = oracle.node('root',children=[(.1,block),(.5,late),(.4,other)])
    oracle._initialize({'alight':root,'continue':late})
    assert oracle._next('alight') is other
    assert oracle._next('continue') is late
    oracle._expand(late)
    assert oracle._next('alight') is final
    frontier = oracle._frontiers['alight']
    assert frontier.weights[block] == LD(.1)+LD(.5)
    frontier.reference_check()


def test_both_arms_early_stop_and_paused_work_match_eager(monkeypatch):
    graph,model,params,history,node,row = forced_case(monkeypatch)
    results, traces = [], []
    for solver in (FrontierOracle,LazyFrontierOracle):
        oracle = solver(graph,model,params,100,history,max_compiles=10000,
            verify_frontier=True,trace=True)
        result = oracle.forced(node,row,stop_on_certificate=True,certificate_every=1)
        results.append(result)
        traces.append(oracle.selection_trace)
    a,b = results
    assert a['stop_reason'] == b['stop_reason'] == 'certificate'
    assert a['actions'] == b['actions']
    assert a['first_certificate']['charged_compiles'] == b['first_certificate']['charged_compiles']
    assert traces[0] == traces[1]
    paused = LazyFrontierOracle(graph,model,params,100,history,max_compiles=10000,
        verify_frontier=True,trace=True)
    platform = node.exits[row['cid']]
    entry = row['clock']
    root = paused._platform_node(platform,entry,paused.snapshot)
    paused._initialize({'policy':root})
    for quota in (0,1,3,7,10000):
        paused._phase('policy',quota,None,'pause',(),[])
    once = LazyFrontierOracle(graph,model,params,100,history,max_compiles=10000,
        verify_frontier=True,trace=True)
    original = once.platform(platform,entry)
    assert paused._values() == original['actions']
    assert paused.selection_trace == once.selection_trace


def test_width_increase_is_recorded_and_not_clipped():
    oracle = TinyLazyOracle()
    dependency = oracle.node('dep')
    width = [2.]
    block = BulkDPNode(0.,10.,lambda memo:np.array([0.,width[0],1.]),[dependency])
    oracle.nodes['block'] = block
    oracle.verify_frontier = False
    oracle._initialize({'alight':block})
    assert oracle._next('alight') is dependency
    frontier = oracle._frontiers['alight']
    width[0] = 2.+1e-12
    frontier._dirty_blocks.add(block)
    frontier.peek()
    assert frontier._widths[block] == width[0]
    assert oracle.stats['lazy_width_increases'] == 1
    assert oracle.stats['lazy_max_observed_width_increase'] > 0
