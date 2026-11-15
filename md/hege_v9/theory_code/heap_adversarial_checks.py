"""Independent DAG cases for the sibling agent's incremental occupancy heap."""
import sys,json
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier_efficiency'))
from efficient_priority import RootFrontier
from experiments.reconstruction_v47.incremental import ValueNode
from experiments.reconstruction_v47.bulk import BulkDPNode

class FakeOracle:
    def __init__(self):
        self.stats=defaultdict(int);self._creation_order={};self._expansion_events=[]
    def node(self,build=None,kind='paid',constant=None):
        n=ValueNode(0.,10.,build=build,kind=kind,constant=constant,expanded=constant is not None)
        self._creation_order[n]=len(self._creation_order);return n
    def register(self,n):self._creation_order[n]=len(self._creation_order);return n
    def _cpu_ok(self):return True
    def _expand(self,n):
        if n.expanded:return
        n.constant,n.children=n.build();n.build=None;n.expanded=True
        self._expansion_events.append(n)

def sync(frontier):
    assert frontier.sync();frontier.reference_check()

rows=[]
for external_first in (False,True):
    o=FakeOracle()
    leaf=o.node(build=lambda:(np.array([3.,3.,0.]),[]))
    other=o.node(build=lambda:(np.array([7.,7.,0.]),[]))
    shared=o.node(build=lambda:(None,[(1.,leaf)]))
    p=o.node(build=lambda:(None,[(.8,shared),(.2,other)]))
    q=o.node(build=lambda:(None,[(.5,shared),(.5,other)]))
    root=o.node(build=lambda:(None,[(.6,p),(.4,q)]),kind=None)
    f=RootFrontier(o,root);sync(f)
    if external_first:o._expand(shared);sync(f)
    o._expand(p);sync(f)
    if not external_first:o._expand(shared);sync(f)
    assert abs(float(f.weights[leaf])-.48)<1e-15
    o._expand(q);sync(f)
    assert abs(float(f.weights[shared])-.68)<1e-15
    assert abs(float(f.weights[leaf])-.68)<1e-15
    assert abs(float(f.weights[other])-.32)<1e-15
    o._expand(leaf);sync(f)
    assert f.peek() is other
    o._expand(other);sync(f)
    assert f.peek() is None
    rows.append(dict(case='external_expansion_before_reach' if external_first else 'late_parent_after_child_expansion',passed=True,checks=o.stats['priority_reference_checks']))

o=FakeOracle()
leaf=o.node(build=lambda:(np.array([3.,3.,0.]),[]))
shared=o.node(build=lambda:(None,[(1.,leaf)]))
block=o.register(BulkDPNode(0.,10.,lambda memo:shared.value(memo),[shared]))
other=o.node(build=lambda:(None,[(1.,shared)]))
root=o.node(build=lambda:(None,[(.5,block),(.5,other)]),kind=None)
f=RootFrontier(o,root);sync(f)
assert float(f.weights[shared])==0. # opaque dependency is not a physical edge
o._expand(shared);sync(f)
o._expand(leaf);sync(f)
assert f.peek() is other # block width invalidated through an indirect dependency
o._expand(other);sync(f)
assert abs(float(f.weights[shared])-.5)<1e-15
assert abs(float(f.weights[leaf])-.5)<1e-15
assert f.peek() is None
rows.append(dict(case='opaque_zero_weight_watch_and_later_physical_parent',passed=True,checks=o.stats['priority_reference_checks']))

out=dict(all_passed=True,cases=rows,scope='Independent hand-constructed DAGs; every step checks root masses and heap winner against full recomputation')
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
