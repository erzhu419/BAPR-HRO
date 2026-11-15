"""Diagnostic fixed-quota global-frontier ordering; no model/policy changes.

Rule frozen before the natural-state runs: ALIGHT (and standalone test policy)
chooses the unresolved frontier with maximum root occupancy times interval
width. All arithmetic-only reachable nodes are expanded to expose that frontier.
All positive physical masses stay represented. Bulk reducer nodes are treated
as opaque physical frontier blocks: their scheduling-only zero-weight edges
are NOT treated as probabilities; when selected, their existing DFS yields its
next paid task. CONTINUE uses the original walker. Complete CPU includes all
graph scans, reductions, and scheduling. No natural-state outcome-based tuning.
"""
import os,sys,json
from pathlib import Path
from collections import defaultdict,deque
from time import process_time,perf_counter
import numpy as np

REPO=Path(os.environ.get('BAPR_HRO_REPO',str(Path.cwd()))).resolve()
sys.path.insert(0,str(REPO))
from experiments.reconstruction_v47.bulk import BulkIncrementalOracle,BulkDPNode
from experiments.reconstruction_v47.hybrid import TerminalDPNode
from experiments.reconstruction_v34.discover import context,read
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node
from experiments.reconstruction_v41.run import restored,engine,score
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner

OUT=Path(__file__).resolve().parent/'priority_diagnostic'
OUT.mkdir(exist_ok=True)

class PriorityOracle(BulkIncrementalOracle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._bulk_walkers={};self._bulk_pending={};self._priority_cache={}
        self.stats.update(priority_scans=0,priority_node_visits=0,priority_bulk_steps=0)

    def _cpu_ok(self):
        if self.max_process_cpu is not None and process_time()-self.cpu_start>=self.max_process_cpu:
            self._stopped_cpu=True;return False
        return True

    def _next_bulk(self,node):
        walker=self._bulk_walkers.setdefault(node,self._walk(node))
        while self._cpu_ok():
            task=self._bulk_pending.get(node)
            if task is None or task.expanded:
                try:task=next(walker)
                except StopIteration:return None
            self._bulk_pending[node]=task
            if task.expanded:
                self._bulk_pending[node]=None;continue
            if task.kind is None:
                self._expand(task);self._bulk_pending[node]=None;continue
            return task
        return None

    def _frontier(self,root):
        """Expose arithmetic branches, then sum all incoming physical paths."""
        seen={root};todo=[root];edges={};indegree=defaultdict(int)
        while todo:
            if not self._cpu_ok():return None
            n=todo.pop()
            if not n.expanded and n.build is not None and n.kind is None:
                self._expand(n)
            if isinstance(n,(BulkDPNode,TerminalDPNode)):
                edges[n]=[]
            else:
                edges[n]=[(p,c) for p,c in n.children if p>0]
                for p,c in edges[n]:
                    indegree[c]+=1
                    if c not in seen:seen.add(c);todo.append(c)
        self.stats['priority_scans']+=1
        self.stats['priority_node_visits']+=len(seen)
        ready=deque(n for n in seen if indegree[n]==0)
        weights=defaultdict(np.longdouble);weights[root]=np.longdouble(1)
        processed=0
        while ready:
            n=ready.popleft();processed+=1
            for p,c in edges[n]:
                weights[c]+=weights[n]*np.longdouble(p)
                indegree[c]-=1
                if indegree[c]==0:ready.append(c)
        assert processed==len(seen),'Physical value graph must be acyclic'
        # Stable insertion order breaks equal priorities; no outcome information.
        order={n:i for i,n in enumerate(self.nodes.values())}
        memo={};candidates=[]
        for n in seen:
            if isinstance(n,(BulkDPNode,TerminalDPNode)):
                if n.drained:continue
                interval=n.value(memo);width=max(0.,float(interval[1]-interval[0]))
            elif not n.expanded and n.build is not None:
                width=max(0.,n.upper-n.lower)
            else:continue
            if width>0:
                candidates.append((weights[n]*np.longdouble(width),-order[n],n))
        return max(candidates,key=lambda t:(t[0],t[1]))[2] if candidates else None

    def _next(self,arm):
        if arm not in ('alight','policy'):
            return super()._next(arm)
        revision=self.stats['dag_expansions']
        cached=self._priority_cache.get(arm)
        if cached and cached[0]==revision and not cached[1].expanded:
            return cached[1]
        while self._cpu_ok():
            n=self._frontier(self._roots[arm])
            if n is None:return None
            if isinstance(n,(BulkDPNode,TerminalDPNode)):
                task=self._next_bulk(n)
                if task is None:
                    n.drained=True
                    continue
                self.stats['priority_bulk_steps']+=1
            else:task=n
            self._priority_cache[arm]=(self.stats['dag_expansions'],task)
            return task
        return None

def run(indices):
    base=REPO/'experiments'
    manifest=read(base/'reconstruction_v38/seed_manifest.json')
    data=read(base/'reconstruction_v34/data.json.gz');fit=read(base/'reconstruction_v34/model_fit.json')
    emission=read(base/'reconstruction_v35/fitted_emission.json')
    for index in indices:
        output=OUT/f'{index:03d}_priority.json'
        if output.exists():
            print('Retaining existing diagnostic',index,flush=True);continue
        state=manifest[index]['state'];graph=context(data,state['day'])[0]
        start=process_time();wall=perf_counter()
        model,params=setup(graph,fit);node=restore_node(state['node'])
        history=restored('shallow',model,params,emission,state)
        planner=MatchedContinuationPlanner(graph,model,params,state['deadline'])
        base_action,_=score(engine(planner,'shallow'),node,state['cid'],state['clock'],history)
        oracle=PriorityOracle(graph,model,params,state['deadline'],history,max_compiles=2000,max_process_cpu=120)
        result=oracle.forced(node,{k:state[k] for k in ('kind','cid','clock')},schedule='pooled')
        result.update(index=index,root_index=state['root_index'],base_action=base_action,
            complete_diagnostic_cost={'process_cpu':process_time()-start,'wall':perf_counter()-wall},
            scope='One diagnostic ordering ablation, unchanged P0/W/pi and2000 quota; not a preregistered paper result or novelty claim',
            rule='ALIGHT root occupancy times unresolved interval width; arithmetic closure; bulk opaque block with original DFS; CONTINUE unchanged')
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ['index','root_index','difference_interval','certified_action','complete_diagnostic_cost','counters']},ensure_ascii=False),flush=True)

if __name__=='__main__':run(list(map(int,sys.argv[1:])))
