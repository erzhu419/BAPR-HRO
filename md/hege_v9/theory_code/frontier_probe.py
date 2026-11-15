"""Same-2000-quota diagnostic trace; no repository or archived receipt edits."""
import os, sys, json, gzip
from pathlib import Path
from collections import defaultdict, deque, Counter
import numpy as np

REPO=Path(os.environ.get('BAPR_HRO_REPO',str(Path.cwd()))).resolve()
sys.path.insert(0,str(REPO))
from experiments.reconstruction_v34.discover import context,read
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node
from experiments.reconstruction_v41.run import restored,engine,score
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v47.bulk import BulkIncrementalOracle,BulkDPNode

OUT=Path(__file__).resolve().parent/'diagnostic_replay'
OUT.mkdir(exist_ok=True)

def flow(root):
    seen={root}; todo=[root]; indegree=defaultdict(int)
    while todo:
        n=todo.pop()
        if isinstance(n,BulkDPNode):
            raise RuntimeError('A contains bulk reducer: linear path flow is not directly applicable')
        for p,c in n.children:
            if p<=0: continue
            indegree[c]+=1
            if c not in seen:seen.add(c);todo.append(c)
    ready=deque(n for n in seen if indegree[n]==0)
    weights=defaultdict(np.longdouble);weights[root]=np.longdouble(1)
    ordered=[]
    while ready:
        n=ready.popleft();ordered.append(n)
        for p,c in n.children:
            if p<=0:continue
            weights[c]+=weights[n]*np.longdouble(p)
            indegree[c]-=1
            if indegree[c]==0:ready.append(c)
    assert len(ordered)==len(seen)
    return weights,ordered

class Trace(BulkIncrementalOracle):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.trace=[];self.phase=None;self.audits=[]
    def _phase(self,arm,limit,cpu_limit,name,checkpoints,records):
        self.phase=name
        return super()._phase(arm,limit,cpu_limit,name,checkpoints,records)
    def _expand(self,node):
        if not node.expanded and node.kind is not None:
            self.trace.append((self.stats['charged_compiles']+1,self.phase,node))
        return super()._expand(node)
    def _snapshot(self,phase):
        result=super()._snapshot(phase)
        if self._roots:
            weights,nodes=flow(self._roots['alight'])
            reverse={n:k for k,n in self.nodes.items()}
            frontier=[]
            for n in nodes:
                if not n.expanded:
                    w=weights[n]
                    frontier.append(dict(type=reverse[n][0],kind=n.kind,lower=n.lower,upper=n.upper,
                        probability=float(w),probability_scientific=str(w),width_contribution=float(w*(n.upper-n.lower))))
            frontier.sort(key=lambda x:-x['width_contribution'])
            self.audits.append(dict(phase=phase,count=self.stats['charged_compiles'],reachable_nodes=len(nodes),
                frontier_nodes=len(frontier),frontier_mass=sum(x['probability'] for x in frontier),
                frontier_width=sum(x['width_contribution'] for x in frontier),
                top_frontier=frontier[:25],a_value=result['actions']['alight']))
        return result

def run(index):
    base=REPO/'experiments'
    manifest=read(base/'reconstruction_v38/seed_manifest.json');state=manifest[index]['state']
    data=read(base/'reconstruction_v34/data.json.gz');fit=read(base/'reconstruction_v34/model_fit.json')
    emission=read(base/'reconstruction_v35/fitted_emission.json')
    graph=context(data,state['day'])[0]; model,params=setup(graph,fit)
    node=restore_node(state['node']);history=restored('shallow',model,params,emission,state)
    planner=MatchedContinuationPlanner(graph,model,params,state['deadline'])
    score(engine(planner,'shallow'),node,state['cid'],state['clock'],history)
    oracle=Trace(graph,model,params,state['deadline'],history,max_compiles=2000,max_process_cpu=120)
    result=oracle.forced(node,{k:state[k] for k in ('kind','cid','clock')},schedule='pooled')
    original=read(base/f'reconstruction_v47/natural/{index:03d}_bulk_recycle.json.gz')
    for arm in ('alight','continue'):
        for k in ('lower','upper','unresolved_probability'):
            assert abs(result['actions'][arm][k]-original['actions'][arm][k])<1e-8,(arm,k)
    weights,nodes=flow(oracle._roots['alight'])
    phase_stats={}
    for phase in sorted({p for _,p,_ in oracle.trace}):
        calls=[(i,n) for i,p,n in oracle.trace if p==phase]
        vals=np.array([weights[n] for _,n in calls],dtype=np.longdouble)
        phase_stats[phase]=dict(calls=len(calls),kinds=dict(Counter(n.kind for _,n in calls)),
            positive_a_flow=int(np.sum(vals>0)),
            min_global_a_probability=str(vals.min()),max_global_a_probability=str(vals.max()),
            sum_global_a_probability=str(vals.sum()),
            count_p_below_1e_16=int(np.sum(vals<np.longdouble(1e-16))),
            count_p_below_1e_50=int(np.sum(vals<np.longdouble(1e-50))),
            clocks=[min(n.lower for _,n in calls),max(n.lower for _,n in calls)])
    out=dict(index=index,root_index=state['root_index'],same_archived_scientific_result=True,
        scope='Diagnostic replay of unchanged solver at original2000 quota; instrumentation CPU is not a performance measurement',
        actions=result['actions'],difference_interval=result['difference_interval'],phase_stats=phase_stats,
        snapshots=oracle.audits,
        paid_tasks=[dict(unit=i,phase=p,kind=n.kind,clock=n.lower,global_a_probability=str(weights[n]))
                    for i,p,n in oracle.trace],counters=result['counters'])
    (OUT/f'{index:03d}_frontier.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['index','root_index','same_archived_scientific_result','phase_stats']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    for index in map(int,sys.argv[1:] or ['55','122']):run(index)
