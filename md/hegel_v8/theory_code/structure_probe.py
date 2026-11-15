"""Read-only structure coverage and cross-candidate dependence audit."""
import json, os
from pathlib import Path
from collections import Counter
from experiments.reconstruction_v34.discover import read, context
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node
ROOT=Path(os.environ.get('BAPR_HRO_REPO','/workspace/scratch/fde8f7d289c2/repos/BAPR-HRO-0022df2'))
OUT=Path(__file__).resolve().parent
manifest=read(ROOT/'experiments/reconstruction_v38/seed_manifest.json')
data=read(ROOT/'experiments/reconstruction_v34/data.json.gz')
fit=read(ROOT/'experiments/reconstruction_v34/model_fit.json')
metrics=read(OUT.parent/'metrics/states.json')
byday={day:setup(context(data,day)[0],fit) for day in {e['state']['day'] for e in manifest}}
def inspect(platform,model):
    roots={cid:model._positions[cid,'departure'][0] for cid in platform.rides}
    candidates=[]
    for cid,ride in platform.rides.items():
        pure=bool(ride.exits) and all(v is None for v in ride.exits.values())
        own=roots[cid]
        tails={t for p in ride.exits.values() if p is not None for t in p.trips(model)}
        pos=[model._positions[c,k] for c in ride.path for k in ('departure','arrival')]
        continuous=all(t==own for t,_ in pos) and all(b[1]==a[1]+1 for a,b in zip(pos,pos[1:]))
        others=[j for j,t in roots.items() if j!=cid and t in tails]
        candidates.append(dict(cid=cid,trip=own,pure_terminal=pure,continuous=continuous,
            cross_candidate_roots_in_tail=others,self_trip_in_tail=own in tails,exits=len(ride.exits),
            tail_trip_count=len(tails)))
    return dict(candidate_count=len(roots),distinct_roots=len(set(roots.values()))==len(roots),
        pure_terminal=all(x['pure_terminal'] for x in candidates),
        cross_candidate_edges=sum(len(x['cross_candidate_roots_in_tail']) for x in candidates),
        candidates=candidates)
rows=[]
for i,e in enumerate(manifest):
    s=e['state'];model,_=byday[s['day']];node=restore_node(s['node'])
    p=node.exits[s['cid']]
    info=inspect(p,model)
    rows.append(dict(index=i,root=s['root_index'],changed=metrics[i]['base_action']!=metrics[i]['selected']['markov128'],
        alight=info,all_exits=[dict(cid=cid,platform=inspect(p,model)) for cid,p in node.exits.items() if p is not None]))
(OUT/'structure_probe.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print('all ALIGHT',Counter((r['alight']['pure_terminal'],r['alight']['cross_candidate_edges']) for r in rows))
print('changed states')
for r in rows:
    if r['changed']:print(json.dumps(r,ensure_ascii=False))
