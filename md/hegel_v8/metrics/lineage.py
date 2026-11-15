"""Read-only cross-version quality lineage, without comparing cross-run CPU."""
from pathlib import Path
import os,gzip,json,statistics

ROOT=Path(os.environ.get('BAPR_HRO_REPO','/workspace/scratch/fde8f7d289c2/repos/BAPR-HRO-0022df2'))/'experiments'
OUT=Path(__file__).resolve().parent
paths={
 'v40cheap':('reconstruction_v40','cheap_frontier__combined_rolling'),
 'v40full':('reconstruction_v40','frontier__pairs128'),
 'v41legacycheap':('reconstruction_v41','legacy_cheap'),
 'v41legacyfull':('reconstruction_v41','legacy_rollout128'),
 'v41physicalcheap':('reconstruction_v41','physical_cheap'),
 'v41physicalfull':('reconstruction_v41','physical_rollout128'),
 'v41shallowcheap':('reconstruction_v41','shallow_cheap'),
 'v41shallowfull':('reconstruction_v41','shallow_rollout128'),
 'v45cheap':('reconstruction_v45','rolling_fixed'),
 'v45markov':('reconstruction_v45','markov128'),
}
rows=[]
for i in range(72):
    r={'root_index':i}
    for k,(v,l) in paths.items():
        with gzip.open(ROOT/v/'roots'/f'{i:03}_{l}.json.gz','rt') as f:r[k]=json.load(f)['outcome']['loss']
    rows.append(r)
assert all(r['v45markov']==r['v41physicalfull'] for r in rows)
assert all(r['v45cheap']==r['v41shallowcheap'] for r in rows)
comparisons={}
for a,b in [('v41physicalcheap','v41legacycheap'),('v41physicalfull','v41legacyfull'),('v41shallowcheap','v41physicalcheap'),('v41shallowfull','v41physicalfull'),('v45markov','v40cheap'),('v45markov','v40full'),('v45cheap','v40cheap')]:
    d=[r[b]-r[a] for r in rows]
    comparisons[a+'_vs_'+b]=dict(mean_gain=statistics.mean(d),net=sum(d),wins=sum(x>0 for x in d),harms=sum(x<0 for x in d),ties=sum(x==0 for x in d),nonzero=[dict(root_index=r['root_index'],gain=r[b]-r[a]) for r in rows if r[a]!=r[b]])
result=dict(rows=rows,means={k:statistics.mean(r[k] for r in rows) for k in paths},comparisons=comparisons)
(OUT/'lineage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(comparisons,ensure_ascii=False,indent=2))
