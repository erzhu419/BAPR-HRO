"""Independent receipt arithmetic for fixed commit 6e46267; read-only inputs."""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import gzip
import json
import math
import os

REPO=Path(os.environ.get('BAPR_HRO_REPO','/workspace/scratch/fde8f7d289c2/repos/BAPR-HRO-6e46267'))
OUT=Path(__file__).resolve().parent
EXP=REPO/'experiments'
def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def close(a,b):
    assert abs(a-b)<1e-8,(a,b)
def vals_equal(a,b):
    for arm in ('alight','continue'):
        for k in ('lower','upper','mean','unresolved_probability'):
            x,y=a['actions'][arm][k],b['actions'][arm][k]
            if x is None or y is None: assert x is y
            else:close(x,y)
def gap(r):
    a,c=r['actions']['alight'],r['actions']['continue']
    return [a['lower']-c['upper'],a['upper']-c['lower']]
def sample_equal(r,old,label):
    for k in ('seed','count','generated_count','future_messages','public_history','clock','deadline'):
        if k in old: assert r[k]==old[k],k
    assert r['count']==128
    for arm in ('alight','continue'):
        assert len(r['rows'][arm])==128
        for i,(a,b) in enumerate(zip(r['rows'][arm],old['rows'][arm],strict=True)):
            assert a['world_index']==b['world_index']==i
            if label=='markov128':
                assert a['prefix']==b['prefix']
                for k in ('lower','upper','mean','unresolved_probability'):assert a['value'][k]==b['value'][k]
            else:
                for k in ('loss','status','trace','decisions'):assert a[k]==b[k]
    assert r['paired']==old['paired']
    if 'selected_action' in old:assert r['selected_action']==old['selected_action']

files=sorted((EXP/'reconstruction_v47/natural').glob('*.json.gz'))
assert len(files)==63
rows=[read(p) for p in files]
by=defaultdict(dict)
checks={'receipt_count':len(rows),'all_interval_and_quota_checks':True,'checkpoint_count':0,
        'sample_arm_rows_compared':0,'legacy_values_unchanged':True,'same_kernel_prefixes':True}
for r in rows:
    by[r['index']][r['label']]=r
    assert r['complete_cost']['process_cpu']>0
    if r['label'] in ('markov128','full128'):
        old=read(EXP/('reconstruction_v45/banks' if r['label']=='markov128' else 'reconstruction_v41/banks')/
                 f'{r["index"]:03}_{"markov128" if r["label"]=="markov128" else "shallow"}.json.gz')
        sample_equal(r,old,r['label']);checks['sample_arm_rows_compared']+=256
        continue
    assert r['counters']['charged_compiles']<=2000
    assert r['counters']['structure_refusals']==r['counters']['unsupported_scores']==0
    for arm,v in r['actions'].items():
        assert r['clock']-1e-8<=v['lower']<=v['upper']+1e-8<=r['deadline']+2e-8
        assert -1e-8<=v['unresolved_probability']<=1+1e-8
        if v['mean'] is not None: close(v['upper'],v['lower']);assert v['unresolved_probability']==0
    for a,b in zip(gap(r),r['difference_interval']):close(a,b)
    assert r['certified_action'] is None
    cps=r.get('checkpoints',[])
    checks['checkpoint_count']+=len(cps)
    for a,b in zip(cps,cps[1:]):
        assert a['counters']['charged_compiles']<=b['counters']['charged_compiles']<=2000
        for arm in ('alight','continue'):
            x,y=a['actions'][arm],b['actions'][arm]
            assert y['lower']>=x['lower']-1e-8 and y['upper']<=x['upper']+1e-8
            assert y['unresolved_probability']<=x['unresolved_probability']+1e-8
    if isinstance(r.get('budget_allocation'),list):
        assert sum(a['charged_compiles'] for a in r['budget_allocation'])==r['counters']['charged_compiles']
        assert r['unspent_compiles']+r['counters']['charged_compiles']==2000
        assert all(a['stop_reason']!='cpu' for a in r['budget_allocation'])
    if r['label']=='legacy46':
        vals_equal(r,read(EXP/'reconstruction_v46/natural'/f'{r["index"]:03}_full_policy_oracle.json.gz'))

aggregates={}
for label in sorted(next(iter(by.values()))):
    rs=[x[label] for x in by.values()]
    value={'cpu_mean':mean(r['complete_cost']['process_cpu'] for r in rs),
           'wall_mean':mean(r['complete_cost']['wall'] for r in rs)}
    if label not in ('markov128','full128'):
        value.update(width_mean=mean(gap(r)[1]-gap(r)[0] for r in rs),
            total_charged=sum(r['counters']['charged_compiles'] for r in rs),
            recycled=sum(r.get('recycled_compiles',0) for r in rs),
            exact_alight=sum(r['actions']['alight']['mean'] is not None for r in rs),
            exact_continue=sum(r['actions']['continue']['mean'] is not None for r in rs),
            certifications=sum(r['certified_action'] is not None for r in rs))
    aggregates[label]=value

states=[]
for index,data in sorted(by.items()):
    for kernel in ('incremental','hybrid','bulk'):
        s,p=data[kernel+'_split'],data[kernel+'_recycle']
        vals_equal(s,next(c for c in p['checkpoints'] if c['phase']=='after_continue'))
    s,p=data['bulk_split'],data['bulk_recycle']
    a,c=p['actions']['alight'],p['actions']['continue']
    ga=gap(p)
    sa=next(x for x in p['checkpoints'] if x['phase']=='after_initial_alight')
    co=next(x for x in p['checkpoints'] if x['phase']=='after_continue')
    cp500=next(x for x in p['checkpoints'] if x['phase']=='initial_alight' and x['counters']['charged_compiles']==500)
    delta_counts={k:p['counters'][k]-co['counters'][k] for k in p['counters']}
    pc=p['actions']['alight']['unresolved_probability']
    full=data['full128']['paired']['complete_pair_statistics']
    pop=read(EXP/'reconstruction_v45/banks'/f'{index:03}_absorbing.json.gz')
    states.append(dict(index=index,root=p['root_index'],gap=ga,width=ga[1]-ga[0],
        alight=a,continue_value=c,cpu=p['complete_cost']['process_cpu'],
        cpu_split=s['complete_cost']['process_cpu'],
        width_split=gap(s)[1]-gap(s)[0],recycled=p['recycled_compiles'],
        recycle_counter_increments=delta_counts,
        A_width_reduction_calls501_1000=(cp500['actions']['alight']['upper']-cp500['actions']['alight']['lower'])-
            (sa['actions']['alight']['upper']-sa['actions']['alight']['lower']),
        frozen_action=data['markov128']['selected_action'],
        lower_increase_needed_for_continue=max(0.,-ga[0]),upper_decrease_needed_for_alight=max(0.,ga[1]),
        average_unresolved_range=(a['upper']-a['lower'])/pc if pc else 0,
        kappa_population_gap=pop['paired']['full_bank_mean_gain_interval'][0],
        full128_descriptive=full))

v46=[]
for p in sorted((EXP/'reconstruction_v46/natural').glob('*_full_policy_oracle.json.gz')):
    r=read(p);index=r['index'];g=gap(r)
    kap=read(EXP/'reconstruction_v45/banks'/f'{index:03}_absorbing.json.gz')
    kg=kap['paired']['full_bank_mean_gain_interval'][0]
    v46.append(dict(index=index,root=r['root_index'],gap=g,
        exact=all(v['mean'] is not None for v in r['actions'].values()),
        kappa_population_gap=kg,gap_residual=[g[0]-kg,g[1]-kg],cpu=r['complete_cost']['process_cpu']))
    for label in ('markov128','full128'):
        fresh=read(p.parent/f'{index:03}_{label}.json.gz')
        old=read(EXP/('reconstruction_v45/banks' if label=='markov128' else 'reconstruction_v41/banks')/
                 f'{index:03}_{"markov128" if label=="markov128" else "shallow"}.json.gz')
        sample_equal(fresh,old,label);checks['sample_arm_rows_compared']+=256

structure=read(EXP/'reconstruction_v46/tail_structure.json')
struct={}
for panel in ('frozen','online'):
    rr=structure[panel]
    critical=[r for r in rr if r['critical']]
    candidates=[c for r in critical for g in r['arms']['alight'] for c in g['structure']['candidates']]
    struct[panel]=dict(states=len(rr),critical=len(critical),
        pure_physical=sum(all(g['structure']['pure_physical_terminal_platform'] for g in r['arms']['alight']) for r in rr),
        critical_pure_physical=sum(all(g['structure']['pure_physical_terminal_platform'] for g in r['arms']['alight']) for r in critical),
        critical_A_candidates=len(candidates),critical_A_declared_terminal=sum(c['pure_terminal'] for c in candidates),
        critical_A_physical_terminal=sum(c['pure_physical_terminal'] for c in candidates))

out=dict(commit='6e46267b56656595d2f996bc206ab65c5ee222d8',checks=checks,aggregates=aggregates,
         states=states,v46=v46,structure_metadata_recount=struct,
         scope='Receipt statistics and comparisons; source timing retained, no timing rerun.')
(OUT/'recomputed.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':checks,'aggregates':aggregates,'structure':struct},ensure_ascii=False,indent=2))
