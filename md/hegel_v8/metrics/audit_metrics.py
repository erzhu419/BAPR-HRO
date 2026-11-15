"""Independent read-only receipt audit for pinned V45; no policy reruns."""
from pathlib import Path
import os,json,gzip,csv,statistics,math,collections

ROOT=Path(os.environ.get('BAPR_HRO_REPO','/workspace/scratch/fde8f7d289c2/repos/BAPR-HRO-0022df2'))
OUT=Path(__file__).resolve().parent
V=ROOT/'experiments/reconstruction_v45'
LABELS=('rolling_fixed','full128','matched128','markov128','absorbing')
def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def dump(name,d): (OUT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def mean(xs):return statistics.mean(xs) if xs else None
def stats(xs):
    a=sorted(xs);n=len(a)
    if not n:return {'n':0}
    def quant(q):
        u=(n-1)*q;i=int(u);return a[i]+(a[min(i+1,n-1)]-a[i])*(u-i)
    return dict(n=n,total=sum(a),mean=mean(a),median=quant(.5),q95=quant(.95),minimum=a[0],maximum=a[-1])
def physical(d):
    o=d['outcome'];return dict(loss=o['loss'],status=o['status'],decisions=o['decisions'],trace=[{k:v for k,v in e.items() if k!='receipt'} for e in o['trace']])
def equal_values(a,b):
    return all(a.get(k)==b.get(k) for k in ('lower','upper','mean','unresolved_probability'))
def bank_parity(a,b):
    for key in ('seed','count','generated_count','future_messages','public_history','clock','deadline'):
        assert a[key]==b[key],key
    assert a['count']==128 and a['generated_count']==128
    for arm in ('alight','continue'):
        assert len(a['rows'][arm])==len(b['rows'][arm])==128
        for j,(x,y) in enumerate(zip(a['rows'][arm],b['rows'][arm])):
            assert x['world_index']==y['world_index']==j
            assert x['prefix']==y['prefix'],('prefix',arm,j)
            assert equal_values(x['value'],y['value']),('value',arm,j)
    return True
def pure_terminal_shape(node):
    if not node or node[0]!='platform' or not node[2]:return False
    return all(ride[0]=='ride' and len(ride[3])==1 and ride[3][0][0]==ride[2][-1] and ride[3][0][1] is None for _,ride in node[2])
def platform_shape(atom):
    if atom['kind']!='platform':return None
    n=atom['node_structure']
    return dict(stop=atom['stop'],candidates=len(n[2]),pure_terminal_shape=pure_terminal_shape(n),
                candidate_path_lengths=[len(r[2]) for _,r in n[2]],
                nonterminal_candidates=[cid for cid,r in n[2] if any(child is not None for _,child in r[3])])
def costs_record(d):
    c=d['cost'];r={k:c.get(k,0.) for k in ('prefix_process_cpu','reference_process_cpu','total_process_cpu','total_wall')}
    for sec,keys in [('prefix_decision_stats',('onboard_requests','onboard_hits','onboard_unique')),('prefix_planner_stats',('ride_compiles',)),('reference_planner_stats',('platform_compiles',))]:
        for key in keys:r[sec+'.'+key]=c.get(sec,{}).get(key,0)
    for key in ('queries','hits','compiles'):r['cache_'+key]=d.get('cache_stats',{}).get(key,0)
    for key in ('canonical_queries','canonical_hits','canonical_histories'):r[key]=d.get('canonical_stats',{}).get(key,0)
    return r
def aggregate(rows):
    ks=sorted(set().union(*(r.keys() for r in rows)))
    return {k:sum(r.get(k,0.) for r in rows) for k in ks}

references=[];ref_costs=collections.defaultdict(list);ref_rows=collections.defaultdict(list)
for i in range(148):
    banks={x:read(V/'banks'/f'{i:03}_{x}.json.gz') for x in ('matched128','markov128','absorbing')}
    old=read(ROOT/'experiments/reconstruction_v44/references'/f'{i:03}.json.gz')
    full=read(ROOT/'experiments/reconstruction_v41/banks'/f'{i:03}_shallow.json.gz')
    assert bank_parity(banks['matched128'],old)
    assert bank_parity(banks['markov128'],banks['matched128'])
    assert banks['markov128']['selected_action']==banks['matched128']['selected_action']
    a=banks['absorbing'];m=banks['markov128'];f=full['paired']['complete_pair_statistics']
    assert full['count']==128 and full['seed']==m['seed']
    assert full['paired']['complete_pairs']==128 and full['paired']['unresolved_pairs']==0
    assert a['count']==a['generated_count']==0
    assert a['structural']['eligible'] and m['structural']['eligible']
    mass={arm:sum(x['probability'] for x in a['atoms'][arm]) for arm in ('alight','continue')}
    assert all(abs(v-1)<1e-8 for v in mass.values())
    assert all(x['probability']>=0 for xs in a['atoms'].values() for x in xs)
    lo,hi=a['full_pi_difference_interval'];assert lo<=0<=hi
    ar=a['atoms']['alight'];assert len(ar)==1 and ar[0]['probability']==1
    gap={lab:b['paired']['mean_gain'] for lab,b in banks.items()}
    gap['full128']=f['mean_gain']
    sel={lab:b['selected_action'] for lab,b in banks.items()};sel['full128']=full['selected_action']
    r=dict(index=i,root_index=m['root_index'],source=m['source'],clock=m['clock'],deadline=m['deadline'],
        base_action=m['base_action'],selected=sel,gaps=gap,
        full_mc_se=f['se'],matched_se=m['paired']['se'],
        matched_minus_full_mc=gap['matched128']-gap['full128'],
        absorbing_minus_full_mc=gap['absorbing']-gap['full128'],
        absorbing_minus_matched=gap['absorbing']-gap['matched128'],
        exact_full_pi_gap_bounds=[lo,hi],bound_width=hi-lo,
        alight_shape=platform_shape(ar[0]),
        continue_terminal_shape_probability=sum(x['probability'] for x in a['atoms']['continue'] if x['kind']=='platform' and pure_terminal_shape(x['node_structure'])),
        continue_platform_probability=sum(x['probability'] for x in a['atoms']['continue'] if x['kind']=='platform'),
        continue_direct_terminal_probability=sum(x['probability'] for x in a['atoms']['continue'] if x['kind']=='terminal'),
        alight_value=a['actions']['alight']['mean'],continue_value=a['actions']['continue']['mean'],
        continue_atoms=len(a['atoms']['continue']),
        max_candidates=a.get('root_independence',{}).get('max_candidates'))
    references.append(r)
    for lab,b in banks.items():
        ref_costs[lab].append(b['complete_cost']);ref_rows[lab].append(costs_record(b))

roots=[];online_gates=[];online_costs=collections.defaultdict(list);counts=collections.Counter()
for i in range(72):
    rec={lab:read(V/'roots'/f'{i:03}_{lab}.json.gz') for lab in LABELS}
    for lab in LABELS:
        d=rec[lab];assert not d['saved_answers_reused'];assert d['root_index']==i
        c=d['cost'];assert abs(c['total_process_cpu']-c['decision_cpu_sum']-c['final_execution_cpu'])<1e-8
        assert abs(c['total_wall']-c['decision_wall_sum']-c['final_execution_wall'])<1e-8
        if lab!='rolling_fixed':assert physical(d)==physical(rec['full128']),('physical',i,lab)
    ng=len(rec['matched128']['gates'])
    assert all(len(rec[x]['gates'])==ng for x in ('full128','markov128','absorbing'))
    for j in range(ng):
        g={lab:rec[lab]['gates'][j] for lab in ('full128','matched128','markov128','absorbing')}
        assert bank_parity(g['markov128'],g['matched128'])
        for lab,x in g.items():
            assert all(x[k]==g['full128'][k] for k in ('cid','clock','public_history','base_action','selected_action')),(i,j,lab)
            if lab!='absorbing':
                assert x['count']==128 and x['seed']==g['full128']['seed']
            online_costs[lab].append(costs_record(x))
        a=g['absorbing'];lo,hi=a['full_pi_difference_interval'];assert lo<=0<=hi
        online_gates.append(dict(root_index=i,gate_index=j,cid=a['cid'],clock=a['clock'],source=a['source'],
            base_action=a['base_action'],selected_action=a['selected_action'],
            changed=a['base_action']!=a['selected_action'],
            gaps={lab:(x['paired'].get('mean_gain') if 'mean_gain' in x['paired'] else x['paired']['complete_pair_statistics']['mean_gain']) for lab,x in g.items()},
            alight_shape=platform_shape(a['atoms']['alight'][0]),
            bound_width=hi-lo,
            continue_terminal_shape_probability=sum(x['probability'] for x in a['atoms']['continue'] if x['kind']=='platform' and pure_terminal_shape(x['node_structure'])),
            continue_platform_probability=sum(x['probability'] for x in a['atoms']['continue'] if x['kind']=='platform')))
    base=rec['rolling_fixed'];r={k:base[k] for k in ('root_index','day','origin','goal')}
    for lab,d in rec.items():
        r[lab]=dict(loss=d['outcome']['loss'],cpu=d['cost']['total_process_cpu'],wall=d['cost']['total_wall'],
            max_decision_cpu=max(x['process_cpu'] for x in d['decision_costs']),
            max_decision_wall=max(x['wall'] for x in d['decision_costs']),decisions=len(d['decision_costs']))
    r['gain']=r['rolling_fixed']['loss']-r['markov128']['loss'];r['changed_gates']=sum(x['base_action']!=x['selected_action'] for x in rec['markov128']['gates'])
    roots.append(r)

bydate=collections.defaultdict(list);byod=collections.defaultdict(list)
for r in roots:bydate[r['day']].append(r['gain']);byod[r['origin']+' -> '+r['goal']].append(r['gain'])
summary=dict(source_commit='0022df22e775c677e0cf6ebeb71901b7fe8d9dbd',scope='All saved V45 receipts read independently; no policy rerun',
    references=len(references),reference_measurements=3*len(references),online_roots=len(roots),online_measurements=5*len(roots),online_comparisons=len(online_gates),
    original_sample_prefix_value_checks=128*2*(148+len(online_gates)),all_assertions_passed=True,
    controls={lab:{field:stats([r[lab][field] for r in roots]) for field in ('loss','cpu','wall','max_decision_cpu','max_decision_wall')} for lab in LABELS},
    gain=stats([r['gain'] for r in roots]),wins=sum(r['gain']>0 for r in roots),losses=sum(r['gain']<0 for r in roots),ties=sum(r['gain']==0 for r in roots),
    by_date={k:stats(v) for k,v in bydate.items()},by_od={k:stats(v) for k,v in byod.items()},
    nonzero_roots=[r['root_index'] for r in roots if r['gain']],changed_ties=[r['root_index'] for r in roots if not r['gain'] and r['changed_gates']],
    slower_roots_vs_matched={lab:{metric:[r['root_index'] for r in roots if r[lab][metric]>r['matched128'][metric]] for metric in ('cpu','wall')} for lab in ('markov128','absorbing')},
    budgets={str(b):{lab:{metric:sum(r[lab]['max_decision_'+metric]<=b for r in roots) for metric in ('cpu','wall')} for lab in LABELS} for b in (.05,.1,.25,.5,1.)},
    reference_costs={lab:{k:stats([r[k] for r in vals]) for k in ('process_cpu','wall')} for lab,vals in ref_costs.items()},
    reference_mechanism={lab:aggregate(v) for lab,v in ref_rows.items()},online_mechanism={lab:aggregate(v) for lab,v in online_costs.items()},
    value_diagnostics={field:stats([r[field] for r in references]) for field in ('matched_minus_full_mc','absorbing_minus_full_mc','absorbing_minus_matched','bound_width')},
    mae_vs_full_mc={x:mean([abs(r[x+'_minus_full_mc']) for r in references]) for x in ('matched','absorbing')},
    pure_terminal_shape_scope='Topology only: every platform candidate has only destination exit; independence, P0/W, forecast selection conditions not certified by this count',
    reference_alight_terminal_shape_indices=[r['index'] for r in references if r['alight_shape'] and r['alight_shape']['pure_terminal_shape']],
    online_alight_terminal_shape=[{'root_index':r['root_index'],'gate_index':r['gate_index'],'changed':r['changed']} for r in online_gates if r['alight_shape'] and r['alight_shape']['pure_terminal_shape']],
    max_candidates_histogram=dict(collections.Counter(r['max_candidates'] for r in references)))
dump('recomputed.json',summary);dump('states.json',references);dump('roots.json',roots);dump('online_gates.json',online_gates)
print(json.dumps({k:summary[k] for k in ('references','online_roots','online_comparisons','all_assertions_passed','gain','wins','losses','ties','nonzero_roots','changed_ties','mae_vs_full_mc','reference_alight_terminal_shape_indices','online_alight_terminal_shape','max_candidates_histogram')},ensure_ascii=False,indent=2))
