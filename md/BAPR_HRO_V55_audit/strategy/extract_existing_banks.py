"""Read-only V51 paired-bank diagnostic; never independent confirmation.

Uses saved physical losses, not the project summarizer.  The same banks were
used to select actions, so the normal intervals are descriptive only.
"""
import os, json, gzip, math, statistics
from pathlib import Path

REPO=Path(os.environ['BAPR_HRO_REPO']).resolve()
HERE=Path(__file__).resolve().parent

def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)

def main():
    rows=[]
    for root in range(72):
        j=read(REPO/f'experiments/reconstruction_v51/roots/{root:03}_full128.json.gz')
        cheap=read(REPO/f'experiments/reconstruction_v51/roots/{root:03}_cheap.json.gz')
        for gi,g in enumerate(j['gates']):
            aa=g['rows']['alight'];cc=g['rows']['continue']
            assert len(aa)==len(cc)==128
            assert all(a['world_index']==c['world_index'] for a,c in zip(aa,cc))
            ds=[a['loss']-c['loss'] for a,c in zip(aa,cc) if a['loss'] is not None and c['loss'] is not None]
            mean=statistics.mean(ds);se=statistics.stdev(ds)/math.sqrt(len(ds))
            s=g['paired']['complete_pair_statistics']
            assert math.isclose(mean,s['mean_gain'],rel_tol=1e-12,abs_tol=1e-10)
            assert math.isclose(se,s['se'],rel_tol=1e-12,abs_tol=1e-10)
            changed=g['base_action']!=g['selected_action']
            sign=1 if g['base_action']=='alight' else -1
            row=dict(root_index=root,day=j['day'],origin=j['origin'],goal=j['goal'],
                gate_index=gi,state_index=g['state_index'],cid=g['cid'],clock=g['clock'],
                baseline_action=g['base_action'],chosen_action=g['selected_action'],changed=changed,
                n=len(ds),mean_A_minus_C=mean,se=se,normal95_A_minus_C=[mean-1.96*se,mean+1.96*se],
                alternative_gain=sign*mean,normal95_alternative_gain=[sign*mean-1.96*se,sign*mean+1.96*se],
                A_less_C=sum(d<0 for d in ds),A_greater_C=sum(d>0 for d in ds),ties=sum(d==0 for d in ds),
                empirical_min=min(ds),empirical_max=max(ds),future_messages=g['future_messages'],
                actual_cheap_loss=cheap['outcome']['loss'],actual_full_loss=j['outcome']['loss'],
                actual_full_gain=cheap['outcome']['loss']-j['outcome']['loss'],
                source=f'experiments/reconstruction_v51/roots/{root:03}_full128.json.gz#gates/{gi}')
            rows.append(row)
    changed=[r for r in rows if r['changed']]
    out=dict(commit='a2b82be9c3fc6b61e2a5826d7cd8691d39ff4491',
        scope='Saved V51 development banks used for action selection. Normal intervals are descriptive, not independent or simultaneous population certificates. Costs are fixed-root P0, future remote messages frozen, cheap W continuation.',
        queries=len(rows),changed_queries=len(changed),changed_root_indices=sorted({r['root_index'] for r in changed}),
        changed_with_descriptive_positive_lower=sum(r['normal95_alternative_gain'][0]>0 for r in changed),
        rows=rows,changed=changed)
    (HERE/'existing_bank_signals.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('rows','changed')},ensure_ascii=False))
    for r in changed:print(json.dumps({k:r[k] for k in ['root_index','cid','clock','baseline_action','chosen_action','alternative_gain','se','normal95_alternative_gain','actual_full_gain']},ensure_ascii=False))

if __name__=='__main__':main()
