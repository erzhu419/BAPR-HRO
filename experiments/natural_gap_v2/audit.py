"""Independent arithmetic over completed saved receipts; never runs a policy."""
import gzip
import json
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
D = parser.parse_args().directory
def read(path):
    with gzip.open(path,'rt') if path.suffix=='.gz' else path.open() as f:
        return json.load(f)
def close(a,b):
    assert abs(a-b)<1e-8,(a,b)
def limits(row):
    y=row['outcome']['loss']
    return [0.,row['loss_cap']] if y is None else [y,y]
def timing(rows,key):
    values=[r['cost'][key] for r in rows]
    return mean(values) if values and all(v is not None for v in values) else None
def comparison(banks,left,right):
    paired=[]
    for bank in banks:
        a,b=bank[left],bank[right]; x,y=limits(a),limits(b)
        lo,hi=x[0]-y[1],x[1]-y[0]
        status='win' if lo>0 else 'harm' if hi<0 else 'tie' if lo==hi==0 else 'unresolved'
        paired.append(dict(root_index=a['root_index'],day=a['day'],gain_bounds=[lo,hi],classification=status,
            complete=a['outcome']['loss'] is not None and b['outcome']['loss'] is not None))
    counts=Counter(p['classification'] for p in paired)
    return dict(counts=dict(counts),mean_gain_bounds=[mean(p['gain_bounds'][j] for p in paired) for j in (0,1)],
        positive=[p for p in paired if p['classification']=='win'],
        negative=[p for p in paired if p['classification']=='harm'],
        unresolved=[p for p in paired if p['classification']=='unresolved'])

contract=read(D/'contract.json'); selection=read(D/'selection.json')['selected']; summary=read(D/'summary.json')
labels=['new_'+p for p in contract['policies']]; expected=contract['roots']
assert len(selection)==expected
banks=[]; audit=Counter(); gate_cpu=[]; gate_kernels=Counter(); unknown_worlds=Counter(); equal_full={}; detailed=[]; evidence=[]
for i,selected in enumerate(selection):
    bank={label:read(D/'roots'/f'{i:03}_{label}.json.gz') for label in labels}
    reference=bank['new_cheap']; initial=[]
    for label,row in bank.items():
        assert (row['root_index'],row['root_id'],row['day'],row['label'])==(i,selected['root_id'],selected['day'],label)
        assert row['loss_cap']==selected['deadline']-selected['start']
        assert row['observation']==reference['observation'] and row['coverage']==reference['coverage']
        assert row['cost']['shared_menu']==reference['cost']['shared_menu']
        assert row['emission_model']=='repaired_fit' and row['emission_source']=='repaired_fit.json'
        assert row['relevant_remote_frames']==reference['relevant_remote_frames']
        for metric in ('process_cpu','wall'):
            raw,total,shared=[row['cost'][k] for k in (metric,metric+'_including_menu','shared_menu')]
            if row.get('executed',True):close(raw+shared[metric],total)
            else: assert raw is total is None
        first=next((s for s in row['states'] if s['kind']=='onboard'),None)
        initial.append(None if first is None else {k:first[k] for k in ('cid','clock','public_history')})
        for state in row['states']:
            for entry in state['public_history']:
                clock=entry.get('receipt_seconds') if entry['evidence']=='forecast' else entry.get('clock')
                if clock is not None: assert clock<=state['clock']
                audit['saved_public_history_entries']+=1
        for trace in row['outcome']['trace']:
            for message in trace.get('message_receipt',{}).get('receipts',[]):
                assert message['receipt_seconds']<=message['consumed_at_clock']
                audit['saved_message_receipts']+=1
        for n,gate in enumerate(row['gates']):
            audit['gates']+=1; gate_cpu.append(gate['cost']['process_cpu'])
            gate_kernels[label+'/'+gate['used_kernel']]+=1
            assert gate['count']==contract['rollout_limits']['worlds_per_query'] and gate['seed']==contract['rollout_limits']['seed']
            assert gate['future_messages']==0 and gate['root_index']==i
            assert set(gate['rows'])=={'alight','continue'}
            for arm,values in gate['rows'].items():
                assert [r['world_index'] for r in values]==list(range(gate['count']))
                unknown_worlds[label]+=sum(v.get('loss') is None if 'loss' in v else v['value']['lower']!=v['value']['upper'] for v in values)
            audit['unfinished_banks']+=gate['completed_pairs']<gate['count']
            state=row['states'][gate['state_index']]
            assert state['gate_index']==n and state['selected_action']==gate['selected_action']
            if gate['used_kernel']=='full_common_observation_pi' and gate['completed_pairs']==gate['count']:
                public=dict({k:v for k,v in state.items() if k not in ('selected_action','base_action','gate_index')},day=row['day'])
                key=json.dumps(public,sort_keys=True,separators=(',',':'))
                science={k:gate[k] for k in ('rows','paired','common_tape')}
                if key in equal_full:
                    assert science==equal_full[key];audit['same_state_full_bank_equalities']+=1
                else:equal_full[key]=science
    assert all(s==initial[0] for s in initial)
    audit['common_first_observation_roots']+=1
    c=reference['coverage']
    evidence.append(dict(root_index=i,day=reference['day'],initial_state=c['initial_state'],
        current_numeric=c.get('predecision_numeric_messages'),selected_numeric=c.get('model_selected_numeric_messages'),
        other_trip_numeric=c.get('other_trip_numeric_messages'),
        incoming_trip=None if initial[0] is None else next((e['trip'] for e in initial[0]['public_history']
            if e['evidence']=='observed' and e.get('cid')==initial[0]['cid'] and e.get('kind')=='arrival'),None)))
    audit['matched_full_equal_actions']+=bank['new_matched128']['actions']==bank['new_full128']['actions']
    matched_loss,full_loss=(bank[label]['outcome']['loss'] for label in ('new_matched128','new_full128'))
    audit['complete_equal_loss']+=matched_loss is not None and full_loss is not None and matched_loss==full_loss
    audit['shared_unknown_loss']+=matched_loss is None and full_loss is None
    for challenger in labels[6:]:
        a,b=reference,bank[challenger]
        if a['actions']!=b['actions']:
            changes=[(j,x,y) for j,(x,y) in enumerate(zip(a['actions'],b['actions'])) if x!=y]
            j,x,y=changes[0] if changes else (min(len(a['actions']),len(b['actions'])),None,None)
            state=next((s for s in b['states'] if y and s['kind']==y['kind'] and s['clock']==y['clock'] and (y['kind']!='onboard' or s['cid']==y['cid'])),None)
            gate=None if state is None or 'gate_index' not in state else b['gates'][state['gate_index']]
            detailed.append(dict(root_index=i,day=a['day'],root_id=a['root_id'],challenger=challenger,
                origin=a['origin'],goal=a['goal'],losses={p:bank[p]['outcome']['loss'] for p in labels},
                first_difference=dict(index=j,cheap=x,challenger=y),
                challenger_gate=None if gate is None else {k:gate[k] for k in ('base_action','selected_action','used_kernel','paired')}))
    banks.append({label:{k:row[k] for k in ('root_index','day','loss_cap','outcome','cost')}
                  for label,row in bank.items()})
    for row in banks[-1].values():
        row['outcome']={k:row['outcome'][k] for k in ('status','loss')}

aggregate={}; all_pair_matches=0
for label in labels:
    rows=[b[label] for b in banks]; bs=[limits(r) for r in rows]
    aggregate[label]=dict(mean_loss_bounds=[mean(v[j] for v in bs) for j in (0,1)],
        mean_billed_cpu=timing(rows,'process_cpu_including_menu'),mean_billed_wall=timing(rows,'wall_including_menu'),
        unknown_roots=[r['root_index'] for r in rows if r['outcome']['loss'] is None],
        unknown_reasons=[dict(root_index=r['root_index'],reason=r['outcome']['status']) for r in rows if r['outcome']['loss'] is None],
        status_counts=dict(Counter(r['outcome']['status'] for r in rows)))
    s=summary['aggregate'][label]
    for j in (0,1):close(aggregate[label]['mean_loss_bounds'][j],s['all_root_mean_loss_bounds'][j])
    for key,name in [('mean_billed_cpu','mean_cpu'),('mean_billed_wall','mean_wall')]:
        if aggregate[label][key] is None:assert s[name] is None
        else:close(aggregate[label][key],s[name])
    assert aggregate[label]['unknown_roots']==s['unresolved_roots']
for a,b in combinations(labels,2):
    own=comparison(banks,a,b); saved=summary['comparisons'][a+'__'+b]
    for j in (0,1):close(own['mean_gain_bounds'][j],saved['all_root_mean_gain_bounds'][j])
    for key,plural in [('win','wins'),('harm','harms'),('tie','ties'),('unresolved','undecided')]:assert own['counts'].get(key,0)==saved[plural]
    all_pair_matches+=1
by_day={}
for day in contract['control_dates']:
    rows=[b for b in banks if b['new_cheap']['day']==day]
    assert len(rows)==summary['by_day'][day]['registered_roots']
    by_day[day]=dict(registered_roots=len(rows),full_vs_each_cheap={p:comparison(rows,p,'new_full128') for p in labels[:6]})
residual={}
for challenger in labels[6:]:
    against={p:comparison(banks,p,challenger) for p in labels[:6]}
    positive=set.intersection(*(set(v['root_index'] for v in c['positive'] if v['complete']) for c in against.values()))
    assert sorted(positive)==summary['residual_diagnostic'][challenger]['strictly_better_than_every_cheap_policy_roots']
    residual[challenger]=dict(versus_each_fixed_cheap=against,strict_positive_roots=sorted(positive))
out=dict(scope='Independent saved-receipt audit; no replay, resampling, refitting or budget extension.',
    dates=len(contract['control_dates']),roots=expected,receipts=expected*len(labels),aggregate=aggregate,
    all_28_paired_comparisons_match=all_pair_matches==28,checks=dict(audit),
    gate_kernels=dict(gate_kernels),nonpoint_or_unresolved_simulated_arm_rows=dict(unknown_worlds),
    maximum_gate_cpu=max(gate_cpu,default=None),declared_limits=contract['rollout_limits'],
    residual=residual,by_day=by_day,action_difference_mechanisms=detailed,
    first_decision_coverage=dict(observed_roots=sum(r['initial_state']['status']=='observed' for r in evidence),
        status_counts=dict(Counter(r['initial_state']['status'] for r in evidence)),
        roots_with_current_numeric=sum(bool(r['current_numeric']) for r in evidence),
        roots_with_selected_numeric=sum(bool(r['selected_numeric']) for r in evidence),
        roots_with_other_trip_numeric=sum(bool(r['other_trip_numeric']) for r in evidence),rows=evidence),
    limitation='Strict complete-root residuals are discovery mechanisms; unknown losses retain full-cap bounds. No population expectation, actual platform guarantee or G2 is inferred.')
out['known_archive_clock_anomaly']=dict(day='2023-03-23',incoming_cid=296,next_cid=314,
    affected_menu_roots=[5,9],saved_outcomes=[dict(root_index=i,
        statuses={label:banks[i][label]['outcome']['status'] for label in labels},
        losses={label:banks[i][label]['outcome']['loss'] for label in labels}) for i in (5,9)],
    interpretation='Both roots have eight arrived outcomes with equal losses. The clock-reversing same-trip segment is an untraversed menu branch; the fixed executor would return service_or_clock_failure if it traversed that segment. It cannot create a strict positive root here.')
assert all(banks[i][label]['outcome']['status']=='arrived' for i in (5,9) for label in labels)
with (D/'audit.json').open('w') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(roots=expected,receipts=expected*len(labels),checks=dict(audit),maximum_gate_cpu=out['maximum_gate_cpu'],
    aggregate=aggregate,residual={k:v['strict_positive_roots'] for k,v in residual.items()},mechanism_roots=sorted(set(v['root_index'] for v in detailed)))))
