"""Read-only V41 error decomposition over every frozen state and online gate.

All signed differences are ALIGHT minus CONTINUE. The FrozenCombined hybrid
replaces only the current ALIGHT score; it is a diagnostic, not a policy run.
"""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V41 = HERE.parent/'reconstruction_v41'


def read(path):
    with gzip.open(path,'rt') if str(path).endswith('.gz') else path.open() as handle:
        return json.load(handle)


def numeric(values):
    return dict(count=len(values),total=float(sum(values)),mean=float(np.mean(values)) if values else None,
        median=float(np.median(values)) if values else None,minimum=min(values,default=None),maximum=max(values,default=None))


def scalar_action(alight, continuation, cap):
    return 'alight' if alight<cap and alight<=continuation+1e-9 else 'continue'


def decompose(receipt, rows, clock, deadline, frozen_alight=None):
    """Recompute paired uncertainty, keeping covariance between both arms."""
    left=[row['loss'] for row in rows['alight']]
    right=[row['loss'] for row in rows['continue']]
    if len(left)!=len(right) or any(value is None for value in left+right):
        raise ValueError('This attribution requires all frozen V41 pairs; do not discard missing mass')
    a,c=np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    delta=a-c
    mean_a,mean_c=float(a.mean()),float(c.mean())
    gap=float(delta.mean())
    se=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None
    cheap_a,cheap_c=receipt['alight']-clock,receipt['continuation']-clock
    cap=deadline-clock
    result=dict(worlds=len(delta),cheap_alight=cheap_a,cheap_continue=cheap_c,
        cheap_gap=cheap_a-cheap_c,mc_alight=mean_a,mc_continue=mean_c,mc_gap=gap,
        mc_paired_se=se,mc_ci95=None if se is None else [gap-1.96*se,gap+1.96*se],
        alight_error=cheap_a-mean_a,continue_error=cheap_c-mean_c,
        gap_error=(cheap_a-cheap_c)-gap,
        cheap_rank=scalar_action(cheap_a,cheap_c,cap),mc_gap_zero=gap==0,
        finite_bank_equal_worlds=int((delta==0).sum()))
    # Match V41 choose(): exact zero bank difference retains the base action.
    result['mc_rank']='alight' if gap<0 else 'continue' if gap>0 else result['cheap_rank']
    result['cheap_mc_boundary_crossing']=result['cheap_rank']!=result['mc_rank']
    if frozen_alight is not None:
        hybrid_rank=scalar_action(frozen_alight,cheap_c,cap)
        result.update(frozen_current_alight=frozen_alight,
            current_alight_correction=frozen_alight-cheap_a,
            frozen_alight_error=frozen_alight-mean_a,
            hybrid_gap=frozen_alight-cheap_c,
            hybrid_gap_error=frozen_alight-cheap_c-gap,
            hybrid_rank=hybrid_rank,
            hybrid_changes_cheap_rank=hybrid_rank!=result['cheap_rank'],
            hybrid_agrees_mc_rank=hybrid_rank==result['mc_rank'])
    return result


def menu_counts(node,cid):
    """Syntactic occurrences; identical physical trips are not counted as independent."""
    counts=dict(ride_occurrences=0,platform_occurrences=0,ride_exit_occurrences=0,
                nonterminal_exit_occurrences=0,terminal_exit_occurrences=0,
                max_ride_exits=0,max_platform_candidates=0,max_transfer_depth=0)
    def visit(item,depth):
        if item['kind']=='ride':
            counts['ride_occurrences']+=1
            exits=item['exits']
            counts['ride_exit_occurrences']+=len(exits)
            counts['max_ride_exits']=max(counts['max_ride_exits'],len(exits))
            for row in exits:
                if row['next'] is None:
                    counts['terminal_exit_occurrences']+=1
                else:
                    counts['nonterminal_exit_occurrences']+=1
                    visit(row['next'],depth)
        else:
            counts['platform_occurrences']+=1
            counts['max_transfer_depth']=max(counts['max_transfer_depth'],depth+1)
            counts['max_platform_candidates']=max(counts['max_platform_candidates'],len(item['rides']))
            for row in item['rides']:
                visit(row['next'],depth+1)
    path=node['path'][node['path'].index(cid):]
    exits=[row for row in node['exits'] if row['cid'] in path]
    view=dict(node,path=path,exits=exits)
    visit(view,0)
    counts.update(current_and_future_exit_count=len(exits),future_exit_count=sum(row['cid']!=cid for row in exits))
    return counts


def attribution_summary(rows):
    names=('cheap_gap','mc_gap','mc_paired_se','alight_error','continue_error','gap_error')
    available=[r for r in rows if 'frozen_current_alight' in r]
    return dict(denominator=len(rows),values={name:numeric([r[name] for r in rows if r[name] is not None]) for name in names},
        absolute_errors={name:numeric([abs(r[name]) for r in rows]) for name in ('alight_error','continue_error','gap_error')},
        cheap_mc_boundary_crossings=sum(r['cheap_mc_boundary_crossing'] for r in rows),
        exactly_zero_mc_gaps=sum(r['mc_gap_zero'] for r in rows),
        frozen_current_alight_available=len(available),
        hybrid_changes_cheap_rank=sum(r['hybrid_changes_cheap_rank'] for r in available),
        hybrid_agrees_mc_rank=sum(r['hybrid_agrees_mc_rank'] for r in available),
        hybrid_resolves_existing_crossing=sum(r['cheap_mc_boundary_crossing'] and r['hybrid_agrees_mc_rank'] for r in available),
        hybrid_introduces_new_crossing=sum(not r['cheap_mc_boundary_crossing'] and not r['hybrid_agrees_mc_rank'] for r in available),
        corrected_gap_absolute_error=numeric([abs(r['hybrid_gap_error']) for r in available]))


def historical_rows():
    source=read(V41/'summary.json')
    labels=('legacy_cheap','physical_cheap','shallow_cheap','shallow_rollout128')
    rows=[]
    for case in source['root_cases']:
        values={label:case['loss'][label] for label in labels}
        if any(value is None for value in values.values()):
            raise ValueError('Historical V41 cohort contains unresolved loss')
        old,physical,shallow,full=(values[label] for label in labels)
        rows.append(dict(root_index=case['root_index'],day=case['day'],origin=case['origin'],goal=case['goal'],
            losses=values,model_gain=old-physical,shallow_gain=physical-shallow,residual_gain=shallow-full,
            final_vs_legacy_gain=old-full,
            returns_to_legacy_loss=full==old and shallow>old))
    return rows


def main():
    banks=[]
    by_source={}
    for index in range(148):
        bank=read(V41/'banks'/f'{index:03}_shallow.json.gz')
        tail=read(V41/'tails'/f'{index:03}.json')
        state=bank['state']
        if bank['index']!=index or tail['index']!=index or bank['count']!=128:
            raise ValueError(f'Frozen state identity mismatch: {index}')
        row=dict(record_type='frozen_state',index=index,root_index=bank['root_index'],source=bank['source'],
            day=state['day'],cid=state['cid'],clock=state['clock'],deadline=state['deadline'],
            base_action=bank['base_action'],selected_action=bank['selected_action'],
            selected_changed=bank['base_action']!=bank['selected_action'],
            **decompose(bank['base_receipt'],bank['rows'],state['clock'],state['deadline'],tail['frozen_combined']),
            first_offer_current_alight=tail['first'],shallow_envelope_current_alight=tail['envelope'],
            frozen_diagnostic_cpu=tail['cost']['process_cpu'],menu=menu_counts(state['node'],state['cid']))
        if abs(row['cheap_alight']-tail['envelope'])>1e-7:
            raise ValueError(f'Tail diagnostic is not the current ALIGHT score: {index}')
        banks.append(row)
        by_source[bank['source']]=(row,bank)
    roots=historical_rows()
    root_by_index={row['root_index']:row for row in roots}
    selection=read(HERE.parent/'reconstruction_v34/selection.json')['selected']
    online=[]
    for index in range(72):
        root=read(V41/'roots'/f'{index:03}_shallow_rollout128.json.gz')
        for gate_index,gate in enumerate(root['gates']):
            frozen=None
            if gate['source'] is not None:
                record,bank=by_source[gate['source']]
                if gate['clock']!=record['clock'] or gate['cid']!=record['cid'] or gate['public_history']!=bank['public_history']:
                    raise ValueError(f'Online source does not identify identical public state: {index}/{gate_index}')
                frozen=record['frozen_current_alight']
            row=dict(record_type='online_gate',root_index=index,gate_index=gate_index,day=root['day'],
                source=gate['source'],cid=gate['cid'],clock=gate['clock'],deadline=selection[index]['deadline'],
                base_action=gate['base_action'],selected_action=gate['selected_action'],
                selected_changed=gate['base_action']!=gate['selected_action'],
                root_gain=root_by_index[index]['residual_gain'])
            row.update(decompose(gate['base_receipt'],gate['rows'],gate['clock'],row['deadline'],frozen))
            online.append(row)
    changed=[row for row in online if row['selected_changed']]
    recoveries=[row for row in roots if row['returns_to_legacy_loss']]
    residual=sum(row['residual_gain'] for row in roots)
    recovery=sum(row['residual_gain'] for row in recoveries)
    result=dict(stage='V43 existing-evidence attribution',source='Frozen V41 receipts; no new worlds or policies executed',
        orientation='ALIGHT minus CONTINUE; signed score error = cheap minus full conditional Monte Carlo',
        frozen_current_scope='Exact fixed-Combined at current platform with frozen child marks. Hybrid only replaces current ALIGHT; not online full future-time matched control.',
        uncertainty_scope='Paired ordinary fixed-sample MC SE and normal interval; not a certificate or independent real-world confirmation.',
        frozen_summary=attribution_summary(banks),online_summary=attribution_summary(online),
        frozen_states=banks,online_gates=online,changed_online_gates=changed,
        changed_roots=[row['root_index'] for row in changed],
        changed_but_zero_outcome_roots=sorted({row['root_index'] for row in changed if row['root_gain']==0}),
        nonzero_outcome_roots=[row['root_index'] for row in roots if row['residual_gain']!=0],
        historical_roots=roots,old_negative_roots=[row for row in roots if row['final_vs_legacy_gain']<0],
        historical_attribution=dict(denominator=len(roots),residual_net=residual,residual_mean=residual/len(roots),
            returns_to_legacy= recoveries,return_to_legacy_gain=recovery,
            return_share_of_net=recovery/residual,other_residual_net=residual-recovery,
            final_vs_legacy_net=sum(row['final_vs_legacy_gain'] for row in roots),
            final_vs_legacy_wins=sum(row['final_vs_legacy_gain']>0 for row in roots),
            final_vs_legacy_harms=sum(row['final_vs_legacy_gain']<0 for row in roots),
            final_vs_legacy_ties=sum(row['final_vs_legacy_gain']==0 for row in roots),
            scope='Arithmetic across retained versions, not a causal partition of algorithm effects.'),
        static_menu_summary={name:numeric([row['menu'][name] for row in banks]) for name in banks[0]['menu']})
    HERE.mkdir(exist_ok=True)
    (HERE/'existing_attribution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    fields=('record_type','index','root_index','gate_index','cid','clock','base_action','selected_action','selected_changed',
        'cheap_alight','cheap_continue','mc_alight','mc_continue','cheap_gap','mc_gap','mc_paired_se',
        'alight_error','continue_error','gap_error','cheap_mc_boundary_crossing','frozen_current_alight',
        'current_alight_correction','hybrid_gap','hybrid_gap_error','hybrid_rank','hybrid_changes_cheap_rank',
        'hybrid_agrees_mc_rank','root_gain')
    with (HERE/'existing_attribution.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(banks+online)
    print(json.dumps({k:result[k] for k in ('frozen_summary','online_summary','changed_roots','changed_but_zero_outcome_roots','nonzero_outcome_roots')},indent=2))


if __name__=='__main__':
    main()
