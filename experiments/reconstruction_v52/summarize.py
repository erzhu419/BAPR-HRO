"""All selected public histories, conditional policy gaps and unresolved mass."""
import argparse
import csv
from math import isfinite
from pathlib import Path

import numpy as np

from experiments.reconstruction_v34.discover import read
from experiments.reconstruction_v38.run import save

HERE=Path(__file__).resolve().parent
TOL=1e-8


def valid_value(value,cap):
    lo,hi=value['lower'],value['upper']
    if not all(isfinite(v) for v in (lo,hi)) or lo < -TOL or hi > cap+TOL or lo > hi+TOL:
        raise ValueError('Invalid causal value range')
    if value['complete'] and (value['mean'] is None or abs(hi-lo)>TOL):
        raise ValueError('Incomplete range advertised as exact')
    if not value['complete'] and value['mean'] is not None:
        raise ValueError('Unresolved causal value advertised as a mean')


def gap(cheap,reference,nonnegative=True):
    lo=cheap['lower']-reference['upper']
    hi=cheap['upper']-reference['lower']
    if nonnegative:
        if hi < -TOL:
            raise ValueError('Partial ranges contradict the policy-class ordering')
        lo,hi=max(0.,lo),max(0.,hi)
    return dict(lower=lo,upper=hi,positive_lower=lo>TOL)


def write_csv(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def summarize(implementation='mass_terminal'):
    contract=read(HERE/'contract.json')
    panel=read(HERE/'panel.json.gz')
    modes=contract['modes']
    folder='natural' if implementation=='chronological' else 'natural_optimized'
    records,flat=[],[]
    for item in panel['rows']:
        query=item['query_index']; state=item['state']
        banks={mode:read(HERE/folder/f'{query:03}_{mode}.json.gz') for mode in modes}
        cap=state['deadline']-state['clock']
        record=dict(query_index=query,source_state_index=item['source_state_index'],
            root_index=state['root_index'],day=state['day'],kind=state['kind'],clock=state['clock'],
            legal_actions=state['legal_actions'],selection_reasons=item['selection_reasons'],
            observed_baseline_actions=item['selected_actions'],results={})
        for mode,bank in banks.items():
            if bank['query_index']!=query or bank['source_state_index']!=item['source_state_index']:
                raise ValueError('Receipt input state differs from the frozen panel')
            if bank['clock']!=state['clock'] or bank['deadline']!=state['deadline'] or bank['future_public_frames']!=0:
                raise ValueError('Reference information contract changed')
            if bank['counters']['states']>contract['limits']['states_per_query']:
                raise ValueError('Reference exceeded its state budget')
            valid_value(bank['value'],cap)
            for value in bank['actions'].values():
                valid_value(value,cap)
            certified=bank['certified_action']
            if certified is not None:
                chosen=bank['actions'][str(certified)]
                others=[v['lower'] for a,v in bank['actions'].items() if a!=str(certified)]
                if not others or not chosen['upper']<min(others)-1e-9:
                    raise ValueError('Unsupported optimal-action certificate')
            result=dict(value=bank['value'],actions=bank['actions'],selected_action=bank['selected_action'],
                certified_action=certified,counters=bank['counters'],reasons=bank['reasons'],
                cost=bank['complete_cost'],screening_workers=bank['screening_workers'],
                status=bank.get('status','bounded_reference'))
            record['results'][mode]=result
            flat.append(dict(query_index=query,root_index=state['root_index'],day=state['day'],
                kind=state['kind'],mode=mode,lower=bank['value']['lower'],upper=bank['value']['upper'],
                width=bank['value']['upper']-bank['value']['lower'],complete=bank['value']['complete'],
                selected_action=bank['selected_action'],certified_action=certified,
                states=bank['counters']['states'],cpu=bank['complete_cost']['process_cpu'],
                wall=bank['complete_cost']['wall'],status=result['status']))
        values={mode:banks[mode]['value'] for mode in modes}
        record['consistent_p0_gaps']={mode:gap(values['cheap_p0'],values[mode])
            for mode in ('one_step_p0','two_step_p0','optimal')}
        record['two_step_remaining_gap']=gap(values['two_step_p0'],values['optimal'])
        record['working_vs_consistent_p0']=gap(values['cheap_working'],values['cheap_p0'],False)
        # All modes are causal policies in the same physical problem. The
        # stronger class cannot cost more, even when intervals are incomplete.
        for weaker,stronger in (('cheap_p0','one_step_p0'),('one_step_p0','two_step_p0'),
                                 ('two_step_p0','optimal'),('cheap_working','optimal')):
            gap(values[weaker],values[stronger])
        records.append(record)
    aggregate={}
    for mode in modes:
        rows=[r['results'][mode] for r in records]
        aggregate[mode]=dict(queries=len(rows),complete=sum(r['value']['complete'] for r in rows),
            certified_actions=sum(r['certified_action'] is not None for r in rows),
            mean_width=float(np.mean([r['value']['upper']-r['value']['lower'] for r in rows])),
            mean_cpu=float(np.mean([r['cost']['process_cpu'] for r in rows])),
            mean_wall=float(np.mean([r['cost']['wall'] for r in rows])),
            max_cpu=max(r['cost']['process_cpu'] for r in rows),
            mean_states=float(np.mean([r['counters']['states'] for r in rows])),
            mean_chance_outcomes=float(np.mean([r['counters']['chance_outcomes'] for r in rows])),
            mean_terminal_integrations=float(np.mean([r['counters'].get('terminal_integrations',0) for r in rows])),
            cpu_above_requested_limit=sum(r['cost']['process_cpu']>contract['limits']['process_cpu_seconds_per_query'] for r in rows),
            state_capped=sum(r['counters']['states']==contract['limits']['states_per_query'] for r in rows),
            budget_unknown_queries=sum(r['counters']['budget_unknowns']>0 for r in rows),
            unsupported_queries=sum(r['counters']['unsupported_histories']>0 or r['status']=='unsupported_root_history' for r in rows))
    positives=[dict(query_index=r['query_index'],root_index=r['root_index'],day=r['day'],
        gap=r['consistent_p0_gaps']['optimal'],two_step_remaining=r['two_step_remaining_gap'])
        for r in records if r['consistent_p0_gaps']['optimal']['positive_lower']]
    remaining=[r for r in positives if r['two_step_remaining']['positive_lower']]
    result=dict(stage='V52',implementation=implementation,original_root_denominator=72,represented_roots=panel['represented_roots'],
        all_v51_state_count=panel['all_state_count'],queries=len(records),calls=len(records)*len(modes),
        missing_roots=panel['missing_first_nontrivial_roots'],aggregate=aggregate,records=records,
        certified_positive_cheap_p0_gaps=positives,certified_positive_after_two_step=remaining,
        g2_passed=False,new_solver_gate_established=False,ordinary_method_intrinsic_barrier_established=False,
        whole_journey_optimality_claim=False,server_transfer_bytes=0,
        scope='Input-selected conditional-state screen. All uncompleted ranges retained; no optimal visitation weighting, no future-publication model, no new solver authorized by timeout alone.')
    suffix='_chronological' if implementation=='chronological' else ''
    if implementation=='mass_terminal':
        result['implementation_comparison']=compare_implementations(read(HERE/'summary_chronological.json'),result,modes)
    save(HERE/f'summary{suffix}.json',result)
    write_csv(HERE/f'paired{suffix}.csv',flat)
    plot(flat,modes,suffix)
    print(dict(aggregate=aggregate,positive_gaps=len(positives),remaining_after_two_step=len(remaining)),flush=True)
    return result


def compare_implementations(original,optimized,modes):
    cases=[]
    if original['queries']!=optimized['queries']:
        raise ValueError('Implementation panels differ')
    for before,after in zip(original['records'],optimized['records'],strict=True):
        if (before['query_index'],before['source_state_index'])!=(after['query_index'],after['source_state_index']):
            raise ValueError('Implementation inputs differ')
        for mode in modes:
            a,b=before['results'][mode],after['results'][mode]
            for action in (None,*a['actions']):
                av,bv=(a['value'],b['value']) if action is None else (a['actions'][action],b['actions'][action])
                if max(av['lower'],bv['lower'])>min(av['upper'],bv['upper'])+TOL:
                    raise ValueError('Two purported bounds for the same policy value are disjoint')
            if a['certified_action'] is not None and b['certified_action'] is not None and a['certified_action']!=b['certified_action']:
                raise ValueError('Implementation action certificates conflict')
            aw=a['value']['upper']-a['value']['lower'];bw=b['value']['upper']-b['value']['lower']
            cases.append(dict(query_index=after['query_index'],root_index=after['root_index'],mode=mode,
                old_width=aw,new_width=bw,width_reduction=aw-bw,
                old_cpu=a['cost']['process_cpu'],new_cpu=b['cost']['process_cpu'],
                old_complete=a['value']['complete'],new_complete=b['value']['complete'],
                old_chance_outcomes=a['counters']['chance_outcomes'],new_chance_outcomes=b['counters']['chance_outcomes']))
    aggregate={}
    for mode in modes:
        rows=[r for r in cases if r['mode']==mode]
        aggregate[mode]=dict(cells=len(rows),narrower=sum(r['width_reduction']>TOL for r in rows),
            wider=sum(r['width_reduction']<-TOL for r in rows),ties=sum(abs(r['width_reduction'])<=TOL for r in rows),
            mean_width_reduction=float(np.mean([r['width_reduction'] for r in rows])),
            mean_cpu_change=float(np.mean([r['new_cpu']-r['old_cpu'] for r in rows])))
    write_csv(HERE/'implementation_paired.csv',cases)
    plot_comparison(cases,modes)
    return dict(cells=len(cases),all_value_and_action_ranges_intersect=True,conflicting_certificates=0,
        aggregate=aggregate,raw='implementation_paired.csv',
        scope='Same declared per-query limits and inputs. Descending chance mass plus exact absorbing-terminal integration; not a new policy. Original interrupted in-flight CPU is not part of paired saved receipts.')


def plot_comparison(cases,modes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(left,right)=plt.subplots(1,2,figsize=(12,5))
    colors=plt.get_cmap('tab10').colors
    limit=max(max(r['old_width'],r['new_width']) for r in cases)*1.03
    left.plot([0,limit],[0,limit],color='gray',linestyle='--',linewidth=1)
    for mode,color in zip(modes,colors):
        rows=[r for r in cases if r['mode']==mode]
        left.scatter([r['old_width'] for r in rows],[r['new_width'] for r in rows],s=14,alpha=.45,color=color)
        x=[np.mean([r[k] for r in rows]) for k in ('old_cpu','new_cpu')]
        y=[np.mean([r[k] for r in rows]) for k in ('old_width','new_width')]
        right.plot(x,y,color=color,alpha=.7,label=mode)
        right.scatter(x[0],y[0],marker='x',color=color,s=55)
        right.scatter(x[1],y[1],marker='o',color=color,s=35)
    left.set(xlabel='Chronological interval width (s)',ylabel='Optimized interval width (s)',
             title='All 440 paired cells; below diagonal is narrower')
    right.set(xlabel='Mean query process CPU (s)',ylabel='Mean conditional interval width (s)',
              title='Same limits: chronological x, mass + terminal o')
    for axis in (left,right):
        axis.grid(alpha=.2)
    right.legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/'implementation_quality_time.png',dpi=160);plt.close(fig)


def plot(rows,modes,suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(9,5))
    for mode in modes:
        values=[row for row in rows if row['mode']==mode]
        axis.scatter([r['cpu'] for r in values],[r['width'] for r in values],s=20,alpha=.6,label=mode)
    axis.set(xlabel='Complete query process CPU (s)',ylabel='Conditional value interval width (s)',
        title='Ordinary bounded causal reference: all selected public histories')
    axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/f'quality_time{suffix}.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation',choices=('chronological','mass_terminal'),default='mass_terminal')
    summarize(parser.parse_args().implementation)
