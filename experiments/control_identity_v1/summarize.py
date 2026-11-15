"""Complete-denominator quality/cost pairs, with separate posterior models."""
from itertools import combinations
from math import isfinite
from pathlib import Path

from experiments.decision_gap_v1.panel import NEW_LABELS, read, write
from experiments.decision_gap_v1.summarize import (
    aggregate_rows, audit_gates, comparison, paired_rows, residual_diagnostic, write_csv, event_signature,
)
from experiments.reconstruction_v50.summarize import bounds, first_divergence
from experiments.control_identity_v1.run import HERE, GAP, MODELS, POLICIES


def validate_root(rows, index, model_name, menu):
    reference = rows['new_cheap']
    for label in NEW_LABELS:
        row = rows[label]
        if row['root_index'] != index or row['label'] != label or row['emission_model'] != model_name:
            raise ValueError('Receipt root, policy or fit identity changed')
        for key in ('root_id','day','origin','goal','loss_cap'):
            if row[key] != reference[key]:
                raise ValueError(f'Paired root context changed: {index}/{key}')
        if row['root_id'] != menu['root']['root_id'] or row['observation'] != menu['new_observation_contract']:
            raise ValueError('Frozen root or expanded observation contract changed')
        if row['relevant_remote_frames'] != reference['relevant_remote_frames']:
            raise ValueError('External publication population differs across policies')
        cost = row['cost']
        if cost['shared_menu'] != menu['cost']:
            raise ValueError('Frozen shared menu charge changed')
        for metric in ('process_cpu','wall'):
            raw, shared, total = cost[metric],cost['shared_menu'][metric],cost[metric+'_including_menu']
            if not all(isfinite(x) and x>=0 for x in (raw,shared,total)) or abs(raw+shared-total)>1e-9:
                raise ValueError('Invalid total computation charge')


def compact(row):
    return {key:row[key] for key in ('root_index','root_id','day','origin','goal','loss_cap',
        'label','emission_model','actions','cost','relevant_remote_frames','observation')} | {
        'outcome':{key:row['outcome'][key] for key in ('loss','status')}}


def unchanged_source_regression(banks, dates):
    """Use saved runs only, on dates whose entire raw-decoded frames matched."""
    rows = []
    for bank in banks:
        for label,row in bank.items():
            if row['day'] not in dates:
                continue
            old = read(GAP/'roots'/f'{row["root_index"]:03}_{label}.json.gz')
            complete_banks = all(g['completed_pairs']==g['count'] for g in row['gates']+old['gates'])
            equal = dict(loss=row['outcome']['loss']==old['outcome']['loss'],
                status=row['outcome']['status']==old['outcome']['status'],
                actions=row['actions']==old['actions'],
                public_physical_trace=event_signature(row['outcome'])==event_signature(old['outcome']))
            rows.append(dict(root_index=row['root_index'],day=row['day'],label=label,
                             complete_banks=complete_banks,equal=equal))
    return dict(rows=rows,checked=len(rows),
        exact_trajectory_matches=sum(all(r['equal'].values()) for r in rows),
        unexpected_complete_bank_differences=[r for r in rows
            if r['complete_banks'] and not all(r['equal'].values())],
        scope='Same old fit, menu and exactly reproduced source frames. Historical CPU is not a timing comparator.')


def summarize(directory=HERE, *, make_plot=True):
    directory = Path(directory)
    contract = read(directory/'contract.json')
    model_results, small, flat, all_pairs = {}, {}, [], []
    for model_name in MODELS:
        banks = []
        for index in range(contract['roots']):
            rows = {label:read(directory/'models'/model_name/'roots'/f'{index:03}_{label}.json.gz')
                    for label in NEW_LABELS}
            menu = read(GAP/'menus'/f'{index:03}.json.gz')
            validate_root(rows,index,model_name,menu)
            banks.append(rows)
            for label,row in rows.items():
                low, high = bounds(row)
                flat.append(dict(model=model_name,policy=label.removeprefix('new_'),root_index=index,
                    day=row['day'],status=row['outcome']['status'],loss=row['outcome']['loss'],
                    loss_lower=low,loss_upper=high,cpu=row['cost']['process_cpu_including_menu'],
                    wall=row['cost']['wall_including_menu'],states=len(row['states']),gates=len(row['gates'])))
        comparisons = {}
        for left,right in combinations(NEW_LABELS,2):
            comparisons[f'{left}__{right}'] = comparison(banks,left,right)
            all_pairs.extend(dict(model_scope=model_name,**r) for r in paired_rows(banks,left,right))
        model_results[model_name] = dict(
            aggregate={label:aggregate_rows(banks,label) for label in NEW_LABELS},
            comparisons=comparisons,
            gate_audit=audit_gates(banks,contract['rollout_limits']['worlds_per_query']),
            residual_diagnostic=residual_diagnostic(banks))
        if model_name=='old_fit':
            dates = read(directory/'source_receipts.json')['unchanged_frame_dates']
            regression = unchanged_source_regression(banks,dates)
            write(directory/'unchanged_source_regression.json',regression)
            if regression['unexpected_complete_bank_differences']:
                raise ValueError('Old-fit execution changed on an unchanged source date; inspect saved regression')
        # Keep model-specific bank checks above separate. A different fit can
        # produce a different conditional law even at identical public facts.
        small[model_name] = [{label:compact(row) for label,row in bank.items()} for bank in banks]
        del banks
    model_comparisons, divergences = {}, []
    for policy in POLICIES:
        label = 'new_'+policy
        paired = [{m:small[m][i][label] for m in MODELS} for i in range(contract['roots'])]
        for bank in paired:
            before,after = bank['old_fit'],bank['repaired_fit']
            if any(before[k] != after[k] for k in ('root_id','day','loss_cap','observation','relevant_remote_frames')):
                raise ValueError('Old/new fit comparison uses different public inputs')
            first = first_divergence(before['actions'],after['actions'])
            if first is not None:
                divergences.append(dict(policy=policy,root_index=before['root_index'],day=before['day'],
                    old_loss=before['outcome']['loss'],repaired_loss=after['outcome']['loss'],first=first))
        model_comparisons[policy] = comparison(paired,'old_fit','repaired_fit')
        all_pairs.extend(dict(model_scope='old_to_repaired:'+policy,**r)
                         for r in paired_rows(paired,'old_fit','repaired_fit'))
    result = dict(stage='control_identity_v1',roots=contract['roots'],archive_calls=len(flat),
        models=model_results,model_comparisons=model_comparisons,first_model_action_divergences=divergences,
        source_preparation='replay_preparation.json',source_identity='source_receipts.json',
        quality_scope='All-root loss bounds retain missing endpoints; complete-pair means are supplementary. '
            'Bounds are missing-value ranges, not confidence intervals.',
        compute_scope=contract['measurement'],g2_passed=False,
        interpretation='Fixed model/input repair with unchanged existing controls; not a new algorithm or independent confirmation.')
    write(directory/'summary.json',result)
    write_csv(directory/'paired_journeys.csv',flat)
    write_csv(directory/'paired_comparisons.csv',all_pairs)
    if make_plot:
        plot(model_results,directory/'quality_time.png')
    print('completed',result['archive_calls'],'journeys; paired comparisons',len(all_pairs),flush=True)
    return result


def plot(models, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,ax = plt.subplots(figsize=(9,5.4))
    for i,policy in enumerate(POLICIES):
        color = f'C{i}'
        points = []
        for model_name,marker in zip(MODELS,('o','D')):
            row = models[model_name]['aggregate']['new_'+policy]
            low,high = row['all_root_mean_loss_bounds']
            x = row['mean_cpu']
            ax.vlines(x,low,high,color=color,lw=1.6)
            ax.plot([x] if low==high else [x,x],[low] if low==high else [low,high],
                    marker=marker,linestyle='none',color=color,markersize=6,
                    markerfacecolor='white' if model_name=='old_fit' else color)
            points.append((x,low,high))
        if all(low==high for _,low,high in points):
            ax.plot([p[0] for p in points],[p[1] for p in points],color=color,lw=.8,alpha=.5)
    handles = [Line2D([0],[0],color=f'C{i}',lw=2,label=policy) for i,policy in enumerate(POLICIES)]
    handles += [Line2D([0],[0],marker=m,color='black',linestyle='none',
                      markerfacecolor='white' if name=='old_fit' else 'black',label=name)
                for name,m in zip(MODELS,('o','D'))]
    ax.legend(handles=handles,loc='best',ncol=2,fontsize=8)
    ax.set(xscale='log',xlabel='Mean billed whole-trip process CPU (s, log scale)',
           ylabel='All-72-root mean truncated travel loss bounds (s)',
           title='Fixed expanded menu and repaired public message stream')
    ax.grid(alpha=.2)
    fig.text(.5,.015,'Bars retain missing outcomes; they are not confidence intervals. Frozen menu charge included.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    summarize()
