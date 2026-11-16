"""All-root development contrasts, execution diagnostics and two-policy hindsight."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean, median

import numpy as np

from experiments.bounded_gate_5d_v1.summarize import compare_pair,write_csv
from experiments.decision_gap_v1.panel import read,write
from experiments.reconstruction_v50.summarize import bounds

HERE=Path(__file__).resolve().parent


def timing(rows):
    cpus=[r['cost']['process_cpu_including_menu'] for r in rows]
    return dict(mean_cpu_seconds=mean(cpus),median_cpu_seconds=median(cpus),
        p99_cpu_seconds=float(np.quantile(cpus,.99)),max_cpu_seconds=max(cpus),
        mean_wall_seconds=mean(r['cost']['wall_including_menu'] for r in rows))


def aggregate(rows):
    known=[r['outcome']['loss'] for r in rows if r['outcome']['loss'] is not None]
    return dict(registered_roots=len(rows),complete_roots=len(known),
        unresolved_roots=[r['root_index'] for r in rows if r['outcome']['loss'] is None],
        complete_mean_loss_seconds=mean(known) if known else None,
        all_root_loss_bounds=[mean(bounds(r)[j] for r in rows) for j in (0,1)],
        status_counts=dict(Counter(r['outcome']['status'] for r in rows)),**timing(rows))


def execution_diagnostics(rows):
    gates=[g for r in rows for g in r['gates']]
    pubs=[p for r in rows for p in r['publications']]
    expired=[dict(root_index=r['root_index'],publication_index=i,
        base_cpu=p.get('base_cpu_seconds'),window=p.get('window_seconds'),
        opens_at=p.get('opens_at'),closes_at=p.get('closes_at'),
        base_proposal_already_late=not p['base_proposal_timely'])
        for r in rows for i,p in enumerate(r['publications'])
        if p.get('kind')=='onboard_publication' and not p['final_proposal_timely']]
    budgets=[dict(root_index=r['root_index'],gate_index=i,
        requested_cpu=g['budget']['requested_process_cpu'],actual_cpu=g['cost']['process_cpu'],
        overshoot_seconds=g['budget']['overshoot_seconds'],interrupted=g['budget']['interrupted'],
        stopped_phase=g['budget']['stopped_phase']) for r in rows for i,g in enumerate(r['gates'])]
    samples=[sample for gate in gates for arm in gate['rows'].values() for sample in arm]
    return dict(gates=len(gates),completed_pairs_counts=dict(Counter(g['completed_pairs'] for g in gates)),
        fully_computed_banks=sum(g['completed_pairs']==g['count'] for g in gates),
        fully_resolved_banks=sum(g['paired']['unresolved_pairs']==0 for g in gates),
        unfinished_computation_banks=sum(g['completed_pairs']<g['count'] for g in gates),
        unresolved_paired_samples=sum(g['paired']['unresolved_pairs'] for g in gates),
        sample_arm_status_counts=dict(Counter(s['status'] for s in samples)),
        unresolved_arm_status_counts=dict(Counter(s['status'] for s in samples if s['loss'] is None)),
        completion_definition='Fully computed means all requested pairs evaluated; fully resolved additionally requires both losses known for every pair. CPU interruption and completed-but-unknown projection outcomes remain distinct.',
        gate_max_cpu=max((g['cost']['process_cpu'] for g in gates),default=0.),
        gate_mean_cpu=mean(g['cost']['process_cpu'] for g in gates) if gates else 0.,
        proposed_action_changes=sum(g.get('proposed_action')!=g.get('base_action') for g in gates),
        accepted_proposal_changes=sum(bool(g.get('publication_accepted'))
            and g.get('proposed_action')!=g.get('base_action')
            and g.get('selected_action')==g.get('proposed_action') for g in gates),
        deadline_default_changes=sum(not g.get('publication_accepted')
            and g.get('selected_action')!=g.get('base_action') for g in gates),
        late_onboard_proposals=expired,
        query_budget_receipts=budgets,
        base_exhausted_query_allowance=sum(s.get('rollout_skipped')=='base_exhausted_query_allowance'
            for r in rows for s in r['states']),
        background_expired=sum(p.get('background_rejection_reason')=='deadline_expired' for p in pubs),
        control_stats_totals=dict(sum((Counter(r['control_stats']) for r in rows),Counter())))


def fixed_policy_hindsight(banks,base,challenger):
    """This selector commits to one entire policy; it is not a control upper bound."""
    rows=[]
    for bank in banks:
        a,b=bank[base]['outcome']['loss'],bank[challenger]['outcome']['loss']
        if a is not None and b is not None:
            rows.append(dict(root_index=bank[base]['root_index'],gain=max(0.,a-b)))
    return dict(base=base,challenger=challenger,complete_pairs=len(rows),
        complete_mean_saving_seconds=mean(r['gain'] for r in rows) if rows else None,
        positive_roots=[r for r in rows if r['gain']>0],
        scope='Post-hoc choice of one complete saved policy per journey on complete pairs only. No claim about state-wise switching, new policies, optimal value or EVPI.')


def paired_comparisons(banks,labels,causal_contrasts):
    pairs=list(combinations(labels,2))
    pairs.extend((a,b) for a,b in causal_contrasts if (a,b) not in pairs)
    return {f'{a}__vs__{b}':compare_pair(banks,a,b) for a,b in pairs}


def summarize(directory=HERE,source=None):
    directory=Path(directory)
    contract,protocol=read(directory/'contract.json'),read(directory/'protocol.json')
    labels=[r['label'] for r in contract['arms']]
    source=Path(source or protocol['source_directory'])
    banks=[];flat=[]
    for i,root in enumerate(contract['root_manifest']):
        bank={label:read(directory/'roots'/f'{i:03}_{label}.json.gz') for label in labels}
        for label,r in bank.items():
            if (r['root_index'],r['root_id'],r['day'],r['label'])!=(i,root['root_id'],root['day'],label):
                raise ValueError('Root or policy changed from the declared panel')
            if not r['development_only']:
                raise ValueError('Exposed roots cannot be independent confirmation')
            c=r['cost']
            if abs(c['process_cpu']+c['shared_menu']['process_cpu']-c['process_cpu_including_menu'])>1e-8:
                raise ValueError('Journey computation charge is incomplete')
            lo,hi=bounds(r)
            flat.append(dict(root_index=i,day=r['day'],arm=label,status=r['outcome']['status'],
                loss=r['outcome']['loss'],loss_lower=lo,loss_upper=hi,
                cpu_seconds=c['process_cpu_including_menu'],wall_seconds=c['wall_including_menu'],
                states=len(r['states']),gates=len(r['gates'])))
        banks.append(bank)
    contrasts=paired_comparisons(banks,labels,protocol['comparison']['causal_contrasts'])
    root_contrasts=[];historical=[]
    for bank in banks:
        for pair in contrasts.values():
            a,b=pair['baseline'],pair['subject']
            base,subject=bank[a],bank[b]
            al,ah=bounds(base);bl,bh=bounds(subject)
            x,y=base['outcome']['loss'],subject['outcome']['loss']
            root_contrasts.append(dict(root_index=base['root_index'],day=base['day'],baseline=a,subject=b,
                baseline_loss=x,subject_loss=y,saving=None if x is None or y is None else x-y,
                saving_lower=al-bh,saving_upper=ah-bl,
                baseline_status=base['outcome']['status'],subject_status=subject['outcome']['status'],
                cpu_delta=subject['cost']['process_cpu_including_menu']-base['cost']['process_cpu_including_menu'],
                same_reduced_trace=base['actions']==subject['actions']))
        for label,new in bank.items():
            if label in ('cheap_plat_ideal','recursive_plat_ideal'):
                continue
            old_label='bounded_cert_plat_async' if label=='screen64_plat_async' else label
            old=read(source/'roots'/f"{new['root_index']:03}_{old_label}.json.gz")
            x,y=old['outcome']['loss'],new['outcome']['loss']
            historical.append(dict(root_index=new['root_index'],day=new['day'],old_arm=old_label,new_arm=label,
                old_status=old['outcome']['status'],new_status=new['outcome']['status'],old_loss=x,new_loss=y,
                saving=None if x is None or y is None else x-y,same_reduced_trace=old['actions']==new['actions'],
                scope='Development contract/policy revision; not an isolated treatment contrast.'))
    all_equal_loss=[];all_equal_actions=[];unknown=[]
    for i,bank in enumerate(banks):
        rows=list(bank.values())
        if any(r['outcome']['loss'] is None for r in rows):unknown.append(i)
        elif len({r['outcome']['loss'] for r in rows})==1:all_equal_loss.append(i)
        if all(r['actions']==rows[0]['actions'] for r in rows):all_equal_actions.append(i)
    sensitivities={}
    for arm in protocol['sensitivity']['arms']:
        sensitivities[arm]={}
        for tau in (60,90,120):
            values=[read(directory/'sensitivity_roots'/f'{i:03}.json')['sensitivity'][arm][str(tau)] for i in range(60)]
            known=[r['loss'] for r in values if r['loss'] is not None]
            sensitivities[arm][str(tau)]=dict(complete_roots=len(known),
                complete_mean_loss=mean(known) if known else None,
                all_root_loss_bounds=[mean(0 if r['loss'] is None else r['loss'] for r in values),
                    mean(banks[i][arm]['loss_cap'] if r['loss'] is None else r['loss'] for i,r in enumerate(values))],
                scope='Only this analytical policy at this tau; no rollout comparison at an unevaluated tau.')
    attempts=[read(p) for p in sorted(directory.glob('development_attempt_*/attempt.json'))]
    result=dict(stage='trc_revision_v1',development_only=True,independent_confirmation=False,
        original_72_additional_budget=0,roots=len(banks),core_receipts=len(flat),
        sensitivity_additional_receipts=240,dates=contract['control_dates'],
        retained_invalid_development_attempts=attempts,
        valid_controller_calls=len(flat)+240,
        total_completed_controller_calls_including_invalid_attempts=len(flat)+240+sum(
            a['core_receipts']+a['sensitivity_additional_calls'] for a in attempts),
        unreceipted_development_interruptions=any(a.get('interrupted_controller_calls',0)!=0 for a in attempts),
        aggregate={label:aggregate([b[label] for b in banks]) for label in labels},
        comparisons=contrasts,causal_contrasts=protocol['comparison']['causal_contrasts'],
        by_day={day:{f'{a}__vs__{b}':compare_pair([bank for bank in banks if bank[a]['day']==day],a,b)
            for a,b in protocol['comparison']['causal_contrasts']} for day in contract['control_dates']},
        execution={label:execution_diagnostics([b[label] for b in banks]) for label in labels},
        equality=dict(all_complete_equal_loss_roots=all_equal_loss,all_equal_reduced_traces_roots=all_equal_actions,
            roots_with_any_unknown=unknown,
            scope='Reduced traces retain command and other non-observation event clocks; equality is not a claim about boarded-service itineraries.'),
        hindsight=fixed_policy_hindsight(banks,'cheap_plat_async','full128_plat_async'),
        shared_preparation=dict(receipts=[read(p) for p in sorted((directory/'worker_preparation').glob('*.json'))],
            charging='Reported once per actual worker load/day construction, separately from paired journey costs.'),
        sensitivity=sensitivities,g2_passed=False,
        scope='All positive, negative, tie and unresolved roots retained. Bootstrap contrasts are descriptive on five already exposed dates; no general near-optimality inference.')
    write(directory/'summary.json',result)
    write_csv(directory/'paired_journeys.csv',flat)
    write_csv(directory/'paired_root_contrasts.csv',root_contrasts)
    write_csv(directory/'historical_changes.csv',historical)
    write_csv(directory/'paired_comparisons.csv',[dict(baseline=v['baseline'],subject=v['subject'],
        complete_pairs=v['complete_pairs'],wins=v['wins_count'],losses=v['losses_count'],ties=v['ties_count'],
        mean_saving=v['complete_mean_saving_seconds'],
        all_root_lower=v['all_roots_missingness_saving_bounds'][0],all_root_upper=v['all_roots_missingness_saving_bounds'][1],
        ci95_low=v['date_cluster_bootstrap']['ci95_saving_seconds'][0],ci95_high=v['date_cluster_bootstrap']['ci95_saving_seconds'][1],
        mean_cpu_delta=v['mean_cpu_delta_seconds']) for v in contrasts.values()])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=HERE)
    p.add_argument('--source-directory',type=Path)
    a=p.parse_args();summarize(a.directory,a.source_directory)
