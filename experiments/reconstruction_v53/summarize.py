"""Complete-panel validity, policy-loss attribution and paired budget curves."""
from collections import Counter
import argparse
import csv
from math import isfinite
from pathlib import Path

import numpy as np

from experiments.reconstruction_v34.discover import read
from experiments.reconstruction_v38.run import save

HERE = Path(__file__).resolve().parent
TOL = 1e-8
ACTION_TOL = 1e-9
GAPS = (
    ('cheap_p0_minus_optimal','cheap_p0','optimal',True),
    ('cheap_working_minus_optimal','cheap_working','optimal',True),
    ('cheap_p0_minus_one_step','cheap_p0','one_step_p0',True),
    ('cheap_p0_minus_two_step','cheap_p0','two_step_p0',True),
    ('one_step_minus_two_step','one_step_p0','two_step_p0',True),
    ('one_step_minus_optimal','one_step_p0','optimal',True),
    ('two_step_minus_optimal','two_step_p0','optimal',True),
    ('working_minus_consistent_p0','cheap_working','cheap_p0',False),
)


def valid_value(value, cap):
    low, high = value['lower'], value['upper']
    if not all(isfinite(v) for v in (low, high)) or low < -TOL or high > cap+TOL or low > high+TOL:
        raise ValueError('Invalid conditional value interval')
    if value['complete']:
        mean = value['mean']
        if mean is None or not isfinite(mean) or max(abs(mean-low),abs(mean-high)) > TOL:
            raise ValueError('Complete computation does not give a contained point value')
    elif value['mean'] is not None:
        raise ValueError('Incomplete computation reported a mean')


def valid_certificate(bank):
    certified = bank['certified_action']
    if certified is None:
        return
    key = str(certified)
    if key not in bank['actions']:
        raise ValueError('Certified action is outside the action set')
    other = [v['lower'] for a,v in bank['actions'].items() if a != key]
    if not other or not bank['actions'][key]['upper'] < min(other)-ACTION_TOL:
        raise ValueError('Action certificate lacks strict interval separation')


def compatible(left, right):
    """Check the same declared value and every forced action, without pooling."""
    if set(left['actions']) != set(right['actions']):
        raise ValueError('Compared computations have different forced action sets')
    checks = [(left['value'],right['value'])]
    checks.extend((left['actions'][a],right['actions'][a]) for a in left['actions'])
    points = 0
    for a,b in checks:
        if max(a['lower'],b['lower']) > min(a['upper'],b['upper'])+TOL:
            raise ValueError('Intervals for the same conditional value are disjoint')
        for exact, bound in ((a,b),(b,a)):
            if exact['complete']:
                if not bound['lower']-TOL <= exact['mean'] <= bound['upper']+TOL:
                    raise ValueError('Complete value is outside an earlier valid interval')
                points += 1
    a,b = left['certified_action'],right['certified_action']
    if a is not None and b is not None and str(a) != str(b):
        raise ValueError('Strict action certificates conflict')
    return dict(value_and_action_intervals=len(checks),complete_point_checks=points)


def gap(left, right, nonnegative=True):
    low, high = left['lower']-right['upper'],left['upper']-right['lower']
    if nonnegative:
        if high < -TOL:
            raise ValueError('Intervals contradict inclusion of causal policy classes')
        low, high = max(0.,low),max(0.,high)
        status = ('certified_positive_loss' if low > TOL else
                  'certified_zero_within_tolerance' if high <= TOL else 'computationally_unresolved')
    else:
        status = ('certified_working_worse' if low > TOL else
                  'certified_working_better' if high < -TOL else
                  'certified_zero_within_tolerance' if max(abs(low),abs(high)) <= TOL else
                  'computationally_unresolved')
    return dict(lower=low,upper=high,status=status,nonnegative_by_policy_inclusion=nonnegative,
        tolerance_seconds=TOL)


def observed_action_gaps(occurrences, optimal):
    """Only choices actually made at this exact V51 public history."""
    rows = []
    for occurrence in occurrences:
        executed = occurrence['selected_action']
        action = 'wait' if executed is None else str(executed)
        if action not in optimal['actions']:
            raise ValueError(f'Observed V51 action {action} has no matching forced-action bound: {occurrence["source"]}')
        loss = gap(optimal['actions'][action],optimal['value'])
        certificate = optimal.get('certified_action')
        basis = 'forced_action_and_optimal_value_intervals'
        if certificate is not None and str(certificate) == action:
            # Strictly optimal a implies Q*(h,a)=V*(h), even while both
            # absolute values remain unresolved. This says nothing about
            # the V51 controller's choices after this first action.
            loss.update(lower=0.,upper=0.,status='certified_zero_loss')
            basis = 'strict_optimal_action_certificate'
        rows.append(dict(policy=occurrence['policy'],selected_action=executed,reference_action=action,
            state_index=occurrence['state_index'],
            v51_source='../reconstruction_v51/'+occurrence['source'],
            action_value=optimal['actions'][action],optimal_value=optimal['value'],
            proof_basis=basis,certified_optimal_action=certificate,**loss))
    return rows


def check_working_root_action(occurrences, bank):
    """Restored W control must reproduce the actual same-history cheap choice."""
    normalize = lambda action:'wait' if action is None else str(action)
    checked = 0
    for occurrence in occurrences:
        if occurrence['policy'] != 'cheap':
            continue
        expected = normalize(occurrence['selected_action'])
        if (normalize(bank['cheap_action']),normalize(bank['selected_action'])) != (expected,expected):
            raise ValueError(f'Restored cheap_working root choice changed from V51 cheap: {occurrence["source"]}')
        checked += 1
    return checked


def joint_attribution(results):
    """Combine five already paid queries using policy-class relations only.

    V* <= V2 <= V1 <= cheap_P0 and V* <= cheap_W. When an existing
    cheap_P0 or one-step certificate selects cheap_P0's root choice, those
    two root policy values are equal. No corresponding equality with V*
    follows. The original per-query ranges and costs are never changed.
    """
    names=('optimal','two_step_p0','one_step_p0','cheap_p0','cheap_working')
    normalize=lambda action:'wait' if action is None else str(action)
    cheap_action=normalize(results['cheap_p0']['cheap_action'])
    evidence=[dict(mode=mode,certified_action=results[mode]['certified_action'])
              for mode in ('cheap_p0','one_step_p0')
              if results[mode]['certified_action'] is not None
              and normalize(results[mode]['certified_action'])==cheap_action]
    lower_sources={
        'optimal':('optimal',),
        'two_step_p0':('optimal','two_step_p0'),
        'one_step_p0':('optimal','two_step_p0','one_step_p0'),
        'cheap_p0':('optimal','two_step_p0','one_step_p0','cheap_p0'),
        'cheap_working':('optimal','cheap_working'),
    }
    upper_sources={
        'optimal':names,
        'two_step_p0':('two_step_p0','one_step_p0','cheap_p0'),
        'one_step_p0':('one_step_p0','cheap_p0'),
        'cheap_p0':('cheap_p0',),
        'cheap_working':('cheap_working',),
    }
    if evidence:
        lower_sources['one_step_p0']=lower_sources['cheap_p0']
        upper_sources['cheap_p0']=upper_sources['one_step_p0']
    values={}
    for name in names:
        low=max(results[source]['value']['lower'] for source in lower_sources[name])
        high=min(results[source]['value']['upper'] for source in upper_sources[name])
        if low>high+TOL:
            raise ValueError('Joint policy ordering/equality contradicts the computed intervals')
        values[name]=dict(lower=min(low,high),upper=high,
            lower_source_modes=list(lower_sources[name]),upper_source_modes=list(upper_sources[name]))
    gaps={}
    for name,left,right,nonnegative in GAPS:
        entry=gap(values[left],values[right],nonnegative)
        entry['proof_basis']='five_query_policy_class_order'
        entry['certificate_modes']=[]
        if name=='cheap_p0_minus_one_step' and evidence:
            entry.update(lower=0.,upper=0.,status='certified_zero_loss',
                proof_basis='same_forced_continuation_and_certified_cheap_root_action',
                certificate_modes=[item['mode'] for item in evidence])
        gaps[name]=entry
    return dict(values=values,gaps=gaps,
        equalities=[] if not evidence else [dict(left='cheap_p0',right='one_step_p0',
            cheap_root_action=cheap_action,certificates=evidence,
            proof_basis='Forced root continuations are identical; certified minimizer is the cheap root action')],
        cost=dict(process_cpu=sum(results[name]['cost']['process_cpu'] for name in names),
                  complete_queries=len(names)),
        scope='Derived from all five complete charged queries at this state/budget. CPU is their sum, not one 2/10-second query. Policy ordering and the stated root equality do not identify the entire cheap policy with the optimal policy.')


def width(value):
    return value['upper']-value['lower']


def mean(rows, call):
    return float(np.mean([call(row) for row in rows]))


def write_csv(path, rows):
    with path.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate(rows, budget):
    threshold = budget['process_cpu_seconds']
    counters = Counter()
    reasons = Counter()
    for row in rows:
        counters.update(row['counters'])
        reasons.update(row['reasons'])
    commitments = [b['commitment'] for row in rows for b in row['root_bounds'].values()
                   if b['commitment'] is not None]
    return dict(queries=len(rows),complete=sum(r['value']['complete'] for r in rows),
        certified_actions=sum(r['certified_action'] is not None for r in rows),
        mean_width=mean(rows,lambda r:width(r['value'])),
        mean_cpu=mean(rows,lambda r:r['cost']['process_cpu']),
        mean_wall=mean(rows,lambda r:r['cost']['wall']),
        max_cpu=max(r['cost']['process_cpu'] for r in rows),
        total_cpu=sum(r['cost']['process_cpu'] for r in rows),
        cpu_threshold=threshold,
        cpu_above_threshold=sum(r['cost']['process_cpu']>threshold for r in rows),
        max_cpu_overshoot=max(0.,max(r['cost']['process_cpu'] for r in rows)-threshold),
        state_cap=budget['states'],max_states=max(r['counters']['states'] for r in rows),
        state_capped=sum(r['counters']['states']==budget['states'] for r in rows),
        mean_states=mean(rows,lambda r:r['counters']['states']),
        mean_cache_hits=mean(rows,lambda r:r['counters']['cache_hits']),
        mean_whole_tail_integrations=mean(rows,lambda r:r['counters']['whole_tail_integrations']),
        mean_whole_tail_hits=mean(rows,lambda r:r['counters']['whole_tail_hits']),
        mean_destination_bounds=mean(rows,lambda r:r['counters']['destination_bounds']),
        mean_destination_bound_hits=mean(rows,lambda r:r['counters']['destination_bound_hits']),
        mean_impact_pruned_branches=mean(rows,lambda r:r['counters'].get('impact_pruned_branches',0)),
        impact_screened_queries=sum(r['counters'].get('impact_pruned_branches',0)>0 for r in rows),
        budget_unknown_queries=sum(r['counters']['budget_unknowns']>0 for r in rows),
        unsupported_queries=sum(r['counters']['unsupported_histories']>0 for r in rows),
        counters_total=dict(counters),reasons_total=dict(reasons),
        commitments=dict(root_actions=len(commitments),examined=sum(r['examined'] for r in commitments),
            admitted=sum(r['admitted'] for r in commitments),
            repeated_trip_rejections=sum(r['repeated_trip_rejections'] for r in commitments),
            stopped=dict(Counter(r['stopped'] for r in commitments if r['stopped'] is not None))))


def summarize(implementation='impact'):
    if implementation not in ('compressed','impact'):
        raise ValueError(implementation)
    folder = 'natural' if implementation=='compressed' else 'natural_priority'
    suffix = '_compressed' if implementation=='compressed' else ''
    kernel = 'compressed_conditional' if implementation=='compressed' else 'impact_conditional'
    protocol = read(HERE/'contract.json')
    panel = read(HERE.parent/'reconstruction_v52/panel.json.gz')
    modes, budgets = protocol['modes'],protocol['budgets']
    if (len(panel['rows']) != 88 or {r['query_index'] for r in panel['rows']} != set(range(88))
            or {r['state']['root_index'] for r in panel['rows']} != set(range(72))):
        raise ValueError('The frozen 88-state/all72-root panel changed')
    if [b['name'] for b in budgets] != ['b2','b10'] or len(modes) != 5:
        raise ValueError('The declared two-budget/five-policy comparison changed')
    # Do not publish a selected subset while the paired experiment is running.
    paths = [HERE/folder/f"{item['query_index']:03}_{b['name']}_{mode}.json.gz"
             for item in panel['rows'] for b in budgets for mode in modes]
    if implementation=='impact':
        paths += [HERE/'natural'/f"{item['query_index']:03}_{b['name']}_{mode}.json.gz"
                  for item in panel['rows'] for b in budgets for mode in modes]
    missing = [str(p.relative_to(HERE)) for p in paths if not p.exists()]
    if missing:
        raise ValueError(f'Complete panel required: {len(missing)} missing receipts; first={missing[0]}')
    v51 = read(HERE.parent/'reconstruction_v51/summary.json')
    v51_by_root = {r['root_index']:r for r in v51['records']}
    records, flat, gap_rows, comparisons, budget_pairs, implementation_pairs = [],[],[],[],[],[]
    observed_choice_rows, joint_rows = [],[]
    audit = Counter()
    for item in panel['rows']:
        query,state = item['query_index'],item['state']
        cap = state['deadline']-state['clock']
        old = {implementation:{mode:read(HERE.parent/'reconstruction_v52'/folder/f'{query:03}_{mode}.json.gz')
               for mode in modes} for implementation,folder in
               (('chronological','natural'),('mass_terminal','natural_optimized'))}
        record = dict(query_index=query,source_state_index=item['source_state_index'],
            root_index=state['root_index'],day=state['day'],kind=state['kind'],clock=state['clock'],
            selection_reasons=item['selection_reasons'],observed_baseline_actions=item['selected_actions'],
            source=f'../reconstruction_v52/panel.json.gz#/rows/{query}',
            actual_v51=f"../reconstruction_v51/summary.json#/records/{state['root_index']}",results={},gaps={})
        record['observed_v51_action_gaps'] = {}
        record['joint_attribution'] = {}
        for b in budgets:
            level = b['name']
            record['results'][level] = {}
            for mode in modes:
                bank = read(HERE/folder/f'{query:03}_{level}_{mode}.json.gz')
                if (bank['query_index'],bank['source_state_index'],bank['root_index'],bank['day'],bank['mode']) != (
                        query,item['source_state_index'],state['root_index'],state['day'],mode):
                    raise ValueError('Receipt differs from the frozen input state or policy')
                if bank['budget'] != b or bank['counters']['states'] > b['states']:
                    raise ValueError('Receipt violates its declared state budget')
                if bank['implementation'] != kernel:
                    raise ValueError('Receipt uses the wrong reference implementation')
                if (bank['clock'],bank['deadline'],bank['future_public_frames']) != (state['clock'],state['deadline'],0):
                    raise ValueError('Physical clock or information contract changed')
                for value in (bank['value'],*bank['actions'].values()):
                    valid_value(value,cap)
                valid_certificate(bank)
                if mode=='cheap_working':
                    audit['original_working_root_action_checks'] += check_working_root_action(item['occurrences'],bank)
                if any(not isfinite(bank['complete_cost'][k]) or bank['complete_cost'][k]<0
                       for k in ('process_cpu','wall')):
                    raise ValueError('Invalid charged query cost')
                for baseline_implementation,old_modes in old.items():
                    baseline = old_modes[mode]
                    if (baseline['query_index'],baseline['source_state_index'],baseline['mode']) != (
                            query,item['source_state_index'],mode):
                        raise ValueError('V52 baseline input differs from V53')
                    for value in (baseline['value'],*baseline['actions'].values()):
                        valid_value(value,cap)
                    valid_certificate(baseline)
                    audit.update(compatible(baseline,bank))
                    audit['cross_version_cells'] += 1
                if implementation=='impact':
                    original = read(HERE/'natural'/f'{query:03}_{level}_{mode}.json.gz')
                    if (original['query_index'],original['source_state_index'],original['mode'],original['budget']) != (
                            query,item['source_state_index'],mode,b):
                        raise ValueError('Compressed baseline differs from the paired input or budget')
                    for value in (original['value'],*original['actions'].values()):
                        valid_value(value,cap)
                    valid_certificate(original)
                    audit.update(compatible(original,bank))
                    audit['cross_implementation_cells'] += 1
                    implementation_pairs.append(dict(query_index=query,root_index=state['root_index'],
                        budget=level,mode=mode,compressed_width=width(original['value']),
                        impact_width=width(bank['value']),width_reduction=width(original['value'])-width(bank['value']),
                        compressed_cpu=original['complete_cost']['process_cpu'],
                        impact_cpu=bank['complete_cost']['process_cpu'],
                        cpu_change=bank['complete_cost']['process_cpu']-original['complete_cost']['process_cpu'],
                        compressed_complete=original['value']['complete'],impact_complete=bank['value']['complete'],
                        compressed_certified=original['certified_action'],impact_certified=bank['certified_action'],
                        impact_pruned_branches=bank['counters'].get('impact_pruned_branches',0),
                        impact_cutoff=bank.get('impact_cutoff')))
                result = dict(value=bank['value'],actions=bank['actions'],selected_action=bank['selected_action'],
                    cheap_action=bank['cheap_action'],
                    certified_action=bank['certified_action'],counters=bank['counters'],reasons=bank['reasons'],
                    root_bounds=bank['root_bounds'],cost=bank['complete_cost'],prefix_cost=bank['prefix_cost'],
                    limits=bank['limits'],screening_workers=bank['screening_workers'],
                    impact_cutoff=bank.get('impact_cutoff'),impact_scope=bank.get('impact_scope'),
                    source=f'{folder}/{query:03}_{level}_{mode}.json.gz')
                record['results'][level][mode] = result
                flat.append(dict(query_index=query,root_index=state['root_index'],day=state['day'],
                    kind=state['kind'],budget=level,mode=mode,lower=bank['value']['lower'],
                    upper=bank['value']['upper'],width=width(bank['value']),complete=bank['value']['complete'],
                    selected_action=bank['selected_action'],certified_action=bank['certified_action'],
                    states=bank['counters']['states'],cache_hits=bank['counters']['cache_hits'],
                    partial_cache_hits=bank['counters']['partial_cache_hits'],
                    whole_tail_integrations=bank['counters']['whole_tail_integrations'],
                    whole_tail_hits=bank['counters']['whole_tail_hits'],
                    destination_bounds=bank['counters']['destination_bounds'],
                    destination_bound_hits=bank['counters']['destination_bound_hits'],
                    impact_pruned_branches=bank['counters'].get('impact_pruned_branches',0),
                    impact_cutoff=bank.get('impact_cutoff'),
                    cpu=bank['complete_cost']['process_cpu'],wall=bank['complete_cost']['wall'],
                    cpu_threshold=b['process_cpu_seconds'],
                    cpu_overshoot=max(0.,bank['complete_cost']['process_cpu']-b['process_cpu_seconds']),
                    source=result['source']))
            values = {mode:r['value'] for mode,r in record['results'][level].items()}
            record['gaps'][level] = {name:gap(values[left],values[right],nonnegative)
                                     for name,left,right,nonnegative in GAPS}
            for name,entry in record['gaps'][level].items():
                gap_rows.append(dict(query_index=query,root_index=state['root_index'],day=state['day'],
                    kind=state['kind'],budget=level,comparison=name,**entry))
            choices = observed_action_gaps(item['occurrences'],record['results'][level]['optimal'])
            record['observed_v51_action_gaps'][level] = choices
            for choice in choices:
                observed_choice_rows.append(dict(query_index=query,root_index=state['root_index'],
                    day=state['day'],kind=state['kind'],budget=level,
                    **{k:v for k,v in choice.items() if k not in ('action_value','optimal_value')},
                    action_lower=choice['action_value']['lower'],action_upper=choice['action_value']['upper'],
                    optimal_lower=choice['optimal_value']['lower'],optimal_upper=choice['optimal_value']['upper']))
            joint=joint_attribution(record['results'][level])
            record['joint_attribution'][level]=joint
            for name,entry in joint['gaps'].items():
                joint_rows.append(dict(query_index=query,root_index=state['root_index'],day=state['day'],
                    kind=state['kind'],budget=level,comparison=name,
                    **{k:v for k,v in entry.items() if k!='certificate_modes'},
                    certificate_modes=','.join(entry['certificate_modes']),
                    joint_process_cpu=joint['cost']['process_cpu'],
                    charged_queries=joint['cost']['complete_queries']))
        for mode in modes:
            small,large = (record['results'][b['name']][mode] for b in budgets)
            audit.update(compatible(small,large)); audit['cross_budget_cells'] += 1
            budget_pairs.append(dict(query_index=query,root_index=state['root_index'],mode=mode,
                b2_width=width(small['value']),b10_width=width(large['value']),
                width_reduction=width(small['value'])-width(large['value']),
                b2_cpu=small['cost']['process_cpu'],b10_cpu=large['cost']['process_cpu'],
                b2_complete=small['value']['complete'],b10_complete=large['value']['complete'],
                b2_certified=small['certified_action'],b10_certified=large['certified_action']))
            chronological,optimized = old['chronological'][mode],old['mass_terminal'][mode]
            comparisons.append(dict(query_index=query,root_index=state['root_index'],mode=mode,
                chronological_width=width(chronological['value']),v52_width=width(optimized['value']),
                b2_width=width(small['value']),
                chronological_cpu=chronological['complete_cost']['process_cpu'],
                v52_cpu=optimized['complete_cost']['process_cpu'],b2_cpu=small['cost']['process_cpu'],
                width_reduction=width(optimized['value'])-width(small['value']),
                cpu_change=small['cost']['process_cpu']-optimized['complete_cost']['process_cpu'],
                v52_complete=optimized['value']['complete'],b2_complete=small['value']['complete'],
                v52_certified=optimized['certified_action'],b2_certified=small['certified_action']))
        records.append(record)
    aggregates = {b['name']:{mode:aggregate([r['results'][b['name']][mode] for r in records],b)
                   for mode in modes} for b in budgets}
    gap_summary = {b['name']:{name:dict(queries=len(records),
        statuses=dict(Counter(r['gaps'][b['name']][name]['status'] for r in records)),
        mean_lower=mean(records,lambda r:r['gaps'][b['name']][name]['lower']),
        mean_upper=mean(records,lambda r:r['gaps'][b['name']][name]['upper']),
        positive_queries=[r['query_index'] for r in records if r['gaps'][b['name']][name]['lower']>TOL],
        zero_queries=[r['query_index'] for r in records
                      if r['gaps'][b['name']][name]['status']=='certified_zero_within_tolerance'])
        for name,*_ in GAPS} for b in budgets}
    paired_summary = {}
    for mode in modes:
        rows = [r for r in comparisons if r['mode']==mode]
        paired_summary[mode] = dict(cells=len(rows),
            narrower=sum(r['width_reduction']>TOL for r in rows),
            wider=sum(r['width_reduction']<-TOL for r in rows),
            tied=sum(abs(r['width_reduction'])<=TOL for r in rows),
            mean_width_reduction=mean(rows,lambda r:r['width_reduction']),
            mean_cpu_change=mean(rows,lambda r:r['cpu_change']))
    root_context = []
    for root in range(72):
        original = v51_by_root[root]
        losses = {label:dict(loss=r['loss'],loss_bounds=r['loss_bounds'],status=r['status'],
                            cost=r['cost'],source=f'../reconstruction_v51/roots/{root:03}_{label}.json.gz')
                  for label,r in original['results'].items()}
        a,c = losses['cheap']['loss'],losses['full128']['loss']
        root_context.append(dict(root_index=root,day=original['day'],origin=original['origin'],goal=original['goal'],
            queries=[r['query_index'] for r in records if r['root_index']==root],
            actual_v51=losses,full128_gain_over_cheap=None if a is None or c is None else a-c))
    observed_choice_summary = {}
    for b in budgets:
        level=b['name'];observed_choice_summary[level]={}
        for policy in v51['aggregate']:
            rows=[r for r in observed_choice_rows if r['budget']==level and r['policy']==policy]
            observed_choice_summary[level][policy]=dict(occurrences=len(rows),
                queries=len({r['query_index'] for r in rows}),
                statuses=dict(Counter(r['status'] for r in rows)),
                certified_positive_queries=sorted({r['query_index'] for r in rows if r['lower']>TOL}),
                certified_positive_roots=sorted({r['root_index'] for r in rows if r['lower']>TOL}),
                certified_zero_queries=sorted({r['query_index'] for r in rows if r['upper']<=TOL}),
                certificate_implied_zero_occurrences=sum(
                    r['proof_basis']=='strict_optimal_action_certificate' for r in rows),
                max_proven_loss_lower=max((r['lower'] for r in rows),default=None))
    alignment = {}
    for b in budgets:
        level=b['name']; alignment[level]={}
        certified=[r for r in records if r['results'][level]['optimal']['certified_action'] is not None]
        for mode in ('cheap_p0','cheap_working'):
            same=[r['query_index'] for r in certified if
                  str(r['results'][level]['optimal']['certified_action'])==
                  str(r['results'][level][mode]['cheap_action'])]
            alignment[level][mode]=dict(optimal_certified_queries=len(certified),
                same_root_choice=len(same),different_root_choice=len(certified)-len(same),
                same_query_indices=same)
    joint_summary={}
    for b in budgets:
        level=b['name']
        joint_summary[level]=dict(states=len(records),queries_per_state=len(modes),
            mean_cpu_per_joint_state=mean(records,lambda r:r['joint_attribution'][level]['cost']['process_cpu']),
            total_charged_cpu=sum(r['joint_attribution'][level]['cost']['process_cpu'] for r in records),
            certified_cheap_one_step_equalities=sum(bool(r['joint_attribution'][level]['equalities']) for r in records),
            gaps={name:dict(states=len(records),
                statuses=dict(Counter(r['joint_attribution'][level]['gaps'][name]['status'] for r in records)),
                mean_lower=mean(records,lambda r:r['joint_attribution'][level]['gaps'][name]['lower']),
                mean_upper=mean(records,lambda r:r['joint_attribution'][level]['gaps'][name]['upper']),
                positive_queries=[r['query_index'] for r in records if r['joint_attribution'][level]['gaps'][name]['lower']>TOL],
                zero_queries=[r['query_index'] for r in records if
                              r['joint_attribution'][level]['gaps'][name]['status'].startswith('certified_zero')])
                for name,*_ in GAPS})
    result = dict(stage='V53',implementation=implementation,queries=len(records),calls=len(flat),modes=modes,budgets=budgets,
        original_root_denominator=72,represented_roots=len({r['root_index'] for r in records}),
        missing_roots=panel['missing_first_nontrivial_roots'],all_v51_state_count=panel['all_state_count'],
        all_same_history_disagreements=panel['same_history_disagreement_states'],
        aggregate=aggregates,gap_summary=gap_summary,records=records,all_root_context=root_context,
        observed_v51_action_gaps=dict(aggregate=observed_choice_summary,
            records=f'observed_action_gaps{suffix}.csv',
            scope='For an actually observed V51 choice a at this exact public state: Q*(h,a)-V*(h), permitting optimal continuation after a with future remote messages frozen. This is neither the V51 full policy value nor realized whole-journey regret. No choices are imputed at unvisited counterfactual histories.'),
        optimal_certificate_root_choice_alignment=dict(aggregate=alignment,
            scope='Agreement at the current root action only. Fixed cheap-P0/W full-continuation policy gaps remain the separately computed intervals.'),
        joint_attribution=dict(aggregate=joint_summary,records=f'joint_attribution{suffix}.csv',
            scope='Five paid queries per state and budget, with summed process CPU. Tightened intervals and explicit equalities are derived without another solve. Original single-query aggregates and plots remain unchanged.'),
        audit=dict(audit,all_value_and_action_ranges_intersect=True,conflicting_certificates=0,
            state_caps_respected=True,cpu_is_atomic_operation_threshold=True),
        same_budget_comparison=dict(baseline='V52 mass_terminal',aggregate=paired_summary,
            records=f'same_budget_paired{suffix}.csv',scope='b2 only: same 2000-state/2-second limits; all costs retained'),
        budget_comparison=dict(records=f'budget_paired{suffix}.csv',
            scope='b10 is a separate larger-budget experiment; no same-budget gain or monotonic-width assumption'),
        gap_scope='Conditional-state bounds, not whole-journey regret. Means across selected states have no optimal occupancy interpretation. W-P0 is signed; zero uses the reported numerical tolerance.',
        g2_passed=False,new_method_contribution_established=False,
        ordinary_method_intrinsic_barrier_established=False,whole_journey_optimality_claim=False,
        server_transfer_bytes=0)
    if implementation=='impact':
        result['implementation_comparison'] = implementation_comparison(implementation_pairs,modes,budgets)
        result['impact_scope'] = ('The per-branch cutoff is a scheduling threshold, not the sum of all '
            'remaining errors. Every unexpanded probability retains its interval; only reported root ranges certify.')
    save(HERE/f'summary{suffix}.json',result)
    write_csv(HERE/f'paired{suffix}.csv',flat)
    write_csv(HERE/f'gaps{suffix}.csv',gap_rows)
    write_csv(HERE/f'same_budget_paired{suffix}.csv',comparisons)
    write_csv(HERE/f'budget_paired{suffix}.csv',budget_pairs)
    write_csv(HERE/f'observed_action_gaps{suffix}.csv',observed_choice_rows)
    write_csv(HERE/f'joint_attribution{suffix}.csv',joint_rows)
    plot(comparisons,aggregates,modes,budgets,suffix)
    if implementation=='impact':
        write_csv(HERE/'implementation_paired.csv',implementation_pairs)
        plot_implementations(implementation_pairs,modes,budgets)
    print(dict(aggregate=aggregates,gap_summary=gap_summary,audit=result['audit']),flush=True)
    return result


def implementation_comparison(rows,modes,budgets):
    aggregates = {}
    for budget in budgets:
        level = budget['name']; aggregates[level] = {}
        for mode in modes:
            cells = [r for r in rows if r['budget']==level and r['mode']==mode]
            aggregates[level][mode] = dict(cells=len(cells),
                narrower=sum(r['width_reduction']>TOL for r in cells),
                wider=sum(r['width_reduction']<-TOL for r in cells),
                tied=sum(abs(r['width_reduction'])<=TOL for r in cells),
                mean_width_reduction=mean(cells,lambda r:r['width_reduction']),
                mean_cpu_change=mean(cells,lambda r:r['cpu_change']),
                compressed_complete=sum(r['compressed_complete'] for r in cells),
                impact_complete=sum(r['impact_complete'] for r in cells),
                compressed_certificates=sum(r['compressed_certified'] is not None for r in cells),
                impact_certificates=sum(r['impact_certified'] is not None for r in cells))
    return dict(cells=len(rows),aggregate=aggregates,records='implementation_paired.csv',
        all_values_and_actions_intersect=True,conflicting_certificates=0,
        impact_cutoffs=sorted({r['impact_cutoff'] for r in rows}),
        scope='Separate full-panel implementations at identical per-cell budgets; wider, negative and unresolved cells retained. No monotonic-width or new-method claim.')


def plot_implementations(rows,modes,budgets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,budget in zip(axes,budgets):
        for mode,color in zip(modes,plt.get_cmap('tab10').colors):
            cells=[r for r in rows if r['budget']==budget['name'] and r['mode']==mode]
            x=[mean(cells,lambda r:r[k]) for k in ('compressed_cpu','impact_cpu')]
            y=[mean(cells,lambda r:r[k]) for k in ('compressed_width','impact_width')]
            ax.plot(x,y,color=color,label=mode,linewidth=1)
            ax.scatter(x[0],y[0],color=color,marker='x',s=45)
            ax.scatter(x[1],y[1],color=color,marker='o',s=35)
        ax.set(xlabel='Mean complete query process CPU (s)',ylabel='Mean conditional interval width (s)',
            title=f"{budget['name']}: compressed x to impact o; all 88 states")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/'implementation_quality_time.png',dpi=170);plt.close(fig)


def plot(comparisons,aggregates,modes,budgets,suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = plt.get_cmap('tab10').colors
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    for mode,color in zip(modes,colors):
        rows = [r for r in comparisons if r['mode']==mode]
        axes[0].scatter([r['v52_width'] for r in rows],[r['b2_width'] for r in rows],
            color=color,s=15,alpha=.5,label=mode)
        axes[1].scatter([r['v52_cpu'] for r in rows],[r['b2_cpu'] for r in rows],
            color=color,s=15,alpha=.5)
    for ax,left,right in ((axes[0],'v52_width','b2_width'),(axes[1],'v52_cpu','b2_cpu')):
        top=max(max(r[left],r[right]) for r in comparisons)*1.03
        ax.plot([0,top],[0,top],'--',color='gray',linewidth=1)
        ax.grid(alpha=.2)
    axes[0].set(xlabel='V52 mass_terminal interval width (s)',ylabel='V53 b2 interval width (s)',
        title='All 440 same-budget cells; below line is narrower')
    axes[1].set(xlabel='V52 complete process CPU (s)',ylabel='V53 b2 complete process CPU (s)',
        title='Same limits; atomic-operation overshoots retained')
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/f'same_budget_quality_time{suffix}.png',dpi=170);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    for mode,color in zip(modes,colors):
        rows=[aggregates[b['name']][mode] for b in budgets]
        axes[0].plot([r['mean_cpu'] for r in rows],[r['mean_width'] for r in rows],
            '-o',color=color,label=mode)
        axes[1].plot([b['process_cpu_seconds'] for b in budgets],[r['certified_actions'] for r in rows],
            '-o',color=color,label=mode)
    axes[0].set(xlabel='Mean complete query process CPU (s)',ylabel='Mean conditional interval width (s)',
        title='b2 to b10: all 88 states in every policy')
    axes[1].set(xlabel='Declared CPU threshold (s)',ylabel='Strictly separated root actions / 88',
        title='Certificates for each declared continuation target',
        xticks=[b['process_cpu_seconds'] for b in budgets])
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/f'quality_budget{suffix}.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation',choices=('compressed','impact'),default='impact')
    summarize(parser.parse_args().implementation)
