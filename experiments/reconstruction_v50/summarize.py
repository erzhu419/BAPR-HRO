"""Full-denominator menu/observation attribution and public-support audit."""
import argparse
from collections import Counter
import csv
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.run import inputs, setup
from experiments.reconstruction_v38.run import restore_node, save
from experiments.reconstruction_v41.run import restored
from experiments.reconstruction_v50.menu import scheduled_early_support
from experiments.reconstruction_v50.run import LABELS

HERE = Path(__file__).resolve().parent


def bounds(row):
    value = row['outcome']['loss']
    return [0., float(row['loss_cap'])] if value is None else [float(value), float(value)]


def comparison(banks, left, right):
    lows, highs, complete = [], [], []
    for rows in banks:
        a, b = bounds(rows[left]), bounds(rows[right])
        lows.append(a[0]-b[1]); highs.append(a[1]-b[0])
        x, y = rows[left]['outcome']['loss'], rows[right]['outcome']['loss']
        if x is not None and y is not None:
            complete.append(float(x-y))
    return dict(roots=len(banks), complete_pairs=len(complete), unresolved_pairs=len(banks)-len(complete),
        wins=sum(x>0 for x in complete), harms=sum(x<0 for x in complete), ties=sum(x==0 for x in complete),
        all_root_mean_gain_bounds=[float(np.mean(lows)), float(np.mean(highs))],
        complete_pair_mean_gain=float(np.mean(complete)) if complete else None,
        orientation=f'{left} loss minus {right} loss')


def first_divergence(left, right):
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else None
        b = right[i] if i < len(right) else None
        if a != b:
            earlier_platform = any(r['kind']=='platform' for r in left[:i]+right[:i])
            return dict(index=i, left=a, right=b,
                onboard_before_platform=bool(a and b and a['kind']==b['kind']=='onboard' and not earlier_platform))
    return None


def support_audit():
    """Use saved decision prefixes only; outcome labels are never supplied."""
    path = HERE/'scheduled_early_support.json.gz'
    if path.exists():
        return read(path)
    tick, cpu = perf_counter(), process_time()
    _, data, _, fit, _ = inputs()
    emission = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    by_day, records = {}, []
    onboard_total, terminal_exits = 0, 0
    for index in range(72):
        source = HERE/'roots'/f'{index:03}_wide_common.json.gz'
        bank = read(source)
        day = bank['day']
        if day not in by_day:
            graph = context(data, day)[0]
            model, parameters = setup(graph, fit)
            by_day[day] = graph, model, parameters
        graph, model, parameters = by_day[day]
        for sequence, state in enumerate(bank['states']):
            if state['kind'] != 'onboard':
                continue
            onboard_total += 1
            node = restore_node(state['node'])
            platform = node.exits[state['cid']]
            if platform is None:
                terminal_exits += 1
                continue
            ready = state['clock']+graph.stops[platform.stop].min_transfer_time
            history = restored('shallow', model, parameters, emission, state)
            row = dict(root_index=index, day=day, source=f'roots/{source.name}#/states/{sequence}',
                state_index=sequence, current_cid=state['cid'], clock=state['clock'],
                stop=platform.stop, ready=ready, deadline=state['deadline'],
                action_cids=list(platform.rides), selected_action=state['selected_action'])
            try:
                row['departures'] = scheduled_early_support(graph, model, history,
                    platform.stop, ready, state['deadline'], set(platform.rides))
                row['error'] = None
            except UnsupportedHistory as error:
                row.update(departures=None, error=str(error))
            records.append(row)
    checked = [r for r in records if r['departures'] is not None]
    candidates = [dict(root_index=r['root_index'], state_index=r['state_index'],
                       day=r['day'], **c) for r in checked for c in r['departures']]
    legal = [c for c in candidates if c['public_support_legal']]
    omitted = [c for c in legal if not c['in_action_menu']]
    result = dict(scope='Every saved wide/common onboard decision with a nonterminal exit; all schedule-early pickup departures at that exit. Current W public conditional support only, no realized labels or future frames.',
        interpretation='A legal departure is a witness against schedule-only physical pruning. It is not proof of a complete route to the goal, nor proof that scheduled earliness was its only exclusion reason, nor an action/travel-gain claim.',
        summary=dict(roots=72, onboard_decisions=onboard_total, terminal_exits=terminal_exits,
            eligible_exit_contexts=len(records), checked_contexts=len(checked),
            unsupported_contexts=len(records)-len(checked), scheduled_early_occurrences=len(candidates),
            legal_support_occurrences=len(legal), omitted_legal_support_occurrences=len(omitted),
            roots_with_omitted_legal_support=sorted({c['root_index'] for c in omitted}),
            unique_day_cid_omitted_legal=len({(c['day'],c['cid']) for c in omitted}),
            max_omitted_public_legal_mass=max((c['public_legal_mass'] for c in omitted), default=0.),
            omitted_legal_mass_quantiles=None if not omitted else dict(zip(('min','median','max'),
                map(float,np.quantile([c['public_legal_mass'] for c in omitted],[0,.5,1]))))),
        cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick,
                  scope='Diagnostic support audit outside online controller timing'), records=records)
    save(path,result)
    return result


def summarize(run_support=True):
    probe = read(HERE.parents[1]/'md/hegel_v10/menu_probe.json')
    supplied = read(HERE.parents[1]/'md/hegel_v10/menu_comparison.json')
    banks, menus, records, flat = [], [], [], []
    menu_counts = Counter()
    reasons, frontiers = Counter(), Counter()
    missing_roots, label_issues = [], []
    for index in range(72):
        rows = {label:read(HERE/'roots'/f'{index:03}_{label}.json.gz') for label in LABELS}
        menu = read(HERE/'menus'/f'{index:03}.json.gz')
        banks.append(rows); menus.append(menu)
        for label in ('narrow_native','wide_native'):
            original = probe['rows'][index]['results']['original' if label=='narrow_native' else 'uncapped']
            row = rows[label]
            if (row['outcome']['loss'], row['outcome']['status'], row['actions']) != (
                    original['loss'], original['status'], original['actions']):
                raise AssertionError(f'Supplied native result changed: {index}/{label}')
            if not all(row['supplied_probe_validation'].values()):
                raise AssertionError('Native validation metadata disagrees')
        if rows['narrow_common']['observation'] != rows['wide_common']['observation']:
            raise AssertionError('Common observation contracts differ')
        for label,row in rows.items():
            if row['root_index']!=index or row['root_id']!=rows['narrow_native']['root_id']:
                raise AssertionError('Root selection changed')
            flat.append(dict(root_index=index,day=row['day'],origin=row['origin'],goal=row['goal'],
                label=label,loss=row['outcome']['loss'],status=row['outcome']['status'],
                cpu=row['cost']['process_cpu'],wall=row['cost']['wall'],
                cpu_including_menu=row['cost']['process_cpu_including_menu'],
                wall_including_menu=row['cost']['wall_including_menu'],
                decisions=len(row['states']),observation_only_epochs=row['outcome']['decisions'].get('observation_only_epochs',0)))
        losses = {label:row['outcome']['loss'] for label,row in rows.items()}
        complete = all(value is not None for value in losses.values())
        terms = None
        if complete:
            n,w,nc,wc = (losses[k] for k in LABELS)
            terms = dict(probe_gain=n-w,narrow_observation=n-nc,common_menu=nc-wc,wide_observation_remainder=wc-w)
            assert terms['probe_gain']==sum(terms[k] for k in ('narrow_observation','common_menu','wide_observation_remainder'))
        record = dict(root_index=index,day=rows['wide_common']['day'],
            origin=rows['wide_common']['origin'],goal=rows['wide_common']['goal'],losses=losses,
            loss_bounds={label:bounds(row) for label,row in rows.items()},decomposition=terms,
            probe_first_divergence=first_divergence(rows['narrow_native']['actions'],rows['wide_native']['actions']),
            common_menu_first_divergence=first_divergence(rows['narrow_common']['actions'],rows['wide_common']['actions']),
            narrow_native_common_actions_equal=rows['narrow_native']['actions']==rows['narrow_common']['actions'],
            wide_native_common_actions_equal=rows['wide_native']['actions']==rows['wide_common']['actions'],
            missing_menu_labels=menu['labels']['issues'])
        records.append(record)
        for hub in menu['wide_root']['hubs']:
            candidates={o['first_cid'] for o in hub['options']}
            caprows=[r for r in menu['cap_ledger'] if r['arrival_cid']==hub['arrival_cid']]
            retained={r['first_cid'] for r in caprows if r['original_retained']}
            menu_counts.update(hubs=1,hubs_capped=int(len(candidates)>len(retained)),
                raw_candidates=len(candidates),retained_candidates=len(retained),
                raw_options=len(caprows),retained_options=sum(r['original_retained'] for r in caprows))
            menu_counts['max_raw_candidates']=max(menu_counts['max_raw_candidates'],len(candidates))
        menu_counts['roots_with_cap']+=any(not r['original_retained'] for r in menu['cap_ledger'])
        reasons.update(menu['generation']['counts']['reasons'])
        frontiers.update(menu['generation']['counts']['frontiers'])
        if menu['labels']['issues']:
            missing_roots.append(index)
            label_issues.append(dict(root_index=index,issues=menu['labels']['issues']))
    expected=probe['contract']['counts']
    for name in ('hubs','hubs_capped','roots_with_cap','raw_candidates','retained_candidates','raw_options','retained_options','max_raw_candidates'):
        if menu_counts[name]!=expected[name]:
            raise AssertionError(f'Candidate reconstruction changed {name}: {menu_counts[name]} != {expected[name]}')
    if missing_roots!=supplied['roots_with_any_added_menu_missing_labels']:
        raise AssertionError('Missing-label roots were omitted or changed')
    pairs = dict(probe=comparison(banks,'narrow_native','wide_native'),
        narrow_observation=comparison(banks,'narrow_native','narrow_common'),
        common_menu=comparison(banks,'narrow_common','wide_common'),
        wide_observation_remainder=comparison(banks,'wide_common','wide_native'))
    original=supplied['same_policy_menu_expansion']
    if pairs['probe']['all_root_mean_gain_bounds']!=[original['mean_gain']]*2:
        raise AssertionError('Supplied aggregate gain changed')
    nonzero = [r for r in records if r['decomposition'] and r['decomposition']['probe_gain']]
    early = [r['root_index'] for r in nonzero if r['probe_first_divergence'] and
             r['probe_first_divergence']['onboard_before_platform']]
    if early!=supplied['nonzero_loss_roots_first_divergence_onboard_before_platform']:
        raise AssertionError('Supplied first-divergence mechanism changed')
    aggregate = {}
    for label in LABELS:
        aggregate[label]=dict(roots=72,complete=sum(r[label]['outcome']['loss'] is not None for r in banks),
            mean_loss_bounds=[float(np.mean([bounds(r[label])[i] for r in banks])) for i in (0,1)],
            mean_cpu=float(np.mean([r[label]['cost']['process_cpu'] for r in banks])),
            mean_cpu_including_menu=float(np.mean([r[label]['cost']['process_cpu_including_menu'] for r in banks])),
            mean_wall_including_menu=float(np.mean([r[label]['cost']['wall_including_menu'] for r in banks])),
            decisions=sum(len(r[label]['states']) for r in banks))
    support = support_audit() if run_support else None
    summary = dict(stage='V50',archive_calls=288,roots=72,source_probe_reproduced=144,
        aggregate=aggregate,comparisons=pairs,menu_counts=dict(menu_counts),
        candidate_filter_occurrences=dict(reasons),path_frontier_occurrences=dict(frontiers),
        missing_label_roots=missing_roots,missing_labels=label_issues,
        decomposition_complete_roots=sum(r['decomposition'] is not None for r in records),
        native_common_action_parity={kind:sum(r[kind+'_native_common_actions_equal'] for r in records)
                                    for kind in ('narrow','wide')},
        nonzero_gain_first_divergence_onboard_before_platform=early,
        old_rollout_positive_roots_absorbed=supplied['old_rollout_positive_roots_absorbed'],
        scheduled_early_support=None if support is None else support['summary'],records=records,
        g2_passed=False,new_dates=0,new_model=False,server_transfer_bytes=0,
        scope='Development action/observation contract diagnosis; native20.18 gain is not presumed a pure menu effect')
    save(HERE/'summary.json',summary)
    with (HERE/'paired.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(dict(aggregate=aggregate,comparisons=pairs,menu_counts=dict(menu_counts),
               scheduled_early_support=summary['scheduled_early_support']),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-support',action='store_true')
    args=parser.parse_args()
    summarize(run_support=not args.skip_support)
