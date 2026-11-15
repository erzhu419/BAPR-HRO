"""All registered natural roots, with unknown quality and computation retained."""
from collections import Counter, defaultdict
from itertools import combinations
from math import isfinite
from pathlib import Path
from statistics import mean

from experiments.decision_gap_v1.panel import NEW_LABELS, read, write
from experiments.decision_gap_v1.summarize import (
    aggregate_rows, audit_gates, comparison, paired_rows, write_csv,
)
from experiments.reconstruction_v50.summarize import bounds

HERE = Path(__file__).resolve().parent
CHEAP = NEW_LABELS[:6]
CHALLENGERS = NEW_LABELS[6:]
METRICS = ('process_cpu', 'wall')


def quality_view(banks):
    """Existing quality helpers also subtract costs; replace those outputs below.

    A source-failure receipt has no measured run cost. Zero in this temporary
    view is only an inert argument to the unchanged loss-bound calculations;
    every exposed timing field is computed separately from original receipts.
    """
    return [{label:dict(row, cost=dict(process_cpu_including_menu=0., wall_including_menu=0.))
             for label,row in bank.items()} for bank in banks]


def timing(values):
    known = [x for x in values if x is not None]
    return dict(mean=mean(known) if len(known)==len(values) and known else None,
                measured_mean=mean(known) if known else None,
                measured=len(known), unknown=len(values)-len(known))


def aggregate(banks, label):
    result = aggregate_rows(quality_view(banks), label)
    for metric,name in (('process_cpu','cpu'),('wall','wall')):
        receipt = timing([bank[label]['cost'][metric+'_including_menu'] for bank in banks])
        result['mean_'+name] = receipt['mean']
        result[name+'_measurement'] = receipt
    result['status_counts'] = dict(Counter(bank[label]['outcome']['status'] for bank in banks))
    result['unexecuted_roots'] = [bank[label]['root_index'] for bank in banks
                                  if bank[label].get('executed', True) is False]
    return result


def pairs(banks, left, right):
    result = paired_rows(quality_view(banks), left, right)
    for pair,bank in zip(result,banks,strict=True):
        for metric,name in (('process_cpu','cpu'),('wall','wall')):
            a,b = (bank[label]['cost'][metric+'_including_menu'] for label in (left,right))
            pair[name+'_delta'] = None if a is None or b is None else b-a
    return result


def compare(banks, left, right):
    result = comparison(quality_view(banks), left, right)
    for metric,name in (('process_cpu','cpu'),('wall','wall')):
        values = []
        for bank in banks:
            a,b = (bank[label]['cost'][metric+'_including_menu'] for label in (left,right))
            values.append(None if a is None or b is None else b-a)
        receipt = timing(values)
        result['mean_'+name+'_delta'] = receipt['mean']
        result[name+'_difference_measurement'] = receipt
    return result


def panel(banks):
    if not banks:
        return dict(registered_roots=0, aggregate={}, comparisons={})
    return dict(registered_roots=len(banks),
        aggregate={label:aggregate(banks,label) for label in NEW_LABELS},
        comparisons={f'{a}__{b}':compare(banks,a,b) for a,b in combinations(NEW_LABELS,2)})


def validate_root(rows, index, selected):
    reference = rows['new_cheap']
    for label,row in rows.items():
        if (row['root_index'],row['label'],row['root_id'],row['day']) != (
                index,label,selected['root_id'],selected['day']):
            raise ValueError('Registered root/policy identity changed')
        if row['loss_cap'] != selected['deadline']-selected['start']:
            raise ValueError('Registered root horizon changed')
        if row['observation'] != reference['observation'] or row.get('coverage') != reference.get('coverage'):
            raise ValueError('Common source coverage or observation contract changed')
        cost = row['cost']
        if cost['shared_menu'] != reference['cost']['shared_menu']:
            raise ValueError('Shared menu charge differs between policies')
        for metric in METRICS:
            raw,total,shared = cost[metric],cost[metric+'_including_menu'],cost['shared_menu'][metric]
            if row.get('executed', True) is False:
                if raw is not None or total is not None:
                    raise ValueError('An unexecuted policy has no measured run time')
            elif raw is None or total is None or not all(isfinite(x) and x>=0 for x in (raw,total,shared)) or abs(raw+shared-total)>1e-9:
                raise ValueError('Complete menu-inclusive computation charge required')


def event_groups(banks):
    exact, trips, missing, unknown_trip = defaultdict(list), defaultdict(list), [], []
    for bank in banks:
        row = bank['new_cheap']
        state = next((s for s in row['states'] if s['kind']=='onboard'), None)
        if state is None:
            missing.append(dict(root_index=row['root_index'],status=row['outcome']['status'],
                                loss=row['outcome']['loss']))
            continue
        exact[row['day'],state['cid'],state['clock']].append(row['root_index'])
        trip = next((r['trip'] for r in state['public_history']
                     if r.get('cid')==state['cid'] and r.get('kind')=='arrival'
                     and r.get('evidence')=='observed'), None)
        if trip is not None:
            trips[row['day'],trip].append(row['root_index'])
        else:
            unknown_trip.append(row['root_index'])
    return dict(first_event_groups=[dict(day=k[0],cid=k[1],clock=k[2],roots=v) for k,v in exact.items()],
        incoming_trip_groups=[dict(day=k[0],trip=k[1],roots=v) for k,v in trips.items()],
        roots_without_first_onboard_event=missing,unknown_incoming_trip_roots=unknown_trip,
        scope='Shared first physical events and incoming trip-days are descriptive duplicate groups, not independent statistical replications.')


def residual(banks):
    result = {}
    for challenger in CHALLENGERS:
        comparisons = {cheap:compare(banks,cheap,challenger) for cheap in CHEAP} if banks else {}
        common = set.intersection(*(set(v['positive_roots']) for v in comparisons.values())) if banks else set()
        complete = {bank[challenger]['root_index'] for bank in banks
                    if all(bank[label]['outcome']['loss'] is not None for label in (*CHEAP,challenger))}
        result[challenger] = dict(versus_each_fixed_cheap_policy=comparisons,
            strictly_better_than_every_cheap_policy_roots=sorted(common & complete),
            scope='Observed complete-root candidate mechanisms only; not an expected-value gap, deployable hindsight baseline, or G2.')
    return result


def summarize(directory=HERE, *, make_plot=False):
    directory = Path(directory)
    contract,selection = read(directory/'contract.json'),read(directory/'selection.json')
    selected = selection['selected']
    if len(selected)!=contract['roots'] or tuple('new_'+p for p in contract['policies'])!=NEW_LABELS:
        raise ValueError('Registered root denominator or eight-policy panel changed')
    dates = read(directory/'date_manifest.json')['selected_dates']
    if any(root['day'] not in dates for root in selected):
        raise ValueError('Root outside the frozen date panel')
    banks,flat,coverage,by_day = [],[],[],{day:[] for day in dates}
    for index,root in enumerate(selected):
        rows = {label:read(directory/'roots'/f'{index:03}_{label}.json.gz') for label in NEW_LABELS}
        validate_root(rows,index,root)
        banks.append(rows);by_day[root['day']].append(rows)
        coverage.append(dict(root_index=index,day=root['day'],root_id=root['root_id'],
                             coverage=rows['new_cheap'].get('coverage')))
        for label,row in rows.items():
            low,high = bounds(row)
            flat.append(dict(root_index=index,day=row['day'],policy=label.removeprefix('new_'),
                status=row['outcome']['status'],loss=row['outcome']['loss'],loss_lower=low,loss_upper=high,
                executed=row.get('executed',True),cpu=row['cost']['process_cpu_including_menu'],
                wall=row['cost']['wall_including_menu'],states=len(row['states']),gates=len(row['gates'])))
    result = dict(stage='natural_gap_v2',roots=len(banks),archive_receipts=len(flat),
        executed_calls=sum(row['executed'] for row in flat),selected_dates=dates,
        **panel(banks),by_day={day:panel(rows) for day,rows in by_day.items()},
        gate_audit=audit_gates(banks,contract['rollout_limits']['worlds_per_query']),
        residual_diagnostic=residual(banks),coverage=coverage,event_groups=event_groups(banks),
        g2_passed=False,
        quality_scope='All registered roots and dates retained. Missing losses use original full horizon bounds; complete-pair means are supplementary. These are missing-value bounds, not confidence intervals.',
        compute_scope='Every measured cold run includes its frozen menu charge. Missing-source unexecuted cells retain unknown run costs; measured-subset means are labeled and cannot replace whole-panel timing.',
        interpretation='A strict residual positive root is a candidate mechanism for investigation. Actual platform provenance and separate theoretical novelty are required before stronger operational or algorithmic claims.')
    write(directory/'summary.json',result)
    write(directory/'candidate_gap_events.json',dict(challengers=result['residual_diagnostic'],
                                                     **result['event_groups']))
    if flat:
        write_csv(directory/'paired_journeys.csv',flat)
        write_csv(directory/'paired_comparisons.csv',[row for a,b in combinations(NEW_LABELS,2)
                                                     for row in pairs(banks,a,b)])
    if make_plot and banks:
        plot(result,directory/'quality_time.png')
    return result


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(8,5))
    for i,(label,row) in enumerate(result['aggregate'].items()):
        cpu = row['mean_cpu']
        if cpu is None:
            continue
        lo,hi = row['all_root_mean_loss_bounds']
        ax.vlines(cpu,lo,hi,color=f'C{i}')
        ax.plot([cpu] if lo==hi else [cpu,cpu],[lo] if lo==hi else [lo,hi],
                marker='o',linestyle='none',color=f'C{i}',label=label.removeprefix('new_'))
    ax.set(xscale='log',xlabel='Mean billed cold journey CPU (s)',
           ylabel='All-root mean truncated travel loss bounds (s)',
           title=f"Frozen natural panel: {result['roots']} registered roots")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='center',bbox_to_anchor=(.65,.5),fontsize=9)
    else:
        ax.text(.5,.5,'Whole-panel computation contains unmeasured cells',
                ha='center',transform=ax.transAxes)
    ax.grid(alpha=.2)
    fig.text(.5,.015,'Missing-outcome bounds, not confidence intervals; shared menu charge included.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(path,dpi=180);plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=HERE)
    args = parser.parse_args()
    summarize(args.directory,make_plot=True)
