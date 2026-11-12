"""All-state structural census; no new draws, posterior fitting or control.

The active set is a structural envelope of both current actions, with already
passed exits removed. It is not selected using realized losses, forecast
values or support-dependent pruning.
"""
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
from statistics import mean, median

from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.run import setup


HERE = Path(__file__).resolve().parent
V38 = HERE.parent/'reconstruction_v38'
V34 = HERE.parent/'reconstruction_v34'


def summary(values):
    values = list(values)
    return dict(count=len(values),total=sum(values),minimum=min(values,default=None),
        median=median(values) if values else None,mean=mean(values) if values else None,
        maximum=max(values,default=None),histogram=dict(sorted(Counter(values).items())))


def children(node):
    return [(str(row['cid']),row['next']) for row in
            node['exits' if node['kind']=='ride' else 'rides'] if row['next'] is not None]


def trip_set(node,model):
    own = {model._positions[node['first_cid'],'departure'][0]} if node['kind']=='ride' else set()
    return own.union(*(trip_set(child,model) for _,child in children(node)))


def structure(node,model):
    rides,platforms,shared,ancestor_reuse = [],[],[],[]
    slots = defaultdict(set)

    def visit(current,path,ancestors):
        if current['kind']=='ride':
            trip = model._positions[current['first_cid'],'departure'][0]
            terminal = len(current['exits'])==1 and current['exits'][0]['next'] is None
            rides.append(dict(path=path,trip=trip,cids=current['path'],exits=len(current['exits']),
                terminal=terminal,optional_alighting=len(current['exits'])>1))
            if trip in ancestors:
                ancestor_reuse.append(dict(path=path,trip=trip))
            ancestors = ancestors+(trip,)
            for cid in current['path']:
                for kind in ('departure','arrival'):
                    slots[trip].add(model._positions[cid,kind][1])
        else:
            counts = Counter(model._positions[row['cid'],'departure'][0] for row in current['rides'])
            platforms.append(dict(path=path,stop=current['stop'],options=len(current['rides']),
                same_trip_candidates={trip:n for trip,n in counts.items() if n>1}))
        branches = [(label,child,trip_set(child,model)) for label,child in children(current)]
        for left,right in combinations(branches,2):
            common = left[2]&right[2]
            if common:
                shared.append(dict(node_path=path,left=left[0],right=right[0],trips=sorted(common)))
        for label,child,_ in branches:
            visit(child,path+'/'+label,ancestors)

    visit(node,'root',())
    occurrences = Counter(row['trip'] for row in rides)
    return dict(trips=sorted(occurrences),ride_occurrences=len(rides),platform_occurrences=len(platforms),
        repeated_trips={trip:n for trip,n in sorted(occurrences.items()) if n>1},
        sibling_shared_trips=sorted({trip for row in shared for trip in row['trips']}),
        shared_branch_pairs=shared,ancestor_reuse=ancestor_reuse,
        platforms_with_same_trip_candidates=sum(bool(row['same_trip_candidates']) for row in platforms),
        optional_alighting_rides=sum(row['optional_alighting'] for row in rides),
        terminal_rides=sum(row['terminal'] for row in rides),rides=rides,platforms=platforms,
        remaining_slots={trip:sorted(values) for trip,values in sorted(slots.items())})


def remaining_node(state):
    original = state['node']
    position = original['path'].index(state['cid'])
    suffix = original['path'][position:]
    return dict(original,path=suffix,first_cid=suffix[0],
        exits=[row for row in original['exits'] if row['cid'] in suffix])


def history_census(history,active,full_trips,model,remaining_slots):
    groups = defaultdict(list)
    for row in history:
        groups[row['trip']].append(row)
    rows = []
    for trip in sorted(set(groups)|active):
        facts = groups[trip]
        observed = [model._positions[row['cid'],row['kind']][1] for row in facts if row['evidence']=='observed']
        greatest = max(observed,default=-1)
        forecasts = [row for row in facts if row['evidence']=='forecast']
        available = [row for row in forecasts if model._positions[row['cid'],row['kind']][1]>greatest]
        rows.append(dict(trip=trip,active=trip in active,records=len(facts),
            evidence_counts=dict(Counter(row['evidence'] for row in facts)),
            hard_observed_count=len(observed),max_hard_slot=greatest,
            raw_forecasts=len(forecasts),future_slot_forecasts=len(available),
            selected_forecast_slots=[model._positions[row['cid'],row['kind']][1]
                for row in forecasts if row.get('emission_selected')],
            structurally_remaining_slots=remaining_slots.get(trip,[])))
    inactive = [row for row in rows if not row['active']]
    return dict(rows=len(history),trips=len(groups),inactive_trips=sum(bool(row['records']) for row in inactive),
        inactive_rows=sum(row['records'] for row in inactive),active_rows=sum(row['records'] for row in rows if row['active']),
        inactive_observed_rows=sum(row['hard_observed_count'] for row in inactive),
        already_outside_original_node_rows=sum(row['records'] for row in rows if row['trip'] not in full_trips),
        passed_branch_only_rows=sum(row['records'] for row in rows if row['trip'] in full_trips-active),
        active_trips_with_hard_observation=sum(row['active'] and row['hard_observed_count']>0 for row in rows),
        by_trip=rows)


def traces_census(bank,node,cid):
    counters = Counter()
    for arm,outcomes in bank['rows'].items():
        for row in outcomes:
            counters['outcomes']+=1
            counters['complete_outcomes']+=row['loss'] is not None
            boarded = False
            found = Counter()
            ride = node
            platform = (next(exit['next'] for exit in node['exits'] if exit['cid']==cid)
                        if bank['actions'][arm]=='alight' else None)
            for event in row['trace']:
                if event['kind']=='platform':
                    if platform is None or platform['stop']!=event['stop']:
                        raise AssertionError('Saved platform trace does not match its executed menu branch')
                    found['platform_decisions']+=1
                    if event['choice'] is None:
                        found['wait_decisions']+=1
                        found['wait_after_transfer']+=boarded
                    else:
                        found['board_decisions']+=1
                        boarded=True
                        ride = next(option['next'] for option in platform['rides'] if option['cid']==event['choice'])
                        platform = None
                elif event['kind']=='onboard':
                    position = ride['path'].index(event['cid'])
                    optional = any(exit['cid'] in ride['path'][position+1:] for exit in ride['exits'])
                    found['onboard_decisions']+=1
                    found['onboard_after_transfer']+=boarded
                    found['alight_after_transfer']+=boarded and event['choice']=='alight'
                    found['continue_after_transfer']+=boarded and event['choice']=='continue'
                    found['optional_onboard_after_transfer']+=boarded and optional
                    found['optional_alight_after_transfer']+=boarded and optional and event['choice']=='alight'
                    found['optional_continue_after_transfer']+=boarded and optional and event['choice']=='continue'
                    if event['choice']=='alight':
                        platform = next(exit['next'] for exit in ride['exits'] if exit['cid']==event['cid'])
            for key,value in found.items():
                counters[key]+=value
                counters['outcomes_with_'+key]+=value>0
    return dict(counters)


def main():
    manifest = read(V38/'seed_manifest.json')
    data,fit = read(V34/'data.json.gz'),read(V34/'model_fit.json')
    models = {day:setup(context(data,day)[0],fit)[0] for day in sorted({row['state']['day'] for row in manifest})}
    rows = []
    for index,entry in enumerate(manifest):
        state,model = entry['state'],models[entry['state']['day']]
        original,remaining = state['node'],remaining_node(state)
        full_trips = trip_set(original,model)
        tree = structure(remaining,model)
        active = set(tree['trips'])
        now = next(row for row in remaining['exits'] if row['cid']==state['cid'])['next']
        alight = trip_set(now,model) if now is not None else set()
        continuing = {model._positions[state['cid'],'arrival'][0]}
        for row in remaining['exits']:
            if row['cid']!=state['cid'] and row['next'] is not None:
                continuing.update(trip_set(row['next'],model))
        if active != alight|continuing:
            raise AssertionError('Remaining tree differs from the union of the two current action subtrees')
        name = entry['source'].replace('/','_').removesuffix('.json.gz')+'_fast.json.gz'
        saved = read(V38/'banks'/name)
        rows.append(dict(index=index,source=entry['source'],root_index=state['root_index'],day=state['day'],
            cid=state['cid'],clock=state['clock'],seed=entry['seed'],count=entry['count'],
            source_position=original['path'].index(state['cid']),
            passed_retained_exits=len(original['exits'])-len(remaining['exits']),
            full_node_trip_count=len(full_trips),active_trip_count=len(active),
            active_trips=sorted(active),passed_branch_only_trips=sorted(full_trips-active),
            alight_trips=sorted(alight),continue_trips=sorted(continuing),
            action_shared_trips=sorted(alight&continuing),structure=tree,
            public_history=history_census(state['public_history'],active,full_trips,model,tree['remaining_slots']),
            saved_trace_census=traces_census(saved,original,state['cid'])))
    active_histories = [trip for row in rows for trip in row['public_history']['by_trip'] if trip['active']]
    output = dict(scope='All 148 original fixed states; structural envelopes and saved traces only. No new MC or loss-based selection.',
        denominator=len(rows),reference_states=sum(row['source'].startswith('reference_cases/') for row in rows),
        policy_states=sum(row['source'].startswith('policy_cases/') for row in rows),
        quantities={key:summary(row[key] for row in rows) for key in
            ('full_node_trip_count','active_trip_count','passed_retained_exits','source_position')},
        structures={key:summary(row['structure'][key] for row in rows) for key in
            ('ride_occurrences','platform_occurrences','optional_alighting_rides','terminal_rides','platforms_with_same_trip_candidates')},
        history={key:summary(row['public_history'][key] for row in rows) for key in
            ('rows','trips','inactive_trips','inactive_rows','active_rows','inactive_observed_rows',
             'already_outside_original_node_rows','passed_branch_only_rows','active_trips_with_hard_observation')},
        active_trip_history=dict(state_trip_pairs=len(active_histories),
            max_hard_slot_distribution=dict(sorted(Counter(trip['max_hard_slot'] for trip in active_histories).items())),
            observed_records=sum(trip['hard_observed_count'] for trip in active_histories),
            raw_forecasts=sum(trip['raw_forecasts'] for trip in active_histories),
            future_slot_forecast_distribution=dict(sorted(Counter(trip['future_slot_forecasts'] for trip in active_histories).items())),
            records_per_trip=summary(trip['records'] for trip in active_histories)),
        states_with=dict(passed_branch_only_trips=sum(bool(row['passed_branch_only_trips']) for row in rows),
            action_shared_trips=sum(bool(row['action_shared_trips']) for row in rows),
            repeated_trip_occurrences=sum(bool(row['structure']['repeated_trips']) for row in rows),
            sibling_shared_trip=sum(bool(row['structure']['sibling_shared_trips']) for row in rows),
            ancestor_trip_reuse=sum(bool(row['structure']['ancestor_reuse']) for row in rows),
            inactive_public_history=sum(bool(row['public_history']['inactive_rows']) for row in rows),
            tail_onboard_in_saved_bank=sum(bool(row['saved_trace_census'].get('onboard_after_transfer')) for row in rows),
            tail_alight_in_saved_bank=sum(bool(row['saved_trace_census'].get('alight_after_transfer')) for row in rows),
            optional_tail_onboard_in_saved_bank=sum(bool(row['saved_trace_census'].get('optional_onboard_after_transfer')) for row in rows),
            tail_wait_in_saved_bank=sum(bool(row['saved_trace_census'].get('wait_after_transfer')) for row in rows)),
        saved_trace_totals=dict(sum((Counter(row['saved_trace_census']) for row in rows),Counter())),
        interpretation=[
            'The existing planner already excludes unrelated trips from each subtree cache key; deleting these facts alone does not produce a new sufficient-state result.',
            'The remaining subtree can remove passed exits and their exclusively reachable trips. Preserve all evidence on active trips, including earlier observations and unselected raw forecasts needed after reselection.',
            'Do not split repeated physical trips into independent branch copies; sibling sharing and the original joint trip draws are retained.',
            'Both action arms use the union of their active trips. Removing a trip from sorted TripWorlds sampling would shift later random-number slices; exact fixed-bank comparisons must preserve the original RNG assignment.',
            'Saved traces expose actual continuation decisions within the declared banks, not unseen-state completeness or real-world benefit. Active sets here are structural envelopes, not proven minimal states.'
        ],states=rows)
    HERE.mkdir(exist_ok=True)
    (HERE/'census.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in output.items() if key not in ('states','quantities','structures','history')},indent=2))
    print('quantities',json.dumps(output['quantities']))
    print('history',json.dumps(output['history']))


if __name__=='__main__':
    main()
