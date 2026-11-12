"""Enumerate every saved V36 baseline onboard decision without gain screening."""
from collections import Counter
import json
from pathlib import Path

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.conditional_reference import locate_node
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v35.validate import write

HERE = Path(__file__).resolve().parent
V34 = HERE.parent/'reconstruction_v34'
V36 = HERE.parent/'reconstruction_v36'


def action(row):
    return row['kind'], row.get('cid', row.get('stop')), row['clock'], row.get('choice')


def node_structure(ride):
    return dict(first_cid=ride.first_cid, path=list(ride.path), exits={str(cid):
        None if platform is None else dict(stop=platform.stop,
            rides={str(first):node_structure(child) for first,child in sorted(platform.rides.items())})
        for cid,platform in sorted(ride.exits.items())})


def public_state_key(day, node, row, deadline):
    """A plain state description, not a digest; dates disambiguate local CIDs."""
    return json.dumps(dict(day=day, node=node_structure(node), current_cid=row['cid'],
        clock=row['clock'], deadline=deadline, public_history=row['public_history']),
        sort_keys=True, separators=(',', ':'))


def main():
    data, selection = read(V34/'data.json.gz'), read(V34/'selection.json')
    graphs = {day:context(data,day)[0] for day in sorted({r['day'] for r in selection['selected']})}
    records, by_root, by_day, seen = [], [], {}, {}
    for index, root in enumerate(selection['selected']):
        source = V36/'roots'/f'{index:03}.json.gz'
        saved = read(source)['rows']['public_messages']
        trace = saved['combined_rolling']['outcome']['trace']
        nominal = saved['rolling_nominal']['outcome']['trace']
        gated = saved['standard_conditional_rollout']['outcome']['trace']
        graph = graphs[root['day']]
        ride = root_ride(graph,root)
        counts, eligible_call = Counter(), 0
        for trace_index, row in enumerate(trace):
            if row['kind']!='onboard':
                continue
            node = locate_node(ride,trace,trace_index)
            position = node.path.index(row['cid'])
            later = [cid for cid in node.path[position+1:] if cid in node.exits]
            current = node.exits[row['cid']]
            eligible = current is not None and bool(current.rides) and bool(later)
            matches = {}
            for name, other in (('nominal',nominal),('old_rollout',gated)):
                same_prefix = (trace_index<len(other) and
                    [action(r) for r in trace[:trace_index]]==[action(r) for r in other[:trace_index]])
                same_state = (same_prefix and action(row)[:3]==action(other[trace_index])[:3]
                    and row['public_history']==other[trace_index].get('public_history'))
                matches[name] = dict(same_public_state=same_state,
                    action=other[trace_index]['choice'] if same_state else None,
                    agrees_with_baseline=(other[trace_index]['choice']==row['choice']) if same_state else None,
                    score_receipt=other[trace_index]['receipt'] if same_state else None)
            receipt = row['receipt']
            state_id = f'{index:03}:{trace_index:03}'
            key = public_state_key(root['day'],node,row,root['deadline'])
            duplicate = seen.get(key)
            seen.setdefault(key,state_id)
            record = dict(state_id=state_id, root_index=index,root_id=root['root_id'],day=root['day'],
                source=f'../reconstruction_v36/roots/{index:03}.json.gz',
                method='combined_rolling',trace_index=trace_index,current_cid=row['cid'],clock=row['clock'],
                deadline=root['deadline'],node_first_cid=node.first_cid,node_path=list(node.path),
                current_path_index=position,current_alighting_menu_present=current is not None and bool(current.rides),
                later_retained_exits=later,two_structural_menu_actions=eligible,
                exclusion_reason=None if eligible else 'no_later_retained_exit_or_current_platform',
                eligible_call_index=eligible_call if eligible else None,
                seed=37050+1000*index+eligible_call if eligible else None,
                duplicate_of_state_id=duplicate,
                baseline_action=row['choice'],baseline_score_receipt=receipt,
                alight_minus_continue_seconds=receipt['alight']-receipt['continuation'],
                absolute_score_gap_seconds=abs(receipt['alight']-receipt['continuation']),
                comparisons=matches,public_history=row['public_history'],
                trace_locator=dict(source=f'../reconstruction_v36/roots/{index:03}.json.gz',
                    path=['rows','public_messages','combined_rolling','outcome','trace',trace_index]))
            records.append(record)
            counts['onboard_decisions']+=1
            counts['eligible_two_action_states']+=eligible
            counts['nominal_agreement']+=matches['nominal']['agrees_with_baseline'] is True
            counts['nominal_unmatched_public_state']+=not matches['nominal']['same_public_state']
            if eligible:
                eligible_call+=1
        by_root.append(dict(root_index=index,root_id=root['root_id'],day=root['day'],**dict(counts)))
        by_day.setdefault(root['day'],Counter()).update(counts)
    summary = dict(roots=len(selection['selected']),onboard_decisions=len(records),
        unique_full_public_states=len(seen),
        eligible_two_action_states=sum(r['two_structural_menu_actions'] for r in records),
        excluded_no_second_menu_action=sum(not r['two_structural_menu_actions'] for r in records),
        nominal_same_state_and_action=sum(r['comparisons']['nominal']['agrees_with_baseline'] is True for r in records),
        old_rollout_same_state_and_action=sum(r['comparisons']['old_rollout']['agrees_with_baseline'] is True for r in records),
        duplicate_full_public_states=sum(r['duplicate_of_state_id'] is not None for r in records))
    output = dict(scope='Complete saved baseline onboard census; no sample or counterfactual actual outcome.',
        selection='All 72 roots and every combined_rolling onboard receipt. Eligibility uses only current '
            'retained platform plus at least one later retained exit on that ride, not score gaps or losses.',
        feasibility='Two actions means two structural menu choices; it does not assert either continuation '
            'succeeds in the realized world or under every conditional draw.',
        history_restoration='SingleMessageHistory with unchanged V35 fitted emission; restore saved own/local '
            'facts, raw forecasts and notices, clear laws, and attach no future frame stream.',
        summary=summary,by_day={day:dict(c) for day,c in by_day.items()},by_root=by_root,states=records)
    write(HERE/'census.json',output)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
