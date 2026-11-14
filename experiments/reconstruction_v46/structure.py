"""Input-derived mixed-tail structure, retaining every positive V45 atom.

Reads existing receipts only. These are eligibility facts, not a certificate
that a proposed smaller state preserves full-policy value.
"""
from collections import Counter
from pathlib import Path
from time import perf_counter, process_time
import json

from experiments.reconstruction_v34.discover import context, read, write
from experiments.reconstruction_v34.run import setup

HERE = Path(__file__).resolve().parent
V45 = HERE.parent/'reconstruction_v45'


def ride_nodes(node):
    if node is None:
        return
    if node[0] == 'ride':
        yield node
        for _, child in node[3]:
            yield from ride_nodes(child)
    else:
        for _, ride in node[2]:
            yield from ride_nodes(ride)


def trips(node, model):
    return {model._positions[cid, 'departure'][0]
            for ride in ride_nodes(node) for cid in ride[2]}


def boardings(node):
    if node is None:
        return 0
    if node[0] == 'platform':
        return max((1+boardings(ride) for _, ride in node[2]), default=0)
    return max((boardings(child) for _, child in node[3]), default=0)


def physical_exits(ride):
    """The original executor terminates at the first destination in path order."""
    exits = dict(ride[3])
    for cid in ride[2]:
        if cid not in exits:
            continue
        yield cid, exits[cid]
        if exits[cid] is None:
            break


def executed_boardings(node):
    if node is None:
        return 0
    if node[0] == 'platform':
        return max((1+executed_boardings(ride) for _, ride in node[2]), default=0)
    return max((executed_boardings(child) for _, child in physical_exits(node)), default=0)


def ride_execution(ride, model):
    exits = dict(ride[3])
    destination = next((cid for cid in ride[2] if cid in exits and exits[cid] is None), None)
    end = len(ride[2]) if destination is None else ride[2].index(destination)+1
    later = [(cid, exits[cid]) for cid in ride[2][end:] if cid in exits]
    return dict(first_destination_cid=destination,
        physical_executable_prefix=list(ride[2][:end]),
        early_destination=destination is not None and end < len(ride[2]),
        exits_after_first_destination=[dict(cid=cid, has_tail=child is not None,
            tail_trips=sorted(trips(child, model))) for cid, child in later],
        policy_dependencies_after_first_destination=sorted(set().union(
            *(trips(child, model) for _, child in later))),
        pure_physical_terminal=destination is not None and all(child is None for _, child in physical_exits(ride)),
        max_executable_boardings=1+executed_boardings(ride),
        execution_scope='Physical path ends at the first destination; original policy scoring retains every declared exit and tail')


def future_entries(ride, physical_reachable=True):
    """Identify later platform departure slots, including upstream slots."""
    executable_exits = {cid for cid, _ in physical_exits(ride)}
    for exit_cid, child in ride[3]:
        if child is None:
            continue
        reachable = physical_reachable and exit_cid in executable_exits
        for cid, future in child[2]:
            yield exit_cid, child[1], cid, future, reachable
            yield from future_entries(future, reachable)


def inspect_platform(node, model, graph):
    roots = {cid: model._positions[cid, 'departure'][0] for cid, _ in node[2]}
    candidate_rows, links = [], []
    tail_counts = Counter()
    for cid, ride in node[2]:
        positions = [model._positions[c, kind] for c in ride[2]
                     for kind in ('departure', 'arrival')]
        future = set().union(*(trips(child, model) for _, child in ride[3]))
        tail_counts.update(future)
        candidate_rows.append(dict(cid=cid, trip=roots[cid], path=list(ride[2]),
            pure_terminal=all(child is None for _, child in ride[3]),
            same_trip=all(trip == roots[cid] for trip, _ in positions),
            contiguous=all(b[1] == a[1]+1 for a, b in zip(positions, positions[1:])),
            future_trips=sorted(future), self_trip_in_tail=roots[cid] in future,
            max_remaining_boardings=1+boardings(ride), **ride_execution(ride, model)))
        for exit_cid, stop, future_cid, future_ride, reachable in future_entries(ride):
            future_trip, future_slot = model._positions[future_cid, 'departure']
            for root_cid, root_trip in roots.items():
                if root_trip != future_trip:
                    continue
                root_slot = model._positions[root_cid, 'departure'][1]
                links.append(dict(selected_candidate_cid=cid, transfer_exit_cid=exit_cid,
                    later_platform_stop=stop, observed_root_cid=root_cid, trip=root_trip,
                    root_departure_slot=root_slot, later_candidate_cid=future_cid,
                    later_departure_slot=future_slot, later_path=list(future_ride[2]),
                    physically_reachable=reachable,
                    slot_direction='backward' if future_slot < root_slot else
                                   'forward' if future_slot > root_slot else 'same'))
    interfaces = []
    for trip in sorted({link['trip'] for link in links}):
        relevant = [link for link in links if link['trip'] == trip]
        interfaces.append(dict(trip=trip,
            observed_departures=sorted({(link['observed_root_cid'], link['root_departure_slot'])
                                       for link in relevant}),
            later_departures=sorted({(link['later_candidate_cid'], link['later_departure_slot'])
                                    for link in relevant}),
            backward_conditioning_required=any(link['slot_direction'] == 'backward' for link in relevant),
            retained_evidence=['observed departure clock or latest absence constraint',
                'completed physical slot for working-forecast reselection',
                'all root raw forecasts, notices and immutable physical basis by reference'],
            probability_rule='Condition P0 on legal observations, including backward queries',
            policy_rule='Evaluate original pi under W; keep chosen cid and original simultaneous rank'))
    return dict(stop=node[1], station=graph.stops[node[1]].name,
        candidate_count=len(roots), distinct_root_trips=len(set(roots.values())) == len(roots),
        max_remaining_boardings=boardings(node), max_executable_boardings=executed_boardings(node),
        candidates=candidate_rows,
        pure_terminal_platform=all(r['pure_terminal'] for r in candidate_rows),
        pure_physical_terminal_platform=all(r['pure_physical_terminal'] for r in candidate_rows),
        early_destination_candidates=sum(r['early_destination'] for r in candidate_rows),
        same_trip_contiguous_candidates=all(r['same_trip'] and r['contiguous'] for r in candidate_rows),
        root_trip_in_own_tail=any(r['self_trip_in_tail'] for r in candidate_rows),
        future_trips_shared_between_candidates={trip: count for trip, count in tail_counts.items() if count > 1},
        root_to_future_links=links,
        physically_reachable_root_to_future_links=[link for link in links if link['physically_reachable']],
        required_observation_interface=interfaces,
        interface_scope='Necessary scoring evidence retains all declared tails, including exits after first physical destination; sufficiency requires a separate proof')


def grouped_platforms(atoms, model, graph):
    """Group equal menus without dropping small probability or entry times."""
    groups = {}
    for index, atom in enumerate(atoms):
        if atom['kind'] != 'platform' or atom['probability'] <= 0:
            continue
        key = json.dumps(atom['node_structure'], separators=(',', ':'))
        if key not in groups:
            groups[key] = dict(structure=inspect_platform(atom['node_structure'], model, graph),
                atom_indices=[], probability=0., entries=[])
        group = groups[key]
        group['atom_indices'].append(index)
        group['probability'] += atom['probability']
        group['entries'].append(dict(entry=atom['entry'], probability=atom['probability']))
    return list(groups.values())


def describe(receipt, model, graph, source):
    arms = {arm: grouped_platforms(atoms, model, graph) for arm, atoms in receipt['atoms'].items()}
    all_platforms = [p for groups in arms.values() for p in groups]
    interface_trips = {interface['trip'] for p in all_platforms
                       for interface in p['structure']['required_observation_interface']}
    existing = [record for record in receipt['public_history']
                if record.get('trip') in interface_trips and record['evidence'] in ('observed', 'unseen')]
    return dict(source=source, clock=receipt['clock'], deadline=receipt['deadline'],
        base_action=receipt['base_action'], selected_action=receipt['selected_action'],
        critical=receipt['base_action'] != receipt['selected_action'],
        public_history_reference=source+('/public_history' if '#/' in source else '#/public_history'),
        root_interface_hard_evidence=existing, arms=arms,
        all_positive_platform_atoms_retained=True,
        terminal_mass={arm: sum(a['probability'] for a in values if a['kind'] == 'terminal')
                       for arm, values in receipt['atoms'].items()},
        unresolved_prefix_mass={arm: sum(a['probability'] for a in values if a['kind'] == 'unresolved_prefix')
                                for arm, values in receipt['atoms'].items()},
        maximum_tail_boardings=max((p['structure']['max_remaining_boardings'] for p in all_platforms), default=0),
        maximum_executable_tail_boardings=max((p['structure']['max_executable_boardings'] for p in all_platforms), default=0),
        pure_terminal_alight=all(p['structure']['pure_terminal_platform'] for p in arms['alight']),
        pure_physical_terminal_alight=all(p['structure']['pure_physical_terminal_platform'] for p in arms['alight']),
        early_destination_alight_candidates=sum(p['structure']['early_destination_candidates'] for p in arms['alight']),
        alight_interface_trip_count=len({i['trip'] for p in arms['alight']
                                        for i in p['structure']['required_observation_interface']}),
        physically_reachable_alight_interface_trip_count=len({link['trip'] for p in arms['alight']
            for link in p['structure']['physically_reachable_root_to_future_links']}))


def panel_summary(rows):
    critical = [r for r in rows if r['critical']]
    def execution_counts(selected):
        platforms = [p['structure'] for r in selected for groups in r['arms'].values() for p in groups]
        candidates = [c for p in platforms for c in p['candidates']]
        alight_candidates = [c for r in selected for p in r['arms']['alight'] for c in p['structure']['candidates']]
        return dict(grouped_platform_occurrences=len(platforms), candidate_occurrences=len(candidates),
            early_destination_candidates=sum(c['early_destination'] for c in candidates),
            declared_terminal_candidates=sum(c['pure_terminal'] for c in candidates),
            physical_terminal_candidates=sum(c['pure_physical_terminal'] for c in candidates),
            alight_candidate_occurrences=len(alight_candidates),
            early_destination_alight_candidates=sum(c['early_destination'] for c in alight_candidates),
            declared_terminal_alight_candidates=sum(c['pure_terminal'] for c in alight_candidates),
            physical_terminal_alight_candidates=sum(c['pure_physical_terminal'] for c in alight_candidates),
            states_with_early_destination_alight=sum(r['early_destination_alight_candidates'] > 0 for r in selected),
            pure_physical_terminal_alight=sum(r['pure_physical_terminal_alight'] for r in selected),
            max_executable_tail_boardings_histogram=dict(Counter(r['maximum_executable_tail_boardings'] for r in selected)))
    return dict(count=len(rows), critical_count=len(critical),
        pure_terminal_alight=sum(r['pure_terminal_alight'] for r in rows),
        critical_pure_terminal_alight=sum(r['pure_terminal_alight'] for r in critical),
        max_tail_boardings_histogram=dict(Counter(r['maximum_tail_boardings'] for r in rows)),
        alight_interface_trip_count_histogram=dict(Counter(r['alight_interface_trip_count'] for r in rows)),
        physically_reachable_alight_interface_trip_count_histogram=dict(Counter(
            r['physically_reachable_alight_interface_trip_count'] for r in rows)),
        critical_no_alight_interface=[r['root_index'] for r in critical if r['alight_interface_trip_count'] == 0],
        critical_with_alight_interface=[r['root_index'] for r in critical if r['alight_interface_trip_count'] > 0],
        critical_with_physically_reachable_alight_interface=[r['root_index'] for r in critical
            if r['physically_reachable_alight_interface_trip_count'] > 0],
        critical_backward_alight_roots=[r['root_index'] for r in critical if any(
            i['backward_conditioning_required'] for p in r['arms']['alight']
            for i in p['structure']['required_observation_interface'])],
        physical_execution=execution_counts(rows), critical_physical_execution=execution_counts(critical),
        execution_scope='Physical terminal shape does not remove original policy scoring dependencies and is not a three-mark exactness certificate')


def main():
    tick, cpu = perf_counter(), process_time()
    manifest = read(HERE.parent/'reconstruction_v38/seed_manifest.json')
    data = read(HERE.parent/'reconstruction_v34/data.json.gz')
    fit = read(HERE.parent/'reconstruction_v34/model_fit.json')
    selection = read(HERE.parent/'reconstruction_v34/selection.json')['selected']
    contexts = {day: context(data, day) for day in {row['day'] for row in selection}}
    models = {day: setup(ctx[0], fit)[0] for day, ctx in contexts.items()}
    frozen, online = [], []
    for index, entry in enumerate(manifest):
        day = entry['state']['day']
        source = f'experiments/reconstruction_v45/banks/{index:03}_absorbing.json.gz'
        receipt = read(HERE.parents[1]/source)
        row = describe(receipt, models[day], contexts[day][0], source)
        row.update(index=index, root_index=receipt['root_index'], day=day,
                   manifest_source=entry['source'])
        frozen.append(row)
    for index, root in enumerate(selection):
        day = root['day']
        source = f'experiments/reconstruction_v45/roots/{index:03}_absorbing.json.gz'
        receipt = read(HERE.parents[1]/source)
        for gate_index, gate in enumerate(receipt['gates']):
            gate_source = source+f'#/gates/{gate_index}'
            row = describe(gate, models[day], contexts[day][0], gate_source)
            row.update(root_index=index, gate_index=gate_index, cid=gate['cid'], day=day,
                       manifest_source=gate['source'])
            online.append(row)
    result = dict(scope='Read-only current V45 menus and exact absorption receipts; no policy evaluation or new worlds',
        fixed_model='V41_exact_integer_anchor_finite_chain_v2',
        no_probability_cutoff=True, frozen_summary=panel_summary(frozen), online_summary=panel_summary(online),
        frozen=frozen, online=online,
        cost=dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick),
        claim='Menu eligibility and candidate observation interfaces only; no value equivalence or error bound is claimed')
    write(HERE/'tail_structure.json', result)
    print(json.dumps({key: result[key] for key in ('frozen_summary', 'online_summary', 'cost')}, indent=2))


if __name__ == '__main__':
    main()
