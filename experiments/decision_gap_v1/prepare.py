"""Freeze all public menus before running any new archived policy outcomes."""
from pathlib import Path
from time import perf_counter, process_time
import csv
import json

from experiments.decision_gap_v1.candidate_coverage import expand_menu, endpoint_coverage
from experiments.decision_gap_v1.panel import read, write
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context
from experiments.reconstruction_v34.run import inputs
from experiments.reconstruction_v37.controller import node_structure
from experiments.reconstruction_v38.run import save, restore_node
from experiments.reconstruction_v50.execute import ObservationContract

HERE = Path(__file__).resolve().parent


def observations(value):
    return ObservationContract({row['stop']: row['cids'] for row in value['station_to_cids']})


def trip_set(graph, node):
    return {graph.connections[cid].trip_id for path in node.paths() for cid in path}


def coverage_artifacts():
    """Export the frozen public construction ledger, without rebuilding menus."""
    contract = read(HERE/'contract.json')
    manifest, rows = [], []
    for index in range(contract['roots']):
        menu = read(HERE/'menus'/f'{index:03}.json.gz')
        manifest.append(dict(root_index=index, day=menu['root']['day'],
            source=f'menus/{index:03}.json.gz', counts=menu['ledger']['counts'],
            old_menu_cost=menu['old_menu_cost'], closure_cost=menu['closure_cost']))
        rows.extend(dict(root_index=index, day=menu['root']['day'], **row)
                    for row in menu['ledger']['candidate_occurrences'])
    write(HERE/'candidate_coverage_manifest.json', dict(rule=contract['candidate_rule'],
        roots=manifest, actual_labels_used=False))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (HERE/'coverage_ledger.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:json.dumps(value) if isinstance(value,(list,dict)) else value
                         for key,value in row.items()} for row in rows)


def run():
    contract = read(HERE/'contract.json')
    tick, cpu = perf_counter(), process_time()
    _, data, _, _, selection = inputs()
    preparation = dict(archive=dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick),
                       graphs={}, roots=contract['roots'])
    if len(selection['selected']) != contract['roots']:
        raise AssertionError('Frozen root denominator changed')
    graphs, counts = {}, []
    for index, selected in enumerate(selection['selected']):
        old = read(HERE.parent/'reconstruction_v50/menus'/f'{index:03}.json.gz')
        root = old['wide_root']
        if root['root_id'] != selected['root_id']:
            raise AssertionError('Original72 root identity changed')
        if root['day'] not in graphs:
            tick, cpu = perf_counter(), process_time()
            graphs[root['day']] = context(data, root['day'])
            preparation['graphs'][root['day']] = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        graph = graphs[root['day']][0]
        path = HERE/'menus'/f'{index:03}.json.gz'
        if path.exists():
            counts.append(read(path)['ledger']['counts'])
            continue
        tick, cpu = perf_counter(), process_time()
        original = root_ride(graph, root)
        expanded, ledger = expand_menu(graph, original, root['goal'])
        old_o, new_o = ObservationContract.from_menu(original), ObservationContract.from_menu(expanded)
        union = {stop:set(cids) for stop,cids in new_o.station_to_cids.items()}
        for stop,cids in old_o.station_to_cids.items():
            union.setdefault(stop, set()).update(cids)
        new_o = ObservationContract(union)
        if trip_set(graph, original) != trip_set(graph, expanded):
            raise AssertionError('Closure changed the remote vehicle population')
        original_json, expanded_json = node_structure(original), node_structure(expanded)
        old_description, new_description = old_o.describe(), new_o.describe()
        closure_cost = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        cost = {metric: old['cost'][metric]+closure_cost[metric] for metric in ('process_cpu','wall')}
        menu = dict(root_index=index, root=root, original_node=original_json,
            expanded_node=expanded_json, old_observation_contract=old_description,
            new_observation_contract=new_description, ledger=ledger,
            old_menu_cost=old['cost'], closure_cost=closure_cost, cost=cost,
            scope='Public construction only; actual endpoint audit saved separately after all menus are frozen')
        save(path, menu)
        counts.append(ledger['counts'])
        print('menu', index, ledger['counts'], flush=True)
    # Label inspection occurs only after the entire population is constructed.
    tick, cpu = perf_counter(), process_time()
    coverage = []
    for index in range(contract['roots']):
        menu = read(HERE/'menus'/f'{index:03}.json.gz')
        graph, _, _, _, metadata, _ = graphs[menu['root']['day']]
        coverage.append(dict(root_index=index, **endpoint_coverage(graph, metadata,
            restore_node(menu['expanded_node']), observations(menu['new_observation_contract']))))
    preparation['offline_endpoint_audit'] = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
    write(HERE/'preparation.json', preparation)
    write(HERE/'coverage.json', dict(roots=contract['roots'], counts=counts, endpoint_audits=coverage,
        selection='All original72, frozen before actual labels; labels never remove a candidate or root'))
    coverage_artifacts()


if __name__ == '__main__':
    run()
