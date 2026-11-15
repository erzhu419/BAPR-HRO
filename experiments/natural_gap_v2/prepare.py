"""Freeze public roots/menus before attaching any new actual outcomes."""
import argparse
import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time

from experiments.decision_calibration_v2.run_calibration import select_public
from experiments.decision_gap_v1.candidate_coverage import expand_menu, endpoint_coverage
from experiments.decision_gap_v1.panel import read, write
from experiments.decision_gap_v1.prepare import observations, trip_set
from experiments.decision_gap_v1.transfer_events import register_opportunities
from experiments.reconstruction_data_audit.build_dated_replay import attach_outcomes
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context, static_catalog
from experiments.reconstruction_v37.controller import node_structure
from experiments.reconstruction_v38.run import restore_node
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v50.menu import wide_root

HERE = Path(__file__).resolve().parent


def prepare(directory):
    stage_tick,stage_cpu = perf_counter(),process_time()
    directory = Path(directory)
    manifest = read(directory/'date_manifest.json')
    rules = read(directory/'protocol.json')
    exposed = set(read(directory/'exposure_inventory.json')['known_exposed_service_dates'])
    dates = manifest['selected_dates']
    if not manifest['dates_frozen'] or len(dates) != 14 or exposed.intersection(dates):
        raise ValueError('Expected fourteen frozen unused dates')
    if (directory/'selection.json').exists():
        raise ValueError('Public selection already frozen; do not rebuild menus')
    protocol = read(HERE.parent/'reconstruction_v34/protocol.json')
    protocol['main_dates'] = dates
    source = read(directory/'scheduled.json.gz')
    data = dict(stops=source['stops'],dates={day:source['dates'][day] for day in dates})
    for day in dates:
        attach_outcomes(day,data['dates'][day]['connections'],[])
    write(directory/'scheduled_placeholder.json.gz',data)
    catalog = static_catalog(data,protocol)
    write(directory/'static_catalog.json.gz',catalog)
    selected = select_public(catalog,dates,protocol['structure']['max_selected_per_day'])
    graphs, ledger, roots = {}, [], []
    for index, row in enumerate(selected['selected']):
        day = row['day']
        if day not in graphs:
            graphs[day] = context(data,day)[0]
        graph = graphs[day]
        tick,cpu = perf_counter(),process_time()
        root,caps = wide_root(graph,row,protocol['structure'])
        original = root_ride(graph,root)
        expanded,coverage = expand_menu(graph,original,root['goal'])
        old_o,new_o = ObservationContract.from_menu(original),ObservationContract.from_menu(expanded)
        union = {stop:set(cids) for stop,cids in new_o.station_to_cids.items()}
        for stop,cids in old_o.station_to_cids.items():
            union.setdefault(stop,set()).update(cids)
        sight = ObservationContract(union)
        if trip_set(graph,original) != trip_set(graph,expanded):
            raise AssertionError('Prefix closure changed the remote vehicle set')
        original_json,expanded_json = node_structure(original),node_structure(expanded)
        description = sight.describe()
        cost = dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
        cid = next(cid for cid in original.path if original.exits.get(cid) is not None)
        registration = register_opportunities(graph,original,cid,graph.connections[cid].arr_time,expanded)
        registration.update(root_index=index,day=day,
                            clock_source='schedule placeholder; actual initial state evaluated later')
        write(directory/'menus'/f'{index:03}.json.gz',dict(root_index=index,root=root,
            original_node=original_json,expanded_node=expanded_json,
            new_observation_contract=description,cost=cost,ledger=coverage,cap_ledger=caps,
            registration=registration,actual_values_used=False))
        roots.append(dict(root_index=index,**{k:root[k] for k in ('root_id','day','start','deadline')}))
        ledger.append(dict(root_index=index,day=day,opportunities=len(registration['rows']),
            candidate_pairs=len(registration['candidate_pairs']),no_opportunity=registration['no_opportunity']))
        print('public menu',index,day,len(registration['rows']),flush=True)
    stamp = datetime.now(timezone.utc).isoformat()
    selected.update(frozen_at_utc=stamp,registered=ledger,selection_uses_target_values=False)
    write(directory/'selection.json',selected)
    write(directory/'contract.json',dict(stage='natural_gap_v2',roots=len(roots),root_manifest=roots,
        control_dates=dates,policies=rules['fixed']['policies'],
        labels=['new_'+p for p in rules['fixed']['policies']],
        rollout_limits=rules['fixed']['rollout_limits'],frozen_at_utc=stamp,
        emission_snapshots={'repaired_fit':'repaired_fit.json'},
        physical_fit='physical_fit.json',archive_calls=len(roots)*len(rules['fixed']['policies']),
        measurement='One cold rotated pass; all model/history/menu restore/control decisions and internal trace publication timed. Fresh public wide-menu plus closure charge included equally. Source loading, graph preparation and public/endpoint-audit stages have separate receipts; output serialization and summaries are outside per-query measurements.',
        original_72_additional_budget=0))
    write(directory/'preparation_public.json',dict(process_cpu=process_time()-stage_cpu,
        wall=perf_counter()-stage_tick,roots=len(roots),target_values_read=False,
        scope='Common source loading, public catalogue, menus, registration and frozen output writes. Per-menu construction is already billed separately to every arm.'))
    print('public cohort frozen',len(roots),'roots',flush=True)


def attach(directory):
    stage_tick,stage_cpu = perf_counter(),process_time()
    directory = Path(directory)
    selection = read(directory/'selection.json')
    data = read(directory/'scheduled_placeholder.json.gz')
    summary,graphs,audits = {},{},[]
    for day in data['dates']:
        path = directory/'actual'/f'{day}.csv.gz'
        receipt = path.with_suffix('.audit.json')
        available = path.exists() and receipt.exists()
        with gzip.open(path,'rt') if available else open('/dev/null') as handle:
            rows = list(csv.DictReader(handle,delimiter=';')) if available else []
        coverage = attach_outcomes(day,data['dates'][day]['connections'],rows)
        summary[day] = dict(coverage,actual_source_available=available,
                           source=str(path),audit=str(receipt))
        graphs[day] = context(data,day)
    for index,root in enumerate(selection['selected']):
        menu = read(directory/'menus'/f'{index:03}.json.gz')
        graph,_,_,_,metadata,_ = graphs[root['day']]
        audits.append(dict(root_index=index,day=root['day'],**endpoint_coverage(
            graph,metadata,restore_node(menu['expanded_node']),observations(menu['new_observation_contract']))))
    write(directory/'data.json.gz',data)
    write(directory/'endpoint_coverage.json',dict(days=summary,roots=audits,
        selection_frozen_at_utc=selection['frozen_at_utc'],missing_roots_dropped=False))
    write(directory/'preparation_attach.json',dict(process_cpu=process_time()-stage_cpu,
        wall=perf_counter()-stage_tick,roots=len(audits),
        scope='Offline actual-source joins, expanded-menu coverage audit and shared dated input writes; not per-policy decision time.'))
    print('actual endpoints attached; all',len(audits),'public roots retained',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('public','attach'))
    parser.add_argument('--directory',type=Path,default=HERE)
    args = parser.parse_args()
    (prepare if args.stage=='public' else attach)(args.directory)
