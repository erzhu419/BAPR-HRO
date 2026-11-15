"""One repaired model, eight frozen controllers, all newly frozen public roots."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter, process_time

from experiments.control_identity_v1.run import one_arm
from experiments.decision_gap_v1.panel import read, write
from experiments.reconstruction_v34.discover import context
from experiments.reconstruction_v38.run import ordered, save, restore_node
from experiments.reconstruction_v51.run import LABELS as POLICIES

HERE = Path(__file__).resolve().parent
RESIDENT = None


def check_inputs(directory):
    contract = read(directory/'contract.json')
    protocol = read(directory/'protocol.json')
    manifest = read(directory/'date_manifest.json')
    selection = read(directory/'selection.json')
    if tuple(contract['policies']) != POLICIES:
        raise ValueError('Frozen controller family changed')
    if contract['rollout_limits'] != protocol['fixed']['rollout_limits']:
        raise ValueError('Original rollout budget changed')
    if contract['roots'] != len(selection['selected']):
        raise ValueError('Public root denominator changed')
    if manifest['selected_dates'] != contract['control_dates'] or not manifest['dates_frozen']:
        raise ValueError('Date cohort changed')
    if read(directory/'repaired_fit.json') != read(HERE.parent/'control_identity_v1/repaired_fit.json'):
        raise ValueError('Frozen repaired candidate changed')
    if read(directory/'physical_fit.json') != read(HERE.parent/'reconstruction_v34/model_fit.json'):
        raise ValueError('V41 physical model changed')
    for item in contract['root_manifest']:
        menu = read(directory/'menus'/f'{item["root_index"]:03}.json.gz')
        if any(menu['root'][key] != item[key] for key in ('root_id','day','start','deadline')):
            raise ValueError('Frozen root context changed')
    return contract


def initialize(directory):
    global RESIDENT
    directory = Path(directory)
    tick,cpu = perf_counter(),process_time()
    contract = read(directory/'contract.json')
    messages,available,reasons = {},{},{}
    for day in contract['control_dates']:
        path = directory/'messages'/f'{day}.json.gz'
        source = read(path) if path.exists() else dict(frames=[],summary={})
        messages[day] = source
        summary = source['summary']
        failed = []
        if not summary.get('source_parse_complete',False):
            failed.append('message_parse_incomplete')
        if not summary.get('identity_mapping_complete',False):
            failed.append('message_identity_unresolved')
        if summary.get('sampled_snapshots',0)==0:
            failed.append('no_predecision_capture')
        available[day],reasons[day] = not failed,failed
    RESIDENT = dict(directory=directory,contract=contract,
        data=read(directory/'data.json.gz'),messages={'dates':messages},
        fit=read(directory/'physical_fit.json'),emissions={'repaired_fit':read(directory/'repaired_fit.json')},
        coverage={r['root_index']:r for r in read(directory/'endpoint_coverage.json')['roots']},
        source_available=available,source_reasons=reasons,graphs={})
    RESIDENT['load_cost'] = dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)


def initial_evidence(row, graph, mapping, frames, first_trip, menu_cids):
    first = next((s for s in row['states'] if s['kind']=='onboard'),None)
    if first is None:
        return dict(initial_state=dict(status=row['outcome']['status'],cid=None,clock=None),
                    predecision_numeric_messages=0,model_selected_numeric_messages=0,
                    other_trip_numeric_messages=0,historical_future_tail_publications=0,
                    historical_forecast_witnesses=[])
    now = first['clock']
    active = [r for r in first['public_history'] if r.get('evidence')=='forecast']
    witnesses = []
    for frame in frames:
        if frame['receipt_seconds']>now:
            continue
        for source_cid,kind,status,_ in frame['endpoint_updates']:
            cid = mapping.get(source_cid)
            if cid not in menu_cids or status!='FORECAST':
                continue
            c = graph.connections[cid]
            if c.trip_id==first_trip:
                continue
            planned = c.dep_time if kind=='departure' else c.arr_time
            if planned>now:
                witnesses.append(dict(cid=cid,kind=kind,receipt=frame['receipt_seconds'],
                                      decision_clock=now,scheduled_target=planned))
    return dict(initial_state=dict(status='observed',cid=first['cid'],clock=now),
                predecision_numeric_messages=len(active),
                model_selected_numeric_messages=sum(r.get('emission_selected',False) for r in active),
                other_trip_numeric_messages=sum(r['trip']!=first_trip for r in active),
                historical_future_tail_publications=len(witnesses),
                historical_forecast_witnesses=witnesses)


def missing_source(index,policy,menu):
    root = menu['root']
    return dict(root_index=index,root_id=root['root_id'],day=root['day'],
        origin=root['origin_name'],goal=root['goal_name'],label='new_'+policy,policy=policy,
        emission_model='repaired_fit',emission_source='repaired_fit.json',
        loss_cap=root['deadline']-root['start'],executed=False,
        outcome=dict(loss=None,status='unresolved:predecision_message_source_unavailable',trace=[]),
        actions=[],states=[],gates=[],publications=[],decision_costs=[],control_stats={},planner_stats={},
        observation=menu['new_observation_contract'],relevant_remote_frames=None,
        cost=dict(process_cpu=None,wall=None,shared_menu=menu['cost'],
                  process_cpu_including_menu=None,wall_including_menu=None),
        coverage=dict(initial_state={'status':'message_source_unavailable'},source_available=False))


def run_root(index):
    resident = RESIDENT
    directory = resident['directory']
    menu = read(directory/'menus'/f'{index:03}.json.gz')
    root,day = menu['root'],menu['root']['day']
    graph_cost = None
    if day not in resident['graphs']:
        tick,cpu = perf_counter(),process_time()
        resident['graphs'][day] = context(resident['data'],day)
        graph_cost = dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
    graph,_,_,mapping,_,_ = resident['graphs'][day]
    completed = []
    for policy in ordered(POLICIES,index):
        path = directory/'roots'/f'{index:03}_new_{policy}.json.gz'
        if path.exists():
            continue
        if resident['source_available'][day]:
            row = one_arm(index,'repaired_fit',policy,menu,resident['graphs'][day],resident)
            row['executed'] = True
            # These are descriptive checks of saved public states, outside query billing.
            menu_cids = {cid for path in restore_node(menu['expanded_node']).paths() for cid in path}
            relevant_trips = {graph.connections[cid].trip_id for cid in menu_cids}
            frames = [f for f in resident['messages']['dates'][day]['frames'] if f['trip_id'] in relevant_trips]
            row['coverage'] = dict(initial_evidence(row,graph,mapping,frames,root['first_trip'],menu_cids),
                source_available=True,full_menu=resident['coverage'][index])
        else:
            row = missing_source(index,policy,menu)
            row['coverage']['full_menu'] = resident['coverage'][index]
            row['coverage']['source_failure_reasons'] = resident['source_reasons'][day]
        row['menu_source'] = f'menus/{index:03}.json.gz'
        row['menu_cost_source'] = 'Fresh public wide-menu plus prefix closure construction on this date'
        save(path,row)
        completed.append(dict(policy=policy,status=row['outcome']['status'],loss=row['outcome']['loss'],
            cpu=row['cost']['process_cpu_including_menu']))
        print(index,policy,row['outcome']['status'],row['outcome']['loss'],
              row['cost']['process_cpu_including_menu'],flush=True)
    write(directory/'preparation_roots'/f'{index:03}.json',dict(root_index=index,day=day,
        archive_load_cost=resident['load_cost'],graph_preparation_if_cold=graph_cost,
        declared_order=list(ordered(POLICIES,index)),completed=completed))
    return index


def run(directory,workers=4):
    directory = Path(directory)
    contract = check_inputs(directory)
    pending = [i for i in range(contract['roots']) if any(
        not (directory/'roots'/f'{i:03}_new_{p}.json.gz').exists() for p in POLICIES)]
    tick = perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(directory,)) as pool:
        for future in as_completed([pool.submit(run_root,i) for i in pending]):
            print('completed root',future.result(),flush=True)
    write(directory/'replay_preparation.json',dict(workers=workers,wall=perf_counter()-tick,
        requested_roots=contract['roots'],executed_roots=pending,original_72_additional_budget=0,
        scope='Shared resident/archive and graph preparation outside per-query cost; fresh shared menu cost included'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=HERE)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    run(args.directory,args.workers)
