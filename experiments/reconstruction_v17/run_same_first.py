"""Decompose V16's realized gap on the original frozen evaluation tapes."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v14.run_bank_traces import oracle_path
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map


V16 = HERE.parent/'reconstruction_v16'


def summarize(rows):
    out = {}
    for name, subset in [('all', rows), ('changed_first', [r for r in rows if r['extra']]),
                         ('unchanged_first', [r for r in rows if not r['extra']])]:
        if not subset:
            continue
        out[name] = dict(n=len(subset), components={
            key: dict(mean=mean(r[key] for r in subset), total=sum(r[key] for r in subset),
                      positive=sum(r[key]>0 for r in subset))
            for key in ('tail','first_ranking','outside_menu')},
            rank_by_clock=dict(Counter(r['earliest_best_vs_chosen'] for r in subset
                                       if r['first_ranking']>0)))
        outside = [r for r in subset if r['outside_menu']>0]
        out[name]['outside_cids'] = dict(Counter(r['outside_witness']['cid'] for r in outside))
        out[name]['outside_exclusions'] = dict(Counter(r['outside_witness']['exclusion'] for r in outside))
    return out


def evaluate(case, receipts, cpu):
    os.sched_setaffinity(0,{cpu})
    started = perf_counter()
    graph, belief, initial, goal, deadline, _, parameters = prepare(case)
    ready = max(initial.time,initial.last_arrival+graph.stops[initial.stop].min_transfer_time)
    panels = {}
    with TemporaryDirectory(prefix='bapr_v17_same_first_') as tmp:
        scan = TemporalScan(graph,build_temporal_kernel(Path(tmp)/'scan.so'))
        transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
        templates = template_trip_map(belief,parameters)
        for stage, receipt in receipts.items():
            clock_tape = sample_clocks(belief,parameters,rng(receipt['seed']),len(receipt['rows']),
                                       templates=templates)
            dep, arr, canceled = clock_tape
            full = scan.platform_arrivals(*clock_tape,initial,goal,deadline,transfers)-initial.time
            assert list(full)==[r['hindsight'] for r in receipt['rows']]
            menu = tuple(sorted(int(cid) for cid in receipt['paths']))
            assert all(graph.connections[cid].dep_stop==initial.stop for cid in menu)
            cap = deadline-initial.time
            rows = []
            for i, archived in enumerate(receipt['rows']):
                policy = archived['policies']['120']
                chosen = policy['first']
                losses = {}

                def forced(cid):
                    if cid not in losses:
                        c = graph.connections[cid]
                        if (canceled[i,cid] or dep[i,cid]<ready or arr[i,cid]>=deadline
                                or not c.pickup_allowed or cid in initial.used_connections):
                            losses[cid] = cap
                        else:
                            at = int(arr[i,cid])
                            state = replace(initial,stop=c.arr_stop,time=at,current_trip=c.trip_id,
                                last_arrival=at,last_connection=cid,local_since=at,
                                used_connections=initial.used_connections | {cid})
                            losses[cid] = int(scan.platform_arrivals(
                                *(x[i:i+1] for x in clock_tape),state,goal,deadline,transfers)[0]-initial.time)
                    return losses[cid]

                same = cap if chosen is None else forced(chosen)
                oracle = int(full[i])
                loss = policy['loss']
                assert oracle <= same <= loss, (case,stage,i,oracle,same,loss)
                assert chosen is None or chosen in menu
                best, best_ids, relation = same, [], None
                if same > oracle:
                    best = min(forced(cid) for cid in menu)
                    best_ids = [cid for cid in menu if forced(cid)==best]
                    assert oracle <= best <= same
                    if best < same:
                        if chosen is None:
                            relation = 'no_board'
                        else:
                            first = min(int(dep[i,cid]) for cid in best_ids)
                            relation = ('earlier' if first<dep[i,chosen] else 'same_tick'
                                        if first==dep[i,chosen] else 'later')
                witness = None
                if best>oracle:
                    scene = scene_from_clocks(graph,tuple(x[i] for x in clock_tape))
                    arrival, path = oracle_path(graph,scene,initial,goal,deadline)
                    assert arrival-initial.time==oracle and path and path[0] not in menu
                    cid = path[0]
                    c = graph.connections[cid]
                    reason = ('scheduled_before_ready' if c.dep_time<ready else
                              'scheduled_after_window' if c.dep_time>ready+720 else 'no_screened_path')
                    witness = dict(cid=cid,trip=c.trip_id,route=c.route,
                        departure=int(dep[i,cid]),scheduled_departure=c.dep_time,exclusion=reason,
                        path=path)
                rows.append(dict(world=i,loss=loss,hindsight=oracle,same_first=same,menu_hindsight=best,
                    first=chosen,extra=bool(policy['expanded']),compared=policy['comparisons'],
                    first_departure=None if chosen is None else int(dep[i,chosen]),
                    first_arrival=None if chosen is None else int(arr[i,chosen]),
                    tail=loss-same,first_ranking=same-best,outside_menu=best-oracle,
                    best_menu_firsts=best_ids,earliest_best_vs_chosen=relation,outside_witness=witness))
            panels[stage] = dict(seed=receipt['seed'],menu=menu,summary=summarize(rows),rows=rows)
    return dict(case=case,panels=panels,total_seconds=perf_counter()-started)


def main():
    stage_data = {stage:json.loads((V16/f'{stage}_policy.json').read_text())
                  for stage in ('held_out','precision')}
    assert all(d['status']=='complete' and d['margins']==[120] for d in stage_data.values())
    output = HERE/'same_first_results.json'
    cpus = sorted(os.sched_getaffinity(0),reverse=True)
    result = dict(status='running',scope='Pathwise point decomposition on V16 original matched tapes. '
                  'All oracle arms use full clairvoyant continuation and the same physical root readiness. '
                  'This is not a decomposition of population causal regret.',roots=[])
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(CASES))) as pool:
        tasks = [pool.submit(evaluate,case,{stage:r for stage,panel in stage_data.items()
                  for r in panel['roots'] if tuple(r['case'])==case},cpus[i%len(cpus)])
                 for i,case in enumerate(CASES)]
        for task in as_completed(tasks):
            root = task.result()
            result['roots'].append(root)
            result['roots'].sort(key=lambda r:CASES.index(tuple(r['case'])))
            output.write_text(json.dumps(result,indent=2)+'\n')
            print(root['case'],{k:v['summary']['all'] for k,v in root['panels'].items()},flush=True)
    result['status'] = 'complete'
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
