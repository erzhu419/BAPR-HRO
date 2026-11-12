"""Freeze a departure-support candidate screen and compare to V16 on fresh worlds."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from statistics import mean, stdev
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v13.platform_run import build_model,replay_arm
from experiments.reconstruction_v15.run_attribution_banks import prepare,rng,scene_from_clocks
from experiments.reconstruction_v16.run_policy_pair import receipt
from src.conditional_rescue_policy import ConditionalItineraries,ConditionalRescuePolicy
from src.public_rescue_candidates import screen_public_itineraries
from src.support_rescue_candidates import screen_support_itineraries
from src.tail_rescue import TailRescuePlan
from src.temporal_scan import TemporalScan,build_temporal_kernel,sample_clocks,template_trip_map


WORLDS = 128
MARGIN = 120
MIN_MASS = .01
MAX_NEW = 12
ARMS = ('v16','support')


def compile_policy(graph, belief, parameters, root, goal, deadline, plan, arm):
    before,cpu = perf_counter(),process_time()
    paths = {path[0]:TailRescuePlan(path,first_action_now=True,departures=plan.departures)
             for path in (plan.primary,plan.rescue)}
    extras = screen_public_itineraries(graph,root,goal,deadline,paths,720)
    additional, audit = {}, {}
    if arm == 'support':
        additional,audit = screen_support_itineraries(graph,belief,parameters,root,goal,deadline,
            set(paths)|set(extras),window=720,min_mass=MIN_MASS,max_new=MAX_NEW)
        assert not (set(additional)&set(extras))
    extras.update(additional)
    paths.update((cid,TailRescuePlan(entry.path,first_action_now=True)) for cid,entry in extras.items())
    scores = ConditionalItineraries(belief,parameters,graph,root,goal,deadline,paths)
    policy = ConditionalRescuePolicy(graph,plan,root,deadline,scores,extras,MARGIN)
    setup = dict(controller=policy,construction_seconds=perf_counter()-before,
                 construction_cpu_seconds=process_time()-cpu)
    return setup,dict(paths={cid:entry.primary for cid,entry in paths.items()},
        additional={cid:entry.path for cid,entry in additional.items()},
        available_additions=sorted(set(additional)&set(scores.rows)),
        screened_without_score=scores.excluded,audit=audit)


def run_root(case,cpu):
    os.sched_setaffinity(0,{cpu})
    before=perf_counter()
    graph,belief,root,goal,deadline,plan,pars=prepare(case)
    shared_context=perf_counter()-before
    compiled, designs = {}, {}
    for arm in ARMS:
        compiled[arm],designs[arm]=compile_policy(graph,belief,pars,root,goal,deadline,plan,arm)
    ready=max(root.time,root.last_arrival+graph.stops[root.stop].min_transfer_time)
    day,query,decision=case
    seed=[20261117,int(day.replace('-','')),query,decision,2]
    with TemporaryDirectory(prefix='bapr_v17_support_') as temp:
        scan=TemporalScan(graph,build_temporal_kernel(Path(temp)/'scan.so'))
        clocks=sample_clocks(belief,pars,rng(seed),WORLDS,templates=template_trip_map(belief,pars))
        transfer=[graph.stops[stop].min_transfer_time for stop in scan.stops]
        h=scan.platform_arrivals(*clocks,root,goal,deadline,transfer)-root.time
        rows=[]
        for i in range(WORLDS):
            scene=scene_from_clocks(graph,tuple(x[i] for x in clocks))
            model=build_model(graph,scene,root,goal,deadline)
            order=ARMS if i%2==0 else tuple(reversed(ARMS))
            paired={arm:receipt(replay_arm(graph,plan,root,goal,deadline,model,model.scenarios[0],arm,
                                          compiled[arm])) for arm in order}
            assert all(x['loss']>=h[i] for x in paired.values())
            rows.append(dict(world=i,hindsight=int(h[i]),policies=paired))
    delta=[r['policies']['v16']['loss']-r['policies']['support']['loss'] for r in rows]
    chosen=[r for r in rows if r['policies']['support']['first'] in designs['support']['available_additions']]
    summary=dict(mean_gain=mean(delta),paired_standard_error=stdev(delta)/WORLDS**.5,
        improved=sum(x>0 for x in delta),harmed=sum(x<0 for x in delta),
        new_candidate_taken=len(chosen),new_candidate_gain_total=sum(delta[r['world']] for r in chosen),
        mean_gain_if_no_new_taken=sum(g for r,g in zip(rows,delta,strict=True)
                                     if r['policies']['support']['first'] not in designs['support']['available_additions'])
                                    /(len(rows)-len(chosen)) if len(chosen)<len(rows) else None)
    for arm in ARMS:
        vals=[r['policies'][arm] for r in rows]
        summary[arm]=dict(mean_loss=mean(r['loss'] for r in vals),arrived=sum(r['reached'] for r in vals),
            mean_policy_minus_hindsight=mean(r['loss']-x['hindsight'] for r,x in zip(vals,rows,strict=True)),
            mean_wall=mean(r['rule_wall'] for r in vals),mean_cpu=mean(r['rule_cpu'] for r in vals),
            latest_release=max(r['publication'] for r in vals),rejects=sum(r['rejected'] for r in vals))
    return dict(case=case,seed=seed,ready=ready,root_clock=root.time,cap=deadline-root.time,
                shared_context=shared_context,designs=designs,summary=summary,rows=rows)


def main():
    output=HERE/'support_pair_results.json'
    cpus=sorted(os.sched_getaffinity(0))
    data=dict(status='running',margin=MARGIN,min_departure_probability=MIN_MASS,
              max_additions=MAX_NEW,worlds_per_root=WORLDS,roots=[])
    with ProcessPoolExecutor(max_workers=min(len(CASES),len(cpus))) as pool:
        futures=[pool.submit(run_root,case,cpus[i%len(cpus)]) for i,case in enumerate(CASES)]
        for task in as_completed(futures):
            row=task.result()
            data['roots'].append(row)
            data['roots'].sort(key=lambda x:CASES.index(tuple(x['case'])))
            output.write_text(json.dumps(data,indent=2)+'\n')
            print(row['case'],row['summary'],flush=True)
    data['status']='complete'
    output.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    main()
