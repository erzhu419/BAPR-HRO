"""Paired fixed-rule platform mechanisms at three observed transfer boundaries."""

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
from math import ceil
import os
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from experiments.reconstruction_v8.budget_run import SOURCE, write_json
from experiments.reconstruction_v8.issued_replay import unresolved_check
from experiments.reconstruction_v9.search_reference import ARCHIVE
from experiments.reconstruction_v10.platform_sensitivity import adapter
from experiments.reconstruction_v12.transfer_inventory import CASES, key
from src.background_policy import IssuedPolicy, execute_background
from src.interruptible_recourse import InterruptibleRecourse
from src.platform_transit import PlatformState, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.tail_rescue import TailRescuePlan
from src.temporal_scan import build_temporal_kernel, sample_clocks, template_trip_map


METHODS = ('primary','fixed_timeout','public_failure','first_visible')
WORLDS = 64
INVENTORY = ROOT / 'experiments/reconstruction_v12/transfer_inventory.json'
V10 = ROOT / 'experiments/reconstruction_v10/tail_rescue_results.json'


def load_plan(row, prefix=()):
    plan = TailRescuePlan(tuple(row['primary']),row['failure_index'],tuple(row['rescue']),
                         row['first_action_now'],tuple(map(tuple,row['departures'])))
    if not prefix:
        return plan
    assert plan.primary[:len(prefix)] == tuple(prefix)
    return replace(plan,primary=plan.primary[len(prefix):],failure_index=plan.failure_index-len(prefix),
                   first_action_now=False)


def prepare_waiting_context(graph, metadata, journey, decision, fit, prefix, receipts):
    """Only observed prefix endpoint receipts condition this waiting posterior."""
    belief,state = context(graph,metadata,journey,decision,fit)
    assert [r['ridden_connection'] for r in receipts[:len(prefix)]] == list(prefix)
    for receipt in receipts[:len(prefix)]:
        cid = receipt['ridden_connection']
        c = graph.connections[cid]
        belief.observe_connection_event(cid,'departure',receipt['time']-c.dep_time)
        belief.observe_connection_event(cid,'arrival',receipt['next_time']-c.arr_time)
        state = replace(state,stop=receipt['next_stop'],time=receipt['next_time'],
            last_arrival=receipt['next_time'],current_trip=c.trip_id,
            unavailable_ids=state.unavailable_ids | {cid},learned_trips=state.learned_trips | {c.trip_id})
    # Explicit immediate ALIGHT starts the station transfer at this same clock.
    initial = PlatformState(state.stop,state.time,None,state.time,last_connection=prefix[-1],
        used_connections=state.unavailable_ids,local_since=state.time)
    return belief,initial


def compile_arms(graph,plan,initial,deadline):
    """Compile once per context; fresh binding state is created for each world."""
    compiled = {}
    for method in METHODS:
        before,cpu = perf_counter(),process_time()
        controller = (None if method == 'primary' else
                      InterruptibleRecourse(graph,plan,initial,deadline,mode=method))
        compiled[method] = dict(controller=controller,construction_seconds=perf_counter()-before,
                                construction_cpu_seconds=process_time()-cpu)
    return compiled


def measured_policy(policy,ledger):
    def measure(function,kind,state):
        before,cpu = perf_counter(),process_time()
        try:
            return function(state)
        finally:
            ledger[kind+'_calls'] += 1
            ledger[kind+'_wall_seconds'] += perf_counter()-before
            ledger[kind+'_cpu_seconds'] += process_time()-cpu
    return IssuedPolicy(lambda s:measure(policy.action,'action',s),
                        lambda s:measure(policy.applicable,'applicable',s),policy.name)


def replay_arm(graph,plan,initial,goal,deadline,model,world,method,compiled,*,actual=None,metadata=None):
    """Same model/world for all arms; actual source gaps retain V8 censoring."""
    before = perf_counter()
    incumbent,incumbent_targets,_ = adapter(graph,plan.primary,initial,'incumbent')
    incumbent_prepare_seconds = perf_counter()-before
    before,cpu = perf_counter(),process_time()
    if method == 'primary':
        policy,targets,_ = adapter(graph,plan.primary,initial,method)
        binding = SimpleNamespace(policy=policy,targets=targets,trace=[],stats={})
    else:
        binding = compiled['controller'].bind(method)
    binding_seconds,binding_cpu_seconds = perf_counter()-before,process_time()-cpu
    worker_seconds = compiled['construction_seconds']+binding_seconds
    publish_at = initial.time+ceil(worker_seconds)
    incumbent_work,candidate_work = Counter(),Counter()
    incumbent = measured_policy(incumbent,incumbent_work)
    candidate = measured_policy(binding.policy,candidate_work)
    before = perf_counter()
    unresolved = (None if actual is None else unresolved_check(graph,metadata,actual,
        {'incumbent':incumbent_targets,method:binding.targets}))
    source_audit_setup_seconds = perf_counter()-before
    before = perf_counter()
    result = execute_background(model,world,incumbent,candidate,publish_at,unresolved=unresolved)
    replay_seconds = perf_counter()-before
    final = result['state']
    censored = result['cost'] is None
    lo = final.time-initial.time if censored else result['cost']
    hi = deadline-initial.time if censored else lo
    actions = dict(incumbent_work+candidate_work)
    triggers = dict(binding.stats)
    branches = [r['branch'] for r in binding.trace]
    triggers['fallback_entries'] = sum(branch=='rescue' and (i==0 or branches[i-1]!='rescue')
                                       for i,branch in enumerate(branches))
    return dict(method=method,status='censored' if censored else 'arrived' if final.stop==goal else 'timeout',
        loss_lower=lo,loss_upper=hi,published=result['published'],rejected=result['rejected'],
        publish_at=publish_at,first_candidate_action_clock=next((t['time'] for t in result['trace']
            if t['policy']==method),None),incumbent_moves=result['incumbent_actions_during_compute'],
        missing_connections=result['missing_connections'],construction_seconds=compiled['construction_seconds'],
        construction_cpu_seconds=compiled['construction_cpu_seconds'],binding_seconds=binding_seconds,
        binding_cpu_seconds=binding_cpu_seconds,worker_seconds=worker_seconds,
        incumbent_prepare_seconds=incumbent_prepare_seconds,rule_work=actions,
        total_rule_pipeline_seconds=incumbent_prepare_seconds+worker_seconds+
            actions.get('action_wall_seconds',0.)+actions.get('applicable_wall_seconds',0.),
        posterior_queries=0,search_calls=0,lost_search_calls=0,replay_seconds=replay_seconds,
        offline_source_audit_setup_seconds=source_audit_setup_seconds,
        triggers=triggers,trigger_trace=list(binding.trace),
        rides=[t for t in result['trace'] if t['ridden_connection'] is not None])


def build_model(graph,scene,initial,goal,deadline):
    return build_platform_model(graph,[(1.,scene)],initial.stop,goal,initial.time,deadline,
                                initial_state=initial,poll_interval=1)


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['panel'],row['source'],row['date'],row['query'],row['decision']),[]).append(row)
    summary = []
    for group,records in groups.items():
        indexed = {(r['world'],r['method']):r for r in records}
        for method in METHODS:
            arm = [r for r in records if r['method']==method]
            comparisons = {}
            for reference in ('primary','fixed_timeout','public_failure'):
                differences = [r['loss_upper']-indexed[r['world'],reference]['loss_lower'] for r in arm]
                comparisons[reference] = dict(mean_upper_difference=mean(differences),
                    mean_lower_difference=mean(r['loss_lower']-indexed[r['world'],reference]['loss_upper'] for r in arm),
                    improved=sum(d<0 for d in differences),harmed=sum(r['loss_lower']>
                        indexed[r['world'],reference]['loss_upper'] for r in arm),
                    equal=sum(r['loss_lower']==r['loss_upper']==indexed[r['world'],reference]['loss_lower']==
                              indexed[r['world'],reference]['loss_upper'] for r in arm),
                    min_difference=min(differences),max_difference=max(differences))
            triggers = Counter()
            for row in arm:
                triggers.update(row['triggers'])
            summary.append(dict(panel=group[0],source=group[1],date=group[2],query=group[3],decision=group[4],
                method=method,n=len(arm),mean_lower=mean(r['loss_lower'] for r in arm),
                mean_upper=mean(r['loss_upper'] for r in arm),arrived=sum(r['status']=='arrived' for r in arm),
                censored=sum(r['status']=='censored' for r in arm),paired=comparisons,triggers=dict(triggers),
                mean_pipeline_seconds=mean(r['total_rule_pipeline_seconds'] for r in arm),
                mean_action_calls=mean(r['rule_work'].get('action_calls',0) for r in arm)))
    return summary


def main():
    cpu = max(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{cpu})
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),cases=[list(k) for k in CASES],
        methods=METHODS,worlds=WORLDS,cpu=cpu,seed_rule='[20261113,int(date),query,decision]',
        primary_panel='Three previously observed prefix-arrival states after immediate ALIGHT. Condition posterior only on public prefix endpoint receipts. Actual plus64 full posterior worlds paired across four policies.',
        bridge_panel='Three original onboard roots, actual replay only, same frozen full V10 plans.',
        controls='Original primary-only adapter; full V10 fixed-timeout interpreter; early public-failure rescue; first-visible rescue. No new path, posterior value query or search at events.',
        early_failure_window='Envelope of the early rescue window and counterfactual fixed-timeout rescue window: earliest opening and latest expiry, with physical station-MCT boarding still required. Early failure does not discard a later rescue that the original window could catch.',
        publication='Compile once per context, bind fresh per world; release at root+ceil(measured compilation+binding wall). Common original-primary incumbent acts during compute. Applicability sees only public state.',
        costs='Accumulate rule action/applicability wall and CPU separately; preissued rules use the existing ideal event-response contract without rounding every microsecond call into one simulated second. Offline sample/model construction and historical V10 search costs are separate.',
        censor='Actual replay retains the V8 issued-target and next same-service source audit. Complete posterior worlds use the modeled trip identity and contain every connection; actual metadata is not substituted for sampled clocks.',
        order='Four Williams orders (0,1,3,2) shifted modulo4 cycle across worlds, offset by root; all64 worlds included once, each arm occupies each position16times in each posterior panel.',
        scope='Waiting-state mechanism diagnostic and actual-prefix bridge; no whole-journey G2 or competitive-search claim.',
        inputs=[str(p.relative_to(ROOT)) for p in (SOURCE,ARCHIVE,INVENTORY,V10)])
    write_json(HERE/'platform_protocol.json',protocol)
    before = perf_counter()
    data,_ = load_city(SOURCE)
    fit = fit_training(data,retain_no_pickup=True)
    result = dict(protocol=protocol,shared_fit_seconds=perf_counter()-before,contexts=[],rows=[],status='running')
    journeys = {(r['date'],r['query']):r for r in json.loads(ARCHIVE.read_text())['rows'] if r['method']=='evolving_rollout'}
    inventory = {key(r):r for r in json.loads(INVENTORY.read_text())['rows']}
    archived = {key(r):r for r in json.loads(V10.read_text())['rows']}
    orders = tuple(tuple(METHODS[(i+j)%4] for j in (0,1,3,2)) for i in range(4))
    with TemporaryDirectory(prefix='bapr_v13_platform_') as tmp:
        before = perf_counter()
        build_temporal_kernel(Path(tmp)/'scan.so')
        result['shared_native_compile_seconds'] = perf_counter()-before
        for root_index,identity in enumerate(CASES):
            day,query,decision = identity
            before = perf_counter()
            graph,actual,stops,_,metadata,_ = dated_graph(data,day,'gtfs_station_max',True)
            journey,previous = journeys[day,query],inventory[identity]
            prefix = tuple(previous['next_transfer_boundary']['prefix'])
            belief,initial = prepare_waiting_context(graph,metadata,journey,decision,fit,prefix,
                previous['recorded_incumbent']['rides'])
            goal,deadline = previous['root']['destination'],journey['od']['deadline_seconds']
            plan = load_plan(archived[identity]['fixed_primary_tail_rescue']['plan'],prefix)
            compiled = compile_arms(graph,plan,initial,deadline)
            setup = dict(date=day,query=query,decision=decision,waiting_stop=initial.stop,
                waiting_clock=initial.time,setup_seconds=perf_counter()-before,
                historical_v10_plan_generation_seconds=archived[identity]['fixed_primary_tail_rescue']['wall_seconds'])
            parameters = {c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
            before = perf_counter()
            packed = sample_clocks(belief,parameters,np.random.default_rng(np.random.SeedSequence(
                [20261113,int(day.replace('-','')),query,decision])),WORLDS,
                templates=template_trip_map(belief,parameters))
            setup['offline_sample_seconds'] = perf_counter()-before
            result['contexts'].append(setup)
            panels = [('waiting','actual',-1,actual,initial,plan,compiled)]
            _,original_state = context(graph,metadata,journey,decision,fit)
            onboard = PlatformState(original_state.stop,original_state.time,original_state.current_trip,
                original_state.last_arrival,last_connection=journey['decisions'][decision-1]['action'],
                used_connections=original_state.unavailable_ids,local_since=original_state.time)
            original_plan = load_plan(archived[identity]['fixed_primary_tail_rescue']['plan'])
            bridge_compiled = compile_arms(graph,original_plan,onboard,deadline)
            panels.append(('onboard_bridge','actual',-1,actual,onboard,original_plan,bridge_compiled))
            for world_index in range(WORLDS+2):
                before = perf_counter()
                if world_index < 2:
                    panel,source,number,scene,root,current_plan,controllers = panels[world_index]
                else:
                    number = world_index-2
                    dep,arr,canceled = (x[number] for x in packed)
                    scene = JourneyScenario({c.id:VehicleEvent(int(dep[c.id])-c.dep_time,bool(canceled[c.id]),
                        int(dep[c.id]),int(arr[c.id])) for c in graph.connections},minute=60,publications={})
                    panel,source,root,current_plan,controllers = 'waiting','posterior',initial,plan,compiled
                model = build_model(graph,scene,root,goal,deadline)
                model_seconds = perf_counter()-before
                order = orders[(world_index+root_index*7)%len(orders)]
                for method in order:
                    try:
                        row = replay_arm(graph,current_plan,root,goal,deadline,model,model.scenarios[0],method,
                            controllers[method],actual=actual if source=='actual' else None,
                            metadata=metadata if source=='actual' else None)
                    except Exception as error:
                        result.update(status='failed',failure=dict(date=day,query=query,decision=decision,
                            panel=panel,source=source,world=number,method=method,error=repr(error)))
                        write_json(HERE/'platform_results.json',result)
                        raise
                    row.update(date=day,query=query,decision=decision,panel=panel,source=source,world=number,
                               root_clock=root.time,order=list(order),model_seconds=model_seconds)
                    if source=='posterior':
                        row['rides'] = [dict(cid=r['ridden_connection'],departure=r['time'],
                                             arrival=r['next_time']) for r in row['rides']]
                    result['rows'].append(row)
                if world_index%8==1:
                    write_json(HERE/'platform_results.json',result)
                    print(identity,'completed panel/world',panel,source,number,flush=True)
            write_json(HERE/'platform_results.json',result)
    result.update(status='complete',summary=summarize(result['rows']))
    write_json(HERE/'platform_results.json',result)
    print('complete',len(result['rows']),'paired policy replays',flush=True)


if __name__=='__main__':
    main()
