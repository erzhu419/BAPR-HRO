"""Pilot and independent matched simulation of conditional public rescue."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
from statistics import mean, stdev
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v13.platform_run import build_model, compile_arms, replay_arm
from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v15.run_attribution_banks import prepare, scene_from_clocks, rng
from src.conditional_rescue_policy import ConditionalItineraries, ConditionalRescuePolicy
from src.public_rescue_candidates import screen_public_itineraries
from src.tail_rescue import TailRescuePlan
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map

BASE = 'first_visible'
MARGINS = (0, 60, 120)
N = {'pilot': 16, 'held_out': 128, 'precision': 512, 'point_diagnostic': 128}
STREAM = {'pilot': 0, 'held_out': 1, 'precision': 2, 'point_diagnostic': 1}


def select(pilot):
    assert pilot['status'] == 'complete'
    return min(MARGINS, key=lambda m: (sum(root['summary'][str(m)]['mean_loss']
                                           for root in pilot['roots']), -m))


def receipt(row):
    return dict(loss=row['loss_upper'], reached=row['status']=='arrived', published=row['published'],
                rejected=row['rejected'], publication=row['publish_at'], first=row['rides'][0]['ridden_connection']
                    if row['rides'] else None, worker_wall=row['worker_seconds'],
                rule_wall=row['total_rule_pipeline_seconds'], rule_cpu=row['construction_cpu_seconds']
                    +row['binding_cpu_seconds']+row['rule_work'].get('action_cpu_seconds',0.)
                    +row['rule_work'].get('applicable_cpu_seconds',0.),
                comparisons=row['triggers'].get('conditional_comparisons',0),
                expanded=row['triggers'].get('conditional_boardings',0),
                decisions=[x for x in row['trigger_trace'] if x['branch']=='conditional_rescue'])


def summary(rows, names):
    result = {}
    baseline = [r['policies'][BASE]['loss'] for r in rows]
    for name in names:
        records = [r['policies'][name] for r in rows]
        delta = [a-r['loss'] for a,r in zip(baseline, records, strict=True)]
        result[name] = dict(mean_loss=mean(r['loss'] for r in records),
            mean_gain=mean(delta), paired_standard_error=stdev(delta)/len(delta)**.5,
            arrivals=sum(r['reached'] for r in records), improvements=sum(d>0 for d in delta),
            harms=sum(d<0 for d in delta), extra_boardings=sum(r['expanded'] for r in records),
            comparisons=sum(r['comparisons'] for r in records),
            rejected=sum(r['rejected'] for r in records), mean_rule_wall=mean(r['rule_wall'] for r in records),
            mean_rule_cpu=mean(r['rule_cpu'] for r in records))
    return result


def run_root(case, cpu, phase, margins):
    os.sched_setaffinity(0, {cpu})
    before = perf_counter()
    graph, belief, root, goal, deadline, plan, pars = prepare(case)
    assert plan.failure_index == 0 and plan.primary and plan.rescue
    shared_setup = perf_counter()-before
    before, cpu_start = perf_counter(), process_time()
    paths = {path[0]: TailRescuePlan(path, first_action_now=True, departures=plan.departures)
             for path in (plan.primary,plan.rescue)}
    extras = screen_public_itineraries(graph, root, goal, deadline, paths, 720)
    paths.update((cid,TailRescuePlan(route.path, first_action_now=True)) for cid,route in extras.items())
    scores = ConditionalItineraries(belief, pars, graph, root, goal, deadline, paths,
                                   point_forecast=phase=='point_diagnostic')
    assert all(cid in scores.rows for cid in (plan.primary[0], plan.rescue[0]))
    prepare_wall, prepare_cpu = perf_counter()-before, process_time()-cpu_start
    names = [BASE, *(str(m) for m in margins)]
    compiled = {BASE:compile_arms(graph, plan, root, deadline)[BASE]}
    for margin in margins:
        before, cpu_start = perf_counter(), process_time()
        controller = ConditionalRescuePolicy(graph, plan, root, deadline, scores, extras, margin)
        compiled[str(margin)] = dict(controller=controller,
            construction_seconds=prepare_wall+perf_counter()-before,
            construction_cpu_seconds=prepare_cpu+process_time()-cpu_start)
    day, query, decision = case
    seed = [20261116,int(day.replace('-','')),query,decision,STREAM[phase]]
    with TemporaryDirectory(prefix='bapr_v16_pair_') as temp:
        before = perf_counter()
        scan = TemporalScan(graph,build_temporal_kernel(Path(temp)/'scan.so'))
        compile_seconds = perf_counter()-before
        before = perf_counter()
        clocks = sample_clocks(belief, pars, rng(seed), N[phase], templates=template_trip_map(belief,pars))
        draw_seconds = perf_counter()-before
        before = perf_counter()
        transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
        h = scan.platform_arrivals(*clocks, root, goal, deadline, transfers)-root.time
        oracle_seconds = perf_counter()-before
        rows, model_seconds, replay_seconds = [], 0., 0.
        for i in range(N[phase]):
            before = perf_counter()
            scene = scene_from_clocks(graph, tuple(x[i] for x in clocks))
            model = build_model(graph, scene, root, goal, deadline)
            model_seconds += perf_counter()-before
            before = perf_counter()
            order = (names if i%2==0 else list(reversed(names)))
            policies = {name:receipt(replay_arm(graph, plan, root, goal, deadline, model,
                        model.scenarios[0], name, compiled[name])) for name in order}
            replay_seconds += perf_counter()-before
            assert all(h[i] <= r['loss'] for r in policies.values())
            rows.append(dict(world=i, hindsight=int(h[i]), policies=policies))
    return dict(case=case, cap=deadline-root.time, floor=scores.ready-root.time, seed=seed,
                shared_setup_seconds=shared_setup, kernel_compile_seconds=compile_seconds,
                sample_seconds=draw_seconds, oracle_seconds=oracle_seconds,
                model_seconds=model_seconds, replay_seconds=replay_seconds,
                candidate_prepare_wall=prepare_wall, candidate_prepare_cpu=prepare_cpu,
                excluded=scores.excluded, paths={cid:p.primary for cid,p in paths.items()},
                conditionals={cid:[asdict(r) for r in rows.values()] for cid,rows in scores.rows.items()},
                rows=rows, summary=summary(rows,names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=N)
    phase = parser.parse_args().phase
    output = HERE/f'{phase}_policy.json'
    margins = MARGINS if phase=='pilot' else (select(json.loads((HERE/'pilot_policy.json').read_text())),)
    cases = (CASES[0],CASES[2]) if phase=='precision' else CASES
    cpus = sorted(os.sched_getaffinity(0),reverse=True)
    results = dict(status='running',phase=phase,margins=margins,roots=[])
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(cases))) as pool:
        pending = [pool.submit(run_root,case,cpus[i%len(cpus)],phase,margins) for i,case in enumerate(cases)]
        for task in as_completed(pending):
            root = task.result()
            results['roots'].append(root)
            results['roots'].sort(key=lambda r: CASES.index(tuple(r['case'])))
            output.write_text(json.dumps(results,indent=2)+'\n')
            print(root['case'],root['summary'],flush=True)
    results['status'] = 'complete'
    output.write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
