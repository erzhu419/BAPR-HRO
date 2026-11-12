"""Paired same-platform wall-time curves for public-event online rollout."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v13.platform_run import build_model, replay_arm
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from experiments.reconstruction_v19.run_policy_pair import RELEASE_OFFSET, compile_arm, root_receipt
from experiments.reconstruction_v20.run_policy_pair import audit_menu, compile_direct
from experiments.reconstruction_v21.menus import candidate_paths
from src.platform_public_particles import PlatformPublicParticles
from src.platform_rollout_search import PlatformRolloutSearch
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map


HERE = Path(__file__).resolve().parent
BASE, DIRECT, SCALAR = 'first_visible', 'direct_zero', 'direct_per_clock'
WORLDS = 256
BUDGETS = (0, 8, 32, 128, 512, 2048)
MATCHED = (0, 32, 512, 2048)
ARMS = (BASE, DIRECT, SCALAR, *(f'rollout_{n}' for n in BUDGETS),
        *(f'direct_rollout_{n}' for n in MATCHED))


def compile_rollout(arm, graph, belief, parameters, root, goal, deadline, plan, seed):
    started, cpu = perf_counter(), process_time()
    budget = int(arm.rsplit('_', 1)[1])
    paths, audit = candidate_paths(graph, belief, parameters, root, goal, deadline, plan,
                                   direct=arm.startswith('direct_'))
    screening_wall = perf_counter()-started
    particles = PlatformPublicParticles(belief, parameters) if budget else None
    planner = PlatformRolloutSearch(graph, belief, parameters, root, goal, deadline, paths, particles)
    controller = planner.for_budget(budget, seed)
    return dict(controller=controller, construction_seconds=perf_counter()-started,
                construction_cpu_seconds=process_time()-cpu, screening_wall=screening_wall,
                menu=audit, paths=paths)


def setup_arm(arm, graph, belief, root, goal, deadline, plan, parameters, seed):
    if arm == BASE:
        return compile_arm(arm, graph, belief, parameters, root, goal, deadline, plan)
    if arm in (DIRECT, SCALAR):
        return compile_direct(graph, belief, parameters, root, goal, deadline, plan,
                              contract=arm == DIRECT)
    return compile_rollout(arm, graph, belief, parameters, root, goal, deadline, plan, seed)


def compact_receipt(replay):
    r = root_receipt(replay)
    return dict(loss=r['loss'], arrived=r['reached'], first=r['first'], wall=r['rule_wall'],
                policy_cpu=r['rule_cpu'], action_wall=replay['rule_work'].get('action_wall_seconds', 0.),
                publication=replay['publish_at'],
                search_work={k: v for k, v in replay['triggers'].items()
                             if k.startswith(('search_', 'rollout_', 'posterior_'))},
                last_arrival=replay['rides'][-1]['next_time'] if replay['rides'] else None)


def readout(rows, compiled):
    summary = {}
    for arm in compiled:
        records = [r['policies'][arm] for r in rows]
        deltas = [r['policies'][DIRECT]['loss']-v['loss'] for r, v in zip(rows, records, strict=True)]
        work = Counter()
        for r in records:
            work.update(r['search_work'])
        summary[arm] = dict(loss=mean(r['loss'] for r in records), arrived=sum(r['arrived'] for r in records),
                            mean_wall=mean(r['wall'] for r in records),
                            mean_policy_cpu=mean(r['policy_cpu'] for r in records),
                            mean_action_wall=mean(r['action_wall'] for r in records),
                            cold_compile_wall=compiled[arm]['construction_seconds'],
                            root_condition_wall=compiled[arm]['root_condition_wall'],
                            screening_wall=compiled[arm].get('screening_wall'),
                            scoring_wall=compiled[arm].get('scoring_wall'),
                            gain_vs_direct=mean(deltas), paired_se=stdev(deltas)/len(deltas)**.5,
                            improved_vs_direct=sum(g > 0 for g in deltas),
                            harmed_vs_direct=sum(g < 0 for g in deltas), work=dict(work),
                            first=Counter(r['first'] for r in records))
    return summary


def run_root(case, cpu, pilot):
    os.sched_setaffinity(0, {cpu})
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    day, query, decision = case
    date = int(day.replace('-', ''))
    evaluation_seed = [20261121, date, query, decision, int(pilot)]
    planning_seed = [2026112101, date, query, decision]
    arms = (BASE, DIRECT, SCALAR, 'rollout_0', 'rollout_128', 'rollout_2048', 'direct_rollout_512') if pilot else ARMS
    compiled = {}
    rollout_setup = {}
    for arm in arms:
        sampling = 'rollout_' in arm and not arm.endswith('_0')
        menu = arm.startswith('direct_')
        if sampling and menu in rollout_setup:
            cached = rollout_setup[menu]
            budget = int(arm.rsplit('_', 1)[1])
            started, cpu_started = perf_counter(), process_time()
            controller = cached['controller'].search.for_budget(budget, planning_seed)
            compiled[arm] = dict(cached, controller=controller,
                                 construction_seconds=cached['construction_seconds']+perf_counter()-started,
                                 construction_cpu_seconds=cached['construction_cpu_seconds']+process_time()-cpu_started)
            continue
        cost = {}
        g, b, s, dest, end, p, pars = prepare(case, conditioning_cost=cost)
        assert (s, dest, end, p) == (root, goal, deadline, plan)
        setup = setup_arm(arm, g, b, s, dest, end, p, pars, planning_seed)
        if arm == BASE:
            cost = dict(wall=0., cpu=0.)  # An already issued first-visible plan needs no posterior construction.
        setup['root_condition_wall'] = cost['wall']
        setup['construction_seconds'] += cost['wall']
        setup['construction_cpu_seconds'] += cost['cpu']
        compiled[arm] = setup
        if sampling:
            rollout_setup[menu] = setup
    a = compiled[SCALAR]['controller']
    b = compiled[DIRECT]['controller']
    parity = audit_menu(a, b, graph)
    direct_paths = {cid: p.primary for cid, p in b.scores.paths.items()}
    generic_paths = [arm['paths'] for name, arm in compiled.items() if name.startswith('rollout_')]
    assert all(paths == generic_paths[0] for paths in generic_paths)
    assert set(direct_paths) <= set(generic_paths[0])
    assert all(arm['paths'] == direct_paths for name, arm in compiled.items() if name.startswith('direct_rollout_'))
    ready = max(root.time, root.last_arrival+graph.stops[root.stop].min_transfer_time)
    release = root.time+RELEASE_OFFSET
    assert release <= ready and all(s['construction_seconds'] < RELEASE_OFFSET for s in compiled.values())
    rows = []
    n = 4 if pilot else WORLDS
    with TemporaryDirectory(prefix='bapr_v21_pair_') as tmp:
        scan = TemporalScan(graph, build_temporal_kernel(Path(tmp)/'scan.so'))
        clocks = sample_clocks(belief, parameters, rng(evaluation_seed), n,
                               templates=template_trip_map(belief, parameters))
        transfers = [graph.stops[s].min_transfer_time for s in scan.stops]
        hindsight = scan.platform_arrivals(*clocks, root, goal, deadline, transfers)-root.time
        for i, h in enumerate(hindsight):
            scene = scene_from_clocks(graph, tuple(x[i] for x in clocks))
            model = build_model(graph, scene, root, goal, deadline)
            shift = i % len(arms)
            results = {}
            for arm in arms[shift:]+arms[:shift]:
                replay = replay_arm(graph, plan, root, goal, deadline, model, model.scenarios[0],
                                    arm, compiled[arm], publish_not_before=release)
                assert replay['publish_at'] == release and replay['published'] and not replay['rejected']
                assert replay['loss_lower'] == replay['loss_upper'] >= h
                results[arm] = compact_receipt(replay)
            assert all(results[DIRECT][k] == results[SCALAR][k] for k in ('first', 'loss', 'arrived'))
            rows.append(dict(world=i, hindsight=int(h), policies=results))
    summary = readout(rows, compiled)
    if pilot:
        return dict(case=case, seed=evaluation_seed,
                    timing={k: {f: v[f] for f in ('mean_wall', 'mean_action_wall', 'cold_compile_wall', 'work')}
                            for k, v in summary.items()})
    return dict(case=case, seed=evaluation_seed, planning_seed=planning_seed, ready=ready, publication=release,
                menu=compiled['rollout_0']['menu'], nominal_paths=generic_paths[0], direct_paths=direct_paths,
                contracted_score_audit=parity, summary=summary, rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    pilot = parser.parse_args().pilot
    output = HERE/('pilot_timing.json' if pilot else 'paired_results.json')
    cpus = sorted(os.sched_getaffinity(0))
    data = dict(status='running', worlds_per_root=4 if pilot else WORLDS, roots=[])
    with ProcessPoolExecutor(max_workers=min(len(CASES), len(cpus))) as pool:
        jobs = [pool.submit(run_root, case, cpus[i % len(cpus)], pilot) for i, case in enumerate(CASES)]
        for job in as_completed(jobs):
            root = job.result()
            data['roots'].append(root)
            data['roots'].sort(key=lambda r: CASES.index(tuple(r['case'])))
            output.write_text(json.dumps(data, indent=2)+'\n')
            summary = root['timing' if pilot else 'summary']
            print(root['case'], {k: tuple(round(v, 5) for v in (r['mean_wall'], r['mean_action_wall']))
                                 for k, r in summary.items()}, flush=True)
    data['status'] = 'complete'
    output.write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()
