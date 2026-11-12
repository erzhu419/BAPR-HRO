"""Fresh fixed-menu ablation and measured costs for the V22 first opportunity."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter, process_time

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v13.platform_run import build_model, replay_arm
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from experiments.reconstruction_v19.run_policy_pair import RELEASE_OFFSET
from experiments.reconstruction_v20.run_policy_pair import compile_direct
from experiments.reconstruction_v21.menus import candidate_paths
from experiments.reconstruction_v21.run_policy_pair import compile_rollout
from src.first_opportunity_rescue import FirstOpportunityRescue, route_plan
from src.temporal_scan import sample_clocks, template_trip_map


HERE = Path(__file__).resolve().parent
WORLDS, COLD_REPEATS = 128, 3
MINIMAL, NOMINAL, EXACT, ROLLOUT = 'first_opportunity', 'direct_rollout_0', 'direct_zero', 'direct_rollout_2048'
ARMS = (MINIMAL, NOMINAL, EXACT, ROLLOUT)


def compile_minimal(graph, belief, parameters, root, goal, deadline, plan):
    wall, cpu = perf_counter(), process_time()
    paths, _ = candidate_paths(graph, belief, parameters, root, goal, deadline, plan, direct=True)
    screened = perf_counter()
    controller = FirstOpportunityRescue(graph, plan, root, goal, deadline, paths)
    return dict(controller=controller, paths=paths, screening_wall=screened-wall, scoring_wall=0.,
                construction_seconds=perf_counter()-wall, construction_cpu_seconds=process_time()-cpu)


def compile_case(case, arm, planning_seed):
    root_cost = {}
    graph, belief, root, goal, deadline, plan, parameters = prepare(case, conditioning_cost=root_cost)
    args = graph, belief, parameters, root, goal, deadline, plan
    if arm == MINIMAL:
        setup = compile_minimal(*args)
    elif arm == EXACT:
        setup = compile_direct(*args)
        controller = setup['controller']
        # This arm is the V20 comparator with its archived root-offer rule.
        for cid in plan.primary[:1] + plan.rescue[:1]:
            controller.suffixes.pop(cid, None)
        setup['paths'] = {cid: p.primary for cid, p in controller.scores.paths.items()}
    else:
        setup = compile_rollout(arm, *args, planning_seed)
        # Use the original V20 A/B continuation clocks during platform execution.
        started, cpu = perf_counter(), process_time()
        for cid in (plan.primary[0], plan.rescue[0]):
            suffix = setup['controller'].search.continuations[cid]
            suffix.plan = route_plan(cid, suffix.plan.primary, plan)
        setup['construction_seconds'] += perf_counter()-started
        setup['construction_cpu_seconds'] += process_time()-cpu
    setup['conditioning_wall'] = root_cost['wall']
    setup['construction_seconds'] += root_cost['wall']
    setup['construction_cpu_seconds'] += root_cost['cpu']
    return setup, (root, goal, deadline, plan)


def cost_receipt(setup):
    result = {field: setup[field] for field in ('conditioning_wall', 'screening_wall',
                                               'construction_seconds', 'construction_cpu_seconds')}
    result['scoring_wall'] = setup.get('scoring_wall')
    result['post_screen_wall'] = (result['construction_seconds']-result['conditioning_wall']
                                  -result['screening_wall'])
    return result


def compact(replay):
    rides = replay['rides']
    work = replay['rule_work']
    return dict(first=rides[0]['ridden_connection'] if rides else None,
                first_clock=rides[0]['time'] if rides else None, loss=replay['loss_upper'],
                arrived=replay['status']=='arrived', publication=replay['publish_at'],
                full_wall=replay['total_rule_pipeline_seconds'],
                warm_wall=replay['total_rule_pipeline_seconds']-replay['construction_seconds'],
                binding_wall=replay['binding_seconds'], action_wall=work.get('action_wall_seconds', 0.),
                applicable_wall=work.get('applicable_wall_seconds', 0.),
                incumbent_setup_wall=replay['incumbent_prepare_seconds'],
                rollout_worlds=replay['triggers'].get('rollout_worlds', 0))


def summarize(rows, cold):
    result = {}
    for arm in ARMS:
        records = [r['policies'][arm] for r in rows]
        base = [r['policies'][MINIMAL] for r in rows]
        differences = [a['loss']-b['loss'] for a, b in zip(base, records, strict=True)]
        result[arm] = dict(loss=mean(r['loss'] for r in records), arrived=sum(r['arrived'] for r in records),
                           mean_gain_vs_minimal=mean(differences), paired_se=stdev(differences)/len(rows)**.5,
                           improved=sum(d > 0 for d in differences), harmed=sum(d < 0 for d in differences),
                           first_different=sum(a['first']!=b['first'] for a,b in zip(base, records, strict=True)),
                           first_action_different=sum((a['first'], a['first_clock'])!=(b['first'], b['first_clock'])
                                                      for a,b in zip(base, records, strict=True)),
                           mean_full_wall=mean(r['full_wall'] for r in records),
                           mean_hot_wall=mean(r['warm_wall'] for r in records),
                           mean_binding_wall=mean(r['binding_wall'] for r in records),
                           mean_action_wall=mean(r['action_wall'] for r in records),
                           mean_applicable_wall=mean(r['applicable_wall'] for r in records),
                           mean_incumbent_setup_wall=mean(r['incumbent_setup_wall'] for r in records),
                           median_cold={field: median(r['arms'][arm][field] for r in cold)
                                        for field in cold[0]['arms'][arm]
                                        if cold[0]['arms'][arm][field] is not None},
                           rollout_worlds=sum(r['rollout_worlds'] for r in records))
        result[arm]['estimated_median_cold_plus_mean_hot_wall'] = (
            result[arm]['median_cold']['construction_seconds']+result[arm]['mean_hot_wall'])
    return result


def run_root(case, cpu):
    os.sched_setaffinity(0, {cpu})
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    day, query, decision = case
    date = int(day.replace('-', ''))
    eval_seed = [20261122, date, query, decision, 101]
    planning_seed = [2026112201, date, query, decision, 101]
    ready = max(root.time, root.last_arrival+graph.stops[root.stop].min_transfer_time)
    release = root.time+RELEASE_OFFSET
    assert release <= ready and plan.failure_index == 0
    cold, compiled = [], {}
    for repeat in range(COLD_REPEATS):
        shift = (repeat+CASES.index(case)) % len(ARMS)
        order = ARMS[shift:]+ARMS[:shift]
        ledger = {}
        for arm in order:
            setup, context = compile_case(case, arm, planning_seed)
            assert context == (root, goal, deadline, plan)
            assert setup['construction_seconds'] < RELEASE_OFFSET
            if repeat == 0:
                compiled[arm] = setup
            else:
                assert setup['paths'] == compiled[arm]['paths']
            ledger[arm] = cost_receipt(setup)
        cold.append(dict(order=order, arms=ledger))
    paths = compiled[MINIMAL]['paths']
    assert all(s['paths'] == paths for s in compiled.values())
    assert all(len({graph.connections[cid].trip_id for cid in path}) == 1 for path in paths.values())
    dep, arr, can = sample_clocks(belief, parameters, rng(eval_seed), WORLDS,
                                  templates=template_trip_map(belief, parameters))
    rows, disagreements = [], []
    multiplicity = Counter()
    for i in range(WORLDS):
        scene = scene_from_clocks(graph, (dep[i], arr[i], can[i]))
        model = build_model(graph, scene, root, goal, deadline)
        policies, offers = {}, ()
        shift = i % len(ARMS)
        for arm in ARMS[shift:]+ARMS[:shift]:
            receipt = replay_arm(graph, plan, root, goal, deadline, model, model.scenarios[0], arm,
                                 compiled[arm], publish_not_before=release)
            assert receipt['publish_at']==release and receipt['published'] and not receipt['rejected']
            assert receipt['loss_lower'] == receipt['loss_upper']
            policies[arm] = compact(receipt)
            if arm == MINIMAL:
                offers = next((r['offers'] for r in receipt['trigger_trace']
                               if r['branch']=='first_opportunity'), ())
        multiplicity[len(offers)] += 1
        baseline = policies[MINIMAL]
        for arm in ARMS[1:]:
            r = policies[arm]
            if (r['first'], r['first_clock'], r['loss']) != (baseline['first'], baseline['first_clock'], baseline['loss']):
                disagreements.append(dict(world=i, arm=arm, first_offer_clock=baseline['first_clock'],
                                          first_offer_cids=offers, minimal_first=baseline['first'],
                                          arm_first=r['first'], arm_first_clock=r['first_clock'],
                                          gain_vs_minimal=baseline['loss']-r['loss']))
        rows.append(dict(world=i, policies=policies))
    return dict(case=case, eval_seed=eval_seed, planning_seed=planning_seed, release=release,
                ready=ready, paths=paths, cold=cold, summary=summarize(rows, cold),
                first_offer_multiplicity=dict(multiplicity), disagreements=disagreements, rows=rows)


def main():
    cpus = sorted(os.sched_getaffinity(0))
    data = dict(status='running', worlds_per_root=WORLDS, cold_repeats_per_root=COLD_REPEATS,
                scope='Independent fixed fitted-model tape. Hot timing reuses an issued same-root compiler, '
                      'with fresh binds and rollout histories. Cold timing includes model conditioning and '
                      'screening; graph loading and prior fitting are shared preparation.', roots=[])
    output = HERE/'baseline_results.json'
    with ProcessPoolExecutor(max_workers=min(len(CASES), len(cpus))) as pool:
        jobs = [pool.submit(run_root, case, cpus[i % len(cpus)]) for i, case in enumerate(CASES)]
        for job in as_completed(jobs):
            row = job.result()
            data['roots'].append(row)
            data['roots'].sort(key=lambda r: CASES.index(tuple(r['case'])))
            output.write_text(json.dumps(data, indent=2)+'\n')
            print(row['case'], {k: (round(v['loss'], 3), v['arrived'], v['first_action_different'],
                                   round(v['estimated_median_cold_plus_mean_hot_wall'], 4))
                                for k,v in row['summary'].items()}, flush=True)
    data['status'] = 'complete'
    output.write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()
