"""Paired V18 comparison of the waiting reference on fixed candidate menus."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v13.platform_run import build_model, compile_arms, replay_arm
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from experiments.reconstruction_v16.run_policy_pair import receipt
from experiments.reconstruction_v17.run_support_pair import MAX_NEW, MIN_MASS
from src.conditional_rescue_policy import ConditionalItineraries
from src.continuation_confidence import continuation_lower
from src.offgrid_conditional import ActualClockRescue, ActualClockScores
from src.public_rescue_candidates import screen_public_itineraries
from src.public_wait_recourse import compile_through_scores, through_itineraries
from src.support_rescue_candidates import screen_support_itineraries
from src.tail_rescue import TailRescuePlan
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map
from src.through_wait_value import ThroughWaitValue


HERE = Path(__file__).resolve().parent
WORLDS, MARGIN, RELEASE_OFFSET = 256, 120, 10
PRIMARY = 'support_through_wait'
ARMS = ('first_visible', 'through_seconds', 'support_old_wait', PRIMARY, 'through_updated_wait')
COMPARISONS = (('first_visible', 'through_seconds'), ('first_visible', PRIMARY),
               ('support_old_wait', PRIMARY), ('through_seconds', PRIMARY),
               ('through_seconds', 'through_updated_wait'))


def root_menu(graph, root, goal, deadline, plan):
    paths = {path[0]: TailRescuePlan(path, first_action_now=True, departures=plan.departures)
             for path in (plan.primary, plan.rescue)}
    return paths, screen_public_itineraries(graph, root, goal, deadline, paths, 720)


def support_additions(graph, belief, parameters, root, goal, deadline, paths, extras):
    new, _ = screen_support_itineraries(graph, belief, parameters, root, goal, deadline,
                                        set(paths) | set(extras), window=720,
                                        min_mass=MIN_MASS, max_new=MAX_NEW)
    return new


def compile_comparator(arm, graph, belief, parameters, root, goal, deadline, plan, *,
                       censor_unknown_extras=True):
    if arm == 'first_visible':
        return compile_arms(graph, plan, root, deadline)[arm]
    started, cpu = perf_counter(), process_time()
    paths, extras = root_menu(graph, root, goal, deadline, plan)
    through = through_itineraries(graph, extras)
    if arm.startswith('support_'):
        extras.update(support_additions(graph, belief, parameters, root, goal, deadline, paths, extras))
        paths.update((cid, TailRescuePlan(option.path, first_action_now=True))
                     for cid, option in extras.items())
        scores = ConditionalItineraries(belief, parameters, graph, root, goal, deadline, paths)
    else:
        scores, extras = compile_through_scores(belief, parameters, graph, root, goal, deadline,
                                                paths, extras)
    wait = None
    if arm in (PRIMARY, 'through_updated_wait'):
        wait = ThroughWaitValue(scores, through.keys() & scores.rows.keys(), margin=MARGIN)
    clock_scores = ActualClockScores(belief, parameters, scores, goal)
    controller = ActualClockRescue(graph, plan, root, deadline, scores, extras, MARGIN,
                                  clock_scores, mode='projection', wait_value=wait,
                                  censor_unknown_extras=censor_unknown_extras)
    return dict(controller=controller, construction_seconds=perf_counter()-started,
                construction_cpu_seconds=process_time()-cpu)


def compile_comparators(graph, belief, parameters, root, goal, deadline, plan, *,
                        censor_unknown_extras=True):
    return {arm: compile_comparator(arm, graph, belief, parameters, root, goal, deadline, plan,
                                    censor_unknown_extras=censor_unknown_extras) for arm in ARMS}


def summarize(rows, width):
    summary = {}
    for arm in ARMS:
        records = [r['policies'][arm] for r in rows]
        gaps = [r['loss']-world['hindsight'] for r, world in zip(records, rows, strict=True)]
        assert all(0 <= gap <= width for gap in gaps)
        summary[arm] = dict(loss=mean(r['loss'] for r in records),
                            arrivals=sum(r['reached'] for r in records),
                            rejects=sum(r['rejected'] for r in records),
                            hindsight_gap=mean(gaps), mean_wall=mean(r['rule_wall'] for r in records),
                            mean_cpu=mean(r['rule_cpu'] for r in records),
                            first=Counter(r['first'] for r in records))
        if arm == PRIMARY:
            summary[arm]['regret_upper'] = width-continuation_lower(
                [width-gap for gap in gaps], width, .05/len(CASES))
    paired = {}
    for base, candidate in COMPARISONS:
        gain = [r['policies'][base]['loss']-r['policies'][candidate]['loss'] for r in rows]
        paired[base+'__'+candidate] = dict(mean=mean(gain), se=stdev(gain)/len(gain)**.5,
                                          improved=sum(g > 0 for g in gain),
                                          harmed=sum(g < 0 for g in gain))
    return summary, paired


def run_root(case, cpu):
    os.sched_setaffinity(0, {cpu})
    started = perf_counter()
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    shared_seconds = perf_counter()-started
    compiled = {}
    for arm in ARMS:
        # Independent preparations keep memoization from charging one arm for another.
        g, b, s, dest, end, p, pars = prepare(case)
        assert (s, dest, end, p) == (root, goal, deadline, plan)
        compiled[arm] = compile_comparator(arm, g, b, pars, s, dest, end, p,
                                           censor_unknown_extras=False)
    menus = {arm: {cid: option.path for cid, option in v['controller'].extras.items()}
             for arm, v in compiled.items() if arm != 'first_visible'}
    assert menus['support_old_wait'] == menus[PRIMARY]
    assert menus['through_seconds'] == menus['through_updated_wait']
    for a, b in (('support_old_wait', PRIMARY), ('through_seconds', 'through_updated_wait')):
        assert compiled[a]['controller'].scores.rows == compiled[b]['controller'].scores.rows
    ready = max(root.time, root.last_arrival+graph.stops[root.stop].min_transfer_time)
    publish = root.time+RELEASE_OFFSET
    assert publish <= ready
    day, query, decision = case
    seed = [20261118, int(day.replace('-', '')), query, decision, 0]
    with TemporaryDirectory(prefix='bapr_v18_pair_') as tmp:
        scan = TemporalScan(graph, build_temporal_kernel(Path(tmp)/'scan.so'))
        clocks = sample_clocks(belief, parameters, rng(seed), WORLDS,
                               templates=template_trip_map(belief, parameters))
        transfers = [graph.stops[s].min_transfer_time for s in scan.stops]
        hindsight = scan.platform_arrivals(*clocks, root, goal, deadline, transfers)-root.time
        assert min(hindsight) >= ready-root.time
        rows = []
        for i, h in enumerate(hindsight):
            scene = scene_from_clocks(graph, tuple(a[i] for a in clocks))
            model = build_model(graph, scene, root, goal, deadline)
            shift = i % len(ARMS)
            results = {}
            for arm in ARMS[shift:]+ARMS[:shift]:
                replay = replay_arm(graph, plan, root, goal, deadline, model, model.scenarios[0],
                                    arm, compiled[arm], publish_not_before=publish)
                assert replay['publish_at'] == publish
                results[arm] = receipt(replay)
            rows.append(dict(world=i, hindsight=int(h), policies=results))
    summary, paired = summarize(rows, deadline-ready)
    return dict(case=case, seed=seed, ready=ready, deadline=deadline, publication=publish,
                shared_setup_seconds=shared_seconds, menus=menus, summary=summary, paired=paired,
                rows=rows)


def main():
    cpus = sorted(os.sched_getaffinity(0))
    data = dict(status='running', worlds_per_root=WORLDS, primary=PRIMARY, roots=[])
    output = HERE/'paired_results.json'
    with ProcessPoolExecutor(max_workers=min(len(CASES), len(cpus))) as pool:
        jobs = [pool.submit(run_root, case, cpus[i % len(cpus)]) for i, case in enumerate(CASES)]
        for job in as_completed(jobs):
            root = job.result()
            data['roots'].append(root)
            data['roots'].sort(key=lambda r: CASES.index(tuple(r['case'])))
            output.write_text(json.dumps(data, indent=2)+'\n')
            print(root['case'], root['summary'], root['paired'], flush=True)
    data['status'] = 'complete'
    output.write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()
