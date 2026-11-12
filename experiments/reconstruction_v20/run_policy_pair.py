"""Charge and compare the direct restriction of V19 on fresh fitted worlds."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from math import isclose
import os
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v13.platform_run import build_model, replay_arm
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from experiments.reconstruction_v18.run_policy_pair import root_menu, support_additions
from experiments.reconstruction_v19.run_policy_pair import (
    PRIMARY as V19, RELEASE_OFFSET, compile_arm, root_receipt,
)
from src.conditional_rescue_policy import ConditionalItineraries
from src.continuation_confidence import continuation_lower
from src.local_transfer_rescue import compile_local_transfer_rescue
from src.local_transfer_value import add_transfer_rescue_scores
from src.matched_transfer_rescue import MatchedTransferRescue
from src.offgrid_conditional import ActualClockRescue, ActualClockScores
from src.public_wait_recourse import compile_through_scores, through_itineraries
from src.recourse_itinerary_router import build_trip_blocks
from src.tail_rescue import TailRescuePlan
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map
from src.through_wait_value import ThroughWaitValue


HERE = Path(__file__).resolve().parent
BASE, DIRECT = 'first_visible', 'direct_zero'
ARMS = (BASE, V19, DIRECT)
BLOCKS, BLOCK_SIZE = 4, 256


def compile_direct(graph, belief, parameters, root, goal, deadline, plan, *, contract=True):
    start, cpu = perf_counter(), process_time()
    original, extras = root_menu(graph, root, goal, deadline, plan)
    additions = support_additions(graph, belief, parameters, root, goal, deadline, original, extras)
    extras.update(additions)
    score_start = perf_counter()
    if contract:
        scores, through = compile_through_scores(belief, parameters, graph, root, goal, deadline,
                                                original, extras)
    else:
        through = through_itineraries(graph, extras)
        paths = {**original, **{cid: TailRescuePlan(option.path) for cid, option in through.items()}}
        scores = ConditionalItineraries(belief, parameters, graph, root, goal, deadline, paths)
    scoring_wall = perf_counter()-score_start
    # Preserve V19's rule on the issued A/B plans even if one has a transfer.
    continuations = {}
    if any(len({graph.connections[cid].trip_id for cid in path.primary}) > 1
           for path in original.values()):
        blocks = build_trip_blocks(graph)
        for cid, path in original.items():
            follow = compile_local_transfer_rescue(graph, path, root, goal, deadline, trip_blocks=blocks)
            if follow.transfer_index is not None and cid in scores.rows:
                continuations[cid] = follow
        add_transfer_rescue_scores(scores, belief, parameters, goal, continuations)
    wait = ThroughWaitValue(scores, through.keys() & scores.rows.keys(), margin=0)
    controller = ActualClockRescue(graph, plan, root, deadline, scores, through, 0,
                                  ActualClockScores(belief, parameters, scores, goal), mode='projection',
                                  censor_unknown_extras=False, wait_value=wait)
    if continuations:
        controller = MatchedTransferRescue(controller, continuations, wait, 0)
    return dict(controller=controller, construction_seconds=perf_counter()-start,
                construction_cpu_seconds=process_time()-cpu, screening_wall=score_start-start,
                scoring_wall=scoring_wall, support_direct=sorted(through.keys() & additions.keys()))


def audit_menu(full, direct, graph):
    assert direct.extras == through_itineraries(graph, full.extras)
    assert direct.waiting_score.__self__.through_ids == full.waiting_score.__self__.through_ids
    assert direct.margin == full.margin == 0
    assert direct.waiting_score.__self__.margin == full.waiting_score.__self__.margin == 0
    original = set(full.scores.paths)-set(full.extras)
    retained = original | set(direct.extras)
    assert set(direct.scores.paths) == retained & set(full.scores.paths)
    assert set(direct.scores.rows) == retained & set(full.scores.rows)
    max_error = 0.
    for cid, rows in direct.scores.rows.items():
        old = full.scores.rows[cid]
        assert rows.keys() == old.keys()
        assert direct.scores.paths[cid] == full.scores.paths[cid]
        for time, row in rows.items():
            reference = old[time]
            max_error = max(max_error, abs(row.expected_arrival_or_deadline-reference.expected_arrival_or_deadline))
            assert row.probability == reference.probability
            assert row.next_arrival_support == reference.next_arrival_support
            assert isclose(row.success_probability, reference.success_probability, abs_tol=1e-12)
    assert max_error <= 1e-8
    return dict(retained=sorted(retained), through=sorted(direct.extras),
                removed=sorted(set(full.extras)-set(direct.extras)), max_score_difference=max_error,
                scored_clocks=sum(map(len, direct.scores.rows.values())))


def summarize(rows, width, menus, compiled):
    summary = {}
    for arm in ARMS:
        records = [r['policies'][arm] for r in rows]
        gaps = [r['loss']-w['hindsight'] for r, w in zip(records, rows, strict=True)]
        assert all(0 <= gap <= width for gap in gaps)
        summary[arm] = dict(loss=mean(r['loss'] for r in records), arrivals=sum(r['reached'] for r in records),
                            mean_wall=mean(r['rule_wall'] for r in records),
                            mean_policy_cpu=mean(r['rule_cpu'] for r in records),
                            cold_compile_wall=compiled[arm]['construction_seconds'],
                            hindsight_gap=mean(gaps), first=Counter(r['first'] for r in records),
                            local_misses=sum(r['local_misses'] for r in records),
                            local_rescues=sum(r['local_rescues'] for r in records))
        if arm == DIRECT:
            summary[arm]['regret_upper'] = width-continuation_lower(
                [width-gap for gap in gaps], width, .05/len(CASES))
    paired = {}
    for base, arm in ((BASE, V19), (BASE, DIRECT), (V19, DIRECT)):
        gains = [r['policies'][base]['loss']-r['policies'][arm]['loss'] for r in rows]
        paired[base+'__'+arm] = dict(mean=mean(gains), se=stdev(gains)/len(gains)**.5,
                                     improved=sum(d>0 for d in gains), harmed=sum(d<0 for d in gains))
    removed = [r for r in rows if r['policies'][V19]['first'] in menus['removed']]
    v19_gain = paired[BASE+'__'+V19]['mean']
    return summary, paired, dict(removed_worlds=len(removed),
                                 removed_first=Counter(r['policies'][V19]['first'] for r in removed),
                                 retained_fraction=paired[BASE+'__'+DIRECT]['mean']/v19_gain,
                                 wall_savings=summary[V19]['mean_wall']-summary[DIRECT]['mean_wall'],
                                 wall_fraction_saved=1-summary[DIRECT]['mean_wall']/summary[V19]['mean_wall'])


def run_root(case, cpu):
    os.sched_setaffinity(0, {cpu})
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    compiled = {}
    for arm in ARMS:
        g, b, s, dest, end, p, pars = prepare(case)
        assert (s, dest, end, p) == (root, goal, deadline, plan)
        compiled[arm] = (compile_direct(g, b, pars, s, dest, end, p) if arm == DIRECT else
                         compile_arm(arm, g, b, pars, s, dest, end, p))
    menus = audit_menu(compiled[V19]['controller'], compiled[DIRECT]['controller'], graph)
    menus['support_direct'] = compiled[DIRECT]['support_direct']
    removed = set(menus['removed'])
    ready = max(root.time, root.last_arrival+graph.stops[root.stop].min_transfer_time)
    release = root.time+RELEASE_OFFSET
    assert release <= ready
    assert all(setup['construction_seconds'] < RELEASE_OFFSET for setup in compiled.values())
    day, query, decision = case
    seeds = [[20261120, int(day.replace('-', '')), query, decision, block] for block in range(BLOCKS)]
    rows = []
    with TemporaryDirectory(prefix='bapr_v20_pair_') as tmp:
        scan = TemporalScan(graph, build_temporal_kernel(Path(tmp)/'scan.so'))
        transfer = [graph.stops[s].min_transfer_time for s in scan.stops]
        templates = template_trip_map(belief, parameters)
        for block, seed in enumerate(seeds):
            clocks = sample_clocks(belief, parameters, rng(seed), BLOCK_SIZE, templates=templates)
            hindsight = scan.platform_arrivals(*clocks, root, goal, deadline, transfer)-root.time
            assert min(hindsight) >= ready-root.time
            for i, h in enumerate(hindsight):
                scene = scene_from_clocks(graph, tuple(a[i] for a in clocks))
                model = build_model(graph, scene, root, goal, deadline)
                shift = len(rows) % len(ARMS)
                results = {}
                for arm in ARMS[shift:]+ARMS[:shift]:
                    replay = replay_arm(graph, plan, root, goal, deadline, model, model.scenarios[0],
                                        arm, compiled[arm], publish_not_before=release)
                    assert replay['publish_at'] == release and replay['published'] and not replay['rejected']
                    results[arm] = root_receipt(replay)
                reference, direct = results[V19], results[DIRECT]
                if reference['first'] not in removed:
                    assert (reference['first'], reference['loss'], reference['reached']) == (
                        direct['first'], direct['loss'], direct['reached'])
                rows.append(dict(block=block, world=i, hindsight=int(h), policies=results))
    summary, paired, savings = summarize(rows, deadline-ready, menus, compiled)
    return dict(case=case, seeds=seeds, ready=ready, publication=release, menus=menus, summary=summary,
                paired=paired, savings=savings, rows=rows)


def main():
    cpus = sorted(os.sched_getaffinity(0))
    data = dict(status='running', worlds_per_root=BLOCKS*BLOCK_SIZE, primary=DIRECT, roots=[])
    output = HERE/'paired_results.json'
    with ProcessPoolExecutor(max_workers=min(len(CASES), len(cpus))) as pool:
        jobs = [pool.submit(run_root, case, cpus[i % len(cpus)]) for i, case in enumerate(CASES)]
        for job in as_completed(jobs):
            root = job.result()
            data['roots'].append(root)
            data['roots'].sort(key=lambda r: CASES.index(tuple(r['case'])))
            output.write_text(json.dumps(data, indent=2)+'\n')
            print(root['case'], root['summary'], root['paired'], root['savings'], flush=True)
    data['status'] = 'complete'
    output.write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()
