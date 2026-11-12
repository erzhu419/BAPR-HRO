"""Fresh paired comparison of a common public transfer-rescue continuation."""

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
from experiments.reconstruction_v16.run_policy_pair import receipt
from experiments.reconstruction_v18.run_policy_pair import compile_comparator
from src.continuation_confidence import continuation_lower
from src.local_transfer_rescue import compile_local_transfer_rescue
from src.local_transfer_value import add_transfer_rescue_scores
from src.matched_transfer_rescue import MatchedTransferRescue
from src.public_wait_recourse import through_itineraries
from src.recourse_itinerary_router import build_trip_blocks
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map
from src.through_wait_value import ThroughWaitValue


HERE = Path(__file__).resolve().parent
WORLDS, RELEASE_OFFSET = 256, 30
V18 = 'support_through_wait'
PRIMARY = 'local_matched'
LOCAL = ('local_execution', 'local_original_menu', 'local_matched_120', PRIMARY)
ARMS = ('first_visible', 'through_seconds', 'support_old_wait', V18, *LOCAL)
COMPARISONS = (('first_visible', PRIMARY), ('through_seconds', PRIMARY), ('support_old_wait', PRIMARY),
               (V18, PRIMARY), (V18, LOCAL[0]), *zip(LOCAL, LOCAL[1:]))


def compile_arm(arm, graph, belief, parameters, root, goal, deadline, plan):
    if arm not in LOCAL:
        return compile_comparator(arm, graph, belief, parameters, root, goal, deadline, plan,
                                  censor_unknown_extras=False)
    start, cpu = perf_counter(), process_time()
    setup = compile_comparator(V18, graph, belief, parameters, root, goal, deadline, plan,
                               censor_unknown_extras=False)
    base = setup['controller']
    assert plan.failure_index == 0
    blocks = build_trip_blocks(graph)
    continuations = {}
    for cid in base.scores.rows:
        fixed = base.scores.paths[cid]
        follow = compile_local_transfer_rescue(graph, fixed, root, goal, deadline, trip_blocks=blocks)
        if follow.transfer_index is not None:
            continuations[cid] = follow
    diagnostics = None
    if arm != LOCAL[0]:
        diagnostics = add_transfer_rescue_scores(base.scores, belief, parameters, goal, continuations)
    margin = 0 if arm == PRIMARY else base.margin
    wait = base.waiting_score.__self__
    if arm in LOCAL[2:]:
        through = through_itineraries(graph, base.extras)
        wait = ThroughWaitValue(base.scores, through.keys() & base.scores.rows.keys(), margin=margin)
    controller = MatchedTransferRescue(base, continuations, wait, margin)
    return dict(controller=controller, construction_seconds=perf_counter()-start,
                construction_cpu_seconds=process_time()-cpu, diagnostics=diagnostics)


def root_receipt(replay):
    return dict(receipt(replay), local_misses=replay['triggers'].get('local_transfer_misses', 0),
                local_rescues=replay['triggers'].get('local_transfer_fallback_boards', 0))


def summarize(rows, width):
    summary = {}
    for arm in ARMS:
        records = [row['policies'][arm] for row in rows]
        gaps = [r['loss']-row['hindsight'] for r, row in zip(records, rows, strict=True)]
        assert all(0 <= gap <= width for gap in gaps)
        summary[arm] = dict(loss=mean(r['loss'] for r in records), arrivals=sum(r['reached'] for r in records),
                            local_misses=sum(r['local_misses'] for r in records),
                            local_rescues=sum(r['local_rescues'] for r in records),
                            mean_wall=mean(r['rule_wall'] for r in records),
                            mean_cpu=mean(r['rule_cpu'] for r in records),
                            hindsight_gap=mean(gaps), first=Counter(r['first'] for r in records))
        if arm == PRIMARY:
            summary[arm]['regret_upper'] = width-continuation_lower(
                [width-gap for gap in gaps], width, .05/len(CASES))
    paired = {}
    for base, candidate in COMPARISONS:
        gain = [r['policies'][base]['loss']-r['policies'][candidate]['loss'] for r in rows]
        paired[base+'__'+candidate] = dict(mean=mean(gain), se=stdev(gain)/len(gain)**.5,
                                          improved=sum(g > 0 for g in gain), harmed=sum(g < 0 for g in gain))
    return summary, paired


def run_root(case, cpu):
    os.sched_setaffinity(0, {cpu})
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    compiled = {}
    for arm in ARMS:
        # Each arm receives its own cold preparation from the frozen context.
        g, b, s, dest, end, p, pars = prepare(case)
        assert (s, dest, end, p) == (root, goal, deadline, plan)
        compiled[arm] = compile_arm(arm, g, b, pars, s, dest, end, p)
    expected = compiled[V18]['controller']
    for arm in LOCAL:
        c = compiled[arm]['controller']
        assert c.scores.paths == expected.scores.paths and c.extras == expected.extras
    assert compiled[LOCAL[0]]['controller'].scores.rows == expected.scores.rows
    for arm in LOCAL[1:]:
        assert compiled[arm]['controller'].scores.rows == compiled[PRIMARY]['controller'].scores.rows
    ready = max(root.time, root.last_arrival+graph.stops[root.stop].min_transfer_time)
    release = root.time+RELEASE_OFFSET
    assert release <= ready
    assert all(setup['construction_seconds'] < RELEASE_OFFSET for setup in compiled.values())
    day, query, decision = case
    seed = [20261119, int(day.replace('-', '')), query, decision, 0]
    with TemporaryDirectory(prefix='bapr_v19_pair_') as tmp:
        scan = TemporalScan(graph, build_temporal_kernel(Path(tmp)/'scan.so'))
        clocks = sample_clocks(belief, parameters, rng(seed), WORLDS,
                               templates=template_trip_map(belief, parameters))
        transfer = [graph.stops[s].min_transfer_time for s in scan.stops]
        hindsight = scan.platform_arrivals(*clocks, root, goal, deadline, transfer)-root.time
        assert min(hindsight) >= ready-root.time
        rows = []
        for i, h in enumerate(hindsight):
            scene = scene_from_clocks(graph, tuple(a[i] for a in clocks))
            model = build_model(graph, scene, root, goal, deadline)
            shift = i % len(ARMS)
            results = {}
            for arm in ARMS[shift:]+ARMS[:shift]:
                replay = replay_arm(graph, plan, root, goal, deadline, model, model.scenarios[0],
                                    arm, compiled[arm], publish_not_before=release)
                assert replay['publish_at'] == release and not replay['rejected']
                results[arm] = root_receipt(replay)
            assert results[LOCAL[0]]['first'] == results[V18]['first']
            assert results[LOCAL[0]]['loss'] <= results[V18]['loss']
            rows.append(dict(world=i, hindsight=int(h), policies=results))
    summary, paired = summarize(rows, deadline-ready)
    continuations = compiled[PRIMARY]['controller'].continuations
    return dict(case=case, seed=seed, ready=ready, publication=release, summary=summary, paired=paired,
                transfer_options={cid: follow.fallback_paths for cid, follow in continuations.items()},
                transfer_estimates=compiled[PRIMARY]['diagnostics'], rows=rows)


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
