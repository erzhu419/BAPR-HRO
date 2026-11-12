"""Fresh paired control of the V16 additional through-service routes."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v13.platform_run import build_model, compile_arms, replay_arm
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from experiments.reconstruction_v16.run_policy_pair import BASE, receipt, summary
from src.conditional_rescue_policy import ConditionalItineraries, ConditionalRescuePolicy
from src.public_rescue_candidates import screen_public_itineraries
from src.public_wait_recourse import compile_through_scores
from src.tail_rescue import TailRescuePlan
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map


N, MARGIN = 128, 120
METHODS = (BASE, 'v16', 'through_exact', 'through_point')
PROTOCOL = {
    'cases': CASES, 'n': N, 'margin_seconds': MARGIN, 'methods': METHODS,
    'seed': '[20261117,int(date),query,decision,1], independent of V16 and other V17 arms',
    'menus': 'Use V16 public timetable screening. Both through controls keep only screened paths '
             'whose complete path stays on one identified trip. Same A/B analytic scores and waiting proxy.',
    'through_exact': 'One backward contraction per fixed single-trip path, equal to V16 conditional '
                     'itinerary scoring for each feasible first departure clock. Check equality before replay.',
    'through_point': 'Conditional operating clock means for screened through paths only. Do not compute '
                     'the unused stochastic path integral. Use the exact A/B scores for the same wait reference.',
    'publisher': 'Same previously issued primary incumbent, MCT, V13 local event clock and ceil(wall) release.',
    'ledger': 'Independently reconstruct the frozen belief before each candidate compiler. Charge screen, '
              'scores, controller construction, binding and rule execution to each arm. Reusable shared setup '
              'and simulation-only fitting input, kernel compile, drawing, oracle and replay infrastructure separate.',
    'scope': 'Prespecified fitted-model diagnostic on three previously examined waiting roots. No threshold '
             'selection from these worlds, no real-date generalization or full causal certificate.',
}


def compile_candidate(case, method):
    before = perf_counter()
    graph, belief, root, goal, deadline, plan, pars = prepare(case)
    shared_setup = perf_counter()-before
    before, cpu_start = perf_counter(), process_time()
    ab = {path[0]: TailRescuePlan(path, first_action_now=True, departures=plan.departures)
          for path in (plan.primary, plan.rescue)}
    extra = screen_public_itineraries(graph, root, goal, deadline, ab, 720)
    screen_seconds = perf_counter()-before
    if method == 'v16':
        paths = {**ab, **{cid: TailRescuePlan(route.path, first_action_now=True)
                          for cid, route in extra.items()}}
        scores = ConditionalItineraries(belief, pars, graph, root, goal, deadline, paths)
    else:
        scores, extra = compile_through_scores(belief, pars, graph, root, goal, deadline, ab, extra,
                                               point=method == 'through_point')
    controller = ConditionalRescuePolicy(graph, plan, root, deadline, scores, extra, MARGIN)
    wall, cpu = perf_counter()-before, process_time()-cpu_start
    return dict(controller=controller, construction_seconds=wall, construction_cpu_seconds=cpu), dict(
        shared_setup_seconds=shared_setup, screen_seconds=screen_seconds, preparation_wall=wall,
        preparation_cpu=cpu, n_score_rows=sum(len(rows) for rows in scores.rows.values()),
        excluded=scores.excluded, extra_paths={cid: route.path for cid,route in extra.items()})


def audit_tables(compiled):
    full = compiled['v16']['controller'].scores
    exact = compiled['through_exact']['controller'].scores
    point = compiled['through_point']['controller'].scores
    if exact.rows.keys() != point.rows.keys():
        raise AssertionError('Through controls scored different candidate IDs')
    count = 0
    error = 0.
    for cid, rows in exact.rows.items():
        if rows.keys() != full.rows[cid].keys() or rows.keys() != point.rows[cid].keys():
            raise AssertionError('Through controls changed modeled departure support')
        for clock, actual in rows.items():
            expected = full.rows[cid][clock]
            error = max(error, abs(actual.expected_arrival_or_deadline-expected.expected_arrival_or_deadline))
            if abs(actual.success_probability-expected.success_probability) > 1e-9:
                raise AssertionError('Backward conditional integration changed success mass')
            count += 1
    if error > 1e-7:
        raise AssertionError(f'Backward conditional integration changed fixed-path score by {error}')
    return dict(checked_rows=count, max_absolute_score_error=error)


def run_case(case, cpu):
    os.sched_setaffinity(0, {cpu})
    before = perf_counter()
    graph, belief, root, goal, deadline, plan, pars = prepare(case)
    shared_setup = perf_counter()-before
    compiled = {BASE: compile_arms(graph, plan, root, deadline)[BASE]}
    ledger = {}
    start = CASES.index(case)
    for method in METHODS[1+start:]+METHODS[1:1+start]:
        compiled[method], ledger[method] = compile_candidate(case, method)
    audit = audit_tables(compiled)
    day, query, decision = case
    seed = [20261117, int(day.replace('-', '')), query, decision, 1]
    with TemporaryDirectory(prefix='bapr_v17_through_') as temp:
        before = perf_counter()
        scan = TemporalScan(graph, build_temporal_kernel(Path(temp)/'scan.so'))
        kernel_seconds = perf_counter()-before
        before = perf_counter()
        packed = sample_clocks(belief, pars, rng(seed), N, templates=template_trip_map(belief,pars))
        draw_seconds = perf_counter()-before
        transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
        before = perf_counter()
        hindsight = scan.platform_arrivals(*packed, root, goal, deadline, transfers)-root.time
        oracle_seconds = perf_counter()-before
        rows, model_seconds, replay_seconds = [], 0., 0.
        for i in range(N):
            before = perf_counter()
            scene = scene_from_clocks(graph, tuple(x[i] for x in packed))
            model = build_model(graph, scene, root, goal, deadline)
            model_seconds += perf_counter()-before
            before = perf_counter()
            shift = (start+i)%len(METHODS)
            order = METHODS[shift:]+METHODS[:shift]
            policies = {name: receipt(replay_arm(graph, plan, root, goal, deadline, model,
                        model.scenarios[0], name, compiled[name])) for name in order}
            replay_seconds += perf_counter()-before
            assert all(hindsight[i] <= row['loss'] for row in policies.values())
            rows.append(dict(world=i, hindsight=int(hindsight[i]), policies=policies))
    summaries = summary(rows, METHODS)
    for name, row in summaries.items():
        differences = [q['policies']['v16']['loss']-q['policies'][name]['loss'] for q in rows]
        row.update(mean_gain_vs_v16=mean(differences), improved_vs_v16=sum(d>0 for d in differences),
                   harmed_vs_v16=sum(d<0 for d in differences), tied_vs_v16=sum(d==0 for d in differences),
                   mean_gap_to_hindsight=mean(q['policies'][name]['loss']-q['hindsight'] for q in rows))
    return dict(case=case, root_clock=root.time, deadline=deadline, seed=seed, cpu=cpu,
                shared_setup_seconds=shared_setup, candidate_ledger=ledger, table_audit=audit,
                kernel_seconds=kernel_seconds, draw_seconds=draw_seconds, oracle_seconds=oracle_seconds,
                model_seconds=model_seconds, replay_seconds=replay_seconds, rows=rows, summary=summaries)


def main():
    path = HERE/'through_controls.json'
    output = dict(status='running', protocol=PROTOCOL, roots=[])
    path.write_text(json.dumps(output, indent=2)+'\n')
    cpus = sorted(os.sched_getaffinity(0), reverse=True)
    with ProcessPoolExecutor(max_workers=min(len(cpus), len(CASES))) as pool:
        jobs = [pool.submit(run_case, case, cpus[i%len(cpus)]) for i, case in enumerate(CASES)]
        for job in as_completed(jobs):
            result = job.result()
            output['roots'].append(result)
            output['roots'].sort(key=lambda row: CASES.index(row['case']))
            path.write_text(json.dumps(output, indent=2)+'\n')
            print(result['case'], result['table_audit'], result['summary'], flush=True)
    output['status'] = 'complete'
    path.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
