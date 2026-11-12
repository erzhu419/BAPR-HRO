"""One instrumented timing diagnosis; no new policy bank or scientific score.

Replays only the first chronological V37 census state with its original seed
and 128 worlds. The original receipts remain authoritative. The profile is
checked against every original per-world loss, status, action and trace.
"""
import cProfile
import gzip
import json
from pathlib import Path
import pstats

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v33.run import jsonable
from experiments.reconstruction_v34.conditional_reference import locate_node
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.forecast import ForecastPlanner
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v37.controller import restore_single_snapshot, evaluate_onboard_actions


HERE = Path(__file__).resolve().parent
V37 = HERE.parent/'reconstruction_v37'
V34 = HERE.parent/'reconstruction_v34'


def scientific_result(result):
    return dict(actions=result['actions'], paired=result['paired'],
        selected_action=result['selected_action'], seed=result['seed'],
        count=result['count'], public_history=result['public_history'],
        rows={label:[{key:row[key] for key in ('loss','status','trace','decisions','world_index')}
                     for row in rows] for label,rows in result['rows'].items()})


def existing_costs():
    rows = []
    for group in ('reference_cases','policy_cases'):
        for path in sorted((V37/group).glob('*.json.gz')):
            result = read(path)
            cost = result['cost']
            actions = list(cost['action_continuations'].values())
            stats = {}
            for action in actions:
                for key,value in action['planner_stats'].items():
                    stats[key] = stats.get(key,0)+value
            rows.append(dict(source=str(path.relative_to(V37)),
                total_cpu=cost['total_process_cpu'], preparation_cpu=cost['preparation']['process_cpu'],
                continuation_cpu=sum(action['process_cpu'] for action in actions),
                world_cpu=sum(row['total_process_cpu'] for outcomes in result['rows'].values() for row in outcomes),
                world_compute_wall=sum(row['compute_wall'] for outcomes in result['rows'].values() for row in outcomes),
                worlds=result['count'], planner_stats=stats))
    aggregate = {key:sum(row[key] for row in rows)
                 for key in ('total_cpu','preparation_cpu','continuation_cpu','world_cpu','world_compute_wall','worlds')}
    aggregate['planner_stats'] = {key:sum(row['planner_stats'][key] for row in rows)
                                  for key in rows[0]['planner_stats']}
    return dict(banks=len(rows), aggregate=aggregate, cases=rows)


def profile_entries(profile):
    result = []
    for (file,line,function),(primitive,calls,self_seconds,cumulative,callers) in pstats.Stats(profile).stats.items():
        try:
            file = str(Path(file).resolve().relative_to(HERE.parent.parent))
        except ValueError:
            pass
        result.append(dict(file=file,line=line,function=function,primitive_calls=primitive,
            calls=calls,self_seconds=self_seconds,cumulative_seconds=cumulative))
    return result


def summarize(probe, profile):
    original = read(V37/'reference_cases/000_000.json.gz')
    equivalent = scientific_result(probe)==scientific_result(original)
    entries = profile_entries(profile)
    findings = [
        dict(component='Continuation compilation', evidence='Across the 148 original banks, continuations use '
             '66.132 of 67.892 CPU seconds; 44,420 cold ride and 34,844 cold platform compiles remain after caching.',
             action='Optimize the continuation planner; sampling, trace recording and snapshot copying are not the main target.'),
        dict(component='Scalar board-score interpolation', evidence='The first-state profile calls clock_nominal_score '
             '32,715 times, consuming 0.100 instrumented seconds including scalar score/interp.',
             action='Compute scores for a clock vector in one interpolation call, or reuse board_scores at the identical grid; '
             'retain the existing off-grid projection, support checks and FullMenuOpportunity integer-clock semantics.',
             source=['experiments/reconstruction_v33/planner.py:68',
                     'experiments/reconstruction_v33/planner.py:231','src/full_menu_opportunity.py:37']),
        dict(component='Unrequested stopping policies and terminal chains', evidence='The first-state profile calls '
             'OnboardStopping.solve 472 times (0.061 instrumented seconds), generating optimal, rolling and rollout '
             'policies although execution requests only rolling_fixed. A one-exit destination ride also takes this path.',
             action='Provide the mathematically identical rolling_fixed recursion and one-exit terminal result directly; '
             'do not remove deadline/killed mass or current public arrival masks.',
             source=['experiments/reconstruction_v33/planner.py:284','src/onboard_stopping.py:125']),
        dict(component='Separate arm caches', evidence='conditional_action_check constructs a fresh ForecastPlanner '
             'inside each action loop; existing keys already include node structure, method and relevant trip evidence.',
             action='Share the immutable planner cache across both arms of the same frozen bank. This is a code-proven '
             'duplicate-cache opportunity; the current profile does not measure the arm-key intersection or its speed gain.',
             source=['experiments/reconstruction_v34/gated_rollout.py:48',
                     'experiments/reconstruction_v33/planner.py:121']),
        dict(component='Repeated posterior construction', evidence='MessageHistory.law uses 0.156 instrumented seconds; '
             '408 departure_evidence calls use 0.120 seconds. These are nested timings, not additive.',
             action='A shared cache may reuse an exactly equal ordered trip history and active emission; preserve '
             'observation order and off-grid projection. First measure the simpler planner changes.',
             source=['experiments/reconstruction_v34/forecast.py:187']),
        dict(component='Trace and snapshot overhead', evidence='699 record calls use 0.000331 seconds; '
             '257 full history clone calls use 0.003419 seconds.',
             action='Retain public traces and independent histories; removing them cannot address the measured bottleneck.')]
    output = dict(scope='Instrumentation-only repetition of frozen first state; original V37 scores are authoritative.',
        state_id='000:000', original='reconstruction_v37/reference_cases/000_000.json.gz',
        seed=37050, worlds=128, scientific_results_identical=equivalent,
        instrumented_cost=probe['cost'], original_uninstrumented_cost=original['cost'],
        original_diagnostic_cost=original['diagnostic_setup_cost'],
        instrumented_diagnostic_cost=probe['diagnostic_setup_cost'],
        instrumented_profile_total_seconds=pstats.Stats(profile).total_tt,
        top_cumulative=sorted(entries,key=lambda row:row['cumulative_seconds'],reverse=True)[:40],
        top_self=sorted(entries,key=lambda row:row['self_seconds'],reverse=True)[:40],
        all_profile_entries=entries, existing_paid_costs=existing_costs(), findings=findings,
        limitations='The cProfile timings include instrumentation and cover one predeclared state. '
            'Cumulative function times overlap. No full-bank speedup or new scientific outcome is claimed; '
            'the 128-world value remains a finite-bank value, not an exact population expectation.')
    (HERE/'profile_summary.json').write_text(json.dumps(output,indent=2,default=jsonable,allow_nan=False)+'\n')
    if not equivalent:
        raise AssertionError('Instrumented first-state replay differs from the original scientific receipt')
    print(json.dumps({key:output[key] for key in ('state_id','seed','worlds','scientific_results_identical',
        'instrumented_profile_total_seconds')}),flush=True)
    for row in output['top_cumulative'][:22]:
        print(row['file'],row['line'],row['function'],'calls',row['calls'],
              'self',round(row['self_seconds'],6),'cum',round(row['cumulative_seconds'],6),flush=True)


def main():
    profile_path, probe_path = HERE/'first_state.pstats', HERE/'profile_probe.json.gz'
    if profile_path.exists() and probe_path.exists():
        # Reporting changes reuse the paid profile instead of running it again.
        summarize(read(probe_path),str(profile_path))
        return
    census = read(V37/'census.json')
    record = next(row for row in census['states'] if row['state_id']=='000:000')
    data,selection,physical_fit = (read(V34/name) for name in ('data.json.gz','selection.json','model_fit.json'))
    fit = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    root = selection['selected'][0]
    graph = context(data,root['day'])[0]
    model,parameters = setup(graph,physical_fit)
    planner = ForecastPlanner(graph,model,parameters,root['deadline'])
    trace = read(HERE.parent/'reconstruction_v36/roots/000.json.gz')['rows']['public_messages']['combined_rolling']['outcome']['trace']
    row = trace[record['trace_index']]
    ride = locate_node(root_ride(graph,root),trace,record['trace_index'])
    history = restore_single_snapshot(model,parameters,fit,row['public_history'],row['clock'])
    profile = cProfile.Profile()
    result = profile.runcall(evaluate_onboard_actions,graph,planner,ride,row['cid'],row['clock'],history,
        root_index=0,day=root['day'],call_index=0,count=128,seed_base=37050)
    profile.dump_stats(profile_path)
    with gzip.open(probe_path,'wt') as handle:
        json.dump(result,handle,default=jsonable,allow_nan=False)
    summarize(result,profile)


if __name__=='__main__':
    main()
