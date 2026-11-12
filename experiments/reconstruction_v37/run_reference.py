"""Evaluate each frozen two-action onboard public state exactly once.

Completed banks are saved immediately and are later reusable by the V37
policy. No realized continuation is passed to the conditional evaluator.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v33.run import jsonable
from experiments.reconstruction_v34.conditional_reference import locate_node
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.forecast import ForecastPlanner
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v37.controller import restore_single_snapshot, evaluate_onboard_actions

HERE = Path(__file__).resolve().parent
V34 = HERE.parent/'reconstruction_v34'
V36 = HERE.parent/'reconstruction_v36'


@lru_cache(maxsize=1)
def inputs():
    return (read(V34/'data.json.gz'), read(V34/'selection.json'), read(V34/'model_fit.json'),
            read(HERE.parent/'reconstruction_v35/fitted_emission.json'))


def case_path(record):
    return HERE/'reference_cases'/f'{record["root_index"]:03}_{record["eligible_call_index"]:03}.json.gz'


def run_case(record):
    preparation_tick, preparation_cpu = perf_counter(), process_time()
    data, selection, physical_fit, emission_fit = inputs()
    root_index, call_index = record['root_index'],record['eligible_call_index']
    root = selection['selected'][root_index]
    graph = context(data,root['day'])[0]
    model, parameters = setup(graph,physical_fit)
    planner = ForecastPlanner(graph,model,parameters,root['deadline'])
    trace = read(V36/'roots'/f'{root_index:03}.json.gz')['rows']['public_messages']['combined_rolling']['outcome']['trace']
    row = trace[record['trace_index']]
    node = locate_node(root_ride(graph,root),trace,record['trace_index'])
    history = restore_single_snapshot(model,parameters,emission_fit,row['public_history'],row['clock'])
    if json.loads(json.dumps(history.public_state(),default=jsonable))!=row['public_history']:
        raise AssertionError('Restoring SingleMessageHistory changed a recorded public fact')
    preparation = dict(wall=perf_counter()-preparation_tick,
        process_cpu=process_time()-preparation_cpu,
        scope='Input loading, dated graph, cold physical model/planner, node location and snapshot restoration')
    result = evaluate_onboard_actions(graph,planner,node,row['cid'],row['clock'],history,
        root_index=root_index,day=root['day'],call_index=call_index,count=128,seed_base=37050)
    result.update(census_state_id=record['state_id'],census_source='census.json',
        source_trace=record['trace_locator'],reference_preparation=preparation,
        original_alight_minus_continue_seconds=record['alight_minus_continue_seconds'],
        original_baseline_action=record['baseline_action'],
        baseline_actions_match_census=(result['actions']['base']==record['baseline_action'] and
            result['actions']['nominal_diagnostic']==record['comparisons']['nominal']['action']))
    # Persist the paid bank before any subsequent reporting or another state.
    with gzip.open(case_path(record),'wt') as handle:
        json.dump(result,handle,default=jsonable,allow_nan=False)
    if not result['baseline_actions_match_census']:
        raise AssertionError('Saved reference bank has different cheap actions from its restored census state')
    paired = result['paired']
    print(record['state_id'],'seed',result['seed'],'complete',paired['complete_pairs'],
        'interval',paired['full_bank_mean_gain_interval'],'choice',result['selected_action'],
        'cpu',round(result['cost']['total_process_cpu'],4),flush=True)
    return record['state_id']


def aggregate(census, new_banks):
    records = [r for r in census['states'] if r['two_structural_menu_actions']]
    summaries = []
    for record in records:
        path = case_path(record)
        if not path.exists():
            continue
        result = read(path)
        summaries.append(dict(state_id=record['state_id'],root_index=record['root_index'],
            root_id=record['root_id'],day=record['day'],clock=record['clock'],
            eligible_call_index=record['eligible_call_index'],source=str(path.relative_to(HERE)),
            seed=result['seed'],count=result['count'],actions=result['actions'],
            selected_action=result['selected_action'],paired=result['paired'],
            original_alight_minus_continue_seconds=record['alight_minus_continue_seconds'],
            baseline_actions_match_census=result['baseline_actions_match_census'],
            reference_preparation=result['reference_preparation'],
            diagnostic_setup_cost=result['diagnostic_setup_cost'],cost=result['cost']))
    output = dict(scope='All frozen eligible baseline public states; model-conditional finite banks, not actual gains.',
        census=census['summary'],expected_states=len(records),completed_states=len(summaries),
        newly_executed_banks_this_invocation=new_banks,
        total_unique_worlds=sum(r['count'] for r in summaries),
        total_two_action_outcomes=sum(2*r['count'] for r in summaries),
        complete_pairs=sum(r['paired']['complete_pairs'] for r in summaries),
        unresolved_pairs=sum(r['paired']['unresolved_pairs'] for r in summaries),
        selected_alternative_states=sum(r['selected_action']!=r['actions']['base'] for r in summaries),
        conditional_evaluation_process_cpu=sum(r['cost']['total_process_cpu'] for r in summaries),
        conditional_evaluation_wall=sum(r['cost']['total_wall'] for r in summaries),
        cold_preparation_process_cpu=sum(r['reference_preparation']['process_cpu'] for r in summaries),
        cold_preparation_wall=sum(r['reference_preparation']['wall'] for r in summaries),
        cheap_diagnostics_process_cpu=sum(r['diagnostic_setup_cost']['process_cpu'] for r in summaries),
        cheap_diagnostics_wall=sum(r['diagnostic_setup_cost']['wall'] for r in summaries),
        accounting='Reference preparation is separate from each saved conditional-evaluation cost. '
            'A policy hit must be charged the original conditional-evaluation CPU; this census does not '
            'make cached conditional work free.',
        interval_scope='Full-bank empirical missing-outcome bounds; complete-pair MCSE is retained inside '
            'paired.complete_pair_statistics. Neither quantity is a population regret guarantee.',
        cases=summaries)
    (HERE/'reference_summary.json').write_text(json.dumps(output,indent=2,default=jsonable,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('cases','census')}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    protocol, census = read(HERE/'protocol.json'),read(HERE/'census.json')
    if (protocol['conditional_evaluation']['worlds_per_state']!=128 or
        census['summary']['eligible_two_action_states']!=protocol['reference_panel']['eligible_states']):
        raise ValueError('Reference census and world count must match the frozen protocol')
    (HERE/'reference_cases').mkdir(exist_ok=True)
    pending = [r for r in census['states'] if r['two_structural_menu_actions'] and not case_path(r).exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_case,r) for r in pending]):
            future.result()
    aggregate(census,len(pending))


if __name__=='__main__':
    main()
