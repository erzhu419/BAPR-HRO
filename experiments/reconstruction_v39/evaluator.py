"""Same fixed-bank executable estimator, with public-decision memo variants."""
from time import perf_counter, process_time

from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v34.conditional_reference import (
    AnchoredWorld, SnapshotExecutor as OriginalSnapshotExecutor, summarize_pairs,
)
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot, select_action
from experiments.reconstruction_v38.planner import FastPlanner
from experiments.reconstruction_v39.executor import SnapshotExecutor


def evaluate_actions(graph, model, parameters, deadline, node, row, history,
                     nominal, combined, *, count, seed, variant='frontier'):
    """Generate the entire original bank and execute both actions in order."""
    began, cpu = perf_counter(), process_time()
    if variant not in ('v38','fullmemo','frontier'):
        raise ValueError(variant)
    public_state = history.public_state()
    snapshot = freeze_snapshot(history,row['clock'])
    paths = node.paths() if hasattr(node,'path') else [path for ride in node.rides.values() for path in ride.paths()]
    bank = TripWorlds(snapshot.as_belief(),parameters,paths,count,seed)
    preparation = dict(wall=perf_counter()-began,process_cpu=process_time()-cpu)
    outcomes,costs = {},{}
    planner,engine = None,None
    for label,action in (('nominal',nominal),('combined',combined)):
        tick,tick_cpu = perf_counter(),process_time()
        if planner is None:
            planner = FastPlanner(graph,model,parameters,deadline)
            if variant!='v38':
                from experiments.reconstruction_v39.state import DecisionEngine
                engine = DecisionEngine(planner,variant=variant)
        prior_planner = dict(planner.stats)
        prior_engine = {} if engine is None else dict(engine.stats)
        values = []
        for index in range(count):
            if engine is None:
                controller = OriginalSnapshotExecutor(graph,planner,deadline,method='rolling_fixed',waiting='combined')
            else:
                controller = SnapshotExecutor(graph,planner,deadline,engine=engine,
                    method='rolling_fixed',waiting='combined')
            result = controller.forced(node,row,action,snapshot.clone(),AnchoredWorld(bank,index,public_state))
            result['world_index']=index
            values.append(result)
        outcomes[label]=values
        costs[label]=dict(wall=perf_counter()-tick,process_cpu=process_time()-tick_cpu,
            planner_stats={key:value-prior_planner.get(key,0) for key,value in planner.stats.items()},
            engine_stats={} if engine is None else
                {key:value-prior_engine.get(key,0) for key,value in engine.stats.items()})
    tick,tick_cpu=perf_counter(),process_time()
    paired=summarize_pairs([r['loss'] for r in outcomes['nominal']],
        [r['loss'] for r in outcomes['combined']],deadline-row['clock'])
    paired['orientation']='nominal action loss minus combined action loss; negative favors nominal'
    selected=select_action(nominal,combined,paired)
    decision=dict(wall=perf_counter()-tick,process_cpu=process_time()-tick_cpu)
    cache_sizes=[{name:len(getattr(planner,name)) for name in
        ('_rides','_platforms','_first_offer','_edges','_kernels','_departures')}]
    return dict(variant=variant,seed=seed,count=count,public_history=public_state,future_messages=0,
        continuation='combined_rolling with currently received messages frozen and new local observations',
        actions=dict(nominal=nominal,combined=combined),paired=paired,rows=outcomes,selected_action=selected,
        cost=dict(preparation=preparation,action_continuations=costs,decision=decision,planner_instances=1,
            planner_stats_total=dict(planner.stats),planner_cache_sizes=cache_sizes,
            engine_stats_total={} if engine is None else dict(engine.stats),
            total_wall=perf_counter()-began,total_process_cpu=process_time()-cpu))
