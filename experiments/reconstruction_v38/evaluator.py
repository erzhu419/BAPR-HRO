"""Complete common-world action evaluation with exact planner-cache variants.

The physical bank, frozen message snapshot, executor, action order and gate are
the V37 definitions. Only planner construction/cache reuse differs. Every
world keeps its full trace, including unresolved outcomes.
"""
from time import perf_counter, process_time

from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v34.conditional_reference import AnchoredWorld, SnapshotExecutor, summarize_pairs
from experiments.reconstruction_v34.forecast import ForecastPlanner
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot, select_action


def planner_class(variant):
    if variant in ('legacy', 'shared'):
        return ForecastPlanner
    if variant == 'fast':
        from experiments.reconstruction_v38.planner import FastPlanner
        return FastPlanner
    raise ValueError(variant)


def evaluate_actions(graph, model, parameters, deadline, node, row, history,
                     nominal, combined, *, count, seed, variant='fast'):
    """Match conditional_action_check's inputs and scientific result schema.

    Historical nominal/combined labels are retained for the V37 controller's
    alternative/base renaming. Both actions use the same conditional worlds;
    neither cached outcomes nor realized continuations enter their selection.
    """
    began, cpu = perf_counter(), process_time()
    planner_type = planner_class(variant)
    public_state = history.public_state()
    snapshot = freeze_snapshot(history, row['clock'])
    paths = node.paths() if hasattr(node, 'path') else [path for ride in node.rides.values() for path in ride.paths()]
    bank = TripWorlds(snapshot.as_belief(), parameters, paths, count, seed)
    preparation = dict(wall=perf_counter()-began, process_cpu=process_time()-cpu)
    outcomes, costs, planners = {}, {}, []
    planner = None
    for label, action in (('nominal', nominal), ('combined', combined)):
        tick, tick_cpu = perf_counter(), process_time()
        if planner is None or variant == 'legacy':
            planner = planner_type(graph, model, parameters, deadline)
            planners.append(planner)
        before = dict(planner.stats)
        values = []
        for index in range(count):
            controller = SnapshotExecutor(graph, planner, deadline, method='rolling_fixed', waiting='combined')
            result = controller.forced(node, row, action, snapshot.clone(), AnchoredWorld(bank, index, public_state))
            result['world_index'] = index
            values.append(result)
        outcomes[label] = values
        costs[label] = dict(wall=perf_counter()-tick, process_cpu=process_time()-tick_cpu,
            planner_stats={key:value-before.get(key,0) for key,value in planner.stats.items()})
    tick, tick_cpu = perf_counter(), process_time()
    paired = summarize_pairs([r['loss'] for r in outcomes['nominal']],
        [r['loss'] for r in outcomes['combined']], deadline-row['clock'])
    paired['orientation'] = 'nominal action loss minus combined action loss; negative favors nominal'
    selected = select_action(nominal, combined, paired)
    decision = dict(wall=perf_counter()-tick, process_cpu=process_time()-tick_cpu)
    keys = set().union(*(planner.stats for planner in planners))
    totals = {key:sum(planner.stats.get(key,0) for planner in planners) for key in sorted(keys)}
    cache_sizes = [{name:len(getattr(planner,name,{})) for name in
        ('_rides','_platforms','_first_offer','_edges','_kernels','_departures')} for planner in planners]
    return dict(variant=variant, seed=seed, count=count, public_history=public_state, future_messages=0,
        continuation='combined_rolling with currently received messages frozen and new local observations',
        actions=dict(nominal=nominal, combined=combined), paired=paired, rows=outcomes,
        selected_action=selected, cost=dict(preparation=preparation, action_continuations=costs,
            decision=decision, planner_instances=len(planners), planner_stats_total=totals,
            planner_cache_sizes=cache_sizes, total_wall=perf_counter()-began,
            total_process_cpu=process_time()-cpu))
