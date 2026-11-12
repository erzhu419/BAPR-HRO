"""Paired continuation evaluation with a measured process-CPU allowance.

The stopping boundary is a complete two-action world. A late pair is retained
and charged, but cannot change the last action available before the boundary.
This cooperative limit is not a hard real-time guarantee or a confidence rule.
"""

from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v34.conditional_reference import (
    AnchoredWorld, SnapshotExecutor as OriginalSnapshotExecutor, summarize_pairs,
)
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot, select_action
from experiments.reconstruction_v38.planner import FastPlanner
from experiments.reconstruction_v39.executor import SnapshotExecutor
from experiments.reconstruction_v39.state import DecisionEngine


class CurrentDecisionEngine(DecisionEngine):
    """Original menu, with the existing planner's current-likelihood state.

    A hit returns only this action. The caller retains every raw prediction and
    hard receipt, so later forecast reselection still builds its proper key.
    """

    def __init__(self, planner):
        super().__init__(planner, variant='fullmemo')

    def _history(self, kind, node, history):
        state = history.signature(self.planner.relevant_trips(node))
        units = sum(len(hard)+len(forecasts)+(notice is not None)+len(statuses)+len(services)
                    for _, hard, forecasts, notice, statuses, services in state)
        self.stats['key_trips'] += len(state)
        self.stats['key_record_units'] += units
        self.stats[f'{kind}_key_trips'] += len(state)
        self.stats[f'{kind}_key_record_units'] += units
        self.stats[f'{kind}_requests'] += 1
        return state


def block_seed(base_seed, block_index):
    return int(base_seed) if block_index == 0 else int(np.random.SeedSequence(
        [int(base_seed), 40040, block_index]).generate_state(1)[0])


def empty_pairs():
    return dict(orientation='nominal action loss minus combined action loss; negative favors nominal',
        worlds=0, complete_pairs=0, unresolved_pairs=0,
        complete_pair_mean_subject_loss=None, complete_pair_mean_baseline_loss=None,
        complete_pair_statistics=None, full_bank_mean_gain_interval=None,
        interval_scope='No completed prefix was available within the CPU allowance.')


def evaluate_actions(graph, model, parameters, deadline, node, row, history,
                     nominal, combined, *, seed, max_count=512, cpu_budget=None,
                     started_cpu=None, variant='frontier'):
    """Run nested 128-world blocks, committing only timely paired prefixes.

    ``started_cpu`` may precede cold model/input construction. All subsequent
    sampling, decisions, traces, prefix bookkeeping and the final summary are
    charged. Future publications and previously saved answers are not inputs.
    The engine and its waiting rule are fixed throughout this call.
    """
    began_wall, began_cpu = perf_counter(), process_time()
    charged_start = began_cpu if started_cpu is None else started_cpu
    if variant not in ('v38', 'currentmemo', 'frontier'):
        raise ValueError(variant)
    if max_count not in (128, 512):
        raise ValueError('The frozen evaluation allows 128 or 512 worlds')
    if cpu_budget is not None and cpu_budget < 0:
        raise ValueError('CPU allowance must be nonnegative')

    def elapsed():
        return process_time()-charged_start

    def expired():
        return cpu_budget is not None and elapsed() >= cpu_budget

    public_state = history.public_state()
    rows = dict(nominal=[], combined=[])
    costs = {label: dict(wall=0., process_cpu=0.) for label in rows}
    blocks, prefixes = [], []
    committed = 0
    low = high = 0.
    selected = combined
    ready_cpu = began_cpu-charged_start
    planner = engine = snapshot = None
    preparation = dict(wall=0., process_cpu=0.)
    remaining = deadline-row['clock']
    stop_reason = 'max_count'
    if expired():
        stop_reason = 'initial_budget_exhausted'
    else:
        tick, cpu = perf_counter(), process_time()
        snapshot = freeze_snapshot(history, row['clock'])
        paths = (node.paths() if hasattr(node, 'path') else
                 [path for ride in node.rides.values() for path in ride.paths()])
        planner = FastPlanner(graph, model, parameters, deadline)
        if variant == 'currentmemo':
            engine = CurrentDecisionEngine(planner)
        elif variant == 'frontier':
            engine = DecisionEngine(planner, variant='frontier')
        preparation = dict(wall=perf_counter()-tick, process_cpu=process_time()-cpu)
        for block_index in range(max_count//128):
            if expired():
                stop_reason = 'budget_before_block'
                break
            tick, cpu = perf_counter(), process_time()
            current_seed = block_seed(seed, block_index)
            bank = TripWorlds(snapshot.as_belief(), parameters, paths, 128, current_seed)
            blocks.append(dict(block_index=block_index, seed=current_seed, count=128,
                first_world_index=128*block_index, wall=perf_counter()-tick,
                process_cpu=process_time()-cpu))
            if expired():
                stop_reason = 'sampling_budget_exhausted'
                break
            late = False
            for index in range(128):
                if expired():
                    stop_reason = 'budget_before_pair'
                    late = True
                    break
                for label, action in (('nominal', nominal), ('combined', combined)):
                    tick, cpu = perf_counter(), process_time()
                    controller = (OriginalSnapshotExecutor(graph, planner, deadline,
                        method='rolling_fixed', waiting='combined') if engine is None else
                        SnapshotExecutor(graph, planner, deadline, engine=engine,
                            method='rolling_fixed', waiting='combined'))
                    result = controller.forced(node, row, action, snapshot.clone(),
                        AnchoredWorld(bank, index, public_state))
                    result.update(world_index=128*block_index+index, block_index=block_index,
                                  index_in_block=index, committed=False)
                    rows[label].append(result)
                    costs[label]['wall'] += perf_counter()-tick
                    costs[label]['process_cpu'] += process_time()-cpu
                a, b = rows['nominal'][-1]['loss'], rows['combined'][-1]['loss']
                low += (0. if a is None else a)-(remaining if b is None else b)
                high += (remaining if a is None else a)-(0. if b is None else b)
                count = len(rows['nominal'])
                interval = [low/count, high/count]
                candidate = select_action(nominal, combined, dict(full_bank_mean_gain_interval=interval))
                prefix = dict(count=count, full_bank_mean_gain_interval=interval,
                              selected_action=candidate)
                prefixes.append(prefix)
                elapsed_cpu = elapsed()
                timely = cpu_budget is None or elapsed_cpu <= cpu_budget
                prefix.update(elapsed_cpu=elapsed_cpu, elapsed_wall=perf_counter()-began_wall,
                              committed=timely)
                if timely:
                    committed, selected, ready_cpu = count, candidate, elapsed_cpu
                    rows['nominal'][-1]['committed'] = rows['combined'][-1]['committed'] = True
                else:
                    stop_reason, late = 'late_pair', True
                    break
            if late:
                break
    tick, cpu = perf_counter(), process_time()
    paired = (summarize_pairs([r['loss'] for r in rows['nominal'][:committed]],
        [r['loss'] for r in rows['combined'][:committed]], remaining) if committed else empty_pairs())
    paired['orientation'] = 'nominal action loss minus combined action loss; negative favors nominal'
    summary_cost = dict(wall=perf_counter()-tick, process_cpu=process_time()-cpu)
    result = dict(variant=variant, seed=int(seed), count=committed, max_count=max_count,
        executed_count=len(rows['nominal']), committed_count=committed,
        late_pair_count=len(rows['nominal'])-committed,
        cpu_budget=cpu_budget, stop_reason=stop_reason, fallback=committed == 0,
        selected_action=selected, answer_available_process_cpu=ready_cpu,
        public_history=public_state, future_messages=0,
        continuation='combined_rolling with currently received messages frozen and new local observations',
        actions=dict(nominal=nominal, combined=combined), paired=paired, rows=rows,
        blocks=blocks, prefixes=prefixes,
        cost=dict(pre_evaluation_process_cpu=began_cpu-charged_start,
            preparation=preparation, action_continuations=costs, summary=summary_cost,
            sampling_process_cpu=sum(b['process_cpu'] for b in blocks),
            sampling_wall=sum(b['wall'] for b in blocks),
            planner_stats_total={} if planner is None else dict(planner.stats),
            engine_stats_total={} if engine is None else dict(engine.stats)))
    total_cpu = elapsed()
    result['cost'].update(total_process_cpu=total_cpu, total_wall=perf_counter()-began_wall,
        cpu_overrun=0. if cpu_budget is None else max(0., total_cpu-cpu_budget))
    result['budget_exceeded'] = cpu_budget is not None and total_cpu > cpu_budget
    return result
