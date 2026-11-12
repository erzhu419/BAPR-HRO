"""Standard conditional rollout only when two current cheap actions disagree.

This diagnostic controller holds received messages fixed inside each rollout.
Its finite-bank interval is not a population guarantee or an optimality claim.
"""

from time import perf_counter, process_time

from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v33.execute import Executor as CheapExecutor
from experiments.reconstruction_v34.conditional_reference import (
    AnchoredWorld, SnapshotExecutor, summarize_pairs,
)
from experiments.reconstruction_v34.execute import Executor
from experiments.reconstruction_v34.forecast import ForecastPlanner


def select_action(nominal, combined, paired):
    """Select nominal only if its full-bank loss difference is strictly negative."""
    return nominal if paired['full_bank_mean_gain_interval'][1] < 0. else combined


def freeze_snapshot(history, clock):
    """Keep the fitted conditioner type, without retaining future publications.

    Clear compiled trip laws so snapshot conditioning is charged again, just
    as in the previous public-state reconstruction. The received facts and
    fixed emission parameters are unchanged.
    """
    snapshot = history.clone()
    snapshot.frames, snapshot.frame_index = (), 0
    snapshot.message_clock = float(clock)
    snapshot._laws = {}
    return snapshot


def conditional_action_check(graph, model, parameters, deadline, node, row, history,
                             nominal, combined, *, count, seed):
    """The evaluator has no actual-world argument, future frame stream or labels."""
    began, cpu = perf_counter(), process_time()
    public_state = history.public_state()
    snapshot = freeze_snapshot(history, row['clock'])
    paths = node.paths() if hasattr(node, 'path') else [path for ride in node.rides.values() for path in ride.paths()]
    bank = TripWorlds(snapshot.as_belief(), parameters, paths, count, seed)
    preparation = dict(wall=perf_counter()-began, process_cpu=process_time()-cpu)
    outcomes, costs = {}, {}
    for label, action in (('nominal', nominal), ('combined', combined)):
        tick, tick_cpu = perf_counter(), process_time()
        planner = ForecastPlanner(graph, model, parameters, deadline)
        values = []
        for index in range(count):
            controller = SnapshotExecutor(graph, planner, deadline, method='rolling_fixed', waiting='combined')
            result = controller.forced(node, row, action, snapshot.clone(), AnchoredWorld(bank, index, public_state))
            result['world_index'] = index
            values.append(result)
        outcomes[label] = values
        costs[label] = dict(wall=perf_counter()-tick, process_cpu=process_time()-tick_cpu,
                            planner_stats=dict(planner.stats))
    paired = summarize_pairs([r['loss'] for r in outcomes['nominal']],
        [r['loss'] for r in outcomes['combined']], deadline-row['clock'])
    paired['orientation'] = 'nominal action loss minus combined action loss; negative favors nominal'
    return dict(seed=seed, count=count, public_history=public_state, future_messages=0,
        continuation='combined_rolling with currently received messages frozen and new local observations',
        actions=dict(nominal=nominal, combined=combined), paired=paired, rows=outcomes,
        selected_action=select_action(nominal, combined, paired),
        cost=dict(preparation=preparation, action_continuations=costs,
                  total_wall=perf_counter()-began, total_process_cpu=process_time()-cpu))


class GatedExecutor(Executor):
    def __init__(self, graph, planner, deadline, *, root_index, count=128, seed_base=34050, trace=True):
        super().__init__(graph, planner, deadline, method='rolling_fixed', waiting='combined', trace=trace)
        self.root_index, self.count, self.seed_base = root_index, count, seed_base
        self.gates = []
        self.nominal_controller = CheapExecutor(graph, planner, deadline, method='nominal', waiting='nominal', trace=False)

    def gate(self, node, row, history, nominal, combined, receipts):
        if nominal == combined:
            return combined, dict(method='standard_conditional_rollout', triggered=False,
                                  nominal_action=nominal, combined_action=combined, cheap_receipts=receipts)
        call = len(self.gates)
        seed = self.seed_base+1000*self.root_index+call
        result = conditional_action_check(self.graph, self.planner.model, self.planner.parameters,
            self.deadline, node, row, history, nominal, combined, count=self.count, seed=seed)
        result.update(gate_call_index=call, trigger=dict(row), cheap_receipts=receipts,
                      interval_scope='Empirical finite-bank missing-outcome bounds, not a population certificate')
        self.gates.append(result)
        return result['selected_action'], dict(method='standard_conditional_rollout', triggered=True,
            gate_call_index=call, seed=seed, nominal_action=nominal, combined_action=combined,
            selected_action=result['selected_action'], paired=result['paired'], cost=result['cost'],
            cheap_receipts=receipts)

    def choose_platform(self, platform, entry, ready, history, visible, observed, clock):
        combined, combined_receipt = super().choose_platform(platform, entry, ready, history, visible, observed, clock)
        reference = self.planner.platform_reference(platform, history, entry, ready, 'nominal')
        nominal, nominal_receipt = self.nominal_controller.choose(reference, visible, tuple(observed), clock)
        row = dict(kind='platform', stop=platform.stop, entry=entry, ready=ready,
                   clock=clock, visible=tuple(visible), observed=dict(observed))
        return self.gate(platform, row, history, nominal, combined,
                         dict(nominal=nominal_receipt, combined=combined_receipt))

    def choose_onboard(self, ride, cid, clock, history):
        combined, combined_receipt = super().choose_onboard(ride, cid, clock, history)
        nominal, nominal_receipt = self.planner.compile_ride(ride, history, 'nominal').decision(cid, clock)
        action, receipt = self.gate(ride, dict(kind='onboard', cid=cid, clock=clock), history,
            'alight' if nominal else 'continue', 'alight' if combined else 'continue',
            dict(nominal=nominal_receipt, combined=combined_receipt))
        return action == 'alight', receipt
