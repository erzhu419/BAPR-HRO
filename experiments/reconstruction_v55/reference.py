"""Same causal policy targets, with goal dominance and terminal subset sums."""
from time import process_time, perf_counter

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v33.execute import Executor
from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine
from experiments.reconstruction_v53.reference import CompressedReference
from experiments.reconstruction_v55.semantic import BarrierPlanner, BarrierReadSets, GoalBarrierViews
from experiments.reconstruction_v55.terminal import terminal_platform


class StructuralReference(CompressedReference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.views = GoalBarrierViews(self.model, self.snapshot)
        self.planner = BarrierPlanner(self.graph, self.model, self.parameters,
                                      self.deadline, views=self.views)
        self.engine = PhysicalDecisionEngine(self.planner, variant='frontier')
        self.policy = Executor(self.graph, self.planner, self.deadline,
                               method='rolling_fixed', waiting='combined', trace=False)
        self.reads = (None if self.observations is None else
                      BarrierReadSets(self.model, self.observations, self.views))
        self.stats.update(terminal_dp_calls=0, terminal_dp_admitted=0,
                          terminal_dp_complete=0, terminal_dp_states=0,
                          terminal_dp_budget_stops=0, terminal_dp_hits=0)

    def _prepare(self, node):
        if self.reads is None:
            self.reads = BarrierReadSets(self.model, self.observations, self.views)
        super()._prepare(node)

    def _platform(self, platform, entry, history, *, depth, after=float('-inf')):
        try:
            value = terminal_platform(self, platform, entry, history, depth, after)
        except UnsupportedHistory as error:
            return self._unknown(entry, str(error))
        if value is None:
            return super()._platform(platform, entry, history, depth=depth, after=after)
        return self._intersect(value, self._lower(platform, history, entry))

    def solve(self, node, row, *, mode='optimal'):
        result = super().solve(node, row, mode=mode)
        result.update(implementation='structural_conditional', semantic=dict(self.views.stats),
            cost=dict(process_cpu=process_time()-self.cpu_start,
                      wall=perf_counter()-self.wall_start),
            scope='Same V53 causal targets and original budgets. Standard goal-dominance scoring reduction and exact eligible terminal-platform subset integration. No new control or independent method novelty claimed.')
        return result
