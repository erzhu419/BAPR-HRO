"""Retain V46's ordinary bulk terminal-platform DP inside the resumable DAG.

WAIT reference scores, observed BOARD scores and true P0 terminal marks remain
separate. An unresolved BOARD comparison retains its complete physical range.
"""
import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v46.oracle import FullPolicyOracle, point, unknown, structural, terminal
from experiments.reconstruction_v47.incremental import IncrementalOracle, ValueNode


class TerminalDPNode(ValueNode):
    """Scoring-task dependencies with the original bounded event-DP reducer."""
    def __init__(self, entry, deadline, reducer, dependencies):
        super().__init__(entry, deadline, expanded=True, children=[(0., n) for n in dependencies])
        self.reducer = reducer

    def value(self, memo):
        if self not in memo:
            result = self.reducer()
            result[0] = max(self.lower, result[0])
            result[1] = min(self.upper, result[1])
            if result[0] > result[1]+1e-8:
                raise ValueError('Terminal DP violates its physical bounds')
            if result[0] > result[1]:
                result[:2] = (result[0]+result[1])/2
            memo[self] = result
        return memo[self]


class HybridIncrementalOracle(IncrementalOracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._terminal_board_scores = {}
        self.stats.update(hybrid_terminal_platforms=0, hybrid_interval_reductions=0,
                          hybrid_board_dependencies=0)

    def forced(self, *args, **kwargs):
        result = super().forced(*args, **kwargs)
        result['primitive_unit'] = ('One platform build, WAIT reference, observed BOARD query, '
                                    'original platform policy query, or original onboard query')
        result['kernel'] = 'Retained event DAG with V46 bulk independent terminal subplatforms'
        return result

    def _board_task(self, ride, departure, history):
        h = history.clone()
        h.observe(ride.first_cid, 'departure', departure)
        key = ('hybrid_board', self._key(ride, h), float(departure))
        if key in self.nodes:
            self.stats['retained_node_hits'] += 1
            return key, self.nodes[key]

        def build():
            try:
                value = self.planner.compile_ride(ride, h, 'rolling_fixed').score(departure)
            except UnsupportedHistory:
                self.stats['unsupported_scores'] += 1
                value = None
            self._terminal_board_scores[key] = value
            # This task has no physical payoff. The reducer reads only its
            # resolved BOARD field; zero-weight dependencies are not mixtures.
            return point(self.deadline), []

        return key, self._new(key, departure, build=build, kind='board_compiles')

    def _platform_node(self, platform, entry, history):
        if not (all(terminal(ride) for ride in platform.rides.values())
                and structural(self.model, platform)['eligible']):
            return super()._platform_node(platform, entry, history)
        self._prepare(platform)
        context = self._key(platform, history), float(entry)
        key = ('platform', context)
        if key in self.nodes:
            self.stats['retained_node_hits'] += 1
            return self.nodes[key]
        if entry >= self.deadline:
            return self._point(key, self.deadline)

        def build_platform():
            self.stats['hybrid_terminal_platforms'] += 1
            ready = entry+self.graph.stops[platform.stop].min_transfer_time

            def compile_wait():
                try:
                    reference = self.planner.platform_reference(platform, history, entry, ready, 'rolling_fixed')
                except UnsupportedHistory:
                    self.stats['unsupported_scores'] += 1
                    return unknown(entry, self.deadline), []
                reducer, dependencies = self._terminal_reducer(platform, entry, ready, history, reference)
                dp_key = ('terminal_dp', context)
                dp = TerminalDPNode(float(entry), self.deadline, reducer, dependencies)
                self.nodes[dp_key] = dp
                self.stats['dag_nodes'] += 1
                return None, [(1., dp)]

            wait = self._new(('terminal_wait', context), entry, build=compile_wait, kind='wait_compiles')
            return None, [(1., wait)]

        return self._new(key, entry, build=build_platform, kind='platform_solves')

    def _terminal_reducer(self, platform, entry, ready, history, reference):
        cids = tuple(reference.cids)
        physical = {cid:self.root_law.departure_distribution(cid, history) for cid in cids}
        times = np.array(sorted({float(t) for source in (reference.distributions, physical)
                                for cid in cids for t, p in zip(*source[cid][:2], strict=True)
                                if p > 0 and entry <= t < self.deadline}))
        nt, nc = len(times), len(cids)
        hazards = []
        for source in (reference.distributions, physical):
            hazard = np.zeros((nt, nc))
            for i, cid in enumerate(cids):
                t, p, cancel = source[cid]
                tails = np.r_[np.cumsum(p[::-1])[::-1], 0.]
                survival = cancel+p[t < entry].sum()+tails[np.searchsorted(t, times, side='left')]
                selected = (p > 0) & (t >= entry) & (t < self.deadline)
                np.add.at(hazard[:, i], np.searchsorted(times, t[selected]), p[selected])
                np.divide(hazard[:, i], survival, out=hazard[:, i], where=survival > 0)
            hazards.append(hazard)
        working_hazard, physical_hazard = hazards
        wait_marks = np.full((nt, nc), self.deadline)
        truth = np.full((nt, nc), self.deadline)
        single = np.full((nt, nc), self.deadline)
        board_keys = {}
        dependencies = []
        for i, cid in enumerate(cids):
            clocks, p, marks = reference.marks[cid]
            wait_marks[np.searchsorted(times, clocks), i] = marks
            gains = np.r_[np.cumsum((p*(self.deadline-marks))[::-1])[::-1], 0.]
            root_times, root_mass, cancel = reference.distributions[cid]
            tails = np.r_[np.cumsum(root_mass[::-1])[::-1], 0.]
            survival = cancel+root_mass[root_times < entry].sum()+tails[np.searchsorted(root_times, times, side='right')]
            gain = gains[np.searchsorted(clocks, times, side='right')]
            single[:, i] -= np.divide(gain, survival, out=np.zeros_like(gain), where=survival > 0)
            ride = platform.rides[cid]
            # terminal() guarantees no actual on-board decision before the
            # first destination. The unchanged old backward pass is arithmetic.
            truth_times, true_values = FullPolicyOracle.ride_marks(self, ride, history)
            if np.any(true_values[:, 2] > 0):
                raise ValueError('Terminal physical marks unexpectedly incomplete')
            trip, slot = self.model._positions[cid, 'departure']
            operating = ~self.root_law.law(trip, history).canceled[slot]
            physical_times, mass, _ = physical[cid]
            for departure in physical_times[(mass > 0) & (physical_times >= ready) & (physical_times < self.deadline)]:
                j = int(np.searchsorted(times, departure))
                selected = np.flatnonzero((truth_times == departure) & operating)
                if len(selected) != 1:
                    raise ValueError('Terminal departure has no unique physical mark')
                truth[j, i] = true_values[selected[0], 0]
                board_key, task = self._board_task(ride, float(departure), history)
                board_keys[j, i] = board_key
                dependencies.append(task)
                self.stats['hybrid_board_dependencies'] += 1
        ns = 1 << nc
        bits = [[i for i in range(nc) if mask & (1 << i)] for mask in range(ns)]
        subsets = []
        for active in bits:
            choices = np.array([0], dtype=int)
            for i in active:
                choices = np.r_[choices, choices | (1 << i)]
            subsets.append(choices)
        ranks = sorted(range(nc), key=lambda i:reference.ranks[cids[i]])
        first = np.full((nt+1, ns), self.deadline)
        rows = {}
        for j in range(nt-1, -1, -1):
            for mask in range(ns):
                wp, pp = np.ones(1), np.ones(1)
                for i in bits[mask]:
                    a, b = working_hazard[j, i], physical_hazard[j, i]
                    wp, pp = np.r_[wp*(1-a), wp*a], np.r_[pp*(1-b), pp*b]
                departed = subsets[mask]
                remaining = mask ^ departed
                first_cost = first[j+1, remaining].copy()
                waiting = np.full(len(departed), self.deadline)
                for k, seen in enumerate(departed):
                    if times[j] < ready or not seen:
                        continue
                    first_rank = next(i for i in ranks if seen & (1 << i))
                    first_cost[k] = wait_marks[j, first_rank]
                    target = min((single[j, i] for i in bits[remaining[k]]), default=self.deadline)
                    waiting[k] = min(target, first[j+1, remaining[k]])
                first[j, mask] = wp @ first_cost
                rows[j, mask] = pp, departed, remaining, waiting

        def reduce():
            self.stats['hybrid_interval_reductions'] += 1
            board = np.full((nt, nc), np.nan)
            for (j, i), board_key in board_keys.items():
                value = self._terminal_board_scores.get(board_key)
                if value is not None:
                    board[j, i] = value
            values = np.tile(point(self.deadline), (nt+1, ns, 1))
            for j in range(nt-1, -1, -1):
                for mask in range(ns):
                    pp, departed, remaining, waiting = rows[j, mask]
                    local = values[j+1, remaining].copy()
                    for k, seen in enumerate(departed):
                        if times[j] < ready or not seen or pp[k] <= 0:
                            continue
                        visible = bits[seen]
                        if any(not np.isfinite(board[j, i]) for i in visible):
                            local[k] = unknown(times[j], self.deadline)
                            continue
                        best = min(visible, key=lambda i:(board[j, i], reference.ranks[cids[i]]))
                        if board[j, best] <= waiting[k]+1e-9:
                            local[k] = point(truth[j, best])
                    values[j, mask] = pp @ local
            return values[0, -1].copy()

        return reduce, dependencies
