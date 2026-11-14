"""Incremental maintenance of the unchanged V47 bulk interval recurrence.

The event DP keeps its original physical probabilities, branch comparisons,
row summation order and task dependencies. Only ancestors of changed inputs
are reduced again; this does not choose, skip or charge physical work.
"""
from collections import defaultdict
import heapq

import numpy as np

from experiments.reconstruction_v46.oracle import point, unknown
from experiments.reconstruction_v48.lazy_frontier import LazyFrontierOracle


class DependencyMemo(dict):
    """Persistent value memo; all child edges are invalidation dependencies."""

    def __init__(self, stats):
        super().__init__()
        self.parents = defaultdict(set)
        self.stats = stats

    def __setitem__(self, node, value):
        # Bulk child edges have zero physical mass but still affect their DP.
        for _, child in node.children:
            self.parents[child].add(node)
        super().__setitem__(node, value)
        self.stats['bulk_value_cache_stores'] += 1

    def invalidate(self, node):
        pending, seen = [node], set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in self:
                del self[current]
                self.stats['bulk_value_cache_invalidations'] += 1
            pending.extend(self.parents.get(current, ()))


class IncrementalBulkReducer:
    """Reverse dependency updates for the original (time, active-mask) DP."""

    def __init__(self, oracle, times, ready, cids, ranks, bits, rows,
                 truth, board_keys):
        self.oracle, self.times, self.ready = oracle, times, ready
        self.cids, self.ranks, self.bits = cids, ranks, bits
        self.rows, self.truth, self.board_keys = rows, truth, board_keys
        self.nt, self.ns = len(times), len(bits)
        self.values = np.tile(point(oracle.deadline), (self.nt+1, self.ns, 1))
        self.board = {key: None for key in board_keys}
        self.board_rows, self.parents = defaultdict(set), defaultdict(set)
        self.truth_rows, self.truth_values = defaultdict(set), {}
        self.initialized = False
        for (j, mask), (pp, departed, remaining, waiting) in rows.items():
            for k, seen in enumerate(departed):
                if pp[k] <= 0:
                    continue
                # This conservative edge can be unused after a BOARD action.
                # Retaining it causes extra updates, never omitted updates.
                self.parents[j+1, int(remaining[k])].add((j, mask))
                if times[j] >= ready and seen:
                    for i in bits[seen]:
                        self.board_rows[j, i].add((j, mask))
        self.stats = oracle.stats

    def _physical(self, key):
        node = self.truth[key]
        cache = self.oracle._bulk_value_cache
        if node in cache:
            self.stats['bulk_value_cache_hits'] += 1
        return node.value(cache)

    def __call__(self, memo):
        self.stats['bulk_interval_reductions'] += 1
        self.stats['bulk_full_rows_equivalent'] += len(self.rows)
        dirty = set() if self.initialized else set(self.rows)
        for key, board_key in self.board_keys.items():
            value = self.oracle._terminal_board_scores.get(board_key)
            if value != self.board[key]:
                self.board[key] = value
                dirty.update(self.board_rows[key])
                self.stats['bulk_changed_board_inputs'] += 1
        # Read only true continuations that some resolved policy branch uses.
        for key, previous in list(self.truth_values.items()):
            value = self._physical(key)
            if not np.array_equal(value, previous):
                self.truth_values[key] = value.copy()
                dirty.update(self.truth_rows[key])
                self.stats['bulk_changed_truth_inputs'] += 1
        queue = [(-j, mask) for j, mask in dirty]
        heapq.heapify(queue)
        queued = set(dirty)
        self.stats['bulk_initial_dirty_rows'] += len(dirty)
        while queue:
            negative_j, mask = heapq.heappop(queue)
            j = -negative_j
            queued.remove((j, mask))
            pp, departed, remaining, waiting = self.rows[j, mask]
            local = self.values[j+1, remaining].copy()
            for k, seen in enumerate(departed):
                if self.times[j] < self.ready or not seen or pp[k] <= 0:
                    continue
                visible = self.bits[seen]
                if any(self.board[j, i] is None or not np.isfinite(self.board[j, i])
                       for i in visible):
                    local[k] = unknown(self.times[j], self.oracle.deadline)
                    continue
                best = min(visible, key=lambda i:(self.board[j, i], self.ranks[self.cids[i]]))
                if self.board[j, best] <= waiting[k]+1e-9:
                    key = (j, best)
                    value = self._physical(key)
                    self.truth_values[key] = value.copy()
                    self.truth_rows[key].add((j, mask))
                    local[k] = value
            # Identical original operation and outcome ordering; no deltas
            # are added to a cached sum, so rounding does not accumulate.
            value = pp @ local
            self.stats['bulk_recomputed_rows'] += 1
            self.stats['bulk_recomputed_outcomes'] += len(pp)
            if not np.array_equal(value, self.values[j, mask]):
                self.values[j, mask] = value
                for parent in self.parents[j, mask]:
                    if parent not in queued:
                        queued.add(parent)
                        heapq.heappush(queue, (-parent[0], parent[1]))
                        self.stats['bulk_propagated_rows'] += 1
        self.initialized = True
        return self.values[0, -1].copy()


class IncrementalBulkOracle(LazyFrontierOracle):
    """V48 schedule and certificates, with retained bulk arithmetic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats.update(bulk_value_cache_stores=0, bulk_value_cache_invalidations=0,
            bulk_value_cache_hits=0, bulk_full_rows_equivalent=0,
            bulk_recomputed_rows=0, bulk_recomputed_outcomes=0,
            bulk_changed_board_inputs=0, bulk_changed_truth_inputs=0,
            bulk_initial_dirty_rows=0, bulk_propagated_rows=0)
        self._bulk_value_cache = DependencyMemo(self.stats)

    def _expand(self, node):
        if not node.expanded and node.build is not None:
            self._bulk_value_cache.invalidate(node)
        return super()._expand(node)

    def _bulk_reducer(self, platform, entry, ready, history, reference):
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
        truth = {}
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
            trip, slot = self.model._positions[cid, 'departure']
            law = self.root_law.law(trip, history)
            operating = ~law.canceled[slot]
            physical_times, mass, _ = physical[cid]
            candidate_boards = []
            for departure in physical_times[(mass > 0) & (physical_times >= ready) & (physical_times < self.deadline)]:
                j = int(np.searchsorted(times, departure))
                selected = np.flatnonzero((law.times[slot] == departure) & operating)
                if len(selected) != 1:
                    raise ValueError('Bulk departure has no unique physical continuation row')
                # These are physical value dependencies, never policy scores.
                truth[j, i] = self._ride(ride, history, slot, int(selected[0]))
                dependencies.append(truth[j, i])
                self.stats['bulk_ride_dependencies'] += 1
                board_key, task = self._board_task(ride, float(departure), history)
                board_keys[j, i] = board_key
                candidate_boards.append(task)
                self.stats['bulk_board_dependencies'] += 1
            # Retain the original candidate-level ordering: its physical
            # continuation work precedes its observed BOARD queries.
            dependencies.extend(candidate_boards)
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

        reducer = IncrementalBulkReducer(self, times, ready, cids, reference.ranks, bits, rows, truth, board_keys)
        return reducer, dependencies
