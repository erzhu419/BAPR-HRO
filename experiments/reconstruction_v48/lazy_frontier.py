"""Demand-driven bulk-width maintenance for the same V48 priority rule.

Dirty cached widths are upper priority bounds under mathematical refinement.
Only a dirty heap leader is reduced, then republished before selecting work.
Root weights, physical reducers, tie-breaking, budgets and certificates are
unchanged. Observed floating width increases are recorded, never clipped.
"""
from time import process_time

from experiments.reconstruction_v47.incremental import IncrementalOracle
from experiments.reconstruction_v48.frontier import FrontierOracle, RootFrontier, OPAQUE


class LazyRootFrontier(RootFrontier):
    def sync(self):
        start = process_time()
        try:
            if not self._discover():
                return False
            while self._event_cursor < len(self.oracle._expansion_events):
                node = self.oracle._expansion_events[self._event_cursor]
                self._event_cursor += 1
                watching = self._watchers.get(node)
                if watching:
                    self._dirty_blocks.update(watching)
                    for _, child in node.children:
                        self._watch(child, watching)
                if node in self.seen and node not in self.edges:
                    self._enqueue(node)
                    if not self._discover():
                        return False
            self._propagate()
            for node in self._refresh:
                if isinstance(node, OPAQUE):
                    # A new opaque block has not yet been reduced. Its full
                    # physical range is the initial priority upper bound.
                    self._widths.setdefault(node, max(0., node.upper-node.lower))
                self._publish(node)
            self._refresh.clear()
            self.oracle.stats['priority_syncs'] += 1
            self.oracle.stats['lazy_dirty_blocks_peak'] = max(
                self.oracle.stats['lazy_dirty_blocks_peak'], len(self._dirty_blocks))
            return True
        finally:
            self.oracle.stats['priority_sync_cpu'] += process_time()-start

    def peek(self):
        memo = {}
        while True:
            node = super().peek()
            if node is None or not isinstance(node, OPAQUE) or node not in self._dirty_blocks:
                return node
            # No charged or uncharged physical task is expanded here.
            start = process_time()
            value = node.value(memo)
            width = max(0., float(value[1]-value[0]))
            previous = self._widths[node]
            increase = width-previous
            if increase > 0:
                self.oracle.stats['lazy_width_increases'] += 1
                self.oracle.stats['lazy_max_observed_width_increase'] = max(
                    self.oracle.stats['lazy_max_observed_width_increase'], increase)
            self._widths[node] = width
            self._dirty_blocks.remove(node)
            self.oracle.stats['priority_block_refreshes'] += 1
            self.oracle.stats['priority_block_refresh_cpu'] += process_time()-start
            self.oracle.stats['lazy_top_refreshes'] += 1
            self._publish(node)


class LazyFrontierOracle(FrontierOracle):
    def _setup_priority(self, *, verify_frontier=False, trace=False):
        super()._setup_priority(verify_frontier=verify_frontier, trace=trace)
        self.stats.update(lazy_dirty_blocks_peak=0, lazy_top_refreshes=0,
            lazy_width_increases=0, lazy_max_observed_width_increase=0.)

    def _initialize(self, roots):
        IncrementalOracle._initialize(self, roots)
        self._frontiers = {arm:LazyRootFrontier(self, root)
            for arm, root in roots.items() if arm in self.priority_arms}
