"""Audit-only incremental implementation of priority_probe's scheduling rule.

No physical probabilities, scores, policies, interval reducers, query units,
phase budgets, or CPU limits are changed.  ALIGHT (and standalone policy)
exposes reachable ordinary arithmetic nodes, then selects max(root mass *
width), with creation order breaking ties. CONTINUE keeps the original DFS.
Bulk/terminal reducers are opaque physical candidates; their zero-weight
scheduling dependencies are traversed only by the original local DFS.

Root masses accumulate contributions from every physical parent. New edges
seed their parent's *old* root mass; the resulting deltas are propagated once
in a topological pass over the affected descendant subgraph. Existing masses
plus those deltas solve the same finite DAG path-sum equation. This is not a
claim of linear total complexity: repeated attachments can revisit a large
descendant subgraph. Versioned heap entries avoid rescanning unrelated nodes.

The prototype is deliberately outside the repository and never monkeypatches
its semantics. Environment: BAPR_HRO_REPO; outputs: this script's directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import heapq
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

import numpy as np

OUT = Path(__file__).resolve().parent
DEFAULT_REPO = (Path.cwd() if (Path.cwd() / 'experiments/reconstruction_v47/incremental.py').exists()
                else OUT.parents[1] / 'repos' / 'BAPR-HRO-6e46267')
REPO = Path(os.environ.get('BAPR_HRO_REPO', str(DEFAULT_REPO))).resolve()
sys.path.insert(0, str(REPO))

from experiments.reconstruction_v47.bulk import BulkIncrementalOracle, BulkDPNode
from experiments.reconstruction_v47.hybrid import TerminalDPNode

OPAQUE = (BulkDPNode, TerminalDPNode)
LD = np.longdouble


class CreationOrderedNodes(dict):
    """Observe direct BulkDPNode insertions as well as ordinary _new calls."""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def __setitem__(self, key, node):
        if key not in self:
            self.owner._creation_order[node] = len(self)
        elif self[key] is not node:
            raise AssertionError('Retained DAG node identities must not be replaced')
        super().__setitem__(key, node)


class RootFrontier:
    """One root's physical occupancy and versioned candidate heap."""

    def __init__(self, oracle, root):
        self.oracle = oracle
        self.root = root
        self.seen = set()
        self.edges = {}
        self.weights = defaultdict(LD)
        self._seeds = defaultdict(LD)
        self._seeds[root] = LD(1)
        self._discovery = []
        self._queued = set()
        self._event_cursor = 0
        self._versions = defaultdict(int)
        self._widths = {}
        self._refresh = set()
        self._heap = []
        self._watchers = defaultdict(set)
        self._dirty_blocks = set()
        self._enqueue(root)

    def _enqueue(self, node):
        if node not in self.seen:
            self.seen.add(node)
            self._refresh.add(node)
        if node not in self._queued and node not in self.edges:
            self._queued.add(node)
            self._discovery.append(node)

    def _watch(self, node, blocks):
        """Track value dependencies, including zero-weight scheduling edges.

        These edges invalidate an opaque block's width; they never carry root
        probability. Each (dependency, block) pair is registered once.
        """
        todo = [(node, blocks)]
        while todo:
            child, wanted = todo.pop()
            fresh = wanted - self._watchers[child]
            if not fresh:
                continue
            self._watchers[child].update(fresh)
            self.oracle.stats['priority_watch_pairs'] += len(fresh)
            if child.expanded:
                todo.extend((grandchild, fresh) for _, grandchild in child.children)

    def _discover(self):
        """Retain the reference's eager arithmetic closure and reverse DFS.

        Only newly reachable/unexpanded structure needs visiting. The stack
        persists at a CPU pause; no physical branch is discarded.
        """
        while self._discovery:
            if not self.oracle._cpu_ok():
                return False
            node = self._discovery.pop()
            self._queued.remove(node)
            if node in self.edges:
                continue
            self.oracle.stats['priority_discovery_visits'] += 1
            if not node.expanded and node.build is not None and node.kind is None:
                self.oracle._expand(node)
            if isinstance(node, OPAQUE):
                self.edges[node] = []
                self._watch(node, {node})
                self._dirty_blocks.add(node)
            elif node.expanded:
                children = [(LD(p), child) for p, child in node.children if p > 0]
                self.edges[node] = children
                self._refresh.add(node)
                for p, child in children:
                    # New edge contributes the pre-update parent occupancy.
                    if self.weights[node] != 0:
                        self._seeds[child] += self.weights[node] * p
                    self._enqueue(child)
                    self.oracle.stats['priority_edges_installed'] += 1
            # Paid unresolved nodes deliberately have no installed edges.
        return True

    def _propagate(self):
        if not self._seeds:
            return
        start = process_time()
        seeds = self._seeds
        self._seeds = defaultdict(LD)
        affected = set(seeds)
        todo = list(seeds)
        indegree = defaultdict(int)
        edge_count = 0
        while todo:
            node = todo.pop()
            for _, child in self.edges.get(node, ()):
                indegree[child] += 1
                edge_count += 1
                if child not in affected:
                    affected.add(child)
                    todo.append(child)
        ready = deque(node for node in affected if indegree[node] == 0)
        delta = seeds
        processed = 0
        while ready:
            node = ready.popleft()
            processed += 1
            change = delta[node]
            self.weights[node] += change
            if change != 0:
                self._refresh.add(node)
            for p, child in self.edges.get(node, ()):
                delta[child] += change * p
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if processed != len(affected):
            raise AssertionError('Physical value graph must be acyclic')
        stats = self.oracle.stats
        stats['priority_weight_passes'] += 1
        stats['priority_weight_node_visits'] += processed
        stats['priority_weight_edge_visits'] += edge_count
        stats['priority_weight_cpu'] += process_time() - start

    def _publish(self, node):
        self._versions[node] += 1
        if isinstance(node, OPAQUE):
            if node.drained:
                return
            width = self._widths.get(node, 0.)
        elif not node.expanded and node.build is not None:
            width = max(0., node.upper - node.lower)
        else:
            return
        if width <= 0:
            return
        priority = self.weights[node] * LD(width)
        heapq.heappush(self._heap, (
            -priority, self.oracle._creation_order[node], self._versions[node], node))
        self.oracle.stats['priority_heap_pushes'] += 1
        self.oracle.stats['priority_heap_peak'] = max(
            self.oracle.stats['priority_heap_peak'], len(self._heap))

    def sync(self):
        start = process_time()
        try:
            if not self._discover():
                return False
            # _discover can expand zero-charge nodes and append more events.
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
            memo = {}
            for block in self._dirty_blocks:
                value = block.value(memo)
                self._widths[block] = max(0., float(value[1] - value[0]))
                self._refresh.add(block)
                self.oracle.stats['priority_block_refreshes'] += 1
            self._dirty_blocks.clear()
            for node in self._refresh:
                self._publish(node)
            self._refresh.clear()
            self.oracle.stats['priority_syncs'] += 1
            return True
        finally:
            self.oracle.stats['priority_sync_cpu'] += process_time() - start

    def peek(self):
        while self._heap:
            _, _, version, node = self._heap[0]
            if version != self._versions[node] or (
                    isinstance(node, OPAQUE) and node.drained) or (
                    not isinstance(node, OPAQUE) and node.expanded):
                heapq.heappop(self._heap)
                self.oracle.stats['priority_heap_stale_pops'] += 1
                continue
            return node
        return None

    def reference_check(self):
        """Expensive independent full recomputation, for small tests only."""
        indegree = defaultdict(int)
        for node in self.seen:
            assert (node.expanded or node.kind is not None or node.build is None)
            for _, child in self.edges.get(node, ()):
                indegree[child] += 1
        ready = deque(node for node in self.seen if indegree[node] == 0)
        weights = defaultdict(LD)
        weights[self.root] = LD(1)
        processed = 0
        while ready:
            node = ready.popleft()
            processed += 1
            for p, child in self.edges.get(node, ()):
                weights[child] += weights[node] * p
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        assert processed == len(self.seen)
        maximum_error = LD(0)
        for node in self.seen:
            error = abs(weights[node] - self.weights[node])
            scale = max(abs(weights[node]), np.finfo(LD).tiny)
            maximum_error = max(maximum_error, error / scale)
            assert error <= LD('1e-15') * scale, (
                'Incremental root mass disagrees', self.oracle._creation_order[node],
                str(self.weights[node]), str(weights[node]))
        memo, candidates = {}, []
        for node in self.seen:
            if isinstance(node, OPAQUE):
                if node.drained:
                    continue
                interval = node.value(memo)
                width = max(0., float(interval[1] - interval[0]))
            elif not node.expanded and node.build is not None:
                width = max(0., node.upper - node.lower)
            else:
                continue
            if width > 0:
                candidates.append((weights[node] * LD(width),
                                   -self.oracle._creation_order[node], node))
        expected = max(candidates, key=lambda item: item[:2])[2] if candidates else None
        actual = self.peek()
        assert actual is expected, ('Heap differs from full frontier',
                                    self.oracle._creation_order.get(actual),
                                    self.oracle._creation_order.get(expected))
        self.oracle.stats['priority_reference_checks'] += 1
        self.oracle.stats['priority_max_relative_mass_error'] = max(
            self.oracle.stats['priority_max_relative_mass_error'], float(maximum_error))


class EfficientPriorityOracle(BulkIncrementalOracle):
    def __init__(self, *args, verify_frontier=False, trace=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_priority(verify_frontier=verify_frontier, trace=trace)

    def _setup_priority(self, *, verify_frontier=False, trace=False):
        assert not self.nodes
        self._creation_order = {}
        self.nodes = CreationOrderedNodes(self)
        self._expansion_events = []
        self._frontiers = {}
        self._bulk_walkers = {}
        self._bulk_pending = {}
        self._priority_cache = {}
        self.verify_frontier = verify_frontier
        self._trace_enabled = trace
        self.selection_trace = []
        self._active_arm = None
        self.stats.update(
            priority_syncs=0, priority_discovery_visits=0,
            priority_edges_installed=0, priority_weight_passes=0,
            priority_weight_node_visits=0, priority_weight_edge_visits=0,
            priority_weight_cpu=0., priority_sync_cpu=0., priority_next_cpu=0.,
            priority_heap_pushes=0, priority_heap_stale_pops=0, priority_heap_peak=0,
            priority_bulk_steps=0, priority_watch_pairs=0, priority_block_refreshes=0,
            priority_reference_checks=0, priority_max_relative_mass_error=0.,
            priority_snapshot_cpu=0.)

    def _cpu_ok(self):
        if (self.max_process_cpu is not None and
                process_time() - self.cpu_start >= self.max_process_cpu):
            self._stopped_cpu = True
            return False
        return True

    def _initialize(self, roots):
        super()._initialize(roots)
        self._frontiers = {arm: RootFrontier(self, root)
                           for arm, root in roots.items() if arm in ('alight', 'policy')}

    def _expand(self, node):
        before = node.expanded
        super()._expand(node)
        if not before and node.expanded:
            self._expansion_events.append(node)
            if self._trace_enabled and node.kind is not None:
                self.selection_trace.append(dict(
                    unit=self.stats['charged_compiles'], arm=self._active_arm,
                    order=self._creation_order[node], kind=node.kind,
                    key=next((repr(key) for key, value in self.nodes.items()
                              if value is node), None)))

    def _next_bulk(self, node):
        walker = self._bulk_walkers.setdefault(node, self._walk(node))
        while self._cpu_ok():
            task = self._bulk_pending.get(node)
            if task is None or task.expanded:
                try:
                    task = next(walker)
                except StopIteration:
                    return None
            self._bulk_pending[node] = task
            if task.expanded:
                self._bulk_pending[node] = None
                continue
            if task.kind is None:
                self._expand(task)
                self._bulk_pending[node] = None
                continue
            return task
        return None

    def _next(self, arm):
        self._active_arm = arm
        if arm not in ('alight', 'policy'):
            return super()._next(arm)
        start = process_time()
        try:
            revision = self.stats['dag_expansions']
            cached = self._priority_cache.get(arm)
            if cached and cached[0] == revision and not cached[1].expanded:
                return cached[1]
            frontier = self._frontiers[arm]
            while self._cpu_ok():
                if not frontier.sync():
                    return None
                if self.verify_frontier:
                    frontier.reference_check()
                node = frontier.peek()
                if node is None:
                    return None
                if isinstance(node, OPAQUE):
                    task = self._next_bulk(node)
                    if task is None:
                        if not self._stopped_cpu:
                            node.drained = True
                        continue
                    self.stats['priority_bulk_steps'] += 1
                else:
                    task = node
                self._priority_cache[arm] = (self.stats['dag_expansions'], task)
                return task
            return None
        finally:
            self.stats['priority_next_cpu'] += process_time() - start

    def _snapshot(self, phase):
        start = process_time()
        result = super()._snapshot(phase)
        self.stats['priority_snapshot_cpu'] += process_time() - start
        result['counters']['priority_snapshot_cpu'] = self.stats['priority_snapshot_cpu']
        return result


def run(index=55, *, output=None, verify_frontier=False, trace=False):
    from experiments.reconstruction_v34.discover import context, read
    from experiments.reconstruction_v34.run import setup
    from experiments.reconstruction_v38.run import restore_node
    from experiments.reconstruction_v41.run import restored, engine, score
    from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner

    output = Path(output) if output else OUT / f'{index:03d}_efficient.json'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite retained evidence: {output}')
    base = REPO / 'experiments'
    start, wall = process_time(), perf_counter()
    manifest = read(base / 'reconstruction_v38/seed_manifest.json')
    data = read(base / 'reconstruction_v34/data.json.gz')
    fit = read(base / 'reconstruction_v34/model_fit.json')
    emission = read(base / 'reconstruction_v35/fitted_emission.json')
    input_cpu = process_time() - start
    state = manifest[index]['state']
    graph_start = process_time()
    graph = context(data, state['day'])[0]
    graph_cpu = process_time() - graph_start
    query_start, query_wall = process_time(), perf_counter()
    model, params = setup(graph, fit)
    node = restore_node(state['node'])
    history = restored('shallow', model, params, emission, state)
    planner = MatchedContinuationPlanner(graph, model, params, state['deadline'])
    base_action, _ = score(engine(planner, 'shallow'), node, state['cid'], state['clock'], history)
    setup_cpu = process_time() - query_start
    oracle = EfficientPriorityOracle(
        graph, model, params, state['deadline'], history,
        max_compiles=2000, max_process_cpu=120,
        verify_frontier=verify_frontier, trace=trace)
    result = oracle.forced(node, {key: state[key] for key in ('kind', 'cid', 'clock')},
                           schedule='pooled')
    result.update(
        index=index, root_index=state['root_index'], base_action=base_action,
        complete_diagnostic_cost=dict(process_cpu=process_time() - start,
                                      wall=perf_counter() - wall,
                                      input_load_cpu=input_cpu,
                                      graph_build_cpu=graph_cpu,
                                      setup_and_baseline_cpu=setup_cpu),
        contract_query_cost=dict(process_cpu=process_time() - query_start,
                                 wall=perf_counter() - query_wall,
                                 definition='Graph already materialized; includes model/history setup, baseline, oracle construction, all queue/propagation and snapshot extraction work'),
        implementation='Incremental root masses plus versioned priority heap',
        scope='Audit-only scheduling/data-structure ablation; unchanged P0/W/pi and 2000 quota; not a novelty claim',
        rule='ALIGHT root occupancy times width; eager ordinary arithmetic closure; opaque bulk with original local DFS; CONTINUE unchanged',
        cpu_accounting=('Original 120 CPU-second oracle limit and 60-second first phase retained. '
                        'Complete diagnostic cost additionally includes input loading, model setup, '
                        'baseline score, all scheduling/propagation, and interval extraction. '
                        'JSON serialization and process imports are reported separately by the shell when available.'),
        frontier_state={arm: dict(physical_nodes=len(frontier.seen),
                                  physical_edges=sum(map(len, frontier.edges.values())),
                                  heap_entries=len(frontier._heap),
                                  pending_events=len(oracle._expansion_events)-frontier._event_cursor)
                        for arm, frontier in oracle._frontiers.items()})
    if trace:
        trace_path = output.with_name(output.stem + '_trace.json')
        trace_path.write_text(json.dumps(oracle.selection_trace, indent=2) + '\n')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in (
        'index', 'root_index', 'difference_interval', 'certified_action',
        'complete_diagnostic_cost', 'contract_query_cost', 'budget_allocation', 'counters', 'frontier_state')},
        ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index', nargs='?', type=int, default=55)
    parser.add_argument('--output')
    parser.add_argument('--verify-frontier', action='store_true')
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    run(args.index, output=args.output, verify_frontier=args.verify_frontier, trace=args.trace)
