"""V48 incremental global frontier, adapted from the supplied v12 audit prototype.

RootFrontier's occupancy propagation, creation-order ties, versioned heap and
opaque-block dependency watchers are reused from frontier_efficiency/
efficient_priority.py. V48 applies the rule to both requested arms, records
complete fees and supports certificate stopping. Physical value builders and
reducers remain V47's unchanged implementation.
"""
from collections import defaultdict, deque
import heapq
from time import process_time

import numpy as np

from experiments.reconstruction_v33.nodes import Ride
from experiments.reconstruction_v45.prefix import structure_gate
from experiments.reconstruction_v46.oracle import receipt
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
            self.owner._node_keys[node] = key
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
            refresh_start = process_time()
            for block in self._dirty_blocks:
                value = block.value(memo)
                self._widths[block] = max(0., float(value[1] - value[0]))
                self._refresh.add(block)
                self.oracle.stats['priority_block_refreshes'] += 1
            self.oracle.stats['priority_block_refresh_cpu'] += process_time() - refresh_start
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


class FrontierOracle(BulkIncrementalOracle):
    def __init__(self, *args, verify_frontier=False, trace=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_priority(verify_frontier=verify_frontier, trace=trace)
        self._instrument_costs()

    def _setup_priority(self, *, verify_frontier=False, trace=False):
        assert not self.nodes
        self._creation_order = {}
        self._node_keys = {}
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
        self.priority_arms = ('alight', 'continue', 'policy')
        self._stop_on_certificate = False
        self._certificate_every = 250
        self._certificate_stop = False
        self._first_certificate = None
        self._profile_depth = defaultdict(int)
        self.profile = defaultdict(float)
        self.stats.update(
            priority_syncs=0, priority_discovery_visits=0,
            priority_edges_installed=0, priority_weight_passes=0,
            priority_weight_node_visits=0, priority_weight_edge_visits=0,
            priority_weight_cpu=0., priority_sync_cpu=0., priority_next_cpu=0.,
            priority_heap_pushes=0, priority_heap_stale_pops=0, priority_heap_peak=0,
            priority_bulk_steps=0, priority_watch_pairs=0, priority_block_refreshes=0,
            priority_block_refresh_cpu=0.,
            priority_reference_checks=0, priority_max_relative_mass_error=0.,
            priority_snapshot_cpu=0., priority_snapshot_count=0,
            paid_task_cpu=0., arithmetic_task_cpu=0., dependency_audit_cpu=0.,
            frontier_summary_cpu=0.)

    def _instrument_costs(self):
        """Instance-local inclusive timers; these nested categories do not add."""
        def instrument(owner, name, category):
            original = getattr(owner, name)
            def measured(*args, **kwargs):
                outer = self._profile_depth[category] == 0
                self._profile_depth[category] += 1
                start = process_time() if outer else 0.
                try:
                    return original(*args, **kwargs)
                finally:
                    self._profile_depth[category] -= 1
                    if outer:
                        self.profile[category] += process_time() - start
            setattr(owner, name, measured)
        for owner, name in ((self.engine, 'platform'), (self.engine, 'onboard'),
                            (self.planner, 'compile_ride'), (self.planner, 'platform_reference')):
            instrument(owner, name, 'policy_scoring_inclusive')
        instrument(self.root_law, 'law', 'physical_conditioning_inclusive')

    def _cpu_ok(self):
        if (self.max_process_cpu is not None and
                process_time() - self.cpu_start >= self.max_process_cpu):
            self._stopped_cpu = True
            return False
        return True

    def _initialize(self, roots):
        super()._initialize(roots)
        self._frontiers = {arm: RootFrontier(self, root)
                           for arm, root in roots.items() if arm in self.priority_arms}

    def _expand(self, node):
        before = node.expanded
        start = process_time()
        super()._expand(node)
        if not before and node.expanded:
            self.stats['arithmetic_task_cpu' if node.kind is None else 'paid_task_cpu'] += process_time() - start
            self._expansion_events.append(node)
            if self._trace_enabled and node.kind is not None:
                self.selection_trace.append(dict(
                    unit=self.stats['charged_compiles'], arm=self._active_arm,
                    order=self._creation_order[node], kind=node.kind,
                    key=repr(self._node_keys[node])))

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
        if arm not in self.priority_arms:
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

    def _values(self):
        actions = super()._values()
        start = process_time()
        for arm, root in self._roots.items():
            seen, todo, pending = set(), [root], 0
            while todo:
                node = todo.pop()
                if node in seen:
                    continue
                seen.add(node)
                if not node.expanded:
                    pending += 1
                else:
                    if node.constant is not None and node.constant[2] > 0:
                        pending += 1
                    # Bulk's zero-weight links are required task dependencies.
                    todo.extend(child for _, child in node.children)
            value = actions[arm]
            value['floating_unresolved_probability'] = value['unresolved_probability']
            value['pending_dependencies'] = pending
            value['dependency_complete'] = pending == 0
            if pending:
                value['mean'] = None
        self.stats['dependency_audit_cpu'] += process_time() - start
        return actions

    def _snapshot(self, phase):
        start = process_time()
        result = super()._snapshot(phase)
        self.stats['priority_snapshot_cpu'] += process_time() - start
        self.stats['priority_snapshot_count'] += 1
        result['counters'] = dict(self.stats)
        result['cost'] = self.cost()
        if result['certified_action'] is not None:
            if self._first_certificate is None:
                self._first_certificate = dict(phase=phase, actions=result['actions'],
                    difference_interval=list(result['difference_interval']),
                    certified_action=result['certified_action'],
                    charged_compiles=self.stats['charged_compiles'], cost=result['cost'])
            if self._stop_on_certificate:
                self._certificate_stop = True
        return result

    def _phase(self, arm, limit, cpu_limit, name, checkpoints, records):
        initial, start = self.stats['charged_compiles'], process_time()
        reason = 'quota'
        while self.stats['charged_compiles'] < limit:
            if self._certificate_stop:
                reason = 'certificate'
                break
            if cpu_limit is not None and process_time()-self.cpu_start >= cpu_limit:
                reason = 'cpu'
                break
            pending = self._next(arm)
            if pending is None:
                reason = 'cpu' if self._stopped_cpu else 'no_pending_work'
                break
            self._pending[arm] = None
            self._expand(pending)
            # Keep the prototype's closure before every saved checkpoint.
            self._next(arm)
            count = self.stats['charged_compiles']
            if count in checkpoints or count % self._certificate_every == 0:
                records.append(self._snapshot(name))
                if self._certificate_stop:
                    reason = 'certificate'
                    break
        return dict(arm=arm, phase=name, charged_compiles=self.stats['charged_compiles']-initial,
            process_cpu=process_time()-start, stop_reason=reason,
            cumulative_compiles=self.stats['charged_compiles'])

    def _boundary(self, phase, records):
        """Reuse an unchanged saved value; settling a phase is not refinement."""
        previous = records[-1]
        if (previous['counters']['charged_compiles'] == self.stats['charged_compiles']
                and previous['counters']['dag_expansions'] == self.stats['dag_expansions']):
            result = dict(previous, phase=phase, counters=dict(self.stats), cost=self.cost(),
                          reused_saved_value=True)
        else:
            result = self._snapshot(phase)
        records.append(result)
        return result

    def frontier_summary(self, top=10):
        """Read-only stopped graph audit: never expand or silently resume work."""
        start = process_time()
        summaries = {}
        memo = {}
        for arm, root in self._roots.items():
            edges, pending, todo, seen = {}, [], [root], set()
            while todo:
                node = todo.pop()
                if node in seen:
                    continue
                seen.add(node)
                if isinstance(node, OPAQUE):
                    edges[node] = []
                    pending.append(node)
                elif node.expanded:
                    edges[node] = [(LD(p), child) for p, child in node.children if p > 0]
                    todo.extend(child for _, child in edges[node])
                else:
                    edges[node] = []
                    pending.append(node)
            indegree = defaultdict(int)
            for row in edges.values():
                for _, child in row:
                    indegree[child] += 1
            queue = deque(node for node in seen if indegree[node] == 0)
            weights = defaultdict(LD)
            weights[root] = LD(1)
            while queue:
                node = queue.popleft()
                for p, child in edges[node]:
                    weights[child] += p*weights[node]
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        queue.append(child)
            rows, grouped = [], defaultdict(lambda: [0, LD(0)])
            for node in pending:
                value = node.value(memo)
                width = max(0., float(value[1]-value[0]))
                contribution = weights[node]*LD(width)
                key = self._node_keys[node]
                kind = 'bulk_block' if isinstance(node, OPAQUE) else node.kind or str(key[0])
                grouped[kind][0] += 1
                grouped[kind][1] += contribution
                rows.append(dict(order=self._creation_order[node], kind=kind,
                    root_weight=float(weights[node]), root_weight_decimal=str(weights[node]),
                    interval=[float(value[0]), float(value[1])], width=width,
                    potential_width=float(contribution), potential_width_decimal=str(contribution)))
            rows.sort(key=lambda row:(-LD(row['potential_width_decimal']), row['order']))
            frontier = self._frontiers.get(arm)
            summaries[arm] = dict(physical_nodes=len(seen),
                physical_edges=sum(map(len, edges.values())),
                exposed_blocks_and_unexpanded_nodes=len(rows),
                positive_width_frontier_nodes=sum(row['width'] > 0 for row in rows),
                groups={kind:dict(nodes=count, potential_width=float(width),
                    potential_width_decimal=str(width)) for kind,(count,width) in grouped.items()},
                top=rows[:top], heap_entries=len(frontier._heap) if frontier else None,
                scope='Potential root-width contributions; physical lower/upper clamps can make their sum exceed actual root width. Bulk zero-weight task links are not physical edges; no probability was discarded.')
        self.stats['frontier_summary_cpu'] += process_time()-start
        return summaries

    def cost(self):
        result = super().cost()
        result['components_cpu'] = dict(self.profile,
            paid_task_inclusive=self.stats['paid_task_cpu'],
            arithmetic_task_inclusive=self.stats['arithmetic_task_cpu'],
            priority_next_inclusive=self.stats['priority_next_cpu'],
            priority_sync_nested=self.stats['priority_sync_cpu'],
            root_weight_nested=self.stats['priority_weight_cpu'],
            bulk_refresh_nested=self.stats['priority_block_refresh_cpu'],
            snapshot_inclusive=self.stats['priority_snapshot_cpu'],
            completion_audit_nested=self.stats['dependency_audit_cpu'],
            stopped_frontier_summary=self.stats['frontier_summary_cpu'])
        result['components_scope'] = 'Inclusive nested timers, not additive. Complete CPU/wall includes all work and remains the billing measure.'
        return result

    def forced(self, node, row, *, schedule='pooled', priority_arms=('alight', 'continue'),
               stop_on_certificate=False, certificate_every=250,
               checkpoints=tuple(range(250, 2001, 250))):
        if schedule not in ('fixed_split', 'pooled') or self.max_compiles is None:
            raise ValueError('A finite query budget and supported schedule are required')
        if certificate_every <= 0:
            raise ValueError('Certificate checkpoint cadence must be positive')
        self.priority_arms = tuple(priority_arms)
        self._stop_on_certificate = stop_on_certificate
        self._certificate_every = certificate_every
        self._prepare(node)
        clock, cid = float(row['clock']), row['cid']
        check = structure_gate(self.model, node, row)
        if not check['eligible']:
            raise ValueError('Head structure not admitted')
        alight = self._platform_node(node.exits[cid], clock, self.snapshot)
        remaining = node.path[node.path.index(cid)+1:]
        if remaining:
            ride = Ride(remaining[0], remaining, {c:node.exits[c] for c in remaining if c in node.exits})
            trip, slot = self.model._positions[remaining[0], 'departure']
            law = self.root_law.roots[trip]
            children = [(float(p), self._ride(ride, self.snapshot, slot, i))
                        for i,p in enumerate(law.marginal(slot)) if p > 0]
            continuation = self._mixture(('head_continue',), clock, children)
        else:
            continuation = self._point(('head_continue',), self.deadline)
        self._initialize({'alight':alight, 'continue':continuation})
        records = [self._snapshot('initial')]
        allocations = []
        half_cpu = None if self.max_process_cpu is None else self.max_process_cpu/2
        if not self._certificate_stop:
            allocations.append(self._phase('alight', self.max_compiles//2, half_cpu,
                'initial_alight', checkpoints, records))
            self._boundary('after_initial_alight', records)
        if not self._certificate_stop:
            allocations.append(self._phase('continue', self.max_compiles, self.max_process_cpu,
                'continue', checkpoints, records))
            self._boundary('after_continue', records)
        continue_phase = next((a for a in allocations if a['arm'] == 'continue'), None)
        recycle_reason = None
        if records[-1]['actions']['continue']['mean'] is not None:
            recycle_reason = 'continue_dependencies_complete'
        elif continue_phase and continue_phase['stop_reason'] == 'no_pending_work':
            recycle_reason = 'continue_actionable_frontier_exhausted'
        if (not self._certificate_stop and schedule == 'pooled'
                and self.stats['charged_compiles'] < self.max_compiles and recycle_reason):
            allocations.append(self._phase('alight', self.max_compiles, self.max_process_cpu,
                'recycled_alight', checkpoints, records))
        final = self._boundary('final', records).copy()
        summary = self.frontier_summary()
        final['cost'] = self.cost()
        final['counters'] = dict(self.stats)
        if self._certificate_stop:
            stop_reason = 'certificate'
        elif self._stopped_cpu or (self.max_process_cpu is not None and
                                  process_time()-self.cpu_start >= self.max_process_cpu):
            stop_reason = 'cpu_limit'
        elif self.stats['charged_compiles'] >= self.max_compiles:
            stop_reason = 'compile_limit'
        elif final['complete']:
            stop_reason = 'exact_complete'
        else:
            stop_reason = 'phase_allocation_end'
        final.update(clock=clock, deadline=self.deadline, root_structure=check,
            schedule=schedule, priority_arms=list(self.priority_arms),
            stop_on_certificate=stop_on_certificate, certificate_every=certificate_every,
            checkpoints=records, budget_allocation=allocations, stop_reason=stop_reason,
            first_certificate=self._first_certificate, frontier_summary=summary,
            recycling_eligibility_reason=recycle_reason,
            unspent_compiles=self.max_compiles-self.stats['charged_compiles'],
            recycled_compiles=sum(a['charged_compiles'] for a in allocations if a['phase']=='recycled_alight'),
            primitive_unit='Unchanged V47 platform/WAIT/BOARD/shared-policy/onboard entry units',
            scope='Fixed V41 P0/W/pi; v12 incremental occupancy priority with original bulk blocks. A deterministic certificate is independent of exact dependency completion.')
        return final
