"""Resumable ordinary full-policy event DP with retained interval dependencies.

The two schedules share this evaluator. Primitive units are platform builds,
original policy platform queries and original policy on-board queries; they
are not equivalent arithmetic work or the V46 bulk WAIT/BOARD units.
"""
from dataclasses import dataclass, field
from time import process_time

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v33.nodes import Ride
from experiments.reconstruction_v41.physical_posterior import posterior_transition
from experiments.reconstruction_v45.prefix import structure_gate
from experiments.reconstruction_v46.oracle import physical_path, point, receipt, unknown
from experiments.reconstruction_v46.shared_oracle import SharedObservationOracle, shared_structure


@dataclass(eq=False)
class ValueNode:
    lower: float
    upper: float
    build: object = None
    kind: str | None = None
    constant: object = None
    children: list = field(default_factory=list)
    expanded: bool = False
    drained: bool = False

    def value(self, memo):
        if self in memo:
            return memo[self]
        if not self.expanded:
            result = np.array([self.lower, self.upper, 1.])
        else:
            result = np.zeros(3) if self.constant is None else self.constant.copy()
            for mass, child in self.children:
                result += mass * child.value(memo)
            # Parent physical-time bounds survive reverse-slot child queries.
            result[0] = max(self.lower, result[0])
            result[1] = min(self.upper, result[1])
            if result[0] > result[1] + 1e-8:
                raise ValueError('Expanded conditional value violates its physical bounds')
            if result[0] > result[1]:
                result[:2] = (result[0] + result[1]) / 2
        memo[self] = result
        return result


class IncrementalOracle(SharedObservationOracle):
    """Persist the exact computation graph, including every unresolved child."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = {}
        self.stats.update(dag_nodes=0, dag_expansions=0, dag_zero_expansions=0,
                          retained_node_hits=0)
        self._walkers = {}
        self._pending = {}
        self._roots = {}
        self._stopped_cpu = False

    def _new(self, key, clock, build=None, kind=None, constant=None):
        if key in self.nodes:
            self.stats['retained_node_hits'] += 1
            return self.nodes[key]
        node = ValueNode(min(float(clock), self.deadline), self.deadline,
                         build=build, kind=kind, constant=constant,
                         expanded=constant is not None)
        self.nodes[key] = node
        self.stats['dag_nodes'] += 1
        return node

    def _point(self, key, value):
        return self._new(key, value, constant=point(value))

    def _expand(self, node):
        if node.expanded or node.build is None:
            return
        constant, children = node.build()
        node.constant = constant
        node.children = children
        node.expanded = True
        node.build = None
        self.stats['dag_expansions'] += 1
        if node.kind is None:
            self.stats['dag_zero_expansions'] += 1
        else:
            self.stats['charged_compiles'] += 1
            self.stats[node.kind] += 1

    def _walk(self, node):
        if node.drained:
            return
        if not node.expanded and node.build is not None:
            yield node
        if node.expanded:
            for _, child in node.children:
                yield from self._walk(child)
        node.drained = True

    def _next(self, arm):
        """Drain arithmetic-only nodes, retaining the next paid continuation."""
        while True:
            if self.max_process_cpu is not None and process_time()-self.cpu_start >= self.max_process_cpu:
                self._stopped_cpu = True
                return None
            pending = self._pending.get(arm)
            if pending is None:
                try:
                    pending = next(self._walkers[arm])
                except StopIteration:
                    return None
            self._pending[arm] = None
            if pending.expanded:
                continue
            if pending.kind is None:
                self._expand(pending)
                continue
            self._pending[arm] = pending
            return pending

    def _mixture(self, key, clock, children):
        ordered = sorted(((float(p), n) for p, n in children if p > 0),
                         key=lambda item: -item[0])
        return self._new(key, clock, build=lambda: (None, ordered))

    def _ride(self, ride, history, slot, state):
        self._prepare(ride)
        history_key = self._key(ride, history)
        key = ('ride', history_key, int(slot), int(state))
        if key in self.nodes:
            self.stats['retained_node_hits'] += 1
            return self.nodes[key]
        trip, first = self.model._positions[ride.first_cid, 'departure']
        law = self.root_law.law(trip, history)
        clock = float(law.times[slot][state])
        if law.canceled[slot][state] or clock >= self.deadline or law.marginal(slot)[state] <= 0:
            return self._point(key, self.deadline)
        last = self.model._positions[physical_path(ride)[-1], 'arrival'][1]
        by_slot = {self.model._positions[cid, kind][1]:(cid, kind)
                   for cid in ride.path for kind in ('departure', 'arrival')}
        cid, kind = by_slot[slot]
        tail = ride.exits.get(cid) if kind == 'arrival' else None
        if kind == 'arrival' and cid in ride.exits and tail is None:
            return self._point(key, clock)

        def following():
            if slot >= last:
                return point(self.deadline), []
            probabilities = posterior_transition(law, slot)[state]
            self.stats['physical_transition_cells'] += probabilities.size
            children = [(float(p), self._ride(ride, history, slot+1, i))
                        for i, p in enumerate(probabilities) if p > 0]
            children.sort(key=lambda pair: -pair[0])
            return None, children

        def build():
            if kind == 'arrival' and cid in ride.exits:
                h = history.clone()
                h.observe(cid, 'arrival', clock)
                try:
                    alight, _ = self.engine.onboard(ride, cid, clock, h, method='rolling_fixed')
                except UnsupportedHistory:
                    self.stats['unsupported_scores'] += 1
                    return unknown(clock, self.deadline), []
                if alight:
                    return None, [(1., self._platform_node(tail, clock, history))]
            return following()

        return self._new(key, clock, build=build,
                         kind='onboard_compiles' if kind == 'arrival' and cid in ride.exits else None)

    def _board_node(self, ride, clock, base, event):
        self.stats['shared_board_calls'] += 1
        trip, slot = self.model._positions[ride.first_cid, 'departure']
        history = self.boarding_history(base, event, trip)
        law = self.root_law.law(trip, history)
        selected = np.flatnonzero((law.times[slot] == clock) & ~law.canceled[slot])
        if len(selected) != 1:
            raise ValueError('No unique operating selected departure')
        return self._ride(ride, history, slot, int(selected[0]))

    def _platform_node(self, platform, entry, history):
        self._prepare(platform)
        context = self._key(platform, history), float(entry)
        key = ('platform', context)
        if key in self.nodes:
            self.stats['retained_node_hits'] += 1
            return self.nodes[key]
        if entry >= self.deadline:
            return self._point(key, self.deadline)
        check = shared_structure(self.model, platform)
        if not check['eligible']:
            self.stats['structure_refusals'] += 1
            return self._new(key, entry, constant=unknown(entry, self.deadline))
        interface = set(check['interface_cids'])
        self.stats['shared_interfaces_max'] = max(self.stats['shared_interfaces_max'], len(interface))

        def build_platform():
            self.stats['shared_platform_solves'] += 1
            cids = tuple(platform.rides)
            nc = len(cids)
            physical = {cid:self.root_law.departure_distribution(cid, history) for cid in cids}
            times = np.array(sorted({float(t) for cid in cids for t, p in zip(*physical[cid][:2], strict=True)
                                     if p > 0 and entry <= t < self.deadline}))
            hazards = np.zeros((len(times), nc))
            for i, cid in enumerate(cids):
                clocks, p, cancel = physical[cid]
                tails = np.r_[np.cumsum(p[::-1])[::-1], 0.]
                survival = cancel+p[clocks < entry].sum()+tails[np.searchsorted(clocks, times, side='left')]
                included = (p > 0) & (clocks >= entry) & (clocks < self.deadline)
                np.add.at(hazards[:, i], np.searchsorted(times, clocks[included]), p[included])
                np.divide(hazards[:, i], survival, out=hazards[:, i], where=survival > 0)
            ready = entry+self.graph.stops[platform.stop].min_transfer_time
            bits = [[i for i in range(nc) if mask & (1 << i)] for mask in range(1 << nc)]
            subsets = []
            for active in bits:
                choices = np.array([0], dtype=int)
                for i in active:
                    choices = np.r_[choices, choices | (1 << i)]
                subsets.append(choices)

            def event(j, mask, z):
                event_key = ('event', context, j, mask, z)
                if j == len(times) or not mask:
                    return self._point(event_key, self.deadline)
                clock = float(times[j])

                def build_event():
                    probability = np.ones(1)
                    for i in bits[mask]:
                        hazard = hazards[j, i]
                        probability = np.r_[probability*(1-hazard), probability*hazard]
                    children = []
                    order = sorted(range(len(probability)),
                                   key=lambda k:(-probability[k], int(subsets[mask][k])))
                    for k in order:
                        mass = float(probability[k])
                        if mass <= 0:
                            continue
                        departed = int(subsets[mask][k])
                        left = mask ^ departed
                        visible = tuple(cids[i] for i in bits[departed])
                        next_z = tuple(sorted((*z, *((cid, clock) for cid in visible if cid in interface))))
                        if not departed or clock < ready:
                            child = event(j+1, left, next_z)
                        else:
                            child = policy(j, mask, departed, left, visible, next_z)
                        children.append((mass, child))
                        self.stats['positive_event_branches'] += 1
                    return None, children

                return self._new(event_key, clock, build=build_event)

            def policy(j, mask, departed, left, visible, z):
                clock = float(times[j])
                policy_key = ('policy', context, j, mask, departed, z)

                def build_policy():
                    remaining = {cids[i] for i in bits[left]}
                    h = self.event_history(platform, history, entry, clock, remaining, visible, z)
                    seen = tuple(cid for cid in cids if cid not in remaining)
                    try:
                        chosen, _ = self.engine.platform(platform, entry, ready, h, visible, seen, clock,
                                                        choose=self.policy.choose, method='rolling_fixed')
                    except UnsupportedHistory:
                        self.stats['unsupported_scores'] += 1
                        return unknown(clock, self.deadline), []
                    child = (event(j+1, left, z) if chosen is None else
                             self._board_node(platform.rides[chosen], clock, history, h))
                    return None, [(1., child)]

                return self._new(policy_key, clock, build=build_policy, kind='shared_policy_compiles')

            return None, [(1., event(0, (1 << nc)-1, ()))]

        return self._new(key, entry, build=build_platform, kind='platform_solves')

    def _initialize(self, roots):
        self._roots = roots
        self._walkers = {arm:self._walk(node) for arm, node in roots.items()}
        self._pending = {arm:None for arm in roots}

    def _values(self):
        memo = {}
        return {arm:receipt(node.value(memo)) for arm, node in self._roots.items()}

    def _snapshot(self, phase):
        actions = self._values()
        if set(actions) == {'alight', 'continue'}:
            a, c = actions['alight'], actions['continue']
            gap = [a['lower']-c['upper'], a['upper']-c['lower']]
            certified = 'alight' if gap[1] < -1e-9 else 'continue' if gap[0] > 1e-9 else None
        else:
            gap, certified = None, None
        return dict(actions=actions, difference_interval=gap, certified_action=certified,
                    complete=all(value['mean'] is not None for value in actions.values()),
                    phase=phase, counters=dict(self.stats), cost=self.cost())

    def _phase(self, arm, limit, cpu_limit, name, checkpoints, records):
        initial = self.stats['charged_compiles']
        start = process_time()
        reason = 'quota'
        while self.stats['charged_compiles'] < limit:
            if cpu_limit is not None and process_time()-self.cpu_start >= cpu_limit:
                reason = 'cpu'
                break
            pending = self._next(arm)
            if pending is None:
                reason = 'cpu' if self._stopped_cpu else 'no_pending_work'
                break
            self._pending[arm] = None
            self._expand(pending)
            # All arithmetic work resulting from a paid query is retained too.
            self._next(arm)
            if self.stats['charged_compiles'] in checkpoints:
                records.append(self._snapshot(name))
        return dict(arm=arm, phase=name, charged_compiles=self.stats['charged_compiles']-initial,
                    process_cpu=process_time()-start, stop_reason=reason,
                    cumulative_compiles=self.stats['charged_compiles'])

    def forced(self, node, row, *, schedule='pooled', checkpoints=(500, 1000, 1500, 2000)):
        if schedule not in ('fixed_split', 'pooled'):
            raise ValueError('Unknown budget schedule')
        if self.max_compiles is None:
            raise ValueError('The finite query budget must be specified')
        self._prepare(node)
        clock, cid = float(row['clock']), row['cid']
        check = structure_gate(self.model, node, row)
        if not check['eligible']:
            raise ValueError('Head structure not admitted')
        alight = self._platform_node(node.exits[cid], clock, self.snapshot)
        remaining = node.path[node.path.index(cid)+1:]
        if not remaining:
            continuation = self._point(('head_continue',), self.deadline)
        else:
            ride = Ride(remaining[0], remaining, {c:node.exits[c] for c in remaining if c in node.exits})
            trip, slot = self.model._positions[remaining[0], 'departure']
            law = self.root_law.roots[trip]
            children = [(float(p), self._ride(ride, self.snapshot, slot, i))
                        for i, p in enumerate(law.marginal(slot)) if p > 0]
            continuation = self._mixture(('head_continue',), clock, children)
        self._initialize({'alight':alight, 'continue':continuation})
        records = [self._snapshot('initial')]
        half_cpu = None if self.max_process_cpu is None else self.max_process_cpu/2
        allocations = [self._phase('alight', self.max_compiles//2, half_cpu, 'initial_alight', checkpoints, records)]
        records.append(self._snapshot('after_initial_alight'))
        allocations.append(self._phase('continue', self.max_compiles, self.max_process_cpu,
                                       'continue', checkpoints, records))
        records.append(self._snapshot('after_continue'))
        if (schedule == 'pooled' and self.stats['charged_compiles'] < self.max_compiles
                and self._values()['continue']['mean'] is not None):
            allocations.append(self._phase('alight', self.max_compiles, self.max_process_cpu,
                                           'recycled_alight', checkpoints, records))
        final = self._snapshot('final')
        records.append(final.copy())
        final.update(clock=clock, deadline=self.deadline, root_structure=check,
                     schedule=schedule, checkpoints=records, budget_allocation=allocations,
                     unspent_compiles=self.max_compiles-self.stats['charged_compiles'],
                     recycled_compiles=sum(a['charged_compiles'] for a in allocations if a['phase']=='recycled_alight'),
                     scope='Original full-pi expectation under frozen P0; retained ordinary event-DP work',
                     primitive_unit='One platform build, original platform policy query, or original onboard policy query')
        return final

    def platform(self, platform, entry, history=None, *, checkpoints=()):
        history = self.snapshot if history is None else history
        root = self._platform_node(platform, float(entry), history)
        self._initialize({'policy':root})
        records = [self._snapshot('initial')]
        cap = self.max_compiles if self.max_compiles is not None else float('inf')
        allocation = self._phase('policy', cap, self.max_process_cpu, 'platform', checkpoints, records)
        result = self._snapshot('final')
        result.update(result['actions']['policy'], checkpoints=records, budget_allocation=allocation)
        return result
