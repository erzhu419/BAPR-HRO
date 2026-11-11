"""Finite-bank certificates for the committed transit execution model."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from graphlib import TopologicalSorter
from heapq import heappop, heappush

from .committed_transit import build_committed_model
from .simulator import JourneyScenario


@dataclass(frozen=True)
class RepairWork:
    changed_arcs: int
    recomputed_labels: int
    changed_labels: int


class _ArrivalLabels:
    def __init__(self, index, scene):
        self.index = index
        self.events = dict(scene.events)
        self.publications = scene.publications
        self.outgoing = defaultdict(dict)
        self.incoming = defaultdict(set)
        self.values = [[index.deadline] * (index.deadline - index.start + 1)
                       for _ in index.rank]
        for cid, event in self.events.items():
            arc = index.arc(cid, event)
            if arc is not None:
                self.add(cid, arc)
        for time in range(index.deadline - 1, index.start - 1, -1):
            for stop in range(len(index.rank) - 1, -1, -1):
                self.values[stop][time - index.start] = self.backup((time, stop))

    def add(self, cid, arc):
        tail, head = arc
        self.outgoing[tail][cid] = head
        self.incoming[head].add((tail, cid))

    def remove(self, cid, arc):
        tail, head = arc
        del self.outgoing[tail][cid]
        self.incoming[head].remove((tail, cid))

    def value(self, node):
        time, stop = node
        return self.values[stop][time - self.index.start]

    def backup(self, node):
        time, stop = node
        if stop == self.index.destination_rank:
            return time
        wait = self.value((time + 1, stop))
        return min(wait, min((self.value(head) for head in self.outgoing.get(node, {}).values()),
                             default=self.index.deadline))

    def update(self, changes):
        arcs = [(cid, self.index.arc(cid, self.events[cid]), self.index.arc(cid, event))
                for cid, event in changes.items()]
        seeds = set()
        edited = 0
        for cid, old, new in arcs:
            if old == new:
                continue
            edited += 1
            if old is not None:
                self.remove(cid, old)
                seeds.add(old[0])
            if new is not None:
                self.add(cid, new)
                seeds.add(new[0])
        self.events.update(changes)

        queue, pending = [], set()

        def enqueue(node):
            if node not in pending:
                pending.add(node)
                heappush(queue, (-node[0], -node[1]))

        for node in seeds:
            enqueue(node)
        recomputed = changed = 0
        while queue:
            time, stop = (-part for part in heappop(queue))
            node = (time, stop)
            pending.remove(node)
            recomputed += 1
            old, new = self.value(node), self.backup(node)
            if old != new:
                self.values[stop][time - self.index.start] = new
                changed += 1
                if time > self.index.start:
                    enqueue((time - 1, stop))
                for tail, _ in self.incoming.get(node, ()):
                    enqueue(tail)
        return RepairWork(edited, recomputed, changed)


class ArrivalRelaxation:
    """Earliest arrivals with actual world knowledge and free transfer/waiting.

    Geometry, horizon, world probabilities and the order of same-minute rides
    stay fixed during repair. An edit may move, remove or add a timed ride.
    Zero-time self-loops have no effect on these labels. The union of allowed
    zero-time inter-stop edges must be acyclic on stops that reach the goal.
    """

    def __init__(self, graph, bank, destination, start, deadline):
        self.graph = graph
        self.connections = {c.id: c for c in graph.connections}
        self.destination, self.start, self.deadline = destination, start, deadline
        bank = tuple(bank)
        self.minute = bank[0][1].minute if bank else 1
        self.weights = tuple(Fraction(p) for p, _ in bank)
        self.total_weight = sum(self.weights)
        if not self.total_weight or min(self.weights) < 0 or deadline < start:
            raise ValueError("Require a positive finite bank and a nonnegative horizon")

        predecessors = defaultdict(set)
        for c in self.connections.values():
            predecessors[c.arr_stop].add(c.dep_stop)
        reachable, pending = {destination}, deque([destination])
        while pending:
            for stop in predecessors[pending.popleft()] - reachable:
                reachable.add(stop)
                pending.append(stop)

        zero_time_order = TopologicalSorter({s: set() for s in sorted(reachable)})
        for c in self.connections.values():
            if c.dep_stop == c.arr_stop or not {c.dep_stop, c.arr_stop} <= reachable:
                continue
            if c.dep_time == c.arr_time or any(
                not (event := scene.events[c.id]).canceled and event.departure == event.arrival
                for _, scene in bank
            ):
                zero_time_order.add(c.arr_stop, c.dep_stop)
        self.rank = {stop: rank for rank, stop in enumerate(zero_time_order.static_order())}
        self.destination_rank = self.rank[destination]
        self.worlds = tuple(_ArrivalLabels(self, scene) for _, scene in bank)

    @property
    def label_count(self):
        return len(self.rank) * (self.deadline - self.start) * len(self.worlds)

    def arc(self, cid, event):
        c = self.connections[cid]
        if event.canceled:
            return None
        if event.arrival < event.departure:
            raise ValueError("A ride cannot arrive before it departs")
        if (c.dep_stop == c.arr_stop or c.dep_stop not in self.rank or c.arr_stop not in self.rank
                or not self.start <= event.departure < self.deadline
                or event.arrival > self.deadline):
            return None
        tail = event.departure, self.rank[c.dep_stop]
        head = event.arrival, self.rank[c.arr_stop]
        if head <= tail:
            raise ValueError("This new same-minute ride requires rebuilding the topological order")
        return tail, head

    def arrival(self, world, stop, time):
        if time >= self.deadline or stop not in self.rank:
            return self.deadline
        if time < self.start:
            raise ValueError("Query predates the relaxation horizon")
        return self.worlds[world].value((time, self.rank[stop]))

    def repair(self, world, changed_events):
        """Update one world by explicit connection-to-event deltas."""
        return self.worlds[world].update(changed_events)

    def committed_model(self, state, lookahead=None):
        """Build a fresh executable snapshot using the same updated world bank."""
        bank = [(float(weight / self.total_weight), JourneyScenario(world.events, self.minute,
                                                                    world.publications))
                for weight, world in zip(self.weights, self.worlds)]
        return build_committed_model(self.graph, bank, state.stop, self.destination,
                                     state.time, self.deadline, lookahead=lookahead,
                                     initial_state=state)


@dataclass(frozen=True)
class CandidateCertificate:
    lower: Fraction
    upper: Fraction
    action_lower: dict
    removed: tuple
    transitions: int

    @property
    def gap(self):
        return self.upper - self.lower


def certify(relaxation, state, policy, *, support=None, policy_menu=None, lookahead=None):
    """Bound the full optimum and certify strictly dominated current actions.

    For a state reached from the fixed bank, `support` contains exactly the
    worlds compatible with its public history. Deterministic observations
    condition the weights by deleting incompatible worlds.
    `policy(state, actions)` must use public state only and be deterministic.
    If supplied, `policy_menu` restricts that witness at every visited state.
    """
    model = relaxation.committed_model(state, lookahead)
    root_actions = tuple(model.actions(state))
    action_lower = dict.fromkeys(root_actions, Fraction(0))
    upper = Fraction(0)
    transitions = 0
    support = tuple(range(len(model.scenarios))) if support is None else tuple(support)
    total_weight = sum(relaxation.weights[i] for i in support)
    if not total_weight:
        raise ValueError("The public history has zero mass in the scenario bank")
    for i in support:
        weight, scenario = relaxation.weights[i], model.scenarios[i]
        if not weight:
            continue
        first = {}
        for action in root_actions:
            outcome = model.transition(scenario, state, action)
            transitions += 1
            first[action] = outcome
            continuation = (model.terminal_cost(scenario, outcome.state) if outcome.terminal else
                            relaxation.arrival(i, outcome.state.stop, outcome.state.time) - outcome.state.time)
            action_lower[action] += weight * Fraction(outcome.cost + continuation)

        current = state
        cost = 0
        for step in range(model.horizon):
            actions = root_actions if step == 0 else tuple(model.actions(current))
            if not actions:
                break
            allowed = tuple(policy_menu(current, actions)) if policy_menu else actions
            action = policy(current, allowed)
            if action not in allowed or action not in actions:
                raise ValueError("Witness policy must choose a legal action in its menu")
            if step == 0:
                outcome = first[action]
            else:
                outcome = model.transition(scenario, current, action)
                transitions += 1
            cost += outcome.cost
            current = outcome.state
            if outcome.terminal:
                break
        upper += weight * Fraction(cost + model.terminal_cost(scenario, current))

    action_lower = {action: value / total_weight for action, value in action_lower.items()}
    upper /= total_weight
    lower = min(action_lower.values(), default=upper)
    removed = tuple(action for action, value in action_lower.items() if value > upper)
    return CandidateCertificate(lower, upper, action_lower, removed, transitions)
