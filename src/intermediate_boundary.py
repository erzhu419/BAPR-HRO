"""First-exit relaxation for an issued same-vehicle intermediate fragment.

Bounds here concern a supplied, frozen physical probability model. External
messages may be arbitrarily correlated with the current vehicle. Runtime and
departure floors must hold on its *entire* admissible support; empirical minima
are not automatically such floors. The construction is a research prototype,
not a new generic macro-action or upper/lower-bound principle.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from math import inf, isclose

from .information_planner import InformationState


@dataclass(frozen=True)
class FirstExit:
    prefix: int
    stop: int
    kind: str
    connection: int | None = None
    footpath: tuple | None = None


@dataclass(frozen=True)
class BoundaryCertificate:
    lower: float
    upper: float
    gap: float
    accepted: bool
    local_trajectories: int
    boundary_checks: int
    local_bounds: tuple


@dataclass(frozen=True)
class PolicyValue:
    value: float
    transitions: int
    information_nodes: int


@dataclass(frozen=True)
class MacroReference:
    advantage: float
    incremental_cost: float
    boundary_optimum: float
    exits: tuple


def macro_advantage_reference(planner, policy, stop_when, node=None):
    """Exact diagnostic, using one model and public-history stopping times.

    This deliberately invokes the full planner at boundaries; it is for
    verification, not the cheap certificate. Every exit tuple contains its
    conditional mass, information state and terminal flag. The macro executes
    at least one action before applying the public ``stop_when`` predicate.
    """
    model = planner.model
    node = planner.root if node is None else node

    def visit(context):
        actions = tuple(model.actions(context.state))
        if context.stage >= model.horizon or not actions:
            return 0., ((1., context, True),)
        action = policy(context)
        if action not in actions:
            raise ValueError('The macro witness chose an unavailable action')
        groups = defaultdict(list)
        for i, p in zip(context.support, context.belief):
            groups[model.transition(model.scenarios[i], context.state, action)].append((i, p))
        cost, exits = 0., []
        for outcome, members in groups.items():
            mass = sum(p for _, p in members)
            event = action, outcome.state, outcome.observation, outcome.cost, outcome.terminal
            child = InformationState(context.stage + 1, outcome.state,
                tuple(i for i, _ in members), tuple(p / mass for _, p in members),
                context.history + (event,))
            if outcome.terminal or stop_when(child):
                suffix_cost, suffix_exits = 0., ((1., child, outcome.terminal),)
            else:
                suffix_cost, suffix_exits = visit(child)
            cost += mass * (outcome.cost + suffix_cost)
            exits.extend((mass * p, state, terminal) for p, state, terminal in suffix_exits)
        return cost, tuple(exits)

    cost, exits = visit(node)
    continuation = 0.
    for p, child, terminal in exits:
        value = (sum(q * model.terminal_cost(model.scenarios[i], child.state)
                     for i, q in zip(child.support, child.belief)) if terminal
                 else planner.solve(child).value)
        continuation += p * value
    return MacroReference(cost + continuation - planner.solve(node).value,
                          cost, continuation, exits)


def evaluate_causal_policy(model, policy, node=None):
    """Evaluate only the supplied policy, without solving optimal subtrees.

    This retains the full public history and all correlations of the bank. Its
    model-transition count is at most worlds times maximum executed length,
    rather than expanding actions the policy never chooses. Transition and
    public-history bookkeeping costs are additional. It is a finite-bank value.
    """
    if node is None:
        support = tuple(i for i, world in enumerate(model.scenarios) if world.probability)
        node = InformationState(0, model.initial_state, support,
                                tuple(model.scenarios[i].probability for i in support))
    transitions = 0
    nodes = 0

    def terminal(context):
        return sum(p * model.terminal_cost(model.scenarios[i], context.state)
                   for i, p in zip(context.support, context.belief))

    @lru_cache(None)
    def visit(context):
        nonlocal transitions, nodes
        nodes += 1
        actions = tuple(model.actions(context.state))
        if context.stage >= model.horizon or not actions:
            return terminal(context)
        action = policy(context)
        if action not in actions:
            raise ValueError('A witness must choose a physical action from public history')
        groups = defaultdict(list)
        for i, p in zip(context.support, context.belief):
            outcome = model.transition(model.scenarios[i], context.state, action)
            transitions += 1
            groups[outcome].append((i, p))
        result = 0.
        for outcome, members in groups.items():
            mass = sum(p for _, p in members)
            event = action, outcome.state, outcome.observation, outcome.cost, outcome.terminal
            child = InformationState(context.stage + 1, outcome.state,
                tuple(i for i, _ in members), tuple(p / mass for _, p in members),
                context.history + (event,))
            result += mass * (outcome.cost + (terminal(child) if outcome.terminal else visit(child)))
        return result

    value = visit(node)
    return PolicyValue(value, transitions, nodes)


class FirstExitEnvelope:
    """Amortized physical lower envelope plus complete first-exit inventory.

    Footpaths are (from_stop, to_stop, duration) in the same integer clock unit.
    ``transfer_floor[(incoming_cid,outgoing_cid)]`` contains physical lower
    bounds for the applicable platform pair; missing entries use the valid
    optimistic zero. ``incoming_cid=None`` denotes the initial onboard state.

    Construction costs O((V+E+W)log V). A K-leg inventory contains every outgoing
    alternative and walk at every prefix, plus alighting/skipping, service
    failure and its intermediate endpoint. It costs O(K+sum prefix degrees).
    Given S local trajectories, integration costs O(S*inventory size). Cost of
    obtaining support floors, the local law and a feasible witness U is extra.
    """

    def __init__(self, graph, destination, runtime_floor, departure_floor, *,
                 footpaths=(), transfer_floor=None):
        self.graph, self.destination = graph, destination
        self.runtime_floor, self.departure_floor = dict(runtime_floor), dict(departure_floor)
        if set(self.runtime_floor) != {c.id for c in graph.connections} or set(self.departure_floor) != set(self.runtime_floor):
            raise ValueError('Support floors must cover every physical connection')
        self.footpaths = tuple(footpaths)
        self._walk_index = defaultdict(list)
        self.transfer_floor = {} if transfer_floor is None else dict(transfer_floor)
        reverse = defaultdict(list)
        for c in graph.connections:
            duration = self.runtime_floor[c.id]
            if duration < 0:
                raise ValueError('Runtime floors must be nonnegative')
            reverse[c.arr_stop].append((c.dep_stop, duration))
        for source, target, duration in self.footpaths:
            if duration <= 0:
                raise ValueError('Walks in this model have positive elapsed duration')
            reverse[target].append((source, duration))
            self._walk_index[source].append((source, target, duration))
        self.distance = {destination: 0.}
        queue = [(0., destination)]
        while queue:
            distance, stop = heappop(queue)
            if distance != self.distance[stop]:
                continue
            for before, duration in reverse[stop]:
                value = distance + duration
                if value < self.distance.get(before, inf):
                    self.distance[before] = value
                    heappush(queue, (value, before))

    def inventory(self, path, stop, used=()):
        """List all first physical deviations; no MCT/support heuristic prunes."""
        path = tuple(path)
        if not path:
            raise ValueError('An intermediate fragment must contain a vehicle leg')
        trip = self.graph.connections[path[0]].trip_id
        seen = set(used)
        result = []
        for index, cid in enumerate(path):
            c = self.graph.connections[cid]
            if c.dep_stop != stop or c.trip_id != trip or cid in seen:
                raise ValueError('Fragment must be a contiguous unused same-trip path')
            result += [FirstExit(index, stop, 'alight_skip', cid),
                       FirstExit(index, stop, 'service_failure', cid)]
            for other in self.graph.get_connections_from(stop):
                if other.id == cid or other.id in seen:
                    continue
                same = other.trip_id == trip
                if other.pickup_allowed or same:
                    result.append(FirstExit(index, stop, 'same_trip_loop' if same else 'transfer', other.id))
            result += [FirstExit(index, stop, 'walk', footpath=walk)
                       for walk in self._walk_index[stop]]
            stop = c.arr_stop
            seen.add(cid)
        result.append(FirstExit(len(path), stop, 'decision_boundary'))
        return tuple(result)

    def _after_skip_lower(self, stop, time, unavailable):
        # A skipped departure cannot be reintroduced by the static distance
        # potential. Respect the first available outgoing action, then relax
        # all later time/service constraints. MCT zero is optimistic here.
        values, work = [], 0
        if stop == self.destination:
            return time, work
        for c in self.graph.get_connections_from(stop):
            work += 1
            if c.id not in unavailable:
                values.append(max(time, self.departure_floor[c.id]) + self.runtime_floor[c.id]
                              + self.distance.get(c.arr_stop, inf))
        for _, target, duration in self._walk_index[stop]:
            work += 1
            values.append(time + duration + self.distance.get(target, inf))
        return min(values, default=inf), work

    def local_lower(self, trajectory, path, start, *, incoming_cid=None, deadline,
                    used=()):
        """Pathwise optimistic final *elapsed* loss for any full causal policy.

        ``trajectory`` contains current-vehicle clocks/service only. A skip of
        an operating departure can reconsider after its integer-clock event;
        simultaneous alternatives are already in the transfer/loop inventory.
        A failure branch receives the more optimistic current prefix time,
        avoiding an invented notification time. No fixed-failure-clock rescue
        identity is assumed after changing the execution contract.
        """
        path = tuple(path)
        stop = self.graph.connections[path[0]].dep_stop
        exits = self.inventory(path, stop, used)
        groups = defaultdict(list)
        for exit_ in exits:
            groups[exit_.prefix].append(exit_)
        time, previous, unavailable = start, incoming_cid, set(used)
        candidates, checks = [], 0
        cap = deadline - start

        def capped(arrival):
            return min(cap, max(0., arrival - start))

        for index, cid in enumerate(path):
            c, event = self.graph.connections[cid], trajectory[cid]
            for exit_ in groups[index]:
                checks += 1
                if exit_.kind == 'walk':
                    _, target, duration = exit_.footpath
                    candidates.append(capped(time + duration + self.distance.get(target, inf)))
                elif exit_.kind in ('transfer', 'same_trip_loop'):
                    other = self.graph.connections[exit_.connection]
                    transfer = (0. if exit_.kind == 'same_trip_loop' else
                                self.transfer_floor.get((previous, other.id), 0.))
                    ready = max(time + transfer, self.departure_floor[other.id])
                    candidates.append(capped(ready + self.runtime_floor[other.id]
                                             + self.distance.get(other.arr_stop, inf)))
                elif exit_.kind == 'alight_skip' and not event.canceled and event.departure >= time:
                    arrival, work = self._after_skip_lower(stop, event.departure + 1, unavailable | {cid})
                    checks += work
                    candidates.append(capped(arrival))
            if event.canceled or event.departure < time or event.departure >= deadline:
                candidates.append(capped(time + self.distance.get(stop, inf)))
                break
            stop, time, previous = c.arr_stop, event.arrival, cid
            unavailable.add(cid)
        else:
            checks += 1
            candidates.append(capped(time + self.distance.get(stop, inf)))
        return min(candidates), checks

    def certify(self, local_law, path, start, witness_upper, *, deadline,
                tolerance, incoming_cid=None, used=()):
        """Compare with a separately established feasible causal policy upper.

        A finite-bank policy value cannot be substituted for a full-model U.
        A new root projection cannot be silently substituted for one coherent
        trajectory model when composing certificates across histories.
        The result bounds the supplied witness's full cost. To bound the
        proposed fragment's advantage, that witness must actually execute this
        fragment up to its observable boundary before following its causal
        tail. The lower relaxation's optimistic failure time is not that
        observable stopping time.
        """
        local_law = tuple(local_law)
        if not isclose(sum(p for p, _ in local_law), 1., abs_tol=1e-12):
            raise ValueError('The local marginal must sum to one')
        lower, checks, records = 0., 0, []
        for probability, trajectory in local_law:
            value, work = self.local_lower(trajectory, path, start, incoming_cid=incoming_cid,
                deadline=deadline, used=used)
            lower += probability * value
            checks += work
            records.append(value)
        if witness_upper < lower - 1e-9:
            raise ValueError('Witness upper and support lower describe inconsistent models')
        gap = max(0., witness_upper - lower)
        return BoundaryCertificate(lower, witness_upper, gap, gap <= tolerance,
                                   len(local_law), checks, tuple(records))
