"""Recursive standard stopping with conditional shared-trip messages.

Every ride retains all its declared exits, including rides after a transfer.
At a platform, the lookahead integrates first opportunity using child ride
scores frozen under the supplied public history. This is a rolling surrogate
when another root candidate also appears in a child's continuation; it is not
an exact joint information tree. Execution supplies updated history and
replans at each later observed arrival or local offer.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v32.run import nominal_policy
from src.evolving_trip_belief import logsumexp
from src.full_menu_opportunity import FullMenuOpportunity
from src.onboard_stopping import OnboardStopping


@dataclass(frozen=True)
class CompiledRide:
    arrival_cids: tuple
    times: tuple
    policy: object
    departure_times: np.ndarray
    board_scores: np.ndarray
    deadline: float
    method: str
    conditional_scope: str

    def score(self, clock):
        if (not len(self.departure_times) or clock < self.departure_times[0]
                or clock > self.departure_times[-1]):
            return None
        return float(np.interp(clock, self.departure_times, self.board_scores))

    def decision(self, arrival_cid, clock):
        stage = self.arrival_cids.index(arrival_cid)
        times = self.times[stage]
        if not times[0] <= clock <= times[-1]:
            raise UnsupportedHistory('Observed arrival lies outside the frozen ride support')
        costs = self.policy.decision_costs[stage]
        alight, continuation = (float(np.interp(clock, times, costs[:, i])) for i in (0, 1))
        choice = alight < self.deadline and alight <= continuation+1e-9
        return choice, dict(method=self.method, stage=stage, alight=alight,
            continuation=continuation, arrival_cid=arrival_cid, clock=clock,
            clock_mode='on_grid' if clock in times else 'adjacent_state_projection',
            value_scope=self.conditional_scope)


class _PlatformCompiler:
    def __init__(self, planner, platform, history, method):
        self.planner, self.platform = planner, platform
        self.method = method
        # History advances during execution. Freeze these arrays with the
        # child scores so a warm query cannot read the previous run's future.
        self.distributions = {cid: planner.departure_distribution(cid, history)
                              for cid in platform.rides}
        self.compiled = {cid: planner.compile_ride(ride, history, method)
                         for cid, ride in platform.rides.items()}

    def departure_distribution(self, cid):
        return self.distributions[cid]

    def clock_nominal_score(self, cid, clock):
        value = self.compiled[cid].score(clock)
        if value is None:
            raise UnsupportedHistory('Visible departure lies outside conditional operating support')
        return value


class Planner:
    """Cache only immutable node structure and relevant public trip evidence.

    No history is removed from the traveler. Excluding unrelated trip evidence
    from a subtree's cache key is valid for the frozen independent-trip model
    and fixed menu: those trips cannot affect any action or observation in that
    subtree. Keys are ordinary tuples, not added digests or fingerprints.
    """

    def __init__(self, graph, model, parameters, deadline):
        self.graph, self.model, self.parameters = graph, model, parameters
        self.deadline = float(deadline)
        self._structures, self._trips, self._ranks = {}, {}, {}
        self._rides, self._platforms, self._first_offer = {}, {}, {}
        self._edges, self._kernels, self._departures = {}, {}, {}
        self.stats = dict(ride_queries=0, ride_hits=0, ride_compiles=0,
            platform_queries=0, platform_hits=0, platform_compiles=0,
            first_offer_queries=0, first_offer_hits=0, first_offer_compiles=0,
            edge_compiles=0, edge_cells=0, kernel_compiles=0,
            prior_future_edge_queries=0, prior_future_kernel_queries=0,
            top_level_compile_seconds=0.)
        self._compile_depth = 0

    def structure(self, node):
        if node not in self._structures:
            if hasattr(node, 'first_cid'):
                self._structures[node] = ('ride', node.first_cid, tuple(node.path),
                    tuple((cid, None if node.exits[cid] is None else self.structure(node.exits[cid]))
                          for cid in node.path if cid in node.exits))
            else:
                self._structures[node] = ('platform', node.stop,
                    tuple((cid, self.structure(ride)) for cid, ride in sorted(node.rides.items())))
        return self._structures[node]

    def relevant_trips(self, node):
        if node not in self._trips:
            if hasattr(node, 'first_cid'):
                trips = {self.model._positions[node.first_cid, 'departure'][0]}
                for child in node.exits.values():
                    if child is not None:
                        trips.update(self.relevant_trips(child))
            else:
                trips = {trip for child in node.rides.values() for trip in self.relevant_trips(child)}
            self._trips[node] = frozenset(trips)
        return self._trips[node]

    def _node_key(self, node, history, method):
        return self.structure(node), history.signature(self.relevant_trips(node)), method

    def _trip_key(self, trip, history):
        return trip, history.signature((trip,))

    def _transition_key(self, trip, left, history):
        # Conditioning at or before X_left does not alter P(X_right|X_left).
        # Keep the complete key if any later evidence exists: its projected
        # likelihood can itself depend on the earlier conditioning history.
        records = getattr(history, '_records', None)
        prior_future = records is not None and all(
            self.model._positions[record[0], record[1]][1] <= left
            for record in records.get(trip, ()))
        return ((trip, ('prior_future',)) if prior_future else self._trip_key(trip, history)), prior_future

    def _edge(self, trip, slot, history):
        trip_key, prior_future = self._transition_key(trip, slot, history)
        key = (*trip_key, slot)
        self.stats['prior_future_edge_queries'] += prior_future
        if key not in self._edges:
            law = self.model._law(trip, self.parameters) if prior_future else history.law(trip)
            backward = law._backward_from(max(slot+1, law._last_effect), slot+1)
            weights = law._transition(slot)+law.emissions[slot+1]+backward
            normalizer = logsumexp(weights, axis=1, keepdims=True)
            edge = np.exp(weights-np.where(np.isfinite(normalizer), normalizer, 0.))
            self._edges[key] = edge
            self.stats['edge_compiles'] += 1
            self.stats['edge_cells'] += edge.size
        return self._edges[key]

    def _kernel(self, trip, left, right, history):
        trip_key, prior_future = self._transition_key(trip, left, history)
        key = (*trip_key, left, right)
        self.stats['prior_future_kernel_queries'] += prior_future
        if key not in self._kernels:
            law = self.model._law(trip, self.parameters) if prior_future else history.law(trip)
            value = np.eye(int((~law.canceled[left]).sum()))
            for slot in range(left, right):
                edge = self._edge(trip, slot, history)
                value = value @ edge[np.ix_(~law.canceled[slot], ~law.canceled[slot+1])]
            self._kernels[key] = value
            self.stats['kernel_compiles'] += 1
        return self._kernels[key]

    def departure_distribution(self, cid, history):
        trip, slot = self.model._positions[cid, 'departure']
        key = (*self._trip_key(trip, history), slot)
        if key not in self._departures:
            law = history.law(trip)
            operating, probability = ~law.canceled[slot], law.marginal(slot)
            self._departures[key] = (law.times[slot][operating], probability[operating],
                                     float(probability[~operating].sum()))
        return self._departures[key]

    def scheduled_arrival(self, ride):
        key = self.structure(ride)
        if key not in self._ranks:
            values = []
            for cid, platform in ride.exits.items():
                arrival = self.graph.connections[cid].arr_time
                if platform is None:
                    values.append(arrival)
                else:
                    ready = arrival+self.graph.stops[platform.stop].min_transfer_time
                    values.extend(self.scheduled_arrival(child) for first, child in platform.rides.items()
                                  if self.graph.connections[first].dep_time >= ready)
            self._ranks[key] = min(values, default=self.deadline)
        return self._ranks[key]

    def ranks(self, platform):
        return {cid: (self.scheduled_arrival(ride), cid) for cid, ride in platform.rides.items()}

    def platform_compiler(self, platform, history, method):
        self.stats['platform_queries'] += 1
        key = self._node_key(platform, history, method)
        if key in self._platforms:
            self.stats['platform_hits'] += 1
            return self._platforms[key]
        compiler = _PlatformCompiler(self, platform, history, method)
        self._platforms[key] = compiler
        self.stats['platform_compiles'] += 1
        return compiler

    def platform_reference(self, platform, history, entry, ready, method='optimal'):
        compiler = self.platform_compiler(platform, history, method)
        return FullMenuOpportunity(compiler, self.ranks(platform), entry, ready, self.deadline)

    def first_offer_values(self, platform, history, ready, method='optimal'):
        """Integrate the first legal offer with the current frozen child scores.

        Children are independent at their root departure slots. Their terminal
        marks can share other child vehicles, so this frozen-mark expression
        is a declared surrogate for that larger correlated continuation.
        """
        self.stats['first_offer_queries'] += 1
        ready = np.asarray(ready, dtype=float)
        shape, flat = ready.shape, ready.reshape(-1)
        key = (*self._node_key(platform, history, method), tuple(flat))
        if key in self._first_offer:
            self.stats['first_offer_hits'] += 1
            return self._first_offer[key].reshape(shape)
        compiler = self.platform_compiler(platform, history, method)
        ranks = self.ranks(platform)
        marks = {}
        for cid in platform.rides:
            clocks, mass, cancellation = compiler.departure_distribution(cid)
            positive = mass > 0
            clocks, mass = clocks[positive], mass[positive]
            scores = np.array([compiler.clock_nominal_score(cid, t) for t in clocks])
            marks[cid] = clocks, mass, cancellation, scores
        gain = np.zeros(len(flat))
        for cid, (clocks, mass, _, scores) in marks.items():
            legal = (clocks[None, :] >= flat[:, None]) & (clocks[None, :] < self.deadline)
            selected = legal*mass[None, :]
            for other, (other_clocks, other_mass, cancellation, _) in marks.items():
                if other == cid:
                    continue
                before = np.r_[0., np.cumsum(other_mass)]
                after = np.r_[np.cumsum(other_mass[::-1])[::-1], 0.]
                lower = before[np.searchsorted(other_clocks, flat, side='left')]
                side = 'right' if ranks[other] < ranks[cid] else 'left'
                upper = after[np.searchsorted(other_clocks, clocks, side=side)]
                selected *= cancellation+lower[:, None]+upper[None, :]
            gain += selected @ (self.deadline-scores)
        result = self.deadline-gain
        self._first_offer[key] = result
        self.stats['first_offer_compiles'] += 1
        return result.reshape(shape)

    def exit_cost(self, ride, cid, history, clocks, method):
        """Cost of this declared exit; the base V33 behavior is unchanged."""
        platform = ride.exits[cid]
        if platform is None:
            return np.minimum(clocks, self.deadline).astype(float)
        ready = clocks+self.graph.stops[platform.stop].min_transfer_time
        return self.first_offer_values(platform, history, ready, method)

    def compile_ride(self, ride, history, method='optimal'):
        if method not in ('optimal', 'rolling_fixed', 'one_step_rollout', 'nominal'):
            raise ValueError(method)
        self.stats['ride_queries'] += 1
        key = self._node_key(ride, history, method)
        if key in self._rides:
            self.stats['ride_hits'] += 1
            return self._rides[key]
        top_level = self._compile_depth == 0
        started = perf_counter()
        self._compile_depth += 1
        try:
            trip, departure_slot = self.model._positions[ride.first_cid, 'departure']
            law = history.law(trip)
            arrivals = tuple(cid for cid in ride.path if cid in ride.exits)
            slots = [self.model._positions[cid, 'arrival'][1] for cid in arrivals]
            times = tuple(law.times[slot][~law.canceled[slot]] for slot in slots)
            costs = []
            for cid, clocks in zip(arrivals, times):
                costs.append(self.exit_cost(ride, cid, history, clocks, method))
            kernels = [self._kernel(trip, left, right, history) for left, right in zip(slots, slots[1:])]
            departure_to_arrival = self._kernel(trip, departure_slot, slots[0], history)
            departure_mass = law.marginal(departure_slot)[~law.canceled[departure_slot]]
            initial = departure_mass @ departure_to_arrival
            chain = OnboardStopping(times, costs, kernels, self.deadline)
            if method == 'nominal':
                policy = nominal_policy(chain, initial)
            else:
                policy = chain.solve(initial).policies[method]
            scores = self.deadline-departure_to_arrival @ (self.deadline-policy.values[0])
            operating = ~law.canceled[departure_slot]
            positive = departure_mass > 0
            result = CompiledRide(arrivals, times, policy, law.times[departure_slot][operating][positive],
                scores[positive], self.deadline, method,
                'Conditional ride chain with fixed-history first-offer child scores; not a joint information-tree optimum')
            self._rides[key] = result
            self.stats['ride_compiles'] += 1
            return result
        finally:
            self._compile_depth -= 1
            if top_level:
                self.stats['top_level_compile_seconds'] += perf_counter()-started
