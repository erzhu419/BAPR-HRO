"""Algebraic reductions of the existing frozen-history continuation planner."""

from time import perf_counter

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v32.run import nominal_policy
from experiments.reconstruction_v33.planner import CompiledRide, _PlatformCompiler
from experiments.reconstruction_v34.forecast import ForecastPlanner
from src.full_menu_opportunity import FullMenuOpportunity
from src.onboard_stopping import OnboardStopping


def rolling_fixed_policy(chain, initial_mass):
    """Compute exactly the rolling-fixed member, without the other two policies."""
    initial_mass = chain._initial(initial_mass)
    m = len(chain.times)
    actions, comparisons = [None]*m, [None]*m
    following = {}
    for stage in range(m-1, -1, -1):
        fixed = {stage: chain.alight_costs[stage].copy()}
        for target in range(stage+1, m):
            fixed[target] = chain._continue(stage, following[target])
        continuation = (np.minimum.reduce([fixed[target] for target in range(stage+1, m)])
                        if stage+1 < m else np.full(len(chain.times[stage]), chain.deadline))
        actions[stage] = chain._choose(chain.alight_costs[stage], continuation)
        comparisons[stage] = np.column_stack((chain.alight_costs[stage], continuation))
        following = fixed
    return chain.evaluate_policy(actions, initial_mass, comparisons)


class BatchPlatformCompiler(_PlatformCompiler):
    def clock_nominal_scores(self, cid, clocks):
        """The same scalar linear interpolation applied to an array of clocks."""
        clocks = np.asarray(clocks)
        self.planner.stats['batch_score_calls'] += 1
        self.planner.stats['batch_score_points'] += clocks.size
        if not clocks.size:
            return np.empty(clocks.shape, dtype=float)
        compiled = self.compiled[cid]
        support = compiled.departure_times
        if (not len(support) or (clocks < support[0]).any() or (clocks > support[-1]).any()):
            raise UnsupportedHistory('Visible departure lies outside conditional operating support')
        return np.interp(clocks, support, compiled.board_scores)


class BatchOpportunity(FullMenuOpportunity):
    """Keep all original waiting/selection routines; batch only initial marks."""
    def __init__(self, compiler, ranks, entry, ready, deadline):
        self.compiler, self.ranks = compiler, dict(ranks)
        self.entry, self.ready, self.deadline = entry, ready, deadline
        self.cids = tuple(ranks)
        self.distributions, self.cdfs, self.absent, self.marks = {}, {}, {}, {}
        self.mass_tails = {}
        for cid in self.cids:
            times, mass, cancel = compiler.departure_distribution(cid)
            times, mass = np.asarray(times), np.asarray(mass)
            self.distributions[cid] = times, mass, cancel
            self.cdfs[cid] = np.r_[0., np.cumsum(mass)]
            self.mass_tails[cid] = np.r_[np.cumsum(mass[::-1])[::-1], 0.]
            self.absent[cid] = float(cancel+mass[times < entry].sum())
            legal = (times >= ready) & (times < deadline) & (mass > 0)
            clocks = times[legal]
            # Preserve the original constructor's int(t) conversion. Actual
            # off-grid public offers still use the inherited scalar score().
            scores = compiler.clock_nominal_scores(cid, clocks.astype(np.int64))
            self.marks[cid] = clocks, mass[legal], scores


class FastPlanner(ForecastPlanner):
    """Same values and policies, without unused terminal matrices or policies."""
    def __init__(self, graph, model, parameters, deadline):
        super().__init__(graph, model, parameters, deadline)
        self.stats.update(terminal_vector_compiles=0, terminal_vector_steps=0,
                          rolling_only_compiles=0, batch_score_calls=0, batch_score_points=0)

    def platform_compiler(self, platform, history, method):
        self.stats['platform_queries'] += 1
        key = self._node_key(platform, history, method)
        if key in self._platforms:
            self.stats['platform_hits'] += 1
            return self._platforms[key]
        compiler = BatchPlatformCompiler(self, platform, history, method)
        self._platforms[key] = compiler
        self.stats['platform_compiles'] += 1
        return compiler

    def platform_reference(self, platform, history, entry, ready, method='optimal'):
        return BatchOpportunity(self.platform_compiler(platform, history, method),
                                self.ranks(platform), entry, ready, self.deadline)

    def first_offer_values(self, platform, history, ready, method='optimal'):
        self.stats['first_offer_queries'] += 1
        ready = np.asarray(ready, dtype=float)
        shape, flat = ready.shape, ready.reshape(-1)
        key = (*self._node_key(platform, history, method), tuple(flat))
        if key in self._first_offer:
            self.stats['first_offer_hits'] += 1
            return self._first_offer[key].reshape(shape)
        compiler = self.platform_compiler(platform, history, method)
        ranks, marks, prefix, suffix = self.ranks(platform), {}, {}, {}
        for cid in platform.rides:
            clocks, mass, cancellation = compiler.departure_distribution(cid)
            positive = mass > 0
            clocks, mass = clocks[positive], mass[positive]
            scores = compiler.clock_nominal_scores(cid, clocks)
            marks[cid] = clocks, mass, cancellation, scores
            prefix[cid] = np.r_[0., np.cumsum(mass)]
            suffix[cid] = np.r_[np.cumsum(mass[::-1])[::-1], 0.]
        gain = np.zeros(len(flat))
        for cid, (clocks, mass, _, scores) in marks.items():
            legal = (clocks[None, :] >= flat[:, None]) & (clocks[None, :] < self.deadline)
            selected = legal*mass[None, :]
            for other, (other_clocks, _, cancellation, _) in marks.items():
                if other == cid:
                    continue
                lower = prefix[other][np.searchsorted(other_clocks, flat, side='left')]
                side = 'right' if ranks[other] < ranks[cid] else 'left'
                upper = suffix[other][np.searchsorted(other_clocks, clocks, side=side)]
                selected *= cancellation+lower[:, None]+upper[None, :]
            gain += selected @ (self.deadline-scores)
        result = self.deadline-gain
        self._first_offer[key] = result
        self.stats['first_offer_compiles'] += 1
        return result.reshape(shape)

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
            costs = [self.exit_cost(ride, cid, history, clocks, method)
                     for cid, clocks in zip(arrivals, times)]
            departure_mass = law.marginal(departure_slot)[~law.canceled[departure_slot]]
            terminal = (len(arrivals) == 1 and arrivals[0] == ride.path[-1]
                        and ride.exits[arrivals[0]] is None)
            if terminal:
                edges = [self._edge(trip, slot, history)[np.ix_(~law.canceled[slot], ~law.canceled[slot+1])]
                         for slot in range(departure_slot, slots[0])]
                initial = departure_mass
                for edge in edges:
                    initial = initial @ edge
                kernels = []
                self.stats['terminal_vector_compiles'] += 1
                self.stats['terminal_vector_steps'] += len(edges)
            else:
                kernels = [self._kernel(trip, left, right, history) for left, right in zip(slots, slots[1:])]
                departure_to_arrival = self._kernel(trip, departure_slot, slots[0], history)
                initial = departure_mass @ departure_to_arrival
            chain = OnboardStopping(times, costs, kernels, self.deadline)
            if method == 'nominal':
                policy = nominal_policy(chain, initial)
            elif method == 'rolling_fixed':
                policy = rolling_fixed_policy(chain, initial)
                self.stats['rolling_only_compiles'] += 1
            else:
                policy = chain.solve(initial).policies[method]
            if terminal:
                gain = self.deadline-policy.values[0]
                for edge in reversed(edges):
                    gain = edge @ gain
                scores = self.deadline-gain
            else:
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
