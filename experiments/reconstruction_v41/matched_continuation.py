"""One-transfer waiting corrections, distinct from full physical policy value.

The inexpensive correction selects between first opportunity and each fixed
target *after integrating* their frozen child marks. The finite reference
instead evaluates every refusal made by CombinedWaitingOpportunity. Neither
integrates future changes to the child marks or future public messages.
"""

from dataclasses import replace

import numpy as np

from experiments.reconstruction_v41.physical_posterior import PhysicalPlanner


class FrozenCombinedReference:
    """Exact evaluation of fixed-Combined on independent finite offer laws.

    This is policy evaluation, not optimal waiting DP. A state contains the
    public clock and the still-unseen root candidates; simultaneous events
    are observed together. Marks remain those of the supplied reference.
    Complexity is O(T 3**K), with O(T 2**K) stored values, for small menus.
    """

    def __init__(self, reference):
        self.reference = reference
        self.cids = tuple(reference.cids)
        self.entry, self.ready, self.deadline = reference.entry, reference.ready, reference.deadline
        self.times = np.array(sorted({float(t) for cid in self.cids
            for t, p in zip(*reference.distributions[cid][:2], strict=True)
            if p > 0 and self.entry <= t < self.deadline}))
        nstates, ntime = 1 << len(self.cids), len(self.times)
        self.full_mask = nstates-1
        self.bit_by_cid = {cid: 1 << i for i, cid in enumerate(self.cids)}
        self.values = np.full((ntime+1, nstates), np.nan)
        self.first_values = np.full_like(self.values, np.nan)
        survival = np.zeros((ntime+1, len(self.cids)))
        hazards = np.zeros((ntime, len(self.cids)))
        single_wait = np.full_like(hazards, self.deadline)
        scores = np.full_like(hazards, self.deadline)
        boundaries = np.r_[self.times, self.deadline]
        supported = np.ones_like(self.values, dtype=bool)
        for i, cid in enumerate(self.cids):
            times, mass, cancel = reference.distributions[cid]
            times, mass = np.asarray(times), np.asarray(mass)
            tail = np.r_[np.cumsum(mass[::-1])[::-1], 0.]
            absent = cancel+mass[times < self.entry].sum()
            survival[:, i] = absent+tail[np.searchsorted(times, boundaries, side='left')]
            for t, p in zip(times, mass, strict=True):
                if p > 0 and self.entry <= t < self.deadline:
                    j = int(np.searchsorted(self.times, t))
                    hazards[j, i] += p
                    if t >= self.ready:
                        scores[j, i] = reference.score(cid, t)
            np.divide(hazards[:, i], survival[:-1, i], out=hazards[:, i],
                      where=survival[:-1, i] > 0)
            clocks, probability, marks = reference.marks[cid]
            gain_tail = np.r_[np.cumsum((probability*(self.deadline-marks))[::-1])[::-1], 0.]
            gain = gain_tail[np.searchsorted(clocks, self.times, side='right')]
            after = absent+tail[np.searchsorted(times, self.times, side='right')]
            single_wait[:, i] -= np.divide(gain, after, out=np.zeros_like(gain), where=after > 0)
            included = np.arange(nstates) & (1 << i) != 0
            supported[:, included] &= survival[:, i, None] > 0
        self.values[-1, supported[-1]] = self.deadline
        self.first_values[-1, supported[-1]] = self.deadline
        bits, subsets = [], []
        for mask in range(nstates):
            active = [i for i in range(len(self.cids)) if mask & (1 << i)]
            outcomes = np.array([0], dtype=int)
            for i in active:
                outcomes = np.r_[outcomes, outcomes | (1 << i)]
            bits.append(active)
            subsets.append(outcomes)
        rank_order = sorted(range(len(self.cids)), key=lambda i: reference.ranks[self.cids[i]])
        transitions = 0
        for j in range(ntime-1, -1, -1):
            legal = self.times[j] >= self.ready
            best_board = np.full(nstates, self.deadline)
            ranked_board = np.full(nstates, self.deadline)
            wait = np.full(nstates, self.deadline)
            for mask in range(1, nstates):
                bit = mask & -mask
                i = bit.bit_length()-1
                best_board[mask] = min(best_board[mask ^ bit], scores[j, i])
                ranked_board[mask] = scores[j, next(i for i in rank_order if mask & (1 << i))]
                target = min(single_wait[j, i] for i in bits[mask])
                wait[mask] = min(target, self.first_values[j+1, mask])
            for mask in range(nstates):
                if not supported[j, mask]:
                    continue
                probability = np.ones(1)
                for i in bits[mask]:
                    hazard = hazards[j, i]
                    probability = np.r_[probability*(1-hazard), probability*hazard]
                positive = probability > 0
                departed = subsets[mask][positive]
                probability = probability[positive]
                remaining = mask ^ departed
                future = self.values[j+1, remaining]
                first_future = self.first_values[j+1, remaining]
                if np.any(~np.isfinite(future)):
                    raise ValueError('Positive event mass reached an unsupported waiting history')
                board = legal & (departed != 0) & (best_board[departed] <= wait[remaining]+1e-9)
                self.values[j, mask] = probability @ np.where(board, best_board[departed], future)
                self.first_values[j, mask] = probability @ np.where(
                    legal & (departed != 0), ranked_board[departed], first_future)
                transitions += len(probability)
        self.supported = supported
        self.value = float(self.values[0, self.full_mask])
        self.first_offer_value = float(self.first_values[0, self.full_mask])
        self.counters = dict(events=ntime, masks=nstates, positive_event_outcomes=transitions)

    def value_after(self, clock, seen):
        """Expected policy loss strictly after an already processed event."""
        mask = self.full_mask
        for cid in seen:
            mask &= ~self.bit_by_cid.get(cid, 0)
        j = int(np.searchsorted(self.times, clock, side='right'))
        if not self.supported[j, mask]:
            raise ValueError('The queried public history has zero probability')
        return float(self.values[j, mask])


class MatchedContinuationPlanner(PhysicalPlanner):
    """Cheap one-transfer commitment envelope; not exact actual-combined G_pi.

    Child ride marks are compiled by a separate unchanged PhysicalPlanner.
    Consequently this correction expands one platform only and never silently
    recurses into a new full continuation algorithm.
    """

    value_scope = ('One-transfer min(first-opportunity, single-target commitment) '
                   'with frozen original child marks; approximate actual combined continuation')

    def __init__(self, graph, model, parameters, deadline):
        super().__init__(graph, model, parameters, deadline)
        self.tail_base = PhysicalPlanner(graph, model, parameters, deadline)
        self._selective_tails = {}
        self.stats.update(selective_tail_queries=0, selective_tail_hits=0,
                          selective_tail_compiles=0, selective_tail_points=0)

    def _tail_stats(self):
        self.stats.update({'tail_base_'+key: value for key, value in self.tail_base.stats.items()})

    def selective_waiting_components(self, platform, history, ready, method='rolling_fixed'):
        self.stats['selective_tail_queries'] += 1
        ready = np.asarray(ready, dtype=float)
        shape, flat = ready.shape, ready.reshape(-1)
        key = (*self._node_key(platform, history, method), tuple(flat), shape)
        if key in self._selective_tails:
            self.stats['selective_tail_hits'] += 1
            return self._selective_tails[key]
        compiler = self.tail_base.platform_compiler(platform, history, method)
        first = self.tail_base.first_offer_values(platform, history, ready, method)
        by_target = {}
        for cid in platform.rides:
            clocks, mass, _ = compiler.departure_distribution(cid)
            positive = mass > 0
            clocks, mass = clocks[positive], mass[positive]
            scores = compiler.clock_nominal_scores(cid, clocks)
            legal = (clocks[None, :] >= flat[:, None]) & (clocks[None, :] < self.deadline)
            by_target[cid] = (self.deadline-legal @ (mass*(self.deadline-scores))).reshape(shape)
        single = np.minimum.reduce(list(by_target.values())) if by_target else np.full(shape, self.deadline)
        result = dict(value=np.minimum(first, single), first_offer=first,
                      single_target=single, by_target=by_target)
        self._selective_tails[key] = result
        self.stats['selective_tail_compiles'] += 1
        self.stats['selective_tail_points'] += flat.size
        self._tail_stats()
        return result

    def exit_cost(self, ride, cid, history, clocks, method):
        if history.announced_unavailable(cid, 'arrival'):
            return np.full_like(clocks, self.deadline, dtype=float)
        platform = ride.exits[cid]
        if platform is None:
            return np.minimum(clocks, self.deadline).astype(float)
        ready = clocks+self.graph.stops[platform.stop].min_transfer_time
        return self.selective_waiting_components(platform, history, ready, method)['value']

    def compile_ride(self, ride, history, method='optimal'):
        compiled = super().compile_ride(ride, history, method)
        if compiled.conditional_scope != self.value_scope:
            compiled = replace(compiled, conditional_scope=self.value_scope)
            self._rides[self._node_key(ride, history, method)] = compiled
        return compiled

    def frozen_combined_reference(self, platform, history, entry, ready, method='rolling_fixed'):
        """Small-menu exact reference, using the same unchanged child marks."""
        reference = self.tail_base.platform_reference(platform, history, entry, ready, method)
        result = FrozenCombinedReference(reference)
        self._tail_stats()
        return result
