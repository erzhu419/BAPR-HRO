"""Exact finite-offer waiting decisions for a frozen clock-nominal menu."""

import numpy as np


class FullMenuWaitingDP:
    """Solve repeated boarding/refusal on independent root departure laws.

    ``values[j, mask]`` is the value just before ``times[j]`` for the unseen
    vehicles in mask. The final row is the deadline. Unsupported zero-survival
    masks have NaN values and ``supported=False``; public queries explicitly
    fall back to the reference rule rather than treating them as certificates.

    Board costs are the reference's clock-nominal marks, not exact expected
    physical journey losses. Root offer events before readiness are observed
    and removed but cannot be boarded. No future actual clocks are read.
    """

    def __init__(self, reference):
        self.reference = reference
        self.cids = tuple(reference.cids)
        self.bit_by_cid = {cid: 1 << i for i, cid in enumerate(self.cids)}
        self.entry, self.ready, self.deadline = reference.entry, reference.ready, reference.deadline
        self.times = np.array(sorted({int(t) for cid in self.cids
            for t, p in zip(*reference.distributions[cid][:2], strict=True)
            if p > 0 and self.entry <= t < self.deadline}), dtype=np.int64)
        nstates = 1 << len(self.cids)
        self.full_mask = nstates-1
        self.values = np.full((len(self.times)+1, nstates), np.nan)
        self.supported = np.ones_like(self.values, dtype=bool)
        self.hazards = np.zeros((len(self.times), len(self.cids)))
        self.survival = np.zeros((len(self.times)+1, len(self.cids)))
        boundaries = np.r_[self.times, self.deadline]
        event_index = {int(time): j for j, time in enumerate(self.times)}
        for i, cid in enumerate(self.cids):
            times, mass, cancel = reference.distributions[cid]
            tail = np.r_[np.cumsum(mass[::-1])[::-1], 0.]
            absent = cancel+mass[times < self.entry].sum()
            self.survival[:, i] = absent+tail[np.searchsorted(times, boundaries, side='left')]
            atoms = np.zeros(len(self.times))
            for time, probability in zip(times, mass, strict=True):
                if probability > 0 and self.entry <= time < self.deadline:
                    atoms[event_index[int(time)]] += probability
            self.hazards[:, i] = np.divide(atoms, self.survival[:-1, i],
                out=np.zeros_like(atoms), where=self.survival[:-1, i] > 0)
            masks = np.arange(nstates) & (1 << i) != 0
            self.supported[:, masks] &= self.survival[:, i, None] > 0
        self.values[-1, self.supported[-1]] = self.deadline
        active_bits, subsets = [], []
        for mask in range(nstates):
            bits = [i for i in range(len(self.cids)) if mask & (1 << i)]
            ids = np.array([0], dtype=np.int64)
            for i in bits:
                ids = np.r_[ids, ids | (1 << i)]
            active_bits.append(bits)
            subsets.append(ids)
        self.counters = dict(events=len(self.times), masks=nstates, subset_outcomes=0,
                             wait_queries=0, decision_queries=0, unsupported_queries=0, offgrid_queries=0)
        for j in range(len(self.times)-1, -1, -1):
            clock = int(self.times[j])
            board = np.full(nstates, float(self.deadline))
            single = np.full(len(self.cids), float(self.deadline))
            if clock >= self.ready:
                for i, cid in enumerate(self.cids):
                    if self.hazards[j, i] > 0:
                        single[i] = reference.score(cid, clock)
                for mask in range(1, nstates):
                    bit = mask & -mask
                    board[mask] = min(board[mask ^ bit], single[bit.bit_length()-1])
            for mask in range(nstates):
                if not self.supported[j, mask]:
                    continue
                probability = np.ones(1)
                for i in active_bits[mask]:
                    hazard = self.hazards[j, i]
                    probability = np.r_[probability*(1-hazard), probability*hazard]
                positive = probability > 0
                departed = subsets[mask][positive]
                continuation = self.values[j+1, mask ^ departed]
                if np.any(~np.isfinite(continuation)):
                    raise ValueError('Positive-probability event leads to an unsupported waiting state')
                self.values[j, mask] = probability[positive] @ np.minimum(board[departed], continuation)
                self.counters['subset_outcomes'] += len(probability)

    def unseen_mask(self, seen):
        mask = self.full_mask
        for cid in seen:
            mask &= ~self.bit_by_cid.get(cid, 0)
        return mask

    def _waiting(self, clock, seen):
        index = int(np.searchsorted(self.times, clock, side='right'))
        mask = self.unseen_mask(seen)
        supported = bool(self.supported[index, mask])
        if supported:
            value = float(self.values[index, mask])
        else:
            self.counters['unsupported_queries'] += 1
            value = self.reference.wait_score(clock, seen)
        return value, supported, index, mask

    def wait_score(self, clock, seen):
        """Value strictly after the current public clock; no time interpolation."""
        self.counters['wait_queries'] += 1
        return self._waiting(clock, seen)[0]

    def choose(self, visible, seen, clock):
        """Return a visible ride or WAIT, with explicit support/fallback status."""
        self.counters['decision_queries'] += 1
        departed = set(seen).union(visible)
        wait, supported, index, mask = self._waiting(clock, departed)
        on_grid = bool(index and self.times[index-1] == clock)
        self.counters['offgrid_queries'] += not on_grid
        receipt = dict(method='multistep_nominal', wait=wait, board=None,
                       unsupported_history=not supported, clock_mode='on_grid' if on_grid else 'off_grid',
                       next_event_index=index, unseen_mask=mask)
        if not visible:
            return None, receipt
        values = {cid: self.reference.score(cid, clock) for cid in visible}
        best = min(visible, key=lambda cid: (values[cid], self.reference.ranks[cid]))
        receipt.update(board=values[best], best_visible=best)
        return (best if values[best] <= wait+1e-9 else None), receipt
