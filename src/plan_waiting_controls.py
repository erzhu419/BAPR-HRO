"""Cheap public-event controls for a frozen root departure menu."""

from copy import copy
from itertools import combinations

import numpy as np

from .full_menu_waiting_dp import FullMenuWaitingDP


class SchedulePriority:
    """Take the best scheduled arrival still compatible with public receipts.

    Passing a scheduled departure clock is not evidence that a vehicle left.
    An observed departure/cancellation removes it; currently visible offers
    remain available even though their departure receipt is already in seen.
    """

    def __init__(self, scheduled_arrivals):
        self.scheduled_arrivals = dict(scheduled_arrivals)

    def choose(self, visible, seen, clock):
        visible = set(visible)
        removed = set(seen)-visible
        target = min((cid for cid in self.scheduled_arrivals if cid not in removed),
                     key=lambda cid: (self.scheduled_arrivals[cid], cid), default=None)
        choice = target if target in visible else None
        return choice, dict(method='schedule_priority', target=target,
                            planned_arrival=None if target is None else self.scheduled_arrivals[target])


class SingleTargetOpportunity:
    """Compare boarding now with waiting for one still-unseen candidate.

    The comparison uses the reference's independent departure probabilities
    and clock-nominal terminal marks. It does not integrate exact board scores
    or optimize subsequent refusals. A future target is diagnostic only:
    every new public offer triggers a fresh comparison.
    """

    def __init__(self, reference):
        self.reference, self.cids = reference, tuple(reference.cids)
        self.deadline = reference.deadline
        self.mass_tails, self.gain_tails = {}, {}
        for cid in self.cids:
            _, mass, _ = reference.distributions[cid]
            self.mass_tails[cid] = np.r_[np.cumsum(mass[::-1])[::-1], 0.]
            _, probability, score = reference.marks[cid]
            gain = probability*(self.deadline-score)
            self.gain_tails[cid] = np.r_[np.cumsum(gain[::-1])[::-1], 0.]

    def _waiting(self, clock, seen):
        costs, unsupported = {}, []
        for cid in self.cids:
            if cid in seen:
                continue
            times, _, _ = self.reference.distributions[cid]
            index = np.searchsorted(times, clock, side='right')
            survival = self.reference.absent[cid]+self.mass_tails[cid][index]
            if survival <= 0:
                unsupported.append(cid)
                continue
            clocks = self.reference.marks[cid][0]
            index = np.searchsorted(clocks, clock, side='right')
            costs[cid] = float(self.deadline-self.gain_tails[cid][index]/survival)
        target = min(costs, key=lambda cid: (costs[cid], self.reference.ranks[cid]), default=None)
        return (float(self.deadline) if target is None else costs[target]), target, unsupported

    def wait_score(self, clock, seen):
        """Best single-candidate wait value, retaining unobserved-past mass."""
        return self._waiting(clock, set(seen))[0]

    def choose(self, visible, seen, clock):
        departed = set(seen).union(visible)
        wait, target, unsupported = self._waiting(clock, departed)
        receipt = dict(method='single_target', wait=wait, board=None, target=target,
                       unsupported_candidates=unsupported)
        if not visible:
            return None, receipt
        values = {cid: self.reference.score(cid, clock) for cid in visible}
        best = min(visible, key=lambda cid: (values[cid], self.reference.ranks[cid]))
        receipt.update(board=values[best], best_visible=best)
        return (best if values[best] <= wait+1e-9 else None), receipt


class CombinedWaitingOpportunity:
    """Use the cheaper of single-target and first-future-offer wait references."""

    def __init__(self, single):
        self.single, self.reference = single, single.reference

    def _waiting(self, clock, seen):
        target_wait, target, unsupported = self.single._waiting(clock, seen)
        first_wait = self.reference.wait_score(clock, seen)
        return min(target_wait, first_wait), dict(single_wait=target_wait, first_offer_wait=first_wait,
            target=target, unsupported_candidates=unsupported)

    def wait_score(self, clock, seen):
        return self._waiting(clock, set(seen))[0]

    def choose(self, visible, seen, clock):
        wait, parts = self._waiting(clock, set(seen).union(visible))
        receipt = dict(method='combined_waiting', wait=wait, board=None, **parts)
        if not visible:
            return None, receipt
        values = {cid: self.reference.score(cid, clock) for cid in visible}
        best = min(visible, key=lambda cid: (values[cid], self.reference.ranks[cid]))
        receipt.update(board=values[best], best_visible=best)
        return (best if values[best] <= wait+1e-9 else None), receipt


class TwoTargetOpportunity:
    """Add optimal two-target waits to the combined rolling reference.

    Each pair shares the frozen board marks and departure laws of the common
    reference. It has four unseen-target states, independent of full menu
    size. Pair compilation costs O(K^2 T) for K candidates and at most T
    departure clocks per pair; all board/tail preparation is additional.

    The selected pair is a waiting reference, not a commitment: every legal
    public event compares all visible offers and all remaining pairs again.
    Independence of root trips and the reference's local observation window
    are required, exactly as for FullMenuWaitingDP.
    """

    def __init__(self, combined):
        self.combined, self.reference = combined, combined.reference
        self.cids = tuple(self.reference.cids)
        self.pairs = []
        for first, second in combinations(self.cids, 2):
            reference = copy(self.reference)
            reference.cids = (first, second)
            reference.ranks = {cid: self.reference.ranks[cid] for cid in reference.cids}
            self.pairs.append((first, second, FullMenuWaitingDP(reference)))

    def _waiting(self, clock, seen):
        combined_wait, parts = self.combined._waiting(clock, seen)
        value, pair_wait, best_pair = combined_wait, self.reference.deadline, None
        for first, second, dp in self.pairs:
            if first in seen or second in seen:
                continue
            candidate = dp.wait_score(clock, ())
            if candidate < pair_wait:
                pair_wait, best_pair = candidate, (first, second)
            value = min(value, candidate)
        return value, dict(combined_wait=combined_wait, pair_wait=float(pair_wait),
                           best_pair=best_pair, **parts)

    def wait_score(self, clock, seen):
        return self._waiting(clock, set(seen))[0]

    def choose(self, visible, seen, clock):
        wait, parts = self._waiting(clock, set(seen).union(visible))
        receipt = dict(method='two_target_waiting', wait=wait, board=None, **parts)
        if not visible:
            return None, receipt
        values = {cid: self.reference.score(cid, clock) for cid in visible}
        best = min(visible, key=lambda cid: (values[cid], self.reference.ranks[cid]))
        receipt.update(board=values[best], best_visible=best)
        return (best if values[best] <= wait+1e-9 else None), receipt
