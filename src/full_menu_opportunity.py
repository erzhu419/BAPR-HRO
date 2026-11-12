"""One-step nominal improvement over a full-menu first-opportunity rule."""

import numpy as np


class FullMenuOpportunity:
    """Compare current offers with the next first opportunity in the same menu.

    The compiler supplies independent root departure laws and the inexpensive
    clock-nominal value of each fixed first-path/downstream-rescue option. The
    WAIT reference boards the first future offer, using scheduled ranks on a
    tie. Execution can reconsider at the next public offer. This is a rolling
    one-step rule, not an exact multi-offer dynamic program.

    ``seen`` contains only departures/cancellations already publicly observed
    during this visit. A departure before entry remains possible, so its mass
    stays in the no-local-occurrence denominator together with cancellation.
    No actual future event or exact conditional score enters this class.
    """

    def __init__(self, compiler, ranks, entry, ready, deadline):
        self.compiler, self.ranks = compiler, dict(ranks)
        self.entry, self.ready, self.deadline = entry, ready, deadline
        self.cids = tuple(ranks)
        self.distributions, self.cdfs, self.absent, self.marks = {}, {}, {}, {}
        for cid in self.cids:
            times, mass, cancel = compiler.departure_distribution(cid)
            times, mass = np.asarray(times), np.asarray(mass)
            self.distributions[cid] = times, mass, cancel
            self.cdfs[cid] = np.r_[0., np.cumsum(mass)]
            self.absent[cid] = float(cancel+mass[times < entry].sum())
            legal = (times >= ready) & (times < deadline) & (mass > 0)
            clocks = times[legal]
            scores = np.array([compiler.clock_nominal_score(cid, int(t)) for t in clocks])
            self.marks[cid] = clocks, mass[legal], scores

    def score(self, cid, clock):
        """Use the precompiled score on-grid; retain actual clocks off-grid."""
        clocks, _, scores = self.marks[cid]
        index = np.searchsorted(clocks, clock)
        if index < len(clocks) and clocks[index] == clock:
            return float(scores[index])
        return float(self.compiler.clock_nominal_score(cid, clock))

    def wait_score(self, clock, seen):
        """Expected nominal value of the first future unseen offer.

        Each candidate's conditional future atom is multiplied by the chance
        that every other candidate has no earlier (or tie-preferred) offer.
        Remaining no-offer probability receives the common deadline cost.
        """
        unseen = tuple(cid for cid in self.cids if cid not in seen)
        survival = {}
        for cid in unseen:
            times, mass, _ = self.distributions[cid]
            survival[cid] = self.absent[cid]+float(mass[times > clock].sum())
        active = tuple(cid for cid in unseen if survival[cid] > 0)
        gain = 0.
        for cid in active:
            clocks, mass, scores = self.marks[cid]
            future = clocks > clock
            times = clocks[future]
            selected = mass[future]/survival[cid]
            for other in active:
                if other == cid:
                    continue
                other_times, _, _ = self.distributions[other]
                cdf = self.cdfs[other]
                lower = cdf[np.searchsorted(other_times, clock, side='right')]
                side = 'right' if self.ranks[other] < self.ranks[cid] else 'left'
                upper = cdf[np.searchsorted(other_times, times, side=side)]
                selected = selected*(1.-np.maximum(0., upper-lower)/survival[other])
            gain += float(selected @ (self.deadline-scores[future]))
        return float(self.deadline-gain)

    def choose(self, visible, seen, clock, method='clock_nominal'):
        """Return ``(connection ID or None, receipt)``; None means WAIT.

        Inputs are the legal public offer set, the public receipt IDs and the
        current clock. A refusal commits to no particular future vehicle.
        """
        if not visible:
            return None, dict(method=method, board=None, wait=None)
        if method == 'first_offer':
            return min(visible, key=self.ranks.__getitem__), dict(method=method, board=None, wait=None)
        if method != 'clock_nominal':
            raise ValueError(f'Unknown full-menu rule: {method}')
        values = {cid: self.score(cid, clock) for cid in visible}
        best = min(visible, key=lambda cid: (values[cid], self.ranks[cid]))
        departed = set(seen).union(visible)
        wait = self.wait_score(clock, departed)
        choice = best if values[best] <= wait+1e-9 else None
        return choice, dict(method=method, board=values[best], wait=wait,
                            best_visible=best, unseen=len(set(self.cids)-departed))
