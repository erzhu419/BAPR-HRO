"""Two root offers, with common downstream first-opportunity continuations."""

from functools import cache

import numpy as np


METHODS = ('first_offer', 'clock_nominal', 'mean_arrival', 'exact')


class PairDecision:
    def __init__(self, compiler, ranks, entry, ready, deadline):
        self.compiler, self.ranks = compiler, ranks
        self.entry, self.ready, self.deadline = entry, ready, deadline
        self.cids = tuple(ranks)
        if len(self.cids) != 2:
            raise ValueError('The mechanism comparison has exactly two distinct first trips')
        self.distributions = {cid: compiler.departure_distribution(cid) for cid in self.cids}

    @cache
    def score(self, cid, clock, method):
        if method == 'clock_nominal':
            value = self.compiler.clock_nominal_score(cid, clock)
        else:
            row = self.compiler.board_score(cid, clock)
            value = None if row is None else getattr(row, method)
        return None if value is None else float(value)

    @cache
    def wait_score(self, cid, clock, method):
        times, mass, cancel = self.distributions[cid]
        future = (times > clock) & (times < self.deadline)
        # A departure before platform entry was never observed. It does not
        # become a future vehicle merely because it has not appeared locally.
        survival = cancel + mass[(times < self.entry) | (times > clock)].sum()
        if survival <= 0:
            return float(self.deadline)
        gain = sum(float(p) * (self.deadline-self.score(cid, int(t), method))
                   for t, p in zip(times[future], mass[future], strict=True) if p > 0)
        return float(self.deadline-gain/survival)

    def choose(self, visible, other, clock, method):
        """Return a selected first trip, waiting if it is not visible yet."""
        if method == 'first_offer':
            return min(visible, key=self.ranks.__getitem__), None
        values = {cid: self.score(cid, clock, method) for cid in visible}
        if any(value is None for value in values.values()):
            return min(visible, key=self.ranks.__getitem__), 'outside_score_support'
        cid = min(values, key=lambda c: (values[c], self.ranks[c]))
        if other is not None and self.wait_score(other, clock, method) < values[cid]-1e-9:
            cid = other
        return cid, None

    def expectation(self):
        """Integrate all first legal offers and their observable histories.

        Tail randomness has already been integrated in the board scores.
        There is no sampled path bank or revelation of future root departures.
        """
        deadline = self.deadline
        eligible = {cid: float(mass[(times >= self.ready) & (times < deadline)].sum())
                    for cid, (times, mass, _) in self.distributions.items()}
        none = (1-eligible[self.cids[0]])*(1-eligible[self.cids[1]])
        totals = {method: none*deadline for method in METHODS}
        changed = {method: 0. for method in METHODS}
        total_mass = none
        for cid in self.cids:
            other = next(c for c in self.cids if c != cid)
            times, mass, _ = self.distributions[cid]
            bt, bp, bc = self.distributions[other]
            observed_unboardable = float(bp[(bt >= self.entry) & (bt < self.ready)].sum())
            for time, probability in zip(times, mass, strict=True):
                time, probability = int(time), float(probability)
                if not self.ready <= time < deadline or probability <= 0:
                    continue
                q = float(bc+bp[(bt < self.entry) | (bt > time)].sum())
                # B already departed during our visit before physical readiness.
                known_mass = probability*observed_unboardable
                total_mass += known_mass
                for method in METHODS:
                    totals[method] += known_mass*self.score(cid, time, 'exact')
                if q > 0:
                    weight = probability*q
                    total_mass += weight
                    for method in METHODS:
                        choice, _ = self.choose((cid,), other, time, method)
                        value = (self.score(cid, time, 'exact') if choice == cid
                                 else self.wait_score(other, time, 'exact'))
                        totals[method] += weight*value
                        changed[method] += weight*(choice != cid)
                # Simultaneous offers are visited once; IDs do not break the
                # policy tie, which uses each declared method's scores.
                tied = float(bp[bt == time].sum())
                if cid == self.cids[0] and tied > 0:
                    weight = probability*tied
                    total_mass += weight
                    first, _ = self.choose(self.cids, None, time, 'first_offer')
                    for method in METHODS:
                        choice, _ = self.choose(self.cids, None, time, method)
                        totals[method] += weight*self.score(choice, time, 'exact')
                        changed[method] += weight*(choice != first)
        if not np.isclose(total_mass, 1., atol=1e-9):
            raise ValueError(f'Public history masses sum to {total_mass}')
        return dict(expected_arrival=totals, disagreement_probability=changed,
                    no_offer_probability=none, history_mass=total_mass)
