"""Witness-cutoff variant, kept separate from the archived sparse baseline."""
from fractions import Fraction
from time import perf_counter

from .sparse_certificate import SparseCertificateIndex as _SparseCertificateIndex


class SparseCertificateIndex(_SparseCertificateIndex):
    def action_lower(self, state, actions, support=None, *, cutoff=None):
        """Return valid, possibly weaker bounds while avoiding unnecessary EA.

        The cutoff is a public causal witness's remaining-cost upper bound.
        If a world's first action already costs at least that number, its
        elapsed cost alone is used as a lower bound. This can save work but
        never increases a lower bound or proves more removals than full EA.
        """
        if cutoff is None:
            return super().action_lower(state, actions, support)
        started = perf_counter()
        cutoff = Fraction(cutoff)
        actions = tuple(actions)
        support = tuple(range(len(self.bank))) if support is None else tuple(support)
        weight = sum(self.weights[i] for i in support)
        if weight <= 0:
            raise ValueError('Public history has zero scenario mass')
        result = dict.fromkeys(actions, Fraction(0))
        for world in support:
            if not self.weights[world]:
                continue
            for action in actions:
                stop, time = self.first_projection(world, state, action)
                elapsed = Fraction(time - state.time)
                if elapsed >= cutoff:
                    self.stats['ea_skipped_by_cutoff'] += 1
                    bound = elapsed
                else:
                    bound = Fraction(self.arrival(world, stop, time) - state.time)
                result[action] += self.weights[world] * bound
        self.stats['action_lower_seconds'] += perf_counter() - started
        self.stats['root_actions_evaluated'] += len(actions)
        return {action: value / weight for action, value in result.items()}
