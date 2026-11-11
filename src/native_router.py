"""Durner strict strategy in the corrected fixed-world execution contract.

This secondary baseline shares causal estimates, observation updates, candidate
limits and waiting windows with MatchedRouter. It retains the native conditional
mean-arrival objective, so it is not part of the matched-objective attribution.
"""
from time import perf_counter

from .matched_router import MatchedRouter
from .durner.native import native_topocsa
from .pmf import PMF


class NativeDurnerRouter(MatchedRouter):
    def __init__(self, graph, priors, *, candidate_limit=5, lookahead=25, max_time=120):
        super().__init__(graph, priors, candidate_mode='refresh', online_mean=True,
                         online_cancel=True, beta=0.0, candidate_limit=candidate_limit,
                         lookahead=lookahead, max_time=max_time)

    def _value(self):
        started = perf_counter()
        self._assign()
        # MatchedRouter uses missing mass for operation. Native Durner takes
        # conditional unit-mass PMFs and the operation probability separately.
        conditional = {}
        for c in self.graph.connections:
            if c.route not in conditional:
                conditional[c.route] = (1.0 - c.dep_distribution.total_prob,
                                        c.dep_distribution.normalize().probs)
                conditional[c.route][1].setflags(write=False)
            c.cancel_prob, probs = conditional[c.route]
            c.dep_distribution = PMF(probs, c.dep_distribution.offset)
            c.arr_distribution = PMF(probs, c.arr_distribution.offset)
        self.estimate_ms += (perf_counter() - started) * 1000
        started = perf_counter()
        self.cached_result = native_topocsa(self.graph, self.origin, self.destination,
                                            self.departure - 60, self.journey_deadline)
        self.query_ms.append((perf_counter() - started) * 1000)
        self.scans += self.cached_result.n_connections_processed
        self.label_updates += sum(map(len, self.cached_result.stop_labels.values()))
        self._valued_version = self._version

    def _score(self, label, current_time):
        return label.mean_dest_arrival
