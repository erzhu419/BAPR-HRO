"""Matched candidate/valuation ablations for the corrected simulation.

Every variant computes the same updated TopoCSA continuation estimates. Only
the allowed IDs differ for retained/refreshed menus. This attribution experiment
does not claim cheaper retention. Marginal TopoCSA values are not causal optima.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from .bandit_router import BanditRouter
from .durner.topocsa import topocsa
from .pmf import PMF
from .transit_graph import StopLabel


class MatchedRouter(BanditRouter):
    def __init__(self, graph, priors, *, candidate_mode="refresh", online_mean=False,
                 online_cancel=False, beta=0.0, candidate_limit=5, lookahead=25,
                 max_time=120):
        super().__init__(graph, disruption_gate=False, max_time=max_time,
                         use_hierarchical_prior=False, route_priors_override=priors)
        if candidate_mode not in ("retain", "refresh"):
            raise ValueError("candidate_mode must be retain or refresh")
        self.candidate_mode, self.online_mean = candidate_mode, online_mean
        self.online_cancel, self.beta = online_cancel, beta
        self.candidate_limit, self.lookahead = candidate_limit, lookahead
        self.priors = priors
        self.current_trip = self.last_arrival = None
        self.unavailable = set()
        self.decisions = []
        self.scans = self.label_updates = 0
        self.estimate_ms = self.preprocess_ms = 0.0
        self.query_ms = []
        self._version, self._valued_version = 0, -1
        self._initial_cancels = {}

    def _route_parameters(self, route):
        prior = self.priors.get(route, {"mean": 1.0, "std": 2.0})
        belief = self._get_belief(route)
        initial_cancel = self._initial_cancels.setdefault(route, belief.cancel_rate)
        return {"mean": belief.posterior_mean if self.online_mean else prior["mean"],
                "std": prior["std"],
                "cancel_rate": belief.cancel_rate if self.online_cancel else initial_cancel}

    def _assign(self):
        parameters = {route: self._route_parameters(route)
                      for route in {c.route for c in self.graph.connections}}
        offsets = np.arange(-5, 65)
        probabilities = {}
        for route, params in parameters.items():
            probs = np.exp(-0.5 * ((offsets - params["mean"]) / max(params["std"], 0.5)) ** 2)
            probs *= (1.0 - params["cancel_rate"]) / probs.sum()
            probs.setflags(write=False)
            probabilities[route] = probs
        for c in self.graph.connections:
            probs = probabilities[c.route]
            c.dep_distribution = PMF(probs, c.dep_time - 5)
            c.arr_distribution = PMF(probs, c.arr_time - 5)

    def _value(self):
        started = perf_counter()
        self._assign()
        self.estimate_ms += (perf_counter() - started) * 1000
        started = perf_counter()
        self.cached_result = topocsa(self.graph, self.origin, self.destination,
                                     self.departure - 60, self.journey_deadline)
        self.query_ms.append((perf_counter() - started) * 1000)
        self.scans += self.cached_result.n_connections_processed
        self.label_updates += sum(map(len, self.cached_result.stop_labels.values()))
        self._fresh_menu = set(self.cached_result.hyperpath_connections)
        # Retention is a rule on connection IDs, including options that the
        # approximate continuation recursion can no longer value. Such options
        # receive an explicit failure estimate; never silently reuse old values.
        for cid in getattr(self, "initial_menu", set()) - self._fresh_menu:
            connection = self.graph.connections[cid]
            self.cached_result.stop_labels[connection.dep_stop].append(
                StopLabel(cid, PMF.deterministic(self.journey_deadline),
                          float(self.journey_deadline), 0.0))
        for labels in self.cached_result.stop_labels.values():
            labels.sort(key=lambda label: label.mean_dest_arrival, reverse=True)
        self._valued_version = self._version

    def route(self, s_source, s_dest, t_source):
        self.origin, self.destination, self.departure = s_source, s_dest, t_source
        self.journey_deadline = t_source + self.max_time
        started = perf_counter()
        self._value()
        self.initial_menu = set(self.cached_result.hyperpath_connections)
        self.preprocess_ms = (perf_counter() - started) * 1000
        return self.cached_result

    def observe_delay(self, route, delay):
        super().observe_delay(route, delay)
        if self.online_mean:
            self._version += 1

    def observe_cancel(self, route, kind="true"):
        super().observe_cancel(route, kind)
        if self.online_cancel and kind == "true":
            self._version += 1

    def set_execution_context(self, current_trip, last_arrival, unavailable_ids):
        self.current_trip, self.last_arrival = current_trip, last_arrival
        self.unavailable = set(unavailable_ids)

    def _score(self, label, current_time):
        c = self.graph.connections[label.connection_id]
        same_trip = bool(self.current_trip and c.trip_id == self.current_trip)
        ready = current_time
        if self.current_trip is not None and not same_trip and self.last_arrival is not None:
            ready = max(ready, self.last_arrival + self.graph.stops[c.dep_stop].min_transfer_time)
        if c.arr_stop == getattr(self, "destination", None) and not same_trip:
            # In the execution model one delay shifts both departure and
            # arrival. Conditioning on catching a direct trip must condition
            # that same delay, rather than multiplying independent marginals.
            departures = np.arange(len(c.dep_distribution.probs)) + c.dep_distribution.offset
            arrivals = c.arr_time + departures - c.dep_time
            catch = departures >= ready
            weights = c.dep_distribution.probs[catch]
            loss = np.clip(arrivals[catch] - self.departure, 0, self.max_time)
            capped = float(np.dot(weights, loss) + (1 - weights.sum()) * self.max_time)
            return capped + self.beta * self._get_belief(c.route).posterior_var ** .5
        # Operation is counted once here. label.feasibility is continuation
        # success conditional on boarding this connection.
        p_board = 1.0 if same_trip else max(0.0, c.dep_distribution.total_prob - c.dep_distribution.prob_le(ready - 1))
        p_success = p_board * label.feasibility
        pmf = label.dest_arrival.normalize()
        times = np.arange(len(pmf.probs)) + pmf.offset
        conditional = float(np.dot(pmf.probs, np.clip(times - self.departure, 0, self.max_time)))
        capped_loss = p_success * conditional + (1.0 - p_success) * self.max_time
        return capped_loss + self.beta * self._get_belief(c.route).posterior_var ** 0.5

    def select_connection(self, current_stop, current_time, rng, top_k=5):
        if self._valued_version != self._version:
            self._value()
        labels = self.cached_result.stop_labels.get(current_stop, [])
        feasible = []
        for label in reversed(labels):
            c = self.graph.connections[label.connection_id]
            if c.id in self.unavailable or c.dep_time > current_time + self.lookahead:
                continue
            if c.dep_distribution.support_max < current_time:
                continue
            feasible.append(label)
        fresh_ids = getattr(self, "_fresh_menu", self.cached_result.hyperpath_connections)
        refreshed = {label.connection_id for label in feasible if label.connection_id in fresh_ids}
        menu = fresh_ids if self.candidate_mode == "refresh" else self.initial_menu
        allowed = [label for label in feasible if label.connection_id in menu]
        if self.candidate_limit is not None:
            allowed = allowed[:self.candidate_limit]
        if not allowed:
            return None
        scores = {label.connection_id: self._score(label, current_time) for label in allowed}
        best = min(allowed, key=lambda label: scores[label.connection_id])
        self.decisions.append({"stop": current_stop, "time": current_time,
                               "selected": best.connection_id, "candidates": list(scores),
                               "refresh_only_ids": sorted(refreshed - self.initial_menu),
                               "retained_only_ids": sorted({label.connection_id for label in feasible}
                                                           & self.initial_menu - fresh_ids),
                               "score": scores[best.connection_id]})
        return best, scores[best.connection_id]
