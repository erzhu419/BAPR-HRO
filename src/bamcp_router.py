"""BAMCP Router: Bayes-Adaptive Monte Carlo Planning on hyperpath.

The theoretically strongest practical algorithm for single-episode
Bayesian adaptive SSP-MDP. At each stop:

1. Use MCTS to simulate future journeys under sampled delay models
2. Each simulation: sample delays from posterior → trace hyperpath to destination
3. Pick the route whose simulations yield the best expected total arrival time
4. After boarding: observe actual delay → update posterior → repeat at next stop

References:
- Guez, Silver, Dayan (2012): "Efficient Bayes-Adaptive RL using Sample-Based Search"
- Rigter, Lacerda, Hawes (2021): "Risk-Averse Bayes-Adaptive RL" (CVaR extension)

Connection to BAPR:
- BAPR uses posterior to drive reactive policy (LCB)
- BAMCP uses posterior to drive forward planning (MCTS)
- Both are solutions to the same BA-MDP; BAMCP is more accurate but slower
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .transit_graph import TransitGraph, StopLabel
from .durner.topocsa import topocsa, HyperpathResult
from .ssp_mdp import DelayPosterior


class BAMCPRouter:
    """Bayes-Adaptive Monte Carlo Planning on Durner's hyperpath.

    At each stop, runs N forward simulations through the hyperpath
    to estimate the total journey time for each candidate route,
    accounting for delay uncertainty and cancellation risk.
    """

    def __init__(
        self,
        graph: TransitGraph,
        n_simulations: int = 60,
        max_rollout_depth: int = 8,
    ):
        self.graph = graph
        self.n_simulations = n_simulations
        self.max_depth = max_rollout_depth
        self.cached_result: Optional[HyperpathResult] = None
        self.posteriors: dict[str, DelayPosterior] = {}
        self.total_observations: int = 0
        self.dest: Optional[int] = None

    def _get_posterior(self, route: str) -> DelayPosterior:
        if route not in self.posteriors:
            self.posteriors[route] = DelayPosterior()
        return self.posteriors[route]

    def route(self, s_source: int, s_dest: int, t_source: int):
        self.dest = s_dest
        self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)
        return self.cached_result

    def observe_delay(self, route: str, delay: float):
        post = self._get_posterior(route)
        post.observe_delay(delay)
        post.observe_no_cancel()
        self.total_observations += 1

    def observe_cancel(self, route: str, kind: str = "true"):
        post = self._get_posterior(route)
        post.observe_cancel()
        self.total_observations += 1

    def select_connection(
        self,
        stop_id: int,
        current_time: int,
        rng: np.random.Generator,
        top_k: int = 5,
    ) -> Optional[tuple]:
        """BAMCP: Monte Carlo forward search through the hyperpath.

        For each candidate route at current stop:
          Run N simulated journeys (sample delays → trace hyperpath → reach dest)
          Estimate mean total arrival time

        Pick the candidate with the best (lowest) simulated arrival.

        This is better than LCB because it accounts for DOWNSTREAM effects:
        - "Route A is slower now but gives better transfers at the next stop"
        - "Route B is fast but leads to a disrupted corridor downstream"
        """
        if self.cached_result is None:
            return None

        labels = self.cached_result.stop_labels.get(stop_id, [])
        if not labels:
            return None

        # Collect top-K candidate routes
        candidates = []
        seen_routes = set()
        for label in reversed(labels):
            c = self.graph.connections[label.connection_id]
            if c.dep_time < current_time - 1:
                continue
            if c.dep_time > current_time + 25:
                continue
            if c.route in seen_routes:
                continue
            seen_routes.add(c.route)
            candidates.append((label, c))
            if len(candidates) >= top_k:
                break

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0], candidates[0][0].mean_dest_arrival

        # MCTS: simulate N rollouts per candidate
        best_label = None
        best_score = float('inf')

        for label, c in candidates:
            arrivals = self._simulate_rollouts(
                c, label, stop_id, current_time, rng)
            # Use a risk-aware metric: mean + 0.3 * std (slight pessimism)
            if arrivals:
                score = np.mean(arrivals) + 0.3 * np.std(arrivals)
            else:
                score = float('inf')

            if score < best_score:
                best_score = score
                best_label = label

        return best_label, best_score

    def _simulate_rollouts(
        self,
        first_conn,
        first_label: StopLabel,
        start_stop: int,
        start_time: int,
        rng: np.random.Generator,
    ) -> list[float]:
        """Simulate N complete journeys starting with first_conn.

        Each simulation:
        1. Sample delay for first_conn from posterior
        2. If canceled, return infinity
        3. Arrive at next stop at sampled time
        4. At next stop, greedily pick best label (under sampled delays)
        5. Repeat until destination or max depth
        """
        arrivals = []
        dest = self.dest

        for _ in range(self.n_simulations):
            # Sample delay model for first connection
            post = self._get_posterior(first_conn.route)
            if post.sample_cancel(rng):
                arrivals.append(start_time + 120)  # penalty for cancel
                continue

            delay = post.sample_delay(rng)
            current_time = first_conn.arr_time + delay
            current_stop = first_conn.arr_stop

            if current_stop == dest:
                arrivals.append(current_time)
                continue

            # Forward rollout: greedily follow hyperpath
            for depth in range(self.max_depth):
                stop_labels = self.cached_result.stop_labels.get(current_stop, [])
                if not stop_labels:
                    current_time += 30  # penalty for dead end
                    break

                # Pick best available label under sampled delays
                best_arr = float('inf')
                best_c = None
                for lab in reversed(stop_labels):
                    conn = self.graph.connections[lab.connection_id]
                    if conn.dep_time < current_time - 1:
                        continue
                    if conn.dep_time > current_time + 20:
                        continue

                    p = self._get_posterior(conn.route)
                    if p.cancel_rate > 0.5 and p.n > 3:
                        continue  # skip routes we know are mostly canceled

                    d = p.sample_delay(rng)
                    arr = conn.arr_time + d
                    total_est = arr if conn.arr_stop == dest else arr + 20  # rough downstream est

                    if total_est < best_arr:
                        best_arr = total_est
                        best_c = conn
                    break  # just take the first good one for speed

                if best_c is None:
                    current_time += 15  # no connection, wait penalty
                    break

                d = self._get_posterior(best_c.route).sample_delay(rng)
                current_time = best_c.arr_time + d
                current_stop = best_c.arr_stop

                if current_stop == dest:
                    break

            arrivals.append(current_time)

        return arrivals
