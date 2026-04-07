"""BAPR-HRO LCB routers for VRP — V1 (Normal-Gamma) and V2 (Ensemble + dynamic β).

Architecture (faithful to BAPR-HRO transit routing):
  1. Pre-compute K candidate routes = "hyperpath" (via perturbations of NN/2opt)
  2. Score each route using LCB of total travel time
  3. Pick the route with lowest pessimistic estimated cost
  4. Execute it, observe actual delays, update beliefs
  5. Next episode: re-rank routes using updated beliefs

This maps exactly to transit BAPR-HRO:
  - K candidate routes ↔ K candidate buses at each stop
  - LCB score of route ↔ LCB score of bus connection
  - Beliefs per map zone ↔ beliefs per route
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from vrp_env import VRPInstance, VRPExecutor, StepResult


# ── Candidate route generation ──────────────────────────────────────────────

def generate_nn_route(instance: VRPInstance, rng: np.random.Generator,
                      noise: float = 0.0) -> list[int]:
    """Generate a nearest-neighbor route with optional noise.

    noise=0: deterministic NN. noise>0: add Gaussian noise to distances
    before NN selection, producing diverse candidates.
    """
    unvisited = set(range(1, instance.n_customers + 1))
    route = []
    current = 0  # depot
    remaining_cap = instance.capacity

    while unvisited:
        feasible = [c for c in unvisited if instance.demands[c - 1] <= remaining_cap]
        if not feasible:
            break

        # Score by distance + noise
        scores = []
        for c in feasible:
            d = instance.get_distance(current, c)
            noisy_d = d + rng.normal(0, noise * d) if noise > 0 else d
            scores.append((noisy_d, c))
        scores.sort()
        chosen = scores[0][1]

        route.append(chosen)
        remaining_cap -= instance.demands[chosen - 1]
        unvisited.remove(chosen)
        current = chosen

    return route


def _cluster_customers(instance: VRPInstance, customer_ids: list[int],
                       n_clusters: int = 3) -> list[list[int]]:
    """Split customers into geographic clusters via simple k-means."""
    locs = np.array([instance.locations[c] for c in customer_ids])
    rng = np.random.default_rng(0)

    # Simple k-means
    centers = locs[rng.choice(len(locs), n_clusters, replace=False)]
    for _ in range(20):
        dists = np.linalg.norm(locs[:, None] - centers[None], axis=2)
        labels = dists.argmin(axis=1)
        for j in range(n_clusters):
            mask = labels == j
            if mask.any():
                centers[j] = locs[mask].mean(axis=0)

    clusters = [[] for _ in range(n_clusters)]
    for i, cid in enumerate(customer_ids):
        clusters[labels[i]].append(cid)
    return [c for c in clusters if c]


def _nn_within(instance: VRPInstance, customers: list[int],
               start: int) -> list[int]:
    """NN ordering of a customer subset, starting from a given node."""
    order = []
    remaining = set(customers)
    current = start
    while remaining:
        nearest = min(remaining, key=lambda c: instance.get_distance(current, c))
        order.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return order


def generate_candidate_routes(instance: VRPInstance, k: int = 10,
                              seed: int = 0) -> list[list[int]]:
    """Generate K diverse candidate routes — same customers, different visit ORDER.

    Strategy: split customers into 2-3 geographic clusters, then generate
    routes that visit clusters in different orders. Within each cluster,
    use NN ordering. This produces routes that are similar in total distance
    but differ in WHEN congested areas are traversed (early vs late).

    This is the VRP analog of "hyperpath alternatives" — same destination
    set, different orderings, time-dependent costs make the ordering matter.
    """
    rng = np.random.default_rng(seed)

    # Step 1: get the feasible customer subset
    base_route = generate_nn_route(instance, rng, noise=0.0)
    customer_set = list(base_route)

    if len(customer_set) < 4:
        return [customer_set]

    # Step 2: cluster customers geographically
    n_clust = min(3, max(2, len(customer_set) // 4))
    clusters = _cluster_customers(instance, customer_set, n_clust)

    # Step 3: generate routes with different cluster orderings
    from itertools import permutations
    candidates = []

    cluster_perms = list(permutations(range(len(clusters))))
    rng.shuffle(cluster_perms)

    for perm in cluster_perms[:k]:
        route = []
        current = 0  # depot
        for ci in perm:
            ordered = _nn_within(instance, clusters[ci], current)
            route.extend(ordered)
            current = ordered[-1] if ordered else current
        candidates.append(route)

    # Also add the base NN route if not already present
    if base_route not in candidates:
        candidates.insert(0, base_route)

    # Add 2-opt variants of best candidates
    if len(candidates) < k:
        for route in list(candidates[:3]):
            for _ in range(2):
                new = route.copy()
                a = rng.integers(0, max(1, len(new) - 2))
                b = rng.integers(a + 2, min(len(new), a + 5))
                new[a:b] = new[a:b][::-1]
                if new not in candidates:
                    candidates.append(new)
                if len(candidates) >= k:
                    break

    return candidates[:k]


# ── Edge beliefs ─────────────────────────────────────────────────────────────

def _zone_key(src: int, dst: int, locations: np.ndarray,
              time: float = 0.0) -> tuple[int, int, int]:
    """Key for belief lookup: (grid_x, grid_y, time_bucket).

    Time buckets capture time-dependent congestion:
      0: early morning (< 420 = 7am)
      1: rush hour (420-540 = 7am-9am)
      2: late morning (>= 540 = 9am+)
    This lets beliefs distinguish "zone X during rush hour" from
    "zone X after rush hour" — critical for route ordering decisions.
    """
    mx = (locations[src][0] + locations[dst][0]) / 2
    my = (locations[src][1] + locations[dst][1]) / 2
    gx = min(int(mx / 20), 4)
    gy = min(int(my / 20), 4)
    if time < 420:
        tb = 0
    elif time < 540:
        tb = 1
    else:
        tb = 2
    return (gx, gy, tb)


@dataclass
class EdgeBeliefNG:
    """Normal-Gamma conjugate posterior over edge delay."""
    mu: float = 0.0
    kappa: float = 1.0
    alpha: float = 2.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def std(self) -> float:
        if self.alpha <= 1:
            return 1.0
        return float(np.sqrt(self.beta / (self.kappa * (self.alpha - 1))))

    @property
    def n_obs(self) -> int:
        return max(0, int(self.kappa - 1))

    def update(self, delay: float):
        kn = self.kappa + 1
        mn = (self.kappa * self.mu + delay) / kn
        an = self.alpha + 0.5
        bn = self.beta + 0.5 * self.kappa * (delay - self.mu) ** 2 / kn
        self.mu, self.kappa, self.alpha, self.beta = mn, kn, an, bn

    def sample(self, rng: np.random.Generator) -> float:
        if self.alpha <= 0:
            return self.mu
        tau = rng.gamma(self.alpha, 1.0 / max(self.beta, 1e-8))
        sigma = 1.0 / np.sqrt(max(tau * self.kappa, 1e-8))
        return rng.normal(self.mu, sigma)


@dataclass
class EdgeBeliefEnsemble:
    """Ensemble belief (V2) — Poisson bootstrap."""
    n_estimators: int = 5
    _means: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _vars: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _counts: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self):
        if len(self._means) == 0:
            self._means = np.zeros(self.n_estimators)
            self._vars = np.full(self.n_estimators, 1.0)  # moderate prior
            self._counts = np.ones(self.n_estimators)  # start with 1 pseudo-count

    @property
    def mean(self) -> float:
        return float(self._means.mean())

    @property
    def std(self) -> float:
        return float(self._means.std()) if len(self._means) > 1 else 0.0

    @property
    def ood_score(self) -> float:
        avg_int = float(np.sqrt(self._vars).mean())
        return min(self.std / max(avg_int, 1e-6), 3.0)

    def update(self, delay: float, rng: np.random.Generator):
        weights = rng.poisson(1, self.n_estimators)
        for k in range(self.n_estimators):
            for _ in range(weights[k]):
                self._counts[k] += 1
                n = self._counts[k]
                d = delay - self._means[k]
                self._means[k] += d / n
                self._vars[k] += (d * (delay - self._means[k]) - self._vars[k]) / n


# ── Route-level scoring ─────────────────────────────────────────────────────

class BeliefManager:
    """Manages zone-level delay beliefs, shared across routers."""

    def __init__(self, instance: VRPInstance, belief_type: str = "ng",
                 n_estimators: int = 5):
        self.inst = instance
        self.belief_type = belief_type
        self.n_estimators = n_estimators
        self.beliefs: dict[tuple, EdgeBeliefNG | EdgeBeliefEnsemble] = {}

    def _get(self, src: int, dst: int, time: float = 0.0):
        key = _zone_key(src, dst, self.inst.locations, time)
        if key not in self.beliefs:
            if self.belief_type == "ng":
                self.beliefs[key] = EdgeBeliefNG()
            else:
                self.beliefs[key] = EdgeBeliefEnsemble(n_estimators=self.n_estimators)
        return self.beliefs[key]

    def update(self, step: StepResult, rng: np.random.Generator | None = None):
        # Use arrival_time to key the time bucket
        b = self._get(step.src, step.dst, step.arrival_time - step.travel_time)
        if isinstance(b, EdgeBeliefEnsemble) and rng is not None:
            b.update(step.delay, rng)
        elif isinstance(b, EdgeBeliefNG):
            b.update(step.delay)

    def _simulate_route_times(self, route: list[int],
                              start_time: float = 360.0) -> list[float]:
        """Estimate when each edge would be traversed (for time-aware scoring)."""
        times = []
        t = start_time
        prev = 0
        for cid in route:
            times.append(t)
            dist = self.inst.get_distance(prev, cid)
            b = self._get(prev, cid, t)
            t += dist + b.mean  # estimated travel time
            prev = cid
        times.append(t)  # return-to-depot time
        return times

    def score_route_lcb(self, route: list[int], beta: float,
                        start_time: float = 360.0) -> float:
        """LCB score with time-dependent beliefs."""
        times = self._simulate_route_times(route, start_time)
        total = 0.0
        prev = 0
        for i, cid in enumerate(route):
            dist = self.inst.get_distance(prev, cid)
            b = self._get(prev, cid, times[i])
            total += dist + b.mean + beta * b.std
            prev = cid
        dist = self.inst.get_distance(prev, 0)
        b = self._get(prev, 0, times[-1])
        total += dist + b.mean + beta * b.std
        return total

    def score_route_ts(self, route: list[int], rng: np.random.Generator,
                       start_time: float = 360.0) -> float:
        """Thompson Sampling score with time-dependent beliefs."""
        times = self._simulate_route_times(route, start_time)
        total = 0.0
        prev = 0
        for i, cid in enumerate(route):
            dist = self.inst.get_distance(prev, cid)
            b = self._get(prev, cid, times[i])
            assert isinstance(b, EdgeBeliefNG)
            total += dist + b.sample(rng)
            prev = cid
        dist = self.inst.get_distance(prev, 0)
        b = self._get(prev, 0, times[-1])
        total += dist + b.sample(rng)
        return total

    def score_route_static(self, route: list[int]) -> float:
        """Static score: sum of distances only (no learning)."""
        total = 0.0
        prev = 0
        for cid in route:
            total += self.inst.get_distance(prev, cid)
            prev = cid
        total += self.inst.get_distance(prev, 0)
        return total

    def dynamic_beta(self, route: list[int], beta_base: float,
                     beta_ood: float, start_time: float = 360.0) -> float:
        """V2 dynamic beta with time-dependent OOD."""
        times = self._simulate_route_times(route, start_time)
        max_ood = 0.0
        prev = 0
        for i, cid in enumerate(route):
            b = self._get(prev, cid, times[i])
            if isinstance(b, EdgeBeliefEnsemble):
                max_ood = max(max_ood, b.ood_score)
            prev = cid
        return beta_base + beta_ood * max_ood


# ── Routers ──────────────────────────────────────────────────────────────────

def _execute_route(instance: VRPInstance, route: list[int],
                   start_time: float, seed: int) -> tuple[dict, list[StepResult]]:
    """Execute a route and return (metrics, steps)."""
    executor = VRPExecutor(instance, start_time=start_time, seed=seed)
    steps = []
    for cid in route:
        if cid not in executor.unvisited:
            continue
        if instance.demands[cid - 1] > executor.remaining_cap:
            continue
        step = executor.step(cid)
        steps.append(step)
    step = executor.return_to_depot()
    steps.append(step)
    return executor.get_metrics(), steps


class StaticNNRouter:
    """Baseline: always pick the shortest-distance route (no learning)."""

    def __init__(self, instance: VRPInstance, candidates: list[list[int]]):
        self.inst = instance
        self.candidates = candidates
        self.bm = BeliefManager(instance, belief_type="ng")

    def select_route(self) -> list[int]:
        best_route = self.candidates[0]
        best_score = float("inf")
        for route in self.candidates:
            score = self.bm.score_route_static(route)
            if score < best_score:
                best_score = score
                best_route = route
        return best_route

    def observe(self, steps: list[StepResult]):
        pass  # no learning


class LCBRouterV1:
    """BAPR-HRO V1: Normal-Gamma + fixed β.

    First len(candidates) episodes: round-robin exploration (try each route once).
    After that: pick route with lowest LCB score.
    """

    def __init__(self, instance: VRPInstance, candidates: list[list[int]],
                 beta: float = 1.0, explore_top: int = 4):
        self.inst = instance
        self.candidates = candidates
        self.beta = beta
        self.bm = BeliefManager(instance, belief_type="ng")
        self._episode = 0
        # Only explore the top candidates by static distance
        scored = sorted(range(len(candidates)),
                        key=lambda i: self.bm.score_route_static(candidates[i]))
        self._explore_order = scored[:explore_top]

    def select_route(self) -> list[int]:
        # Exploration phase: try top routes once each
        if self._episode < len(self._explore_order):
            return self.candidates[self._explore_order[self._episode]]

        best_route = self.candidates[0]
        best_score = float("inf")
        for route in self.candidates:
            score = self.bm.score_route_lcb(route, self.beta)
            if score < best_score:
                best_score = score
                best_route = route
        return best_route

    def observe(self, steps: list[StepResult]):
        self._episode += 1
        for step in steps:
            self.bm.update(step)


class LCBRouterV2:
    """BAPR-HRO V2: Ensemble + dynamic β."""

    def __init__(self, instance: VRPInstance, candidates: list[list[int]],
                 beta_base: float = 0.8, beta_ood: float = 0.8,
                 n_estimators: int = 5, seed: int = 0, explore_top: int = 4):
        self.inst = instance
        self.candidates = candidates
        self.beta_base = beta_base
        self.beta_ood = beta_ood
        self.rng = np.random.default_rng(seed)
        self.bm = BeliefManager(instance, belief_type="ensemble",
                                n_estimators=n_estimators)
        self._episode = 0
        bm_tmp = BeliefManager(instance, belief_type="ng")
        scored = sorted(range(len(candidates)),
                        key=lambda i: bm_tmp.score_route_static(candidates[i]))
        self._explore_order = scored[:explore_top]

    def select_route(self) -> list[int]:
        if self._episode < len(self._explore_order):
            return self.candidates[self._explore_order[self._episode]]

        best_route = self.candidates[0]
        best_score = float("inf")
        for route in self.candidates:
            beta = self.bm.dynamic_beta(route, self.beta_base, self.beta_ood)
            score = self.bm.score_route_lcb(route, beta)
            if score < best_score:
                best_score = score
                best_route = route
        return best_route

    def observe(self, steps: list[StepResult]):
        self._episode += 1
        for step in steps:
            self.bm.update(step, self.rng)


class TSRouter:
    """Thompson Sampling — optimistic baseline (Lagos et al. analog)."""

    def __init__(self, instance: VRPInstance, candidates: list[list[int]],
                 seed: int = 0, explore_top: int = 4):
        self.inst = instance
        self.candidates = candidates
        self.rng = np.random.default_rng(seed)
        self.bm = BeliefManager(instance, belief_type="ng")
        self._episode = 0
        scored = sorted(range(len(candidates)),
                        key=lambda i: self.bm.score_route_static(candidates[i]))
        self._explore_order = scored[:explore_top]

    def select_route(self) -> list[int]:
        if self._episode < len(self._explore_order):
            return self.candidates[self._explore_order[self._episode]]

        best_route = self.candidates[0]
        best_score = float("inf")
        for route in self.candidates:
            score = self.bm.score_route_ts(route, self.rng)
            if score < best_score:
                best_score = score
                best_route = route
        return best_route

    def observe(self, steps: list[StepResult]):
        self._episode += 1
        for step in steps:
            self.bm.update(step)


# ── Run one episode ──────────────────────────────────────────────────────────

def run_episode(instance: VRPInstance, router,
                start_time: float = 360.0, seed: int = 0) -> dict:
    """Select a route, execute it, observe delays, return metrics."""
    route = router.select_route()
    metrics, steps = _execute_route(instance, route, start_time, seed)
    router.observe(steps)
    return metrics
