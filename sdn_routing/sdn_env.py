"""SDN Routing environment with stochastic link delays and regime shifts.

Uses NSFNet/GEANT2 topologies from DRL-GNN with K=4 shortest paths.
Replaces bandwidth allocation with delay-based routing:
  - Each link has a stochastic delay (base + congestion + noise)
  - Regime shifts: link failures, burst traffic on random links
  - K candidate paths per (src, dst) pair = "hyperpath" alternatives
  - Packet sent on a path → irrecoverable (delay is realized)

Mapping to BAPR-HRO transit routing:
  K paths ↔ K bus alternatives at a stop
  Link delay ↔ bus travel time
  Regime shift ↔ bus line disruption
  LCB picks lowest pessimistic end-to-end delay
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from itertools import islice


# ── Topologies (from DRL-GNN) ───────────────────────────────────────────────

def create_nsfnet() -> nx.Graph:
    """NSFNet: 14 nodes, 21 edges."""
    G = nx.Graph()
    G.add_nodes_from(range(14))
    G.add_edges_from([
        (0,1),(0,2),(0,3),(1,2),(1,7),(2,5),(3,8),(3,4),(4,5),(4,6),
        (5,12),(5,13),(6,7),(7,10),(8,9),(8,11),(9,10),(9,12),(10,11),
        (10,13),(11,12),
    ])
    return G


def create_geant2() -> nx.Graph:
    """GEANT2: 24 nodes, 37 edges."""
    G = nx.Graph()
    G.add_nodes_from(range(24))
    G.add_edges_from([
        (0,1),(0,2),(1,3),(1,6),(1,9),(2,3),(2,4),(3,6),(4,7),(5,3),
        (5,8),(6,9),(6,8),(7,11),(7,8),(8,11),(8,20),(8,17),(8,18),(8,12),
        (9,10),(9,13),(9,12),(10,13),(11,20),(11,14),(12,13),(12,19),(12,21),
        (14,15),(15,16),(16,17),(17,18),(18,21),(19,23),(21,22),(22,23),
    ])
    return G


# ── SDN Environment ─────────────────────────────────────────────────────────

@dataclass
class LinkState:
    """Current state of a network link."""
    base_delay: float = 1.0   # ms, base propagation delay
    congestion: float = 0.0   # additional congestion delay
    noise_std: float = 0.2    # ms, stochastic noise
    failed: bool = False      # link failure


@dataclass
class RegimeShift:
    """A regime shift event."""
    time_start: int       # episode when shift starts
    time_end: int         # episode when shift ends
    affected_links: list  # list of (i,j) edges affected
    shift_type: str       # "congestion", "failure"
    severity: float = 5.0 # delay multiplier or added delay


class SDNEnv:
    """SDN routing environment with stochastic delays."""

    def __init__(
        self,
        topology: str = "nsfnet",
        k_paths: int = 4,
        seed: int = 0,
        n_regime_shifts: int = 2,
        total_episodes: int = 100,
    ):
        self.rng = np.random.default_rng(seed)

        # Build topology
        if topology == "nsfnet":
            self.graph = create_nsfnet()
        elif topology == "geant2":
            self.graph = create_geant2()
        else:
            raise ValueError(f"Unknown topology: {topology}")

        self.n_nodes = len(self.graph.nodes())
        self.n_edges = len(self.graph.edges())
        self.k_paths = k_paths
        self.topology = topology

        # Initialize link states
        self.link_states: dict[tuple, LinkState] = {}
        for i, j in self.graph.edges():
            base = 1.0 + self.rng.exponential(0.5)
            self.link_states[(i, j)] = LinkState(base_delay=base, noise_std=0.2)
            self.link_states[(j, i)] = self.link_states[(i, j)]

        # Pre-compute K shortest paths for all (src, dst) pairs
        self.all_paths: dict[tuple, list[list[int]]] = {}
        for src in self.graph.nodes():
            for dst in self.graph.nodes():
                if src != dst:
                    paths = list(islice(
                        nx.shortest_simple_paths(self.graph, src, dst),
                        k_paths,
                    ))
                    self.all_paths[(src, dst)] = paths

        # Generate regime shifts
        self.regime_shifts = self._generate_regime_shifts(
            n_regime_shifts, total_episodes,
        )

        self.current_episode = 0

    def _generate_regime_shifts(self, n_shifts: int,
                                total_eps: int) -> list[RegimeShift]:
        """Generate random regime shift events."""
        shifts = []
        edges = list(self.graph.edges())
        for _ in range(n_shifts):
            start = int(self.rng.integers(10, total_eps - 20))
            duration = int(self.rng.integers(10, 30))
            n_affected = self.rng.integers(1, 4)
            affected = [tuple(edges[i]) for i in
                        self.rng.choice(len(edges), n_affected, replace=False)]
            shift_type = self.rng.choice(["congestion", "failure"])
            severity = float(self.rng.uniform(3, 10))
            shifts.append(RegimeShift(
                time_start=start, time_end=start + duration,
                affected_links=affected, shift_type=shift_type,
                severity=severity,
            ))
        return shifts

    def _apply_regime(self, episode: int):
        """Apply/remove regime shift effects for current episode."""
        # Reset all links to normal
        for link in self.link_states.values():
            link.congestion = 0.0
            link.failed = False

        # Apply active shifts
        for shift in self.regime_shifts:
            if shift.time_start <= episode < shift.time_end:
                for i, j in shift.affected_links:
                    link = self.link_states.get((i, j))
                    if link is None:
                        continue
                    if shift.shift_type == "congestion":
                        link.congestion = shift.severity
                    elif shift.shift_type == "failure":
                        link.failed = True

    def get_paths(self, src: int, dst: int) -> list[list[int]]:
        """Get K candidate paths from src to dst."""
        return self.all_paths.get((src, dst), [])

    def sample_path_delay(self, path: list[int], episode: int) -> float:
        """Sample the end-to-end delay of a path under current conditions."""
        self._apply_regime(episode)
        total_delay = 0.0
        for i in range(len(path) - 1):
            link = self.link_states.get((path[i], path[i + 1]))
            if link is None or link.failed:
                return 1000.0  # failed link → very high delay
            delay = (link.base_delay + link.congestion
                     + self.rng.normal(0, link.noise_std))
            total_delay += max(delay, 0.1)
        return total_delay

    def get_path_edges(self, path: list[int]) -> list[tuple[int, int]]:
        """Get list of edges in a path."""
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def step_episode(self):
        """Advance to next episode."""
        self.current_episode += 1


# ── Path-level beliefs and routers ──────────────────────────────────────────

@dataclass
class DelayBeliefNG:
    """Normal-Gamma posterior over path delay."""
    mu: float = 0.0
    kappa: float = 0.1
    alpha: float = 2.0
    beta: float = 1.0
    n_obs: int = 0

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def std(self) -> float:
        if self.n_obs == 0:
            return 10.0
        if self.alpha <= 1:
            return 5.0
        return float(np.sqrt(self.beta / (self.kappa * (self.alpha - 1))))

    def update(self, delay: float):
        self.n_obs += 1
        if self.n_obs == 1:
            self.mu = delay
            self.kappa = 1.0
            self.beta = max(delay * 0.1, 0.1)
            return
        kn = self.kappa + 1
        mn = (self.kappa * self.mu + delay) / kn
        an = self.alpha + 0.5
        bn = self.beta + 0.5 * self.kappa * (delay - self.mu) ** 2 / kn
        self.mu, self.kappa, self.alpha, self.beta = mn, kn, an, bn

    def sample(self, rng: np.random.Generator) -> float:
        if self.n_obs == 0:
            return rng.uniform(1, 20)
        tau = rng.gamma(self.alpha, 1.0 / max(self.beta, 1e-8))
        sigma = 1.0 / np.sqrt(max(tau * self.kappa, 1e-8))
        return rng.normal(self.mu, sigma)


@dataclass
class DelayBeliefEnsemble:
    """Ensemble belief over path delay — V2."""
    n_estimators: int = 5
    _means: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _vars: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_obs: int = 0

    def __post_init__(self):
        if len(self._means) == 0:
            self._means = np.full(self.n_estimators, 5.0)
            self._vars = np.full(self.n_estimators, 10.0)
            self._counts = np.ones(self.n_estimators)

    @property
    def mean(self) -> float:
        return float(self._means.mean())

    @property
    def std(self) -> float:
        return float(self._means.std()) if self.n_obs > 1 else 10.0

    @property
    def ood_score(self) -> float:
        if self.n_obs < 3:
            return 0.0
        avg_int = float(np.sqrt(self._vars).mean())
        return min(self.std / max(avg_int, 1e-6), 3.0)

    def update(self, delay: float, rng: np.random.Generator):
        self.n_obs += 1
        weights = rng.poisson(1, self.n_estimators)
        for k in range(self.n_estimators):
            for _ in range(weights[k]):
                self._counts[k] += 1
                n = self._counts[k]
                d = delay - self._means[k]
                self._means[k] += d / n
                self._vars[k] += (d * (delay - self._means[k]) - self._vars[k]) / n


class StaticRouter:
    """Always pick shortest path (by hop count)."""
    def select_path(self, paths, **kw) -> int:
        return 0  # first path = shortest

    def observe(self, path_idx, delay, **kw):
        pass


class _LinkBeliefRouter:
    """Base: link-level beliefs. Observing one path updates all links on it.
    Scoring a path sums LCB scores of its links. This enables fast learning:
    all paths sharing a congested link get penalized immediately."""

    def _link_key(self, i, j):
        return (min(i, j), max(i, j))

    def _score_path(self, path, beta):
        total = 0.0
        for k in range(len(path) - 1):
            key = self._link_key(path[k], path[k + 1])
            b = self.link_beliefs.get(key)
            if b is None:
                total += 5.0  # prior: unknown link ~5ms
            else:
                total += b.mean + beta * b.std
        return total

    def _observe_path(self, path, delay):
        """Distribute observed delay across links as per-link delay."""
        n_links = len(path) - 1
        if n_links == 0:
            return
        per_link = delay / n_links
        for k in range(n_links):
            key = self._link_key(path[k], path[k + 1])
            if key not in self.link_beliefs:
                self.link_beliefs[key] = DelayBeliefNG()
            self.link_beliefs[key].update(per_link)


class LCBRouter(_LinkBeliefRouter):
    """BAPR-HRO V1: link-level LCB."""
    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.link_beliefs: dict[tuple, DelayBeliefNG] = {}

    def select_path(self, paths, **kw) -> int:
        best, best_score = 0, float("inf")
        for i, path in enumerate(paths):
            score = self._score_path(path, self.beta)
            if score < best_score:
                best_score = score
                best = i
        return best

    def observe(self, path_idx, delay, paths=None, **kw):
        if paths:
            self._observe_path(paths[path_idx], delay)


class LCBRouterV2(_LinkBeliefRouter):
    """BAPR-HRO V2: Ensemble link beliefs + dynamic beta."""
    def __init__(self, beta_base=0.8, beta_ood=0.8, seed=0):
        self.beta_base = beta_base
        self.beta_ood = beta_ood
        self.rng = np.random.default_rng(seed)
        self.link_beliefs: dict[tuple, DelayBeliefEnsemble] = {}

    def _score_path(self, path, beta):
        total = 0.0
        for k in range(len(path) - 1):
            key = self._link_key(path[k], path[k + 1])
            b = self.link_beliefs.get(key)
            if b is None:
                total += 5.0
            else:
                total += b.mean + beta * b.std
        return total

    def _get_ood(self, paths):
        max_ood = 0.0
        for path in paths:
            for k in range(len(path) - 1):
                key = self._link_key(path[k], path[k + 1])
                b = self.link_beliefs.get(key)
                if b and isinstance(b, DelayBeliefEnsemble):
                    max_ood = max(max_ood, b.ood_score)
        return max_ood

    def select_path(self, paths, **kw) -> int:
        beta = self.beta_base + self.beta_ood * self._get_ood(paths)
        best, best_score = 0, float("inf")
        for i, path in enumerate(paths):
            score = self._score_path(path, beta)
            if score < best_score:
                best_score = score
                best = i
        return best

    def observe(self, path_idx, delay, paths=None, **kw):
        if not paths:
            return
        path = paths[path_idx]
        n_links = len(path) - 1
        if n_links == 0:
            return
        per_link = delay / n_links
        for k in range(n_links):
            key = self._link_key(path[k], path[k + 1])
            if key not in self.link_beliefs:
                self.link_beliefs[key] = DelayBeliefEnsemble()
            self.link_beliefs[key].update(per_link, self.rng)


class ReactUCBRouter(_LinkBeliefRouter):
    """React-UCB (Santana & Moura, 2023): link-level UCB (optimistic).

    Mirror of LCB: score = mean - c·std (hoping delay is low).
    Includes discounted updates for non-stationarity.
    """
    def __init__(self, c: float = 1.0, gamma: float = 0.95):
        self.c = c
        self.gamma = gamma
        self.link_beliefs: dict[tuple, DelayBeliefNG] = {}

    def select_path(self, paths, **kw) -> int:
        best, best_score = 0, float("inf")
        for i, path in enumerate(paths):
            score = self._score_path_ucb(path)
            if score < best_score:
                best_score = score
                best = i
        return best

    def _score_path_ucb(self, path):
        total = 0.0
        for k in range(len(path) - 1):
            key = self._link_key(path[k], path[k + 1])
            b = self.link_beliefs.get(key)
            if b is None:
                total += 1.0  # optimistic prior: assume fast
            else:
                total += max(b.mean - self.c * b.std, 0.1)
        return total

    def observe(self, path_idx, delay, paths=None, **kw):
        if not paths:
            return
        path = paths[path_idx]
        n_links = len(path) - 1
        if n_links == 0:
            return
        per_link = delay / n_links
        for k in range(n_links):
            key = self._link_key(path[k], path[k + 1])
            if key not in self.link_beliefs:
                self.link_beliefs[key] = DelayBeliefNG()
            b = self.link_beliefs[key]
            b.kappa *= self.gamma
            b.beta *= self.gamma
            b.update(per_link)


class TSRouter(_LinkBeliefRouter):
    """Thompson Sampling: link-level, sample delay per link."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.link_beliefs: dict[tuple, DelayBeliefNG] = {}

    def select_path(self, paths, **kw) -> int:
        best, best_score = 0, float("inf")
        for i, path in enumerate(paths):
            score = self._score_path_ts(path)
            if score < best_score:
                best_score = score
                best = i
        return best

    def _score_path_ts(self, path):
        total = 0.0
        for k in range(len(path) - 1):
            key = self._link_key(path[k], path[k + 1])
            b = self.link_beliefs.get(key)
            if b is None:
                total += self.rng.uniform(1, 10)
            else:
                total += b.sample(self.rng)
        return total

    def observe(self, path_idx, delay, paths=None, **kw):
        if not paths:
            return
        self._observe_path(paths[path_idx], delay)


class HybridRouter(_LinkBeliefRouter):
    """Hybrid UCB→LCB: explore with UCB first, then exploit with LCB.

    Phase 1 (episode < switch_ep): UCB (optimistic, score = mean - β·std)
    Phase 2: LCB (pessimistic, score = mean + β·std)
    Adaptive β = β₀ / √(episode)
    """
    def __init__(self, beta0: float = 2.0, switch_ep: int = 15):
        self.beta0 = beta0
        self.switch_ep = switch_ep
        self.link_beliefs: dict[tuple, DelayBeliefNG] = {}
        self._episode = 0

    def select_path(self, paths, **kw) -> int:
        beta = self.beta0 / max(self._episode + 1, 1) ** 0.5

        best, best_score = 0, float("inf")
        for i, path in enumerate(paths):
            if self._episode < self.switch_ep:
                # UCB: optimistic about delay (hope it's low)
                score = self._score_path(path, -beta)
            else:
                # LCB: pessimistic about delay (prepare for worst)
                score = self._score_path(path, beta)
            if score < best_score:
                best_score = score
                best = i
        return best

    def observe(self, path_idx, delay, paths=None, **kw):
        self._episode += 1
        if paths:
            self._observe_path(paths[path_idx], delay)


class FlowLCBRouter(_LinkBeliefRouter):
    """Flow-level LCB: once a path is chosen for a (src,dst) flow,
    it's locked for flow_duration episodes (irrecoverable).

    This makes LCB advantageous: picking a bad path is costly for
    flow_duration episodes, not just one packet.
    """
    def __init__(self, beta0: float = 2.0, flow_duration: int = 5):
        self.beta0 = beta0
        self.flow_duration = flow_duration
        self.link_beliefs: dict[tuple, DelayBeliefNG] = {}
        self._episode = 0
        self._committed: dict[str, tuple[int, int]] = {}  # (src:dst) → (path_idx, expire_ep)

    def _pair_key(self, src, dst):
        return f"{src}:{dst}"

    def select_path(self, paths, src=0, dst=0, **kw) -> int:
        pk = self._pair_key(src, dst)

        # If flow is committed and not expired, keep using same path
        if pk in self._committed:
            committed_idx, expire = self._committed[pk]
            if self._episode < expire and committed_idx < len(paths):
                return committed_idx

        # Select new path using LCB
        beta = self.beta0 / max(self._episode + 1, 1) ** 0.5

        best, best_score = 0, float("inf")
        for i, path in enumerate(paths):
            score = self._score_path(path, beta)
            if score < best_score:
                best_score = score
                best = i

        # Commit to this path for flow_duration episodes
        self._committed[pk] = (best, self._episode + self.flow_duration)
        return best

    def observe(self, path_idx, delay, paths=None, src=0, dst=0, **kw):
        self._episode += 1
        if paths:
            self._observe_path(paths[path_idx], delay)
