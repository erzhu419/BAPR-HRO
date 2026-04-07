"""Dynamic VRP execution environment built on SVRPBench's travel time model.

SVRPBench solvers plan a static route and execute it blindly.
This environment enables STEP-BY-STEP execution where at each customer
the router decides which unvisited customer to visit next, observes
the actual (stochastic) travel time, and learns from it.

Mapping to BAPR-HRO transit routing:
  Hyperpath (candidate connections at each stop) → unvisited customer set
  LCB re-ranking of connections → LCB selection of next customer
  Regime shift (bus disruption) → time-of-day traffic + accidents
  Irrecoverable (boarded bus) → drove to customer (fuel/time spent)
"""

from __future__ import annotations

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── SVRPBench travel time model (from vrp_bench/travel_time_generator.py) ────

def _normal_pdf(x: float, mean: float, std: float) -> float:
    return math.exp(-((x - mean) ** 2) / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))


def time_factor(current_time: float) -> float:
    """Time-of-day multiplier: peaks at 8am (480) and 5pm (1020)."""
    morning = _normal_pdf(current_time, 480, 90)
    evening = _normal_pdf(current_time, 1020, 90)
    return 0.5 + 2 * (morning + evening)


def random_factor(current_time: float, rng: random.Random) -> float:
    """Log-normal noise, heavier during rush hour."""
    rush = _normal_pdf(current_time, 480, 90) + _normal_pdf(current_time, 1020, 90)
    mu = 0 + 0.1 * rush
    sigma = 0.3 + 0.2 * rush
    return rng.lognormvariate(mu, sigma)


def sample_accidents(current_time: float, rng: np.random.Generator) -> int:
    rate = 0.05 * _normal_pdf(current_time, 1260, 120)
    return int(rng.poisson(max(rate, 0)))


def sample_travel_time(
    src: int,
    dst: int,
    distances: dict[tuple[int, int], float],
    current_time: float,
    py_rng: random.Random,
    np_rng: np.random.Generator,
    velocity: float = 1.0,
    congestion_zones: list | None = None,
    locations: np.ndarray | None = None,
) -> float:
    """Sample stochastic travel time (SVRPBench model + congestion zones).

    congestion_zones: list of (cx, cy, radius, time_start, time_end, multiplier)
        Edges whose midpoint falls in a zone during active hours get extra delay.
    locations: (N+1, 2) array of node coordinates (for zone checking).
    """
    if src == dst:
        return 0.0
    dist = distances.get((src, dst), 0.0)
    if dist == 0.0:
        return 0.0

    tf = time_factor(current_time)
    df = 1 - math.exp(-dist / 50)
    base_delay = 0.25 * tf * df
    rf = random_factor(current_time, py_rng)
    delay = base_delay * rf

    n_acc = sample_accidents(current_time, np_rng)
    if n_acc > 0:
        delay += float(np_rng.uniform(30, 120, size=n_acc).sum())

    # Zone-based congestion: edges through congested areas get additive delay
    if congestion_zones and locations is not None:
        mx = (locations[src][0] + locations[dst][0]) / 2
        my = (locations[src][1] + locations[dst][1]) / 2
        for cx, cy, radius, t_start, t_end, severity in congestion_zones:
            d_to_center = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
            if d_to_center < radius and t_start <= current_time <= t_end:
                # Additive delay: 10-30 min depending on severity and proximity
                proximity = 1.0 - d_to_center / radius  # 1.0 at center, 0 at edge
                zone_delay = severity * proximity * (dist / 20)  # scale with edge length
                # Add stochastic noise to zone delay
                zone_delay *= max(0.3, py_rng.lognormvariate(0, 0.3))
                delay += zone_delay

    return dist / velocity + delay


# ── VRP Instance ─────────────────────────────────────────────────────────────

@dataclass
class VRPInstance:
    """A single CVRP instance."""
    depot: np.ndarray        # (2,) depot coordinates
    customers: np.ndarray    # (N, 2) customer coordinates
    demands: np.ndarray      # (N,) customer demands
    capacity: float          # vehicle capacity
    n_customers: int = 0

    # Pre-computed
    distances: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)
    locations: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)), repr=False)

    # Congestion zones: [(cx, cy, radius, time_start, time_end, multiplier)]
    congestion_zones: list = field(default_factory=list)

    def __post_init__(self):
        self.n_customers = len(self.customers)
        self._build_distances()

    def _build_distances(self):
        """Build all-pairs distance dict. Node 0 = depot, 1..N = customers."""
        self.locations = np.vstack([self.depot.reshape(1, 2), self.customers])
        n = len(self.locations)
        self.distances = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.distances[(i, j)] = float(np.linalg.norm(
                        self.locations[i] - self.locations[j]))

    def get_distance(self, i: int, j: int) -> float:
        return self.distances.get((i, j), 0.0)


def generate_instance(
    n_customers: int = 20,
    seed: int = 42,
    n_congestion_zones: int = 2,
) -> VRPInstance:
    """Generate a random CVRP instance with congestion zones.

    Congestion zones are circular areas that have 2-4x travel time during
    rush hour (7:30-9:00am). This creates heterogeneous uncertainty that
    BAPR-HRO can learn and exploit.
    """
    rng = np.random.default_rng(seed)
    depot = rng.uniform(10, 90, size=2)
    customers = rng.uniform(0, 100, size=(n_customers, 2))
    demands = rng.integers(1, 10, size=n_customers).astype(float)
    capacity = float(max(40, demands.sum() / 2.5))

    # Generate congestion zones: random circles that are bad during rush hour
    zones = []
    for _ in range(n_congestion_zones):
        cx, cy = rng.uniform(10, 90, size=2)
        radius = rng.uniform(15, 30)
        # Rush hour window: 6:00am-10:00am (360-600 minutes)
        t_start = 360.0
        t_end = 600.0
        severity = rng.uniform(20.0, 50.0)  # 20-50 min additive penalty per unit distance
        zones.append((float(cx), float(cy), float(radius), t_start, t_end, severity))

    return VRPInstance(
        depot=depot, customers=customers, demands=demands,
        capacity=capacity, congestion_zones=zones,
    )


# ── Step-by-step VRP execution ───────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of one routing step."""
    src: int
    dst: int
    travel_time: float
    base_distance: float
    delay: float  # travel_time - base_distance (excess)
    arrival_time: float
    current_time: float


class VRPExecutor:
    """Execute a VRP instance step-by-step with stochastic travel times.

    At each step the router picks the next customer. The executor:
      1. Samples the actual travel time
      2. Returns the observation (delay)
      3. Updates current position and time
    This enables online learning routers (LCB, TS) to adapt.
    """

    def __init__(self, instance: VRPInstance, start_time: float = 360.0, seed: int = 0):
        """
        Args:
            instance: VRP problem instance
            start_time: departure time in minutes from midnight (default 6am=360)
            seed: random seed for travel time sampling
        """
        self.inst = instance
        self.start_time = start_time
        self.py_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # State
        self.current_loc: int = 0  # depot
        self.current_time: float = start_time
        self.remaining_cap: float = instance.capacity
        self.unvisited: set[int] = set(range(1, instance.n_customers + 1))
        self.route: list[int] = [0]
        self.steps: list[StepResult] = []
        self.total_demand_served: float = 0.0

    def get_feasible_customers(self) -> list[int]:
        """Return unvisited customers that fit in remaining capacity."""
        feasible = []
        for cid in self.unvisited:
            demand = self.inst.demands[cid - 1]
            if demand <= self.remaining_cap:
                feasible.append(cid)
        return feasible

    def step(self, next_customer: int) -> StepResult:
        """Drive to next_customer, sample travel time, return observation."""
        assert next_customer in self.unvisited, f"Customer {next_customer} already visited"

        base_dist = self.inst.get_distance(self.current_loc, next_customer)
        tt = sample_travel_time(
            self.current_loc, next_customer,
            self.inst.distances, self.current_time,
            self.py_rng, self.np_rng,
            congestion_zones=self.inst.congestion_zones,
            locations=self.inst.locations,
        )
        delay = tt - base_dist

        self.current_time += tt
        self.current_loc = next_customer
        self.remaining_cap -= self.inst.demands[next_customer - 1]
        self.total_demand_served += self.inst.demands[next_customer - 1]
        self.unvisited.remove(next_customer)
        self.route.append(next_customer)

        result = StepResult(
            src=self.route[-2],
            dst=next_customer,
            travel_time=tt,
            base_distance=base_dist,
            delay=delay,
            arrival_time=self.current_time,
            current_time=self.current_time,
        )
        self.steps.append(result)
        return result

    def return_to_depot(self) -> StepResult:
        """Return vehicle to depot."""
        base_dist = self.inst.get_distance(self.current_loc, 0)
        tt = sample_travel_time(
            self.current_loc, 0,
            self.inst.distances, self.current_time,
            self.py_rng, self.np_rng,
            congestion_zones=self.inst.congestion_zones,
            locations=self.inst.locations,
        )
        delay = tt - base_dist
        self.current_time += tt
        self.current_loc = 0
        self.route.append(0)

        result = StepResult(
            src=self.route[-2], dst=0,
            travel_time=tt, base_distance=base_dist, delay=delay,
            arrival_time=self.current_time, current_time=self.current_time,
        )
        self.steps.append(result)
        return result

    def is_done(self) -> bool:
        return len(self.unvisited) == 0

    def get_metrics(self) -> dict:
        """Summary metrics after route completion."""
        total_tt = self.current_time - self.start_time
        total_dist = sum(s.base_distance for s in self.steps)
        total_delay = sum(s.delay for s in self.steps)
        return {
            "total_time": total_tt,
            "total_distance": total_dist,
            "total_delay": total_delay,
            "n_served": self.inst.n_customers - len(self.unvisited),
            "n_customers": self.inst.n_customers,
            "route": self.route.copy(),
        }
