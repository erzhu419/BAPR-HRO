"""Transit graph data structures for stochastic hyperpath routing.

Models a public transit network as a set of stops and connections,
following Durner (2024) Section 3. Compatible with GTFS data format
but does not require it — can be built from any source including SUMO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .pmf import PMF


@dataclass
class Stop:
    """A transit stop (bus stop, train station, etc.)."""
    id: int
    name: str
    lat: float = 0.0
    lon: float = 0.0
    # Minimum transfer time at this stop (seconds→minutes in our model)
    min_transfer_time: int = 2  # minutes


@dataclass
class Connection:
    """A single scheduled vehicle departure from one stop to the next.

    In Durner's formulation, a connection c is defined by:
    - dep_stop, arr_stop: departure and arrival stops
    - route: which line/route this belongs to
    - t_dep, t_arr: scheduled departure/arrival times
    - dep_distribution, arr_distribution: stochastic actual times (PMFs)
    - cancel_prob: probability that this connection is canceled

    The PMF fields are set during distribution mapping (Section 5.4).
    """
    id: int
    route: str
    trip_id: str
    dep_stop: int       # stop id
    arr_stop: int       # stop id
    dep_time: int       # scheduled departure, minutes from midnight
    arr_time: int       # scheduled arrival, minutes from midnight
    # Stochastic distributions (set later from historical data / regime)
    dep_distribution: Optional[PMF] = None
    arr_distribution: Optional[PMF] = None
    cancel_prob: float = 0.0
    # Computed by preprocessing
    topo_order: int = -1

    @property
    def scheduled_travel_time(self) -> int:
        return self.arr_time - self.dep_time


@dataclass
class StopLabel:
    """A non-dominated departure option at a stop (Durner Alg 5.3-5.4).

    For each stop, we maintain a list of non-dominated connections sorted
    by mean destination arrival time (earlier = better).
    """
    connection_id: int
    dest_arrival: PMF               # T_dest(c): distribution of arrival at destination
    mean_dest_arrival: float         # E[T_dest(c)]
    feasibility: float              # P_feasible(c): prob user is still at this stop


@dataclass
class TransitGraph:
    """The complete transit network for routing queries.

    Built from GTFS data or synthetic generation. Stores all stops and
    connections, with methods for querying and filtering.
    """
    stops: dict[int, Stop] = field(default_factory=dict)
    connections: list[Connection] = field(default_factory=list)
    # Index: stop_id → list of connection ids departing from this stop
    _dep_index: dict[int, list[int]] = field(default_factory=dict)
    # Index: stop_id → list of connection ids arriving at this stop
    _arr_index: dict[int, list[int]] = field(default_factory=dict)

    def add_stop(self, stop: Stop):
        self.stops[stop.id] = stop

    def add_connection(self, conn: Connection):
        conn.id = len(self.connections)
        self.connections.append(conn)
        self._dep_index.setdefault(conn.dep_stop, []).append(conn.id)
        self._arr_index.setdefault(conn.arr_stop, []).append(conn.id)

    def get_connections_from(self, stop_id: int) -> list[Connection]:
        """All connections departing from a stop."""
        return [self.connections[i] for i in self._dep_index.get(stop_id, [])]

    def get_connections_to(self, stop_id: int) -> list[Connection]:
        """All connections arriving at a stop."""
        return [self.connections[i] for i in self._arr_index.get(stop_id, [])]

    def get_connections_in_window(self, t_start: int, t_end: int) -> list[Connection]:
        """All connections with scheduled departure in [t_start, t_end]."""
        return [c for c in self.connections if t_start <= c.dep_time <= t_end]

    def get_routes_at_stop(self, stop_id: int) -> set[str]:
        """All route names serving a stop (departing or arriving)."""
        routes = set()
        for cid in self._dep_index.get(stop_id, []):
            routes.add(self.connections[cid].route)
        for cid in self._arr_index.get(stop_id, []):
            routes.add(self.connections[cid].route)
        return routes

    def get_transfer_stops(self) -> list[int]:
        """Stops served by more than one route (transfer points)."""
        return [sid for sid in self.stops if len(self.get_routes_at_stop(sid)) > 1]

    def assign_distributions(self, regime_dists: dict[str, dict[str, np.ndarray]],
                             default_std: float = 2.0):
        """Assign delay distributions to all connections based on regime.

        Args:
            regime_dists: {route: {"delay_probs": array, "delay_offset": int}}
                          If a route is not in the dict, use a default Gaussian.
            default_std: Std dev (minutes) for default delay distribution.
        """
        for conn in self.connections:
            if conn.route in regime_dists:
                info = regime_dists[conn.route]
                conn.dep_distribution = PMF.from_delays(
                    conn.dep_time, info["delay_probs"], info.get("delay_offset", 0))
                conn.arr_distribution = PMF.from_delays(
                    conn.arr_time, info["delay_probs"], info.get("delay_offset", 0))
            else:
                # Default: Gaussian-like delay centered at 0, discretized
                delays = np.arange(-3, 11)  # -3 to +10 min
                probs = np.exp(-0.5 * (delays / default_std) ** 2)
                probs = probs / probs.sum()
                conn.dep_distribution = PMF.from_delays(conn.dep_time, probs, -3)
                conn.arr_distribution = PMF.from_delays(conn.arr_time, probs, -3)

    def summary(self) -> str:
        n_stops = len(self.stops)
        n_conn = len(self.connections)
        routes = set(c.route for c in self.connections)
        transfers = self.get_transfer_stops()
        return (f"TransitGraph: {n_stops} stops, {n_conn} connections, "
                f"{len(routes)} routes, {len(transfers)} transfer stops")
