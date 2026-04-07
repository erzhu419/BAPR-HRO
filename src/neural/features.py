"""Feature extraction for neural value network and GCN pruner.

Converts transit graph state (stop, time, delay info, regime) into
fixed-size feature vectors suitable for neural network input.
"""

from __future__ import annotations

import numpy as np
from ..transit_graph import TransitGraph, Connection


def extract_stop_features(
    graph: TransitGraph,
    stop_id: int,
    current_time: int,
    regime_id: int = 0,
) -> np.ndarray:
    """Extract features for a stop at a given time.

    Features (14-dim):
    - time_of_day (sin, cos encoding)
    - n_departures_soon (connections departing within 15 min)
    - n_routes_serving
    - is_transfer_stop
    - regime_id (one-hot, 4 regimes)
    - mean_delay_at_stop (from recent observations, placeholder)
    - stop_id_normalized
    """
    n_stops = len(graph.stops)

    # Time encoding (cyclic)
    t_norm = current_time / 1440.0  # normalize by minutes in day
    time_sin = np.sin(2 * np.pi * t_norm)
    time_cos = np.cos(2 * np.pi * t_norm)

    # Stop connectivity
    deps = graph.get_connections_from(stop_id)
    n_deps_soon = sum(1 for c in deps if current_time <= c.dep_time <= current_time + 15)
    n_routes = len(graph.get_routes_at_stop(stop_id))
    is_transfer = float(n_routes > 1)

    # Regime one-hot
    regime_onehot = np.zeros(4)
    regime_onehot[min(regime_id, 3)] = 1.0

    # Stop ID normalized
    stop_norm = stop_id / max(n_stops, 1)

    # Placeholder for mean delay (would come from observations)
    mean_delay = 0.0

    features = np.array([
        time_sin, time_cos,
        n_deps_soon / 10.0,  # normalize
        n_routes / 5.0,
        is_transfer,
        *regime_onehot,
        mean_delay / 10.0,
        stop_norm,
        current_time / 1440.0,
        float(n_stops),
    ], dtype=np.float32)
    return features


def extract_connection_features(
    graph: TransitGraph,
    conn: Connection,
    current_time: int,
    regime_id: int = 0,
) -> np.ndarray:
    """Extract features for a connection (used by GCN pruner).

    Features (12-dim):
    - time_until_departure
    - scheduled_travel_time
    - route_id_hash (normalized)
    - dep_stop features (compact)
    - arr_stop features (compact)
    - regime one-hot
    """
    time_until_dep = (conn.dep_time - current_time) / 60.0
    travel_time = conn.scheduled_travel_time / 60.0
    route_hash = hash(conn.route) % 100 / 100.0

    dep_routes = len(graph.get_routes_at_stop(conn.dep_stop)) / 5.0
    arr_routes = len(graph.get_routes_at_stop(conn.arr_stop)) / 5.0

    regime_onehot = np.zeros(4)
    regime_onehot[min(regime_id, 3)] = 1.0

    features = np.array([
        time_until_dep,
        travel_time,
        route_hash,
        dep_routes,
        arr_routes,
        float(conn.cancel_prob),
        *regime_onehot,
        conn.dep_time / 1440.0,
        conn.arr_time / 1440.0,
    ], dtype=np.float32)
    return features


def generate_training_data(
    graph: TransitGraph,
    exact_results: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (features, labels) from Durner's exact solutions.

    Args:
        graph: Transit network.
        exact_results: List of dicts with keys:
            "stop_id", "time", "regime_id", "mean_arrival" (from TopoCSA)

    Returns:
        X: Feature matrix (N, feature_dim)
        y: Labels (N,) = mean destination arrival times
    """
    X_list = []
    y_list = []

    for result in exact_results:
        feat = extract_stop_features(
            graph, result["stop_id"], result["time"], result["regime_id"])
        X_list.append(feat)
        y_list.append(result["mean_arrival"])

    return np.array(X_list), np.array(y_list)
