"""Training pipeline: generate data from Durner oracle → train V-hat + pruner."""

from __future__ import annotations

import numpy as np
import time
from typing import Optional

from ..transit_graph import TransitGraph
from ..synthetic_network import create_bus_story_network, create_regime_distributions
from ..durner.topocsa import topocsa
from .features import extract_stop_features, extract_connection_features
from .value_network import ValueEnsemble
from .gcn_pruner import ConnectionPruner


def generate_oracle_data(
    n_queries: int = 500,
    regimes: list[str] = None,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Generate training data by running Durner's exact algorithm.

    Returns:
        stop_data: List of {stop_id, time, regime_id, mean_arrival, features}
        conn_data: List of {conn_id, time, regime_id, in_hyperpath, features}
    """
    if regimes is None:
        regimes = ["normal", "rush_hour", "disrupted_402", "weather"]

    rng = np.random.default_rng(seed)
    stop_data = []
    conn_data = []

    for i in range(n_queries):
        regime_id = rng.integers(0, len(regimes))
        regime_name = regimes[regime_id]

        graph = create_bus_story_network()
        dists = create_regime_distributions(regime_name)
        graph.assign_distributions(dists)

        # Random query
        t_source = 360 + rng.integers(0, 360)  # 6:00 to 12:00
        s_source = 0
        s_dest = 9

        result = topocsa(graph, s_source, s_dest, t_source)

        if result.mean_arrival == float('inf'):
            continue

        # Extract stop-level data
        for stop_id, labels in result.stop_labels.items():
            if not labels:
                continue
            best_label = min(labels, key=lambda l: l.mean_dest_arrival)
            feat = extract_stop_features(graph, stop_id, t_source, regime_id)
            stop_data.append({
                "stop_id": stop_id,
                "time": t_source,
                "regime_id": regime_id,
                "mean_arrival": best_label.mean_dest_arrival,
                "features": feat,
            })

        # Extract connection-level data (for pruner)
        hp_conns = result.hyperpath_connections
        for c in graph.get_connections_in_window(t_source, t_source + 180):
            feat = extract_connection_features(graph, c, t_source, regime_id)
            conn_data.append({
                "conn_id": c.id,
                "time": t_source,
                "regime_id": regime_id,
                "in_hyperpath": 1.0 if c.id in hp_conns else 0.0,
                "features": feat,
            })

        if verbose and (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_queries} queries "
                  f"({len(stop_data)} stop samples, {len(conn_data)} conn samples)")

    return stop_data, conn_data


def train_value_ensemble(
    stop_data: list[dict],
    n_models: int = 5,
    epochs: int = 80,
    verbose: bool = True,
) -> ValueEnsemble:
    """Train V-hat ensemble on oracle data."""
    X = np.array([d["features"] for d in stop_data])
    y = np.array([d["mean_arrival"] for d in stop_data])

    # Normalize targets
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / (y_std + 1e-6)

    input_dim = X.shape[1]
    ensemble = ValueEnsemble(input_dim=input_dim, n_models=n_models)

    if verbose:
        print(f"Training V-hat ensemble: {len(X)} samples, {input_dim} features")

    ensemble.train_ensemble(X, y_norm, epochs=epochs, verbose=verbose)

    # Store normalization params for de-normalization at inference
    ensemble._y_mean = y_mean
    ensemble._y_std = y_std

    return ensemble


def train_connection_pruner(
    conn_data: list[dict],
    epochs: int = 80,
    verbose: bool = True,
) -> ConnectionPruner:
    """Train connection pruner on oracle data."""
    X = np.array([d["features"] for d in conn_data])
    y = np.array([d["in_hyperpath"] for d in conn_data])

    input_dim = X.shape[1]
    pruner = ConnectionPruner(input_dim=input_dim)

    if verbose:
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        print(f"Training pruner: {len(X)} samples ({n_pos} positive, {n_neg} negative)")

    pruner.train_pruner(X, y, epochs=epochs, verbose=verbose)

    return pruner


def full_training_pipeline(
    n_queries: int = 300,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[ValueEnsemble, ConnectionPruner]:
    """Run the complete offline training pipeline.

    1. Generate oracle data from Durner exact
    2. Train V-hat ensemble
    3. Train connection pruner
    """
    t0 = time.time()
    if verbose:
        print("=" * 50)
        print("BAPR-HRO Offline Training Pipeline")
        print("=" * 50)

    # Step 1: Generate data
    if verbose:
        print("\n[1/3] Generating oracle data...")
    stop_data, conn_data = generate_oracle_data(n_queries, seed=seed, verbose=verbose)

    # Step 2: Train V-hat
    if verbose:
        print(f"\n[2/3] Training V-hat ensemble...")
    ensemble = train_value_ensemble(stop_data, verbose=verbose)

    # Step 3: Train pruner
    if verbose:
        print(f"\n[3/3] Training connection pruner...")
    pruner = train_connection_pruner(conn_data, verbose=verbose)

    elapsed = time.time() - t0
    if verbose:
        print(f"\nTraining complete in {elapsed:.1f}s")

    return ensemble, pruner
