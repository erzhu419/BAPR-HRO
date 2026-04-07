"""Incremental hyperpath update: re-weight stop_labels under new distributions.

Instead of recomputing the entire hyperpath from scratch (which may produce
an over-conservative result), we keep the original hyperpath structure and
re-evaluate each connection's feasibility and expected arrival time using
the new regime's delay distributions.

This preserves Durner's key insight: the hyperpath is a robust SET of
alternatives. We just re-rank them based on current conditions.

The update is O(|labels|) per stop — much faster than full TopoCSA O(|C|^2·|T|).
"""

from __future__ import annotations

import numpy as np
from ..transit_graph import TransitGraph, StopLabel
from ..pmf import PMF, prob_reachable, componentwise_sum
from .topocsa import HyperpathResult


def reweight_hyperpath(
    graph: TransitGraph,
    original: HyperpathResult,
    s_dest: int,
    current_stop: int,
    current_time: int,
) -> HyperpathResult:
    """Re-weight an existing hyperpath using the graph's current distributions.

    The graph should already have updated distributions (from new regime).
    We re-evaluate each label's feasibility and arrival distribution
    WITHOUT changing which connections are in the hyperpath.

    Algorithm:
    1. For connections arriving at destination: update T_dest from new arr_distribution
    2. For other connections: re-compute P_reachable and P_feasible using new distributions
    3. Re-sort stop_labels by updated mean_dest_arrival

    This is a simplified single-pass update (not full Bellman propagation),
    but preserves the hyperpath structure.
    """
    new_stop_labels: dict[int, list[StopLabel]] = {}

    # First pass: update labels for connections arriving at destination
    for sid, labels in original.stop_labels.items():
        new_labels = []
        for label in labels:
            c = graph.connections[label.connection_id]

            # Skip connections that have already departed
            if c.dep_time < current_time - 1:
                continue

            if c.arr_distribution is None or c.dep_distribution is None:
                continue

            if c.arr_stop == s_dest:
                # Direct to destination: just use updated arrival distribution
                new_dest_arrival = c.arr_distribution
                new_mean = new_dest_arrival.mean()
                new_feas = 1.0 - c.cancel_prob
            else:
                # For non-destination connections: re-evaluate based on
                # downstream labels. Use a lightweight approximation:
                # scale the original arrival estimate by the ratio of
                # new vs old delay expectations.
                if c.arr_distribution is not None:
                    old_mean = label.mean_dest_arrival
                    # Adjust by new delay at this connection
                    new_arr_mean = c.arr_distribution.mean()
                    old_arr_mean = c.arr_time  # scheduled
                    delay_shift = new_arr_mean - old_arr_mean
                    new_mean = old_mean + delay_shift
                    new_feas = label.feasibility * (1.0 - c.cancel_prob)
                    new_dest_arrival = label.dest_arrival  # keep original shape
                else:
                    new_mean = label.mean_dest_arrival
                    new_feas = label.feasibility
                    new_dest_arrival = label.dest_arrival

            if new_feas < 1e-6:
                continue

            new_labels.append(StopLabel(
                connection_id=label.connection_id,
                dest_arrival=new_dest_arrival,
                mean_dest_arrival=new_mean,
                feasibility=new_feas,
            ))

        # Re-sort: descending mean_dest_arrival (worst first, best last)
        new_labels.sort(key=lambda l: -l.mean_dest_arrival)
        new_stop_labels[sid] = new_labels

    # Compute new result stats
    source_labels = new_stop_labels.get(current_stop, [])
    if source_labels:
        best = min(source_labels, key=lambda l: l.mean_dest_arrival)
        mean_arrival = best.mean_dest_arrival
        dest_arrival = best.dest_arrival
    else:
        mean_arrival = float('inf')
        dest_arrival = None

    return HyperpathResult(
        stop_labels=new_stop_labels,
        dest_arrival=dest_arrival,
        mean_arrival=mean_arrival,
        hyperpath_connections=original.hyperpath_connections,
        n_connections_processed=0,  # no full recompute
        cuts=original.cuts,
    )
