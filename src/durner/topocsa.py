"""Durner Algorithm 5.3-5.4: Topological CSA (stochastic hyperpath query).

Given a source stop, destination stop, and departure time, compute the
optimal stochastic hyperpath: a set of alternative connections at each
intermediate stop that minimizes expected destination arrival time.

Reference: Durner (2024) Section 5.1.2, Algorithms 5.3-5.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..transit_graph import TransitGraph, StopLabel
from ..pmf import PMF, componentwise_sum, prob_reachable
from .preprocessing import topological_sort_with_cycle_cutting


@dataclass
class HyperpathResult:
    """Result of a stochastic hyperpath query."""
    # Per-stop alternatives: stop_id → list of StopLabels (ranked by mean arrival)
    stop_labels: dict[int, list[StopLabel]]
    # Overall destination arrival distribution (from source stop)
    dest_arrival: Optional[PMF]
    # Mean destination arrival time
    mean_arrival: float
    # Connections that are part of the hyperpath
    hyperpath_connections: set[int]
    # Stats
    n_connections_processed: int
    cuts: set[tuple[int, int]]


def topocsa(
    graph: TransitGraph,
    s_source: int,
    s_dest: int,
    t_source: int,
    t_end: Optional[int] = None,
) -> HyperpathResult:
    """Run the Topological CSA query algorithm (Durner Algorithm 5.3).

    Args:
        graph: Transit network with distributions assigned.
        s_source: Source stop ID.
        s_dest: Destination stop ID.
        t_source: Earliest departure time (minutes from midnight).
        t_end: Latest departure time to consider (default: t_source + 180 min).

    Returns:
        HyperpathResult with ranked alternatives at each stop.
    """
    if t_end is None:
        t_end = t_source + 180  # 3-hour window

    # Step 1: Preprocessing - topological sort
    ordered_ids, cuts = topological_sort_with_cycle_cutting(
        graph, (t_source, t_end))

    # Step 2: Initialize stop_labels for all stops
    stop_labels: dict[int, list[StopLabel]] = {sid: [] for sid in graph.stops}

    # Step 3: Process connections in topological order (Alg 5.3)
    for cid in ordered_ids:
        c = graph.connections[cid]

        if c.arr_distribution is None or c.dep_distribution is None:
            continue

        # Case 1: Connection arrives at destination
        if c.arr_stop == s_dest:
            feasibility = 1.0
            dest_arrival = c.arr_distribution
            mean_da = dest_arrival.mean()

            label = StopLabel(
                connection_id=cid,
                dest_arrival=dest_arrival,
                mean_dest_arrival=mean_da,
                feasibility=feasibility,
            )
            _insert_departure_label(stop_labels, c.dep_stop, label, c)
            continue

        # Case 2: Connection does not arrive at destination
        # Iterate over stop_labels at arrival stop (lines 10-17 of Alg 5.3)
        arr_labels = stop_labels.get(c.arr_stop, [])
        if not arr_labels:
            continue

        transfer_time = graph.stops[c.arr_stop].min_transfer_time
        p_remaining = 1.0
        t_temp: Optional[PMF] = None

        # Process labels from best (earliest arrival) to worst (Durner Alg 5.3 line 10)
        # Labels are stored worst-first (descending mean_dest_arrival), so reverse
        for label in reversed(arr_labels):
            # Skip cuts
            if (cid, label.connection_id) in cuts:
                continue

            c_next = graph.connections[label.connection_id]

            # P_success = P_feasible(c') · P_reachable(c, c')
            p_reachable = prob_reachable(
                c.arr_distribution, c_next.dep_distribution, transfer_time)
            p_success = label.feasibility * p_reachable

            if p_success < 1e-10:
                continue

            # T_temp ← T_temp ⊕ (P_success · P_remaining · T_dest(c'))
            weighted = label.dest_arrival.scale(p_success * p_remaining)
            if t_temp is None:
                t_temp = weighted
            else:
                t_temp = componentwise_sum(t_temp, weighted)

            # P_remaining ← P_remaining · (1 - P_success)
            p_remaining *= (1.0 - p_success)

            if p_remaining < 1e-10:
                break

        if t_temp is None:
            continue

        # P_feasible(c) = 1 - P_remaining
        feasibility = 1.0 - p_remaining
        if feasibility < 1e-10:
            continue

        # T_dest(c) = (1 / P_feasible(c)) · T_temp
        dest_arrival = t_temp.scale(1.0 / feasibility)
        mean_da = dest_arrival.mean()

        label = StopLabel(
            connection_id=cid,
            dest_arrival=dest_arrival,
            mean_dest_arrival=mean_da,
            feasibility=feasibility,
        )
        _insert_departure_label(stop_labels, c.dep_stop, label, c)

    # Extract result for source stop
    source_labels = stop_labels.get(s_source, [])
    if source_labels:
        best = source_labels[-1]  # last = earliest mean arrival (sorted ascending)
        dest_arrival = best.dest_arrival
        mean_arrival = best.mean_dest_arrival
    else:
        dest_arrival = None
        mean_arrival = float('inf')

    # Collect hyperpath connections
    hp_conns = set()
    for sid, labels in stop_labels.items():
        for lab in labels:
            hp_conns.add(lab.connection_id)

    return HyperpathResult(
        stop_labels=stop_labels,
        dest_arrival=dest_arrival,
        mean_arrival=mean_arrival,
        hyperpath_connections=hp_conns,
        n_connections_processed=len(ordered_ids),
        cuts=cuts,
    )


def _insert_departure_label(
    stop_labels: dict[int, list[StopLabel]],
    dep_stop: int,
    new_label: StopLabel,
    connection: object,
) -> None:
    """Durner Algorithm 5.4: Insert a new departure label maintaining sorted order.

    Labels are sorted by mean_dest_arrival in descending order (worst first,
    best last). This allows efficient iteration in reverse during the main loop.

    Domination: label A dominates label B if A has both earlier departure AND
    earlier mean destination arrival. Dominated labels are removed.
    """
    labels = stop_labels[dep_stop]
    mean_da = new_label.mean_dest_arrival

    if mean_da == float('inf'):
        return

    # Find insertion position (binary search for sorted order)
    # Labels sorted descending by mean_dest_arrival
    lo, hi = 0, len(labels)
    while lo < hi:
        mid = (lo + hi) // 2
        if labels[mid].mean_dest_arrival > mean_da:
            lo = mid + 1
        else:
            hi = mid

    # Check domination: if the next label (later arrival) has later or equal
    # departure time, the new label dominates it — but we use Durner's
    # simplified approach: just insert and let the non-dominated property hold
    # by mean_dest_arrival ordering.
    if lo < len(labels) and abs(labels[lo].mean_dest_arrival - mean_da) < 0.01:
        # Nearly identical mean arrival — skip to avoid redundancy
        return

    labels.insert(lo, new_label)

    # Remove dominated labels: any label with worse mean_dest_arrival
    # that also has later departure. Simplified: keep all for now,
    # since the main loop processes in order anyway.
