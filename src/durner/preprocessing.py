"""Durner Algorithm 5.1-5.2: Preprocessing (DFS + cycle cutting + topological ordering).

The transit graph with stochastic distributions can have cycles (a connection c'
that departs before c arrives, yet c' is a "successor" because transfer is possible
with some probability). We must cut these cycles and produce a topological ordering
so that TopoCSA can process connections in the correct order.

Reference: Durner (2024) Section 5.1.1, Algorithms 5.1-5.2.
"""

from __future__ import annotations

from typing import Optional
from ..transit_graph import TransitGraph, Connection
from ..pmf import prob_reachable


def build_successor_graph(graph: TransitGraph, query_window: tuple[int, int]
                          ) -> dict[int, list[int]]:
    """Build C_reach(c): for each connection c, find reachable continuing connections.

    c' is in C_reach(c) if:
    - s_arr(c) == s_dep(c')  (same stop)
    - P_reachable(c, c') > 0  (can transfer in time)

    We filter to connections in the query time window.

    Returns:
        Dict mapping connection_id → list of successor connection_ids.
    """
    t_start, t_end = query_window
    # Filter connections in window
    active_ids = set()
    for c in graph.connections:
        if t_start <= c.dep_time <= t_end:
            active_ids.add(c.id)

    # Build arrival_stop → departing connections index (for active connections)
    stop_to_deps: dict[int, list[int]] = {}
    for cid in active_ids:
        c = graph.connections[cid]
        stop_to_deps.setdefault(c.dep_stop, []).append(cid)

    successors: dict[int, list[int]] = {}
    for cid in active_ids:
        c = graph.connections[cid]
        arr_stop = c.arr_stop
        transfer_time = graph.stops[arr_stop].min_transfer_time
        succs = []
        for cid2 in stop_to_deps.get(arr_stop, []):
            if cid2 == cid:
                continue
            c2 = graph.connections[cid2]
            # Same trip connections (next segment) are always successors
            if c.trip_id == c2.trip_id:
                succs.append(cid2)
                continue
            # Check if transfer is possible
            if c.arr_distribution is not None and c2.dep_distribution is not None:
                p = prob_reachable(c.arr_distribution, c2.dep_distribution, transfer_time)
                if p > 0:
                    succs.append(cid2)
            else:
                # Fallback: check scheduled times
                if c.arr_time + transfer_time <= c2.dep_time:
                    succs.append(cid2)
        successors[cid] = succs

    return successors


def topological_sort_with_cycle_cutting(
    graph: TransitGraph,
    query_window: tuple[int, int]
) -> tuple[list[int], set[tuple[int, int]]]:
    """Durner Algorithms 5.1-5.2: DFS-based topological sort with cycle cutting.

    Returns:
        ordered_ids: Connection IDs in topological order.
        cuts: Set of (c_a, c_b) pairs where cycles were cut.
    """
    successors = build_successor_graph(graph, query_window)
    active_ids = set(successors.keys())

    # Also include connections that are only successors (not in successors keys yet)
    for succs in successors.values():
        for s in succs:
            active_ids.add(s)
            if s not in successors:
                successors[s] = []

    visited: dict[int, int] = {}  # 0=unvisited, 1=in-progress, 2=completed
    order: dict[int, int] = {}     # connection_id → topological index
    topo_index = 0
    cuts: set[tuple[int, int]] = set()

    for cid in active_ids:
        visited.setdefault(cid, 0)

    def dfs_iterative(start: int):
        nonlocal topo_index

        stack = [(start, 0)]  # (connection_id, successor_index)
        visited[start] = 1

        while stack:
            c, si = stack[-1]
            succs = successors.get(c, [])

            if si < len(succs):
                stack[-1] = (c, si + 1)
                c_next = succs[si]

                # Skip cuts
                if (c, c_next) in cuts:
                    continue

                state = visited.get(c_next, 0)
                if state == 2:
                    # Already completed, skip
                    continue
                elif state == 1:
                    # Cycle detected! Find the cut point.
                    # Cut the edge with the lowest transfer time in the cycle.
                    # Simplified: just cut (c, c_next)
                    cuts.add((c, c_next))
                else:
                    # Unvisited: push onto stack
                    visited[c_next] = 1
                    stack.append((c_next, 0))
            else:
                # All successors processed: assign topological order
                order[c] = topo_index
                topo_index += 1
                visited[c] = 2
                stack.pop()

    # Run DFS from all unvisited connections
    for cid in active_ids:
        if visited.get(cid, 0) == 0:
            dfs_iterative(cid)

    # Sort by topological order
    ordered_ids = sorted(order.keys(), key=lambda x: order[x])

    # Store order in connections
    for cid in ordered_ids:
        graph.connections[cid].topo_order = order[cid]

    return ordered_ids, cuts
