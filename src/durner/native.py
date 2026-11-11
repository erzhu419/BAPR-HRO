"""Independent port of Durner's strict transfer strategy, Algorithms 5.1--5.4.

Reference: Robin Durner (2024), *Stochastic Strategies for Public Transport
Journeys Based on Realtime Delay Predictions*, pp. 32--34, 50--59.

Inputs are normalized conditional departure/arrival PMFs and explicit cancel_prob.
This is the original independent-marginal, strict mean-dominance policy model,
not an optimum for correlated events or a truncated failure-penalized objective.
The local graph has no scalar realtime forecast: PMF means supply the scalar
forecast for cycle-cut ordering; the query window includes possible departures.
Fully canceled connections are removed as unavailable actions.
"""
from collections import defaultdict

from ..pmf import PMF, componentwise_sum, prob_reachable, query_reachability
from ..transit_graph import StopLabel
from .topocsa import HyperpathResult


def _trip_successors(graph):
    trips = defaultdict(list)
    for connection in graph.connections:
        if connection.trip_id:
            trips[connection.trip_id].append(connection)
    following = {}
    for connections in trips.values():
        connections.sort(key=lambda c: (c.dep_time, c.id))
        for first, second in zip(connections, connections[1:]):
            if first.arr_stop == second.dep_stop:
                following[first.id] = second.id
    return following


def _reachable(graph, first, second, following, reach):
    if following.get(first.id) == second.id:
        return 1.0
    transfer = graph.stops[first.arr_stop].min_transfer_time
    return (reach(first.arr_distribution, second.dep_distribution, transfer)
            * (1.0 - second.cancel_prob))


def native_preprocess(graph, t_source, t_end, reach=prob_reachable):
    """Algorithms 5.1--5.2, including lowest-transfer-time cycle cuts/backtrack."""
    active = {c.id for c in graph.connections
              if c.dep_distribution is not None and c.arr_distribution is not None
              and c.dep_distribution.support_max >= t_source
              and c.dep_distribution.support_min <= t_end and c.cancel_prob < 1.0}
    following = _trip_successors(graph)
    by_stop = defaultdict(list)
    for cid in active:
        by_stop[graph.connections[cid].dep_stop].append(cid)
    dep_mean = {cid: graph.connections[cid].dep_distribution.mean() for cid in active}
    arr_mean = {cid: graph.connections[cid].arr_distribution.mean() for cid in active}
    successors = {}
    for cid in sorted(active):
        first = graph.connections[cid]
        successors[cid] = sorted((n for n in by_stop[first.arr_stop]
                                  if n != cid and _reachable(graph, first, graph.connections[n], following, reach) > 0),
                                 key=lambda n: (dep_mean[n], n))
    visited = {cid: 0 for cid in active}
    cuts = set()
    order = []
    for anchor in sorted(active):
        if visited[anchor] == 2:
            continue
        stack = [anchor]
        indices = [len(successors[anchor])]
        while stack:
            current = stack[-1]
            visited[current] = 1
            if indices[-1] == 0:
                order.append(current)
                visited[current] = 2
                stack.pop()
                indices.pop()
                continue
            indices[-1] -= 1
            nxt = successors[current][indices[-1]]
            if (current, nxt) in cuts:
                continue
            if visited[nxt] == 1:
                cycle_start = stack.index(nxt)
                cycle = stack[cycle_start:] + [nxt]
                pairs = list(zip(cycle, cycle[1:]))
                cut = min(pairs, key=lambda edge: dep_mean[edge[1]] - arr_mean[edge[0]])
                cuts.add(cut)
                if cut[1] != nxt:
                    while stack[-1] != cut[0]:
                        visited[stack.pop()] = 0
                        indices.pop()
            elif visited[nxt] != 2:
                stack.append(nxt)
                indices.append(len(successors[nxt]))
    return order, cuts, following


def insert_strict_label(labels, label, graph):
    """Algorithm 5.4 exactly: arrival order plus departure mean dominance.

    Equal departure means remain admissible, as required in the original text.
    """
    if label.feasibility == 0:
        return
    departure = graph.connections[label.connection_id].dep_distribution.mean()
    j = len(labels) - 1
    while j >= 0 and label.mean_dest_arrival > labels[j].mean_dest_arrival:
        j -= 1
    if j + 1 == len(labels) or departure >= graph.connections[labels[j + 1].connection_id].dep_distribution.mean():
        labels.insert(j + 1, label)
        while j >= 0 and departure > graph.connections[labels[j].connection_id].dep_distribution.mean():
            labels.pop(j)
            j -= 1


def _combine(options):
    remaining, mixture = 1.0, None
    for label, reachability in options:
        success = label.feasibility * reachability
        weighted = label.dest_arrival.scale(success * remaining)
        mixture = weighted if mixture is None else componentwise_sum(mixture, weighted)
        remaining *= 1.0 - success
    feasibility = 1.0 - remaining
    if feasibility <= 0:
        return None, 0.0
    return mixture.scale(1.0 / feasibility), feasibility


def native_topocsa(graph, s_source, s_dest, t_source, t_end=None):
    """Strict native policy recursion, on a causal snapshot supplied by caller."""
    if t_end is None:
        t_end = t_source + 180
    # The old recompute baseline used subprobability PMFs for cancellation.
    # Reject that representation rather than counting cancellation twice here.
    for c in graph.connections:
        for distribution in (c.dep_distribution, c.arr_distribution):
            if distribution is not None and abs(distribution.total_prob - 1.0) > 1e-9:
                raise ValueError('native_topocsa needs conditional normalized PMFs and explicit cancel_prob')
    reach = query_reachability()
    order, cuts, following = native_preprocess(graph, t_source, t_end, reach)
    labels = {stop: [] for stop in graph.stops}
    for cid in order:
        c = graph.connections[cid]
        if c.arr_stop == s_dest:
            arrival, feasibility = c.arr_distribution, 1.0
        else:
            options = [(label, _reachable(graph, c, graph.connections[label.connection_id], following, reach))
                       for label in reversed(labels[c.arr_stop])
                       if (cid, label.connection_id) not in cuts]
            arrival, feasibility = _combine(options)
        if arrival is not None:
            label = StopLabel(cid, arrival, arrival.mean(), feasibility)
            insert_strict_label(labels[c.dep_stop], label, graph)
    source_options = [(label, prob_reachable(PMF.deterministic(t_source),
                                            graph.connections[label.connection_id].dep_distribution, 0)
                       * (1.0 - graph.connections[label.connection_id].cancel_prob))
                      for label in reversed(labels[s_source])]
    arrival, _ = _combine(source_options)
    return HyperpathResult(labels, arrival, arrival.mean() if arrival is not None else float('inf'),
                           {label.connection_id for options in labels.values() for label in options},
                           len(order), cuts)
