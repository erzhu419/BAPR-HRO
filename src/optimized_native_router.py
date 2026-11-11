"""Exact implementation optimizations for the existing strict native recursion.

The published comparison model is unchanged: independent minute marginals,
strict mean dominance and the original cycle-cut rule. Its DFS runs in a
small compiled kernel; posterior revisions reuse unaffected trip projections.
Successor positivity uses the same numerical CDF support, not a full integral.
These are implementation savings,
not a new transit-routing algorithm or a comparison with optimized C++.
"""
from collections import defaultdict
import ctypes
from dataclasses import replace
from math import ceil, floor
from time import perf_counter
from pathlib import Path
import subprocess

import numpy as np

from .durner.native import _combine, _reachable, _trip_successors
from .durner.topocsa import HyperpathResult
from .native_event_router import NativeEventRouter
from .pmf import PMF, prob_reachable, query_reachability
from .transit_graph import StopLabel, TransitGraph

FLOAT_TINY = np.finfo(float).tiny


def build_native_kernel(output):
    """Explicit preprocessing; callers charge/report this compilation once."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['c++', '-O3', '-std=c++17', '-shared', '-fPIC',
        str(Path(__file__).with_name('native_order.cpp')), '-o', str(output)], check=True)
    return load_native_kernel(output)


def load_native_kernel(path):
    kernel = ctypes.CDLL(str(Path(path).resolve()))
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    floating = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    counter = np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')
    kernel.native_order.argtypes = [ctypes.c_int, ctypes.c_int, integer, integer,
        integer, floating, floating, integer, integer, integer, counter]
    kernel.native_order.restype = ctypes.c_int
    return kernel


class NativeWorkspace:
    def __init__(self, graph, kernel):
        self.graph = graph
        self.following = _trip_successors(graph)
        self.reach = query_reachability()
        self.means = {}
        self.supports = {}
        self.kernel = kernel
        self.stats = {}

    def mean(self, pmf):
        key = id(pmf.probs), pmf.offset
        entry = self.means.get(key)
        if entry is None:
            entry = pmf.mean(), pmf.probs
            self.means[key] = entry
        return entry[0]

    def support(self, pmf):
        key = id(pmf.probs)
        if key not in self.supports:
            cdf = np.r_[0., np.cumsum(pmf.probs)]
            indices = np.flatnonzero(cdf < cdf[-1])
            last = int(indices[-1]) if len(indices) else -1
            tail = cdf[-1] - cdf[last] if last >= 0 else 0.
            self.supports[key] = last, tail, np.maximum.accumulate(pmf.probs), pmf.probs
        return self.supports[key]

    def candidates(self, first, choices, graph, following, latest, tails, operating):
        choices = choices[choices != first.id]
        arrival = first.arr_distribution
        prefix = self.support(arrival)[2]
        transfer = graph.stops[first.arr_stop].min_transfer_time
        count = latest[choices] - transfer - arrival.offset + 1
        mass = prefix[np.clip(count, 1, len(prefix)) - 1]
        visible = (count > 0) & (mass > 0)
        # A normal positive lower bound proves that the original nonnegative
        # dot product survives underflow. Unresolved tiny products use it.
        proved = visible & (mass * tails[choices] * operating[choices] >= FLOAT_TINY)
        same = choices == following.get(first.id, -1)
        unresolved = visible & ~proved & ~same
        self.stats['zero_support_rejections'] += int(np.count_nonzero(~visible & ~same))
        self.stats['positive_support_proofs'] += int(np.count_nonzero(proved & ~same))
        self.stats['positive_fallback_integrals'] += int(np.count_nonzero(unresolved))
        for index in np.flatnonzero(unresolved):
            second = graph.connections[choices[index]]
            proved[index] = self.reach(arrival, second.dep_distribution, transfer) * operating[second.id] > 0
        return choices[proved | same].tolist()


def optimized_native_preprocess(graph, t_source, t_end, workspace):
    """Same active set, successor order, DFS and cut tie-break as native.py."""
    started = perf_counter()
    workspace.stats.update(zero_support_rejections=0, positive_support_proofs=0, positive_fallback_integrals=0)
    active = {c.id for c in graph.connections
              if c.dep_distribution is not None and c.arr_distribution is not None
              and c.dep_distribution.support_max >= t_source
              and c.dep_distribution.support_min <= t_end and c.cancel_prob < 1.}
    by_stop = defaultdict(list)
    for cid in active:
        by_stop[graph.connections[cid].dep_stop].append(cid)
    dep_mean = {cid: workspace.mean(graph.connections[cid].dep_distribution) for cid in active}
    arr_mean = {cid: workspace.mean(graph.connections[cid].arr_distribution) for cid in active}
    following = workspace.following
    by_stop = {stop: np.array(cids, dtype=np.int32) for stop, cids in by_stop.items()}
    latest = np.zeros(len(graph.connections), dtype=np.int32)
    tails, operating = np.zeros(len(graph.connections)), np.zeros(len(graph.connections))
    for cid in active:
        c = graph.connections[cid]
        last, tail, _, _ = workspace.support(c.dep_distribution)
        latest[cid] = c.dep_distribution.offset + last
        tails[cid], operating[cid] = tail, 1. - c.cancel_prob
    successors = {}
    for cid in sorted(active):
        first = graph.connections[cid]
        choices = by_stop.get(first.arr_stop)
        options = workspace.candidates(first, choices, graph, following, latest, tails, operating) if choices is not None else []
        successors[cid] = sorted(options, key=lambda n: (dep_mean[n], n))
    successor_seconds = perf_counter() - started
    before = perf_counter()
    n = len(graph.connections)
    offsets = np.zeros(n + 1, dtype=np.int32)
    for cid, options in successors.items():
        offsets[cid + 1] = len(options)
    offsets = offsets.cumsum(dtype=np.int32)
    edges = np.empty(int(offsets[-1]), dtype=np.int32)
    for cid, options in successors.items():
        edges[offsets[cid]:offsets[cid + 1]] = options
    active_ids = np.array(sorted(active), dtype=np.int32)
    departures, arrivals = np.zeros(n), np.zeros(n)
    for cid in active:
        departures[cid], arrivals[cid] = dep_mean[cid], arr_mean[cid]
    order = np.empty(len(active_ids), dtype=np.int32)
    cut_sources, cut_targets = np.empty(len(edges), dtype=np.int32), np.empty(len(edges), dtype=np.int32)
    work = np.zeros(2, dtype=np.uint64)
    ncuts = workspace.kernel.native_order(n, len(active_ids), active_ids, offsets, edges,
        departures, arrivals, order, cut_sources, cut_targets, work)
    cuts = set(zip(map(int, cut_sources[:ncuts]), map(int, cut_targets[:ncuts])))
    workspace.stats.update(preprocess_successors_ms=successor_seconds * 1000,
        preprocess_order_ms=(perf_counter() - before) * 1000,
        native_dfs_visits=int(work[0]), original_cycle_edges=int(work[1]), successor_edges=len(edges))
    return order.tolist(), cuts, following


def optimized_native_topocsa(graph, s_source, s_dest, t_source, t_end, workspace):
    if t_end is None:
        t_end = t_source + 180
    for c in graph.connections:
        for distribution in (c.dep_distribution, c.arr_distribution):
            if distribution is not None and abs(distribution.total_prob - 1.) > 1e-9:
                raise ValueError('native_topocsa needs conditional normalized PMFs and explicit cancel_prob')
    order, cuts, following = optimized_native_preprocess(graph, t_source, t_end, workspace)
    started = perf_counter()
    labels = {stop: [] for stop in graph.stops}
    for cid in order:
        c = graph.connections[cid]
        if c.arr_stop == s_dest:
            arrival, feasibility = c.arr_distribution, 1.
        else:
            options = [(label, _reachable(graph, c, graph.connections[label.connection_id], following, workspace.reach))
                       for label in reversed(labels[c.arr_stop]) if (cid, label.connection_id) not in cuts]
            arrival, feasibility = _combine(options)
        if arrival is not None and feasibility:
            label = StopLabel(cid, arrival, arrival.mean(), feasibility)
            options = labels[c.dep_stop]
            departure = workspace.mean(c.dep_distribution)
            j = len(options) - 1
            while j >= 0 and label.mean_dest_arrival > options[j].mean_dest_arrival:
                j -= 1
            if j + 1 == len(options) or departure >= workspace.mean(graph.connections[options[j + 1].connection_id].dep_distribution):
                options.insert(j + 1, label)
                while j >= 0 and departure > workspace.mean(graph.connections[options[j].connection_id].dep_distribution):
                    options.pop(j)
                    j -= 1
    source_options = [(label, prob_reachable(PMF.deterministic(t_source),
        graph.connections[label.connection_id].dep_distribution, 0) *
        (1. - graph.connections[label.connection_id].cancel_prob)) for label in reversed(labels[s_source])]
    arrival, _ = _combine(source_options)
    workspace.stats['propagation_ms'] = (perf_counter() - started) * 1000
    return HyperpathResult(labels, arrival, arrival.mean() if arrival is not None else float('inf'),
        {label.connection_id for options in labels.values() for label in options}, len(order), cuts)


class PosteriorMinuteProjection:
    """Reuse unchanged trip factors of this same evolving public belief."""

    def __init__(self, belief):
        self.belief = belief
        self.graph = TransitGraph()
        for sid, stop in belief.graph.stops.items():
            self.graph.add_stop(replace(stop, min_transfer_time=ceil(stop.min_transfer_time / belief.minute)))
        for c in belief.graph.connections:
            self.graph.add_connection(replace(c, dep_time=c.dep_time / belief.minute,
                                              arr_time=c.arr_time / belief.minute))
        self.shapes, self.keys, self.trip_stats, self.cancellation = {}, {}, {}, {}

    def refresh(self, parameters, unavailable=()):
        started = perf_counter()
        refreshed, events = 0, 0
        for trip, slots in self.belief._trips.items():
            route = self.belief.graph.connections[slots[0].cid].route
            pars = parameters.get(route, {'mean': 1., 'std': 2., 'cancel_rate': .01})
            key = self.belief._revision.get(trip, 0), pars['mean'], pars['std'], pars['cancel_rate']
            if self.keys.get(trip) == key:
                continue
            refreshed += 1
            law = self.belief._law(trip, parameters)
            maximum, error_mass, cells = 0., 0., 0
            for index, slot in enumerate(slots):
                clocks, weights, cancel = self.belief._event_distribution(law, index)
                if len(clocks):
                    bins = np.rint(clocks / self.belief.minute).astype(int)
                    offset = int(bins.min())
                    probs = np.bincount(bins - offset, weights=weights)
                    probs /= probs.sum()
                    probs = self.shapes.setdefault(probs.tobytes(), probs)
                    distribution = PMF(probs, offset)
                    errors = abs(clocks - bins * self.belief.minute)
                    maximum = max(maximum, float(errors[weights > 0].max()))
                    error_mass += float(weights @ errors)
                else:
                    distribution = PMF.deterministic(round(slot.scheduled / self.belief.minute))
                target = self.graph.connections[slot.cid]
                if slot.kind == 'departure':
                    target.dep_distribution = distribution
                    self.cancellation[slot.cid] = cancel
                else:
                    target.arr_distribution = distribution
                cells += len(law.times[index])
                events += 1
            self.keys[trip] = key
            self.trip_stats[trip] = maximum, error_mass, cells, len(slots)
        marginal_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        unavailable = set(unavailable)
        for target, physical in zip(self.graph.connections, self.belief.graph.connections):
            target.cancel_prob = 1. if target.id in unavailable else self.cancellation[target.id]
            target.pickup_allowed = physical.pickup_allowed
        records = list(self.trip_stats.values())
        n_events = sum(item[3] for item in records)
        return self.graph, dict(marginal_ms=marginal_ms, graph_ms=(perf_counter() - started) * 1000,
            refreshed_trips=refreshed, reused_trips=len(records) - refreshed,
            refreshed_posterior_events=events, posterior_events=n_events,
            marginal_cells=sum(item[2] for item in records), connections=len(self.graph.connections),
            unique_exact_operating_shapes=len(self.shapes),
            max_clock_quantization_seconds=max((item[0] for item in records), default=0.),
            mean_event_operating_quantization_seconds=sum(item[1] for item in records) / max(n_events, 1),
            max_transfer_roundup_seconds=max((ceil(s.min_transfer_time / self.belief.minute) * self.belief.minute - s.min_transfer_time
                for s in self.belief.graph.stops.values()), default=0.))


class OptimizedNativeEventRouter(NativeEventRouter):
    """Same native root choices; changed computation only."""

    def __init__(self, *args, native_kernel, **kwargs):
        started = perf_counter()
        super().__init__(*args, **kwargs)
        self.minute_projection = PosteriorMinuteProjection(self.trip_belief)
        self.native_workspace = NativeWorkspace(self.minute_projection.graph, native_kernel)
        self.native_setup_ms = (perf_counter() - started) * 1000
        self.last_native_result = None

    def export_issued_priorities(self):
        """Copy the issued strict conditional-mean decision graph priorities.

        This uses the recursion's reversed label order, not a new posterior
        query. The capped root score is a separately declared adapter choice.
        """
        if self.last_native_result is None:
            return {}
        return {stop: tuple(label.connection_id for label in reversed(labels))
                for stop, labels in self.last_native_result.stop_labels.items()}

    def select_connection(self, current_stop, current_time, rng, top_k=5):
        started = perf_counter()
        state, pars = self.information_state, self._parameters()
        temporary, work = self.minute_projection.refresh(pars, state.unavailable_ids)
        before = perf_counter()
        native = optimized_native_topocsa(temporary, state.stop, self.destination,
            floor(state.time / self.minute), ceil(self.journey_deadline / self.minute), self.native_workspace)
        self.last_native_result = native
        work.update(self.native_workspace.stats, query_ms=(perf_counter() - before) * 1000,
            native_scans=native.n_connections_processed,
            native_labels=sum(map(len, native.stop_labels.values())), native_cycle_cuts=len(native.cuts),
            query_source_rounddown_seconds=state.time % self.minute, native_setup_ms=self.native_setup_ms)
        choices = []
        cap = self.journey_deadline - state.time
        for label in native.stop_labels.get(state.stop, ()):
            c = self.graph.connections[label.connection_id]
            same = state.current_trip is not None and c.trip_id == state.current_trip
            if (c.id in state.unavailable_ids or (not c.pickup_allowed and not same)
                    or (c.dep_distribution is not None and c.dep_distribution.support_max < state.time)):
                continue
            trip, index = self.trip_belief._positions[c.id, 'departure']
            law = self.trip_belief._law(trip, pars)
            dep, weights, cancel = self.trip_belief._event_distribution(law, index)
            if not len(dep):
                continue
            mean = float(dep @ weights)
            attempt = max(state.time, c.dep_time - self.lookahead)
            transfer = 0 if state.current_trip is None or same else self.graph.stops[state.stop].min_transfer_time
            ready = max(attempt, state.last_arrival + transfer)
            if mean > ready + 12 * self.minute:
                attempt = max(attempt, mean - 12 * self.minute)
            attempt = state.time + ceil((attempt - state.time) / self.minute) * self.minute
            ready = max(attempt, state.last_arrival + transfer)
            if ready >= self.journey_deadline:
                continue
            mass = (1 - cancel) * weights[(dep >= ready) & (dep <= min(self.journey_deadline, ready + 12 * self.minute))].sum()
            if mass <= 0:
                continue
            if self.native_objective == 'conditional_mean':
                score = label.mean_dest_arrival * self.minute
            else:
                clocks = (label.dest_arrival.offset + np.arange(len(label.dest_arrival.probs))) * self.minute
                gain = float(label.dest_arrival.probs @ np.maximum(0, self.journey_deadline - np.maximum(clocks, state.time)))
                score = cap - mass * label.feasibility * gain
            choices.append((score, attempt, c.id, float(mass), label.feasibility))
        selected = min(choices, default=None)
        cid = selected[2] if selected is not None and selected[1] == state.time else None
        elapsed = (perf_counter() - started) * 1000
        self.rollout_ms += elapsed
        self.query_number += 1
        self.predictions.append(dict(connection=cid, stop=current_stop, time=current_time,
            planning_ms=elapsed, work=0, value=selected[0] if selected else None,
            completed_rollouts=0, root_actions=len(choices), scored_root_actions=len(choices),
            complete_common_rounds=0, tree_nodes=0, menu_size=len(self.menu),
            native_objective=self.native_objective, native_work=work, native_selected=selected,
            native_model='independent_event_marginals_strict_mean_labels'))
        if cid is None:
            return None
        c = self.graph.connections[cid]
        return StopLabel(cid, c.arr_distribution, c.arr_time, selected[4]), selected[0]
