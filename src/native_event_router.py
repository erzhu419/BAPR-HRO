"""Strict native TopoCSA on a temporary minute projection of event posteriors.

The execution graph stays in seconds. Native continuation values use independent
event marginals, strict mean dominance and unconditional same-trip reachability.
``capped`` changes only root scoring, not the published native recursion.
"""
from dataclasses import replace
from math import ceil, floor
from time import perf_counter

import numpy as np

from .causal_planning_router import CausalPlanningRouter
from .durner.native import native_topocsa
from .evolving_trip_belief import EvolvingTripBelief
from .pmf import PMF
from .transit_graph import TransitGraph, StopLabel


def minute_snapshot(belief, parameters, unavailable=()):
    """Keep every connection; aggregate each operating PMF to nearest minute."""
    started = perf_counter()
    distributions, cancellations, shapes = {}, {}, {}
    events, cells, max_error, error_mass = 0, 0, 0., 0.
    for trip, slots in belief._trips.items():
        law = belief._law(trip, parameters)
        for index, slot in enumerate(slots):
            clocks, weights, cancel = belief._event_distribution(law, index)
            if len(clocks):
                bins = np.rint(clocks / belief.minute).astype(int)
                offset = int(bins.min())
                probs = np.bincount(bins - offset, weights=weights)
                probs /= probs.sum()
                # Exact shape sharing enables native query_reachability's
                # existing identity cache; no mass is rounded or discarded.
                probs = shapes.setdefault(probs.tobytes(), probs)
                distribution = PMF(probs, offset)
                errors = abs(clocks - bins * belief.minute)
                max_error = max(max_error, float(errors[weights > 0].max()))
                error_mass += float(weights @ errors)
            else:
                distribution = PMF.deterministic(round(slot.scheduled / belief.minute))
            distributions[slot.cid, slot.kind] = distribution
            cancellations[slot.cid, slot.kind] = cancel
            events += 1
            cells += len(law.times[index])
    marginal_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    result = TransitGraph()
    transfer_error = 0.
    for sid, stop in belief.graph.stops.items():
        transfer = ceil(stop.min_transfer_time / belief.minute)
        transfer_error = max(transfer_error, transfer * belief.minute - stop.min_transfer_time)
        result.add_stop(replace(stop, min_transfer_time=transfer))
    for c in belief.graph.connections:
        # Fractional scheduled minutes preserve the exact same-trip order;
        # only PMF clock bins, not timetable ordering, are quantized.
        result.add_connection(replace(c, dep_time=c.dep_time / belief.minute,
            arr_time=c.arr_time / belief.minute,
            dep_distribution=distributions[c.id, 'departure'],
            arr_distribution=distributions[c.id, 'arrival'],
            cancel_prob=1. if c.id in unavailable else cancellations[c.id, 'departure']))
    stats = dict(marginal_ms=marginal_ms, graph_ms=(perf_counter() - started) * 1000,
        posterior_events=events, marginal_cells=cells, connections=len(result.connections),
        unique_exact_operating_shapes=len(shapes),
        max_clock_quantization_seconds=max_error,
        mean_event_operating_quantization_seconds=error_mass / max(events, 1),
        max_transfer_roundup_seconds=transfer_error)
    return result, stats


class NativeEventRouter(CausalPlanningRouter):
    """Same public event history and physical execution; native marginal planner."""

    def __init__(self, *args, native_objective='capped', **kwargs):
        super().__init__(*args, **kwargs)
        if native_objective not in ('conditional_mean', 'capped'):
            raise ValueError('native_objective is conditional_mean or capped')
        if not isinstance(self.trip_belief, EvolvingTripBelief):
            raise TypeError('NativeEventRouter requires EvolvingTripBelief')
        self.native_objective = native_objective

    def select_connection(self, current_stop, current_time, rng, top_k=5):
        started = perf_counter()
        state, pars = self.information_state, self._parameters()
        temporary, work = minute_snapshot(self.trip_belief, pars, state.unavailable_ids)
        before = perf_counter()
        native = native_topocsa(temporary, state.stop, self.destination,
                               floor(state.time / self.minute), ceil(self.journey_deadline / self.minute))
        work.update(query_ms=(perf_counter() - before) * 1000,
                    native_scans=native.n_connections_processed,
                    native_labels=sum(map(len, native.stop_labels.values())),
                    native_cycle_cuts=len(native.cuts),
                    query_source_rounddown_seconds=state.time % self.minute)
        choices = []
        cap = self.journey_deadline - state.time
        labels = native.stop_labels.get(state.stop, ())
        for label in labels:
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
            # Wait in the original minute phase until a nominally reachable
            # commitment; future replanning receives only actual observations.
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
            native_objective=self.native_objective, native_work=work,
            native_selected=selected, native_model='independent_event_marginals_strict_mean_labels'))
        if cid is None:
            return None
        c = self.graph.connections[cid]
        return StopLabel(cid, c.arr_distribution, c.arr_time, selected[4]), selected[0]
