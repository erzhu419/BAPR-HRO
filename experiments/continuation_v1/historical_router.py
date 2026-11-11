"""Frozen router and tail policy for continuation_v1_public_history_rollout.

This experiment copy preserves the archived selector, candidate cap, and tail
when the production router changes. The shared physical model and TripBelief
remain the same implementations used for the archived experiment.
"""
from __future__ import annotations

import heapq
import itertools
from time import perf_counter
import numpy as np

from src.committed_transit import CommittedState, WAIT, build_committed_model
from src.matched_router import MatchedRouter
from src.trip_belief import TripBelief


def nominal_action(graph, state, destination, nominal, allowed_ids):
    """Earliest nominal arrival, using public conditional means, never a world."""
    if not allowed_ids:
        return WAIT
    counter = itertools.count()
    queue = [(float(state.time), next(counter), state.stop, state.current_trip, None)]
    best = {(state.stop, state.current_trip): float(state.time)}
    while queue:
        time, _, stop, trip, first = heapq.heappop(queue)
        if time > best[(stop, trip)]:
            continue
        if stop == destination:
            return ("select", first)
        for c in graph.get_connections_from(stop):
            if c.id in state.unavailable_ids or (first is None and c.id not in allowed_ids):
                continue
            event = nominal[c.id]
            if event.departure is None:
                continue
            if first is None:
                ready = max(time, state.last_arrival + (
                    0 if trip is None or trip == c.trip_id else graph.stops[stop].min_transfer_time))
            else:
                ready = time + (0 if trip == c.trip_id else graph.stops[stop].min_transfer_time)
            if event.departure < ready:
                continue
            next_trip = c.trip_id or f"connection:{c.id}"
            key = c.arr_stop, next_trip
            if event.arrival < best.get(key, float("inf")):
                best[key] = event.arrival
                heapq.heappush(queue, (event.arrival, next(counter), c.arr_stop,
                                       next_trip, c.id if first is None else first))
    # An estimated missed vehicle can be removed by a zero-time causal attempt.
    # This also prevents stale nominal top-k options from blocking later ones.
    return ("select", allowed_ids[0])


class HistoricalContinuationRouter(MatchedRouter):
    def __init__(self, graph, priors, *, conditioning=True, rescue=True,
                 samples=8, seed=0, legacy=False, candidate_limit=5, max_time=120):
        super().__init__(graph, priors, online_mean=True, online_cancel=True,
                         candidate_limit=candidate_limit, max_time=max_time)
        self.conditioning, self.rescue, self.legacy = conditioning, rescue, legacy
        self.samples, self.seed = samples, seed
        self.trip_belief = TripBelief(graph)
        self.predictions = []
        self.polls = []
        self.rollout_ms = 0.0
        self.rollout_steps = 0
        self.query_number = 0
        self._scores = {}

    def observe_public_messages(self, stop, time, messages, seen_before):
        self.trip_belief.observe_poll(stop, time, messages, seen_before)
        self.polls.append({"stop": stop, "receipt_time": time,
                           "event_ids": [message.event_id for message in messages]})

    def observe_vehicle_event(self, cid, canceled, delay):
        self.trip_belief.observe_vehicle_event(cid, canceled, delay)

    def set_information_context(self, stop, time, trip, last_arrival, unavailable, seen, learned):
        self.information_state = CommittedState(
            stop, time, trip, last_arrival, frozenset(unavailable),
            frozenset(seen), frozenset(learned), initialized=True)

    def observe_attempt_outcome(self, cid, kind, started, time):
        if kind in ("missed", "wait_late"):
            self.trip_belief.observe_attempt(cid, "missed" if kind == "missed" else "timeout", time)
        if self.predictions and self.predictions[-1]["connection"] == cid:
            self.predictions[-1].update(actual_kind=kind, action_end=time)

    def _parameters(self):
        return {route: {"mean": self._get_belief(route).posterior_mean if self.online_mean else prior["mean"],
                        "std": prior["std"],
                        "cancel_rate": self._get_belief(route).cancel_rate if self.online_cancel
                        else self._initial_cancels[route]}
                for route, prior in self.priors.items()}

    def _snapshot_candidates(self, state):
        ids = []
        for label in reversed(self.cached_result.stop_labels.get(state.stop, [])):
            c = self.graph.connections[label.connection_id]
            if (c.id not in self._fresh_menu or c.id in state.unavailable_ids
                    or c.dep_time > state.time + self.lookahead
                    or c.dep_distribution.support_max < state.time):
                continue
            ids.append(c.id)
        return ids if self.candidate_limit is None else ids[:self.candidate_limit]

    def select_connection(self, current_stop, current_time, rng, top_k=5):
        if self._valued_version != self._version:
            self._value()
        self._scores = {}
        if not self.legacy:
            started = perf_counter()
            self._parameters_now = self._parameters()
            self._initial_belief = self.trip_belief.copy() if self.conditioning else TripBelief(self.graph)
            private_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.query_number]))
            scenes = self._initial_belief.sample(self._parameters_now, private_rng, self.samples)
            # Copies share immutable unchanged nominal events; each sampled
            # history invalidates only the trips affected by its observations.
            self._initial_belief.mean_events(self._parameters_now)
            self._model = build_committed_model(
                self.graph, [(1 / self.samples, scene) for scene in scenes],
                self.origin, self.destination, self.departure, self.journey_deadline,
                candidate_ids=self._fresh_menu, initial_state=self.information_state)
            self.rollout_ms += (perf_counter() - started) * 1000
        selected = super().select_connection(current_stop, current_time, rng, top_k)
        if selected is not None:
            cid = selected[0].connection_id
            prediction = self._scores.get(cid, {"score": selected[1]})
            self.predictions.append({"stop": current_stop, "time": current_time,
                                     "connection": cid, "candidates": self._scores.copy(),
                                     "private_model_query": self.query_number,
                                     **prediction})
        self.query_number += 1
        return selected

    def _advance_belief(self, belief, before, after, terminal):
        new_feedback = after.feedback[len(before.feedback):]
        for kind, cid, time in new_feedback:
            if kind == "board":
                belief.observe_vehicle_event(cid, False, time - self.graph.connections[cid].dep_time)
            elif kind == "cancel":
                belief.observe_vehicle_event(cid, True, 0)
            else:
                belief.observe_attempt(cid, "missed" if kind == "missed" else "timeout", time)
        if not terminal:
            belief.observe_poll(after.stop, after.time,
                                after.messages[len(before.messages):], set(before.seen_events))
        return new_feedback

    def rollout_one(self, world, root_connection):
        """Return total journey loss and first/any failure flags on one model world."""
        state = self.information_state
        belief = self._initial_belief.copy()
        action = ("select", root_connection)
        any_failure, first_failure = False, False
        for step in range(self._model.horizon):
            outcome = self._model.transition(world, state, action)
            feedback = self._advance_belief(belief, state, outcome.state, outcome.terminal)
            bad = (not feedback or feedback[-1][0] != "board") if action != WAIT else False
            any_failure |= bad
            if step == 0:
                first_failure = bad
            state = outcome.state
            self.rollout_steps += 1
            actions = self._model.actions(state) if not outcome.terminal else ()
            if not actions:
                loss = state.time - self.departure + self._model.terminal_cost(world, state)
                break
            allowed = self._snapshot_candidates(state)
            if not allowed:
                action = WAIT
            elif len(allowed) == 1:
                action = ("select", allowed[0])
            else:
                nominal = belief.mean_events(self._parameters_now)
                action = nominal_action(self.graph, state, self.destination, nominal, allowed)
        else:
            loss = self.max_time
        return min(self.max_time, loss), first_failure, any_failure

    def _score(self, label, current_time):
        if self.legacy:
            return super()._score(label, current_time)
        started = perf_counter()
        losses, first_failed, failed = zip(*(
            self.rollout_one(world, label.connection_id) for world in self._model.scenarios))
        losses = np.asarray(losses, dtype=float)
        failure = np.asarray(failed, dtype=bool)
        scored = np.where(failure, self.max_time, losses) if not self.rescue else losses
        value = float(scored.mean())
        self._scores[label.connection_id] = {
            "score": value, "first_attempt_failure_probability": float(np.mean(first_failed)),
            "any_attempt_failure_probability": float(failure.mean()),
            "conditional_no_attempt_failure_loss": float(losses[~failure].mean()) if (~failure).any() else None,
            "conditional_rescue_loss": float(losses[failure].mean()) if failure.any() else None,
            "sample_causal_tail_loss": float(losses.mean()), "samples": self.samples}
        self.rollout_ms += (perf_counter() - started) * 1000
        return value
