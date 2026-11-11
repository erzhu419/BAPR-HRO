"""Public-history-conditioned rollout over boardings and one-minute waiting."""
from __future__ import annotations

import heapq
import itertools
from time import perf_counter
import numpy as np

from .committed_transit import CommittedState, WAIT, build_committed_model
from .matched_router import MatchedRouter
from .simulator import JourneyScenario, VehicleEvent
from .transit_graph import StopLabel
from .trip_belief import TripBelief


def nominal_action(graph, state, destination, nominal, menu, deadline, lookahead, root_boardings=None):
    """First executable action of a nominal path within the frozen menu.

    Waiting can bring a departure into lookahead or commitment range. With
    waiting, an earlier arrival for the same stop and trip dominates a later
    one. Future calls can revise this nominal plan after public observations.
    """
    counter = itertools.count()
    queue = [(state.time, next(counter), state.stop, state.current_trip, None)]
    best = {(state.stop, state.current_trip): state.time}
    immediate = []
    while queue:
        time, _, stop, trip, first = heapq.heappop(queue)
        if time > best[(stop, trip)]:
            continue
        if stop == destination:
            return WAIT if first == WAIT or first is None else ("select", first)
        for c in graph.get_connections_from(stop):
            same_trip = trip is not None and trip == c.trip_id
            if c.id in state.unavailable_ids or (c.id not in menu and not same_trip):
                continue
            event = nominal[c.id]
            if event.departure is None:
                continue
            departure = round(event.departure)
            arrival = departure + c.scheduled_travel_time
            transfer = 0 if trip is None or same_trip else graph.stops[stop].min_transfer_time
            ready = (state.last_arrival if first is None else time) + transfer
            attempt = max(time, c.dep_time - lookahead)
            if departure > max(attempt, ready) + 12:
                attempt = departure - 12
            if (departure < max(attempt, ready) or attempt >= deadline or arrival > deadline
                    or (c.dep_distribution is not None and c.dep_distribution.support_max < attempt)):
                continue
            if first is None and attempt == state.time:
                if root_boardings is not None and c.id not in root_boardings:
                    continue
                immediate.append((arrival, c.id))
            next_trip = c.trip_id or f"connection:{c.id}"
            key = c.arr_stop, next_trip
            if arrival < best.get(key, float("inf")):
                best[key] = arrival
                first_action = first if first is not None else (c.id if attempt == state.time else WAIT)
                heapq.heappush(queue, (arrival, next(counter), c.arr_stop, next_trip, first_action))
    return ("select", min(immediate)[1]) if immediate else WAIT


class ContinuationRouter(MatchedRouter):
    def __init__(self, graph, priors, *, conditioning=True, rescue=True,
                 samples=8, seed=0, legacy=False, candidate_limit=5, max_time=120,
                 candidate_mode="refresh"):
        super().__init__(graph, priors, online_mean=True, online_cancel=True,
                         candidate_limit=candidate_limit, max_time=max_time,
                         candidate_mode=candidate_mode)
        if samples < 1:
            raise ValueError("Continuation requires at least one base sample")
        self.conditioning, self.rescue, self.legacy = conditioning, rescue, legacy
        self.samples, self.seed = samples, seed
        self.trip_belief = TripBelief(graph)
        self.predictions = []
        self.polls = []
        self.rollout_ms = 0.0
        self.rollout_steps = 0
        self.query_number = 0
        self._scores = {}
        self._routes = {c.route for c in graph.connections}
        self._trip_connections = {}
        for c in graph.connections:
            if c.trip_id:
                self._trip_connections.setdefault(c.trip_id, set()).add(c.id)

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
        return {route: self._route_parameters(route) for route in self._routes}

    def _same_trip(self, state):
        if state.current_trip is None:
            return []
        return [c.id for c in self.graph.get_connections_from(state.stop)
                if c.trip_id == state.current_trip and c.id not in state.unavailable_ids]

    def _boarding_support(self, cid, state, belief, parameters):
        c = self.graph.connections[cid]
        transfer = (self.graph.stops[state.stop].min_transfer_time
                    if state.current_trip is not None and c.trip_id != state.current_trip else 0)
        ready = max(state.time, state.last_arrival + transfer)
        delays, weights = belief.posterior(cid, parameters)
        departure = c.dep_time + delays
        caught = ((delays != 999) & (ready < self.journey_deadline) & (ready <= departure)
                  & (departure <= min(self.journey_deadline, ready + 12)))
        return delays, weights, caught

    def _board_probability(self, cid, state, parameters):
        _, weights, caught = self._boarding_support(cid, state, self.trip_belief, parameters)
        return float(weights[caught].sum())

    def has_unlabeled_continuation(self, stop):
        if self.legacy:
            return False
        state = self.information_state
        if state.stop != stop:
            return False
        return any(self.graph.connections[cid].dep_time <= state.time + self.lookahead
                   and self._board_probability(cid, state, {}) > 0 for cid in self._same_trip(state))

    def _prepare_rollout(self, scenes):
        self._menu = self._fresh_menu if self.candidate_mode == "refresh" else self.initial_menu
        trips = {self.graph.connections[cid].trip_id for cid in self._menu}
        trips.add(self.information_state.current_trip)
        self._physical_menu = self._menu.union(*(self._trip_connections.get(trip, ()) for trip in trips))
        self._scenes = scenes
        self._model = self._rollout_model([(1 / len(scenes), scene) for scene in scenes])

    def _rollout_model(self, scenes):
        return build_committed_model(
            self.graph, scenes, self.origin, self.destination, self.departure, self.journey_deadline,
            candidate_ids=self._physical_menu, lookahead=self.lookahead, initial_state=self.information_state)

    def _supplement_branch(self, cid, outcomes, weights, caught):
        """Add a conditional scenario for a missing branch without thinning the bank."""
        caught_delays = set(outcomes[caught])
        draws = [scene.events[cid].delay in caught_delays for scene in self._scenes]
        raw_failure = 1.0 - sum(draws) / len(draws)
        if 0 < raw_failure < 1:
            return raw_failure, None
        connection = self.graph.connections[cid]
        if raw_failure == 1 and connection.arr_stop == self.destination:
            return raw_failure, None  # The direct successful loss is integrated analytically.
        missing = caught if raw_failure == 1 else ~caught
        if not weights[missing].any():
            return raw_failure, None
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.query_number - 1, cid]))
        delay = int(rng.choice(outcomes[missing], p=weights[missing] / weights[missing].sum()))
        events = dict(self._scenes[-1].events)
        for member in self._trip_connections.get(connection.trip_id, (cid,)):
            c = self.graph.connections[member]
            events[member] = VehicleEvent(delay, delay == 999,
                                          c.dep_time if delay == 999 else c.dep_time + delay,
                                          c.arr_time if delay == 999 else c.arr_time + delay)
        return raw_failure, self._rollout_model([(1.0, JourneyScenario(events))])

    def _boarding_candidates(self, state, menu, labels, parameters):
        ordered = dict.fromkeys(self._same_trip(state) + [cid for cid in labels if cid in menu])
        # Gaussian tails can have positive mass too small to change even the
        # representable failure probability. Keep such attempts out of top-k.
        board = [cid for cid in ordered if cid not in state.unavailable_ids
                 and self.graph.connections[cid].dep_time <= state.time + self.lookahead
                 and self.graph.connections[cid].dep_distribution.support_max >= state.time
                 and 1.0 - self._board_probability(cid, state, parameters) < 1.0]
        return board[:self.candidate_limit]

    def select_connection(self, current_stop, current_time, rng, top_k=5):
        if self.legacy:
            selected = super().select_connection(current_stop, current_time, rng, top_k)
            if selected is not None:
                self.predictions.append({"stop": current_stop, "time": current_time,
                                         "connection": selected[0].connection_id, "candidates": {},
                                         "private_model_query": self.query_number, "score": selected[1]})
            self.query_number += 1
            return selected
        if self._valued_version != self._version:
            self._value()
        self._scores = {}
        self._parameters_now = self._parameters()
        state = self.information_state
        menu = self._fresh_menu if self.candidate_mode == "refresh" else self.initial_menu
        labels = {label.connection_id: label
                  for label in reversed(self.cached_result.stop_labels.get(state.stop, []))}
        board = self._boarding_candidates(state, menu, labels, self._parameters_now)
        query = self.query_number
        self.query_number += 1
        if not board:
            return None

        started = perf_counter()
        self._initial_belief = self.trip_belief.copy() if self.conditioning else TripBelief(self.graph)
        private_rng = np.random.default_rng(np.random.SeedSequence([self.seed, query]))
        scenes = self._initial_belief.sample(self._parameters_now, private_rng, self.samples)
        self._initial_belief.mean_events(self._parameters_now)
        self._prepare_rollout(scenes)
        self.rollout_ms += (perf_counter() - started) * 1000
        for cid in board:
            self._scores[cid] = self._score_action(("select", cid))
        self._scores["wait"] = self._score_action(WAIT)
        best = min(self._scores, key=lambda cid: (
            self._scores[cid]["score"], self._scores[cid]["first_attempt_failure_probability"] or 0.0,
            cid == "wait"))
        prediction = {"stop": current_stop, "time": current_time,
                      "connection": None if best == "wait" else best, "candidates": self._scores.copy(),
                      "private_model_query": query, **self._scores[best]}
        self.predictions.append(prediction)
        self.decisions.append({"stop": current_stop, "time": current_time,
                               "selected": prediction["connection"], "candidates": list(self._scores),
                               "score": prediction["score"]})
        if best == "wait":
            return None
        label = labels.get(best)
        if label is None:
            c = self.graph.connections[best]
            label = StopLabel(best, c.arr_distribution, c.arr_time, 0.0)
        return label, prediction["score"]

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

    def _score_action(self, root_action):
        started = perf_counter()
        losses, first_failed = [], []
        boarding = root_action != WAIT
        raw_failure = None
        rollout_worlds = [(self._model, world) for world in self._model.scenarios]
        if boarding:
            cid = root_action[1]
            delays, weights, caught = self._boarding_support(
                cid, self.information_state, self._initial_belief, self._parameters_now)
            p_board, p_fail = float(weights[caught].sum()), float(weights[~caught].sum())
            raw_failure, supplemental = self._supplement_branch(cid, delays, weights, caught)
            if supplemental is not None:
                rollout_worlds.append((supplemental, supplemental.scenarios[0]))
        for model, world in rollout_worlds:
            state = self.information_state
            belief = self._initial_belief.copy()
            action = root_action
            first_failure = False
            for step in range(model.horizon):
                outcome = model.transition(world, state, action)
                feedback = self._advance_belief(belief, state, outcome.state, outcome.terminal)
                if step == 0 and boarding:
                    first_failure = not feedback or feedback[-1][0] != "board"
                state = outcome.state
                self.rollout_steps += 1
                actions = model.actions(state) if not outcome.terminal else ()
                if not actions:
                    loss = state.time - self.departure + model.terminal_cost(world, state)
                    break
                nominal = belief.mean_events(self._parameters_now)
                action = nominal_action(self.graph, state, self.destination, nominal, self._menu,
                                        self.journey_deadline, self.lookahead)
            else:
                loss = self.max_time
            losses.append(min(self.max_time, loss))
            first_failed.append(first_failure)
        losses = np.asarray(losses, dtype=float)
        failure = np.asarray(first_failed, dtype=bool)
        success_cost = failure_cost = p_failure = None
        if boarding:
            c = self.graph.connections[cid]
            if p_board > 0:
                if c.arr_stop == self.destination:
                    direct = np.minimum(self.max_time, c.arr_time + delays[caught] - self.departure)
                    success_cost = float(np.dot(direct, weights[caught]) / p_board)
                else:
                    success_cost = float(losses[~failure].mean())
            if p_fail > 0:
                failure_cost = float(losses[failure].mean())
            success_part = p_board * (success_cost or 0.0)
            value = success_part + p_fail * (failure_cost or 0.0)
            score = value if self.rescue else success_part + p_fail * self.max_time
            p_failure = p_fail
        else:
            score = value = float(losses.mean())
        self.rollout_ms += (perf_counter() - started) * 1000
        return {"score": score, "first_attempt_failure_probability": p_failure,
                "raw_sample_first_failure_probability": raw_failure,
                "conditional_first_success_loss": success_cost, "conditional_first_failure_loss": failure_cost,
                "estimated_causal_tail_loss": value, "samples": len(losses)}
