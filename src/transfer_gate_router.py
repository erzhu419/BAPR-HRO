"""Execute a certified immediate-stay suffix; otherwise use the original UCT64."""
from dataclasses import asdict
from time import perf_counter

from .causal_planning_router import CausalPlanningRouter
from .evolving_trip_belief import EvolvingTripBelief
from .risk_bounds import CommitmentBounds
from .transit_graph import StopLabel


class GateUCTRouter(CausalPlanningRouter):
    """One-second current-grid regret gate with a fixed UCT64 fallback.

    Each actual selector call consumes one query ordinal, even a precompiled
    leg. A fallback therefore uses the same seed ordinal as plain UCT when the
    physical decision histories have equal lengths. The macro proof applies
    at its issue clock, without planning latency; charged execution is measured
    separately, and the newly calculated first stay is never preissued.
    """

    def __init__(self, *args, algorithm='uct', work_budget=64, **kwargs):
        if algorithm != 'uct' or work_budget != 64:
            raise ValueError('GateUCTRouter fixes the UCT64 fallback comparison')
        super().__init__(*args, algorithm=algorithm, work_budget=work_budget, **kwargs)
        if not isinstance(self.trip_belief, EvolvingTripBelief) or self.trip_belief.public_occurrences:
            raise TypeError('GateUCTRouter requires the ordered-trip no-occurrence-feed model')
        self._gate_trip_cids = {}
        for c in sorted(self.graph.connections, key=lambda c: (c.dep_time, c.arr_time, c.id)):
            self._gate_trip_cids.setdefault(c.trip_id or f'connection:{c.id}', []).append(c.id)
        self._gate_path = ()
        self._gate_pending_first = False
        self._gate_certificate = None
        self._gate_issue_clock = None

    def _clear_gate(self):
        self._gate_path = ()
        self._gate_pending_first = False
        self._gate_certificate = None
        self._gate_issue_clock = None

    def _legal_next(self, state):
        if not self._gate_path or state.time >= self.journey_deadline:
            return False
        c = self.graph.connections[self._gate_path[0]]
        return (c.dep_stop == state.stop and c.trip_id == state.current_trip
                and c.id not in state.unavailable_ids and c.dep_time <= state.time + self.lookahead
                and (c.dep_distribution is None or c.dep_distribution.support_max >= state.time))

    def precommitted_connection(self, stop, time, current_trip):
        state = self.information_state
        if (not self._gate_pending_first and stop == state.stop and time == state.time
                and current_trip == state.current_trip and self._legal_next(state)):
            return self._gate_path[0]
        return None

    def _suffix(self, state):
        if state.current_trip is None or state.time != state.last_arrival:
            return ()
        cids = self._gate_trip_cids.get(state.current_trip, ())
        reached = [i for i, cid in enumerate(cids)
                   if self.trip_belief._known_times.get((cid, 'arrival')) == state.last_arrival]
        if not reached:
            return ()
        stop, path = state.stop, []
        for cid in cids[max(reached) + 1:]:
            c = self.graph.connections[cid]
            if c.dep_stop != stop or cid in state.unavailable_ids:
                return ()
            path.append(cid)
            stop = c.arr_stop
            if stop == self.destination:
                return tuple(path)
        return ()

    def observe_attempt_outcome(self, cid, kind, started, time):
        super().observe_attempt_outcome(cid, kind, started, time)
        if self._gate_path and self._gate_path[0] == cid:
            if kind == 'board':
                self._gate_path = self._gate_path[1:]
                self._gate_pending_first = False
            else:
                self._clear_gate()

    def _gate_choice(self, state, started, *, cached, check_ms, work, ordinal):
        elapsed = (perf_counter() - started) * 1000
        cid = self._gate_path[0]
        self.query_number += 1
        self.rollout_ms += elapsed
        self.predictions.append({
            'connection': cid, 'stop': state.stop, 'time': state.time, 'planning_ms': elapsed,
            'work': 0, 'value': None if cached else self._gate_certificate['upper'],
            'completed_rollouts': 0, 'root_actions': 0, 'scored_root_actions': 0,
            'complete_common_rounds': 0, 'tree_nodes': 0, 'menu_size': len(self.menu),
            'gate_used': True, 'gate_cached': cached, 'gate_checked': not cached,
            'gate_check_ms': check_ms, 'gate_work': work,
            'gate_issue_clock': self._gate_issue_clock,
            'gate_certificate_at_issue': self._gate_certificate,
            'gate_tolerance_ticks': self.minute / 60., 'decision_ordinal': ordinal})
        c = self.graph.connections[cid]
        return StopLabel(c.id, c.arr_distribution, c.arr_time, 0.), self.predictions[-1]['value']

    def select_connection(self, current_stop, current_time, rng, top_k=5):
        started = perf_counter()
        state, ordinal = self.information_state, self.query_number
        if self._gate_path:
            if self._legal_next(state):
                return self._gate_choice(state, started, cached=True, check_ms=0., work={}, ordinal=ordinal)
            self._clear_gate()
        path = self._suffix(state)
        checked, work, certificate = bool(path), {}, None
        if path:
            index = CommitmentBounds(self.trip_belief, self._parameters(), self.graph,
                                      self.destination, self.journey_deadline)
            certificate = index.transfer_gate(state, path, lookahead=self.lookahead)
            work = dict(index.stats)
            if certificate is not None and certificate.regret_bound <= self.minute / 60.:
                self._gate_path = path
                self._gate_pending_first = True
                self._gate_certificate = asdict(certificate)
                self._gate_issue_clock = state.time
                check_ms = (perf_counter() - started) * 1000
                return self._gate_choice(state, started, cached=False, check_ms=check_ms,
                                         work=work, ordinal=ordinal)
        check_ms = (perf_counter() - started) * 1000
        result = super().select_connection(current_stop, current_time, rng, top_k)
        prediction = self.predictions[-1]
        prediction.update(fallback_planning_ms=prediction['planning_ms'],
                          planning_ms=(perf_counter() - started) * 1000,
                          gate_used=False, gate_cached=False, gate_checked=checked,
                          gate_check_ms=check_ms, gate_work=work,
                          gate_rejected_certificate=asdict(certificate) if certificate is not None else None,
                          decision_ordinal=ordinal)
        self.rollout_ms += check_ms
        return result
