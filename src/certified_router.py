"""Finite-bank certified allocation of a committed policy-search budget."""
from time import perf_counter

import numpy as np

from .causal_planning_router import CausalPlanningRouter
from .certified_policy_search import CertifiedPolicySearch
from .committed_transit import WAIT, build_committed_model
from .continuation_router import nominal_action
from .sparse_certificate_cutoff import SparseCertificateIndex
from .transit_graph import StopLabel


class CertifiedPlanningRouter(CausalPlanningRouter):
    """Keep every physical root action, then screen by a valid value bound.

    The relaxation knows the sampled vehicle worlds. The nominal witness sees
    only public observations, and uses the same posterior as the rollout
    baseline. A certificate concerns this decision's finite bank and fixed
    witness, not a statistical guarantee for the physical travel day.
    """

    def select_connection(self, current_stop, current_time, rng, top_k=5):
        started = perf_counter()
        state = self.information_state
        parameters = self._parameters()
        private_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.query_number]))
        self.query_number += 1
        scenes = self.trip_belief.sample(parameters, private_rng, self.samples)
        bank = [(1 / self.samples, scene) for scene in scenes]
        model = build_committed_model(
            self.graph, bank, self.origin, self.destination, self.departure,
            self.journey_deadline, candidate_ids=self.menu, lookahead=self.lookahead,
            initial_state=state)
        relaxation = SparseCertificateIndex(self.graph, bank, self.destination,
                                             state.time, self.journey_deadline)
        bound_ms = 0.

        def bounds(upper):
            nonlocal bound_ms
            bound_started = perf_counter()
            lower = relaxation.action_lower(state, tuple(model.actions(state)), cutoff=upper)
            bound_ms = (perf_counter() - bound_started) * 1000
            return lower

        def tail(public_state, belief, actions):
            boardings = {action[1] for action in actions if action[0] == 'select'}
            if not boardings:
                return WAIT
            return nominal_action(self.graph, public_state, self.destination,
                                  belief.mean_events(parameters), self.menu,
                                  self.journey_deadline, self.lookahead,
                                  root_boardings=boardings, minute=self.minute)

        planner = CertifiedPolicySearch(model, self.trip_belief, tail, self._advance_belief,
                                         private_rng, self.work_budget, self.max_time)
        result = planner.plan_certified(bounds, self.journey_deadline - state.time)
        plan = result.planning
        elapsed = (perf_counter() - started) * 1000
        self.rollout_ms += elapsed
        self.rollout_steps += plan.transitions
        cid = None if plan.action == WAIT else plan.action[1]
        self.predictions.append({
            'connection': cid, 'stop': current_stop, 'time': current_time,
            'planning_ms': elapsed, 'work': plan.transitions, 'value': plan.value,
            'completed_rollouts': plan.rollouts, 'root_actions': plan.root_actions,
            'scored_root_actions': plan.scored_root_actions, 'complete_common_rounds': 0,
            'tree_nodes': 0, 'menu_size': len(self.menu),
            'finite_bank_lower': float(result.lower), 'finite_bank_upper': float(result.upper),
            'finite_bank_gap': float(result.gap), 'certified_removed': len(result.removed),
            'bound_ms': bound_ms, 'bound_work': relaxation.stats,
            'action_bounds': [
                {'action': list(action), 'lower': float(result.action_lower[action]),
                 'upper': float(result.action_upper[action]),
                 'completed_worlds': result.evaluated_worlds[action]}
                for action in result.action_lower]})
        if cid is None:
            return None
        c = self.graph.connections[cid]
        return StopLabel(cid, c.arr_distribution, c.arr_time, 0.0), plan.value
