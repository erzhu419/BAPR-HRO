"""Revalue compiled alternatives at the measured pre-dispatch clock."""
from dataclasses import replace
import math
from time import perf_counter

from .committed_transit import WAIT
from .recourse_itinerary_router import CompiledPolicy, RecourseItineraryRouter


class LatencyRecourseRouter(RecourseItineraryRouter):
    """One analytic revaluation pass; dispatch timing error remains recorded.

    Candidate routes and the eight-path budget are unchanged. This corrects
    stale start-clock values, rather than granting free computation. The
    bound refers to the measured clock before this final pass; the executor
    separately records its actual later clock.
    """

    def _dispatch_policy(self, best, policies, bounds, state, started):
        delay = math.ceil((perf_counter() - started) * self.minute / 60)
        execution = replace(state, time=min(self.journey_deadline, state.time + delay))
        delay = execution.time - state.time
        evaluated = []
        for policy in policies:
            if policy.root_action == WAIT:
                after_wait = replace(execution, time=min(self.journey_deadline, execution.time + self.minute))
                value = bounds.itinerary(after_wait, policy.primary, first_action_now=False,
                                          lookahead=self.lookahead)
                upper = self.journey_deadline - state.time - value.success_gain
            else:
                value = bounds.with_rescue(execution, policy.primary, policy.rescue,
                                           lookahead=self.lookahead)
                upper = delay + value.upper
            evaluated.append(CompiledPolicy(policy.primary, policy.rescue, float(upper), policy.root_action))
        selected = min(evaluated, key=lambda p: p.upper)
        # This is the full physical menu at the estimated execution clock.
        # It can contain additional departures that were not legal at query
        # start, so a lower bound over the old candidate subset would be wrong.
        actions = [WAIT]
        for c in self.graph.get_connections_from(execution.stop):
            if (c.id in self.menu and c.id not in execution.unavailable_ids
                    and c.dep_time <= execution.time + self.lookahead
                    and (c.pickup_allowed or c.trip_id == execution.current_trip)
                    and (c.dep_distribution is None or c.dep_distribution.support_max >= execution.time)):
                actions.append(('select', c.id))
        lower = delay + min(bounds.action_lower(execution, actions).values())
        return selected, dict(dispatch_lower=lower, estimated_execution_clock=execution.time,
                              bounds_scope='estimated_execution_clock',
                              dispatch_policy_evaluations=len(evaluated))
