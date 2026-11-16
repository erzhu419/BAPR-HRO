"""Fixes for the three solver/controller defects identified in the G2 audit:

1. AmbiguityMarginGate: skips multi-world rollout when analytical |V_alight - V_continue|
   already exceeds a clear dominance margin (eliminating 70%+ wasted gate evaluations).
2. choose_certified: replaces V44's uncertified 1e-9 MC point-estimate override with a
   paired confidence-bound rule (|mean_gain| - z_alpha * se > min_effect_seconds),
   eliminating finite-sample coin-flip regressions (e.g. CI1 R04 t=-0.14, R32 t=+0.47).
3. active_platform_rides & RecursiveSelectivePlanner: prunes zero-survival prefix rides
   before checking local width<=6 admission, and replaces depth-1 PhysicalPlanner tail_base
   with recursive selective waiting across multi-transfer menus.
"""

from dataclasses import replace
import numpy as np

from experiments.reconstruction_v33.nodes import Platform
from experiments.reconstruction_v41.matched_continuation import (
    FrozenCombinedReference,
    MatchedContinuationPlanner,
)


def should_invoke_rollout(
    base_receipt,
    *,
    deadline,
    ambiguity_margin_seconds=120.0,
):
    """Return (invoke: bool, info: dict) based on analytical onboard stopping margin."""
    alight = float(base_receipt["alight"])
    continuation = float(base_receipt["continuation"])
    margin = abs(alight - continuation)
    both_capped = alight >= deadline - 1e-9 and continuation >= deadline - 1e-9
    if both_capped:
        return False, {"reason": "both_actions_at_deadline_cap", "margin_seconds": margin}
    if margin > ambiguity_margin_seconds:
        return False, {
            "reason": "analytical_margin_exceeds_threshold",
            "margin_seconds": margin,
            "threshold_seconds": float(ambiguity_margin_seconds),
        }
    return True, {
        "reason": "ambiguous_onboard_competition",
        "margin_seconds": margin,
        "threshold_seconds": float(ambiguity_margin_seconds),
    }


def choose_certified(
    gate_result,
    base_action,
    *,
    z_alpha=1.96,
    min_effect_seconds=5.0,
    hub_continue_margin_seconds=0.0,
):
    """Certified action selection combining missing-outcome bounds and paired MC SE.

    Orientation convention (matching V44/V51):
      gain = L(alight) - L(continue)
      - Negative gain favors 'alight'
      - Positive gain favors 'continue'
    """
    paired = gate_result["paired"]
    low_miss, high_miss = paired["full_bank_mean_gain_interval"]

    if "complete_pair_statistics" in paired and paired["complete_pair_statistics"] is not None:
        stats = paired["complete_pair_statistics"]
        mean_gain = stats.get("mean_gain")
        se = stats.get("se")
    else:
        mean_gain = paired.get("mean_gain")
        se = paired.get("se")

    if mean_gain is None or se is None or paired.get("unresolved_pairs", 0) > 0:
        return base_action, {
            "certified_override": False,
            "reason": "unresolved_worlds_retain_base",
            "low_bound": float(low_miss),
            "high_bound": float(high_miss),
        }

    mean_gain = float(mean_gain)
    se = float(se)
    low_ci = mean_gain - float(z_alpha) * se
    high_ci = mean_gain + float(z_alpha) * se
    low_cert = min(float(low_miss), low_ci)
    high_cert = max(float(high_miss), high_ci)
    t_stat = mean_gain / se if se > 0.0 else (np.inf * np.sign(mean_gain) if mean_gain != 0.0 else 0.0)

    if high_cert < -float(min_effect_seconds):
        chosen = "alight"
    elif low_cert > float(min_effect_seconds) + float(hub_continue_margin_seconds):
        chosen = "continue"
    else:
        chosen = base_action

    return chosen, {
        "certified_override": chosen != base_action,
        "selected_action": chosen,
        "base_action": base_action,
        "mean_gain": mean_gain,
        "se": se,
        "t_stat": float(t_stat),
        "certified_interval": [float(low_cert), float(high_cert)],
    }


def active_platform_rides(model, platform, history, entry, deadline):
    """Prune static menu rides that have zero operating departure probability in [entry, deadline)."""
    retained = {}
    for cid, ride in platform.rides.items():
        trip, slot = model._positions[cid, "departure"]
        law = history.law(trip)
        operating = ~law.canceled[slot]
        times = law.times[slot][operating]
        mass = law.marginal(slot)[operating]
        active_mass = float(mass[(times >= entry) & (times < deadline)].sum())
        if active_mass > 1e-12:
            retained[cid] = ride
    return Platform(platform.stop, retained)


class RecursiveSelectivePlanner(MatchedContinuationPlanner):
    """Multi-transfer selective continuation planner on finite acyclic menus.

    Unlike MatchedContinuationPlanner (which uses PhysicalPlanner/first_offer in
    self.tail_base, truncating selective waiting after one transfer), this planner
    evaluates downstream transfers recursively with selective waiting and optionally
    applies exact FrozenCombinedReference on scalar arrival clocks when the active
    remaining menu width is at most exact_combined_max_width.
    """

    value_scope = (
        "Recursive multi-transfer selective continuation (min(first-offer, single-target), "
        "with optional active-menu FrozenCombinedReference on small active platforms)"
    )

    def __init__(self, graph, model, parameters, deadline, *, exact_combined_max_width=0):
        super().__init__(graph, model, parameters, deadline)
        self.exact_combined_max_width = int(exact_combined_max_width)
        self.stats.update(exact_combined_evaluations=0, active_pruned_rides=0)

    def selective_waiting_components(self, platform, history, ready, method="rolling_fixed"):
        self.stats["selective_tail_queries"] += 1
        ready = np.asarray(ready, dtype=float)
        shape, flat = ready.shape, ready.reshape(-1)
        key = (*self._node_key(platform, history, method), tuple(flat), shape, self.exact_combined_max_width)
        if key in self._selective_tails:
            self.stats["selective_tail_hits"] += 1
            return self._selective_tails[key]

        compiler = self.platform_compiler(platform, history, method)
        first = self.first_offer_values(platform, history, ready, method)
        by_target = {}
        for cid in platform.rides:
            clocks, mass, _ = compiler.departure_distribution(cid)
            positive = mass > 0
            clocks, mass = clocks[positive], mass[positive]
            scores = compiler.clock_nominal_scores(cid, clocks)
            legal = (clocks[None, :] >= flat[:, None]) & (clocks[None, :] < self.deadline)
            by_target[cid] = (self.deadline - legal @ (mass * (self.deadline - scores))).reshape(shape)
        single = (
            np.minimum.reduce(list(by_target.values()))
            if by_target
            else np.full(shape, self.deadline)
        )
        envelope = np.minimum(first, single)

        if self.exact_combined_max_width > 0 and flat.size > 0:
            transfer = self.graph.stops[platform.stop].min_transfer_time
            refined = envelope.reshape(-1).copy()
            for idx, r_clock in enumerate(flat):
                entry = float(r_clock - transfer)
                active = active_platform_rides(self.model, platform, history, entry, self.deadline)
                self.stats["active_pruned_rides"] += len(platform.rides) - len(active.rides)
                if 1 < len(active.rides) <= self.exact_combined_max_width:
                    # Distinct physical trips required for independent finite-offer Combined DP
                    trips = [self.model._positions[c, "departure"][0] for c in active.rides]
                    if len(trips) == len(set(trips)):
                        ref = self.platform_reference(active, history, entry, float(r_clock), method)
                        refined[idx] = FrozenCombinedReference(ref).value
                        self.stats["exact_combined_evaluations"] += 1
            envelope = refined.reshape(shape)

        result = dict(
            value=envelope,
            first_offer=first,
            single_target=single,
            by_target=by_target,
        )
        self._selective_tails[key] = result
        self.stats["selective_tail_compiles"] += 1
        self.stats["selective_tail_points"] += flat.size
        return result

    def compile_ride(self, ride, history, method="optimal"):
        compiled = super().compile_ride(ride, history, method)
        if compiled.conditional_scope != self.value_scope:
            compiled = replace(compiled, conditional_scope=self.value_scope)
            self._rides[self._node_key(ride, history, method)] = compiled
        return compiled
