"""Interruptible paired policy evaluation and a descriptive normal screen.

The CPU allowance includes posterior preparation. SIGPROF interrupts Python
work inside a world, rather than waiting for the entire recursive world. A C
kernel can delay signal delivery; actual cost and any overshoot are recorded.
This is neither a hard real-time guarantee nor an action-optimality certificate.
"""

import signal
from time import perf_counter, process_time

from experiments.bounded_gate_5d_v1.pair_contract import (
    PairDecisionEngine, PairMatchedPlanner, PairRecursivePlanner,
)
from experiments.trc_revision_v1.executor import PairSnapshotExecutor
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v41.continuation import empirical_pair_summary
from experiments.reconstruction_v41.physical_posterior import CheckingWorld
from experiments.reconstruction_v51.rollout import StableTripWorlds


class QueryCPUExpired(Exception):
    """The allowance expired; return an unresolved bank and retain the base proposal."""


def _expire(signum, frame):
    raise QueryCPUExpired


def evaluate_pair_rollout(
    graph, model, parameters, deadline, node, row, history, contract, *,
    root_index, count=64, max_process_cpu=1.8, seed=510015,
    rollout_planner_kind="matched", exact_combined_max_width=6,
):
    """Return all requested pairs, including explicitly unresolved pairs.

    A pair is committed only after both forced actions finish. Thus interruption
    never silently changes the comparison bank or leaves unequal sample counts.
    The complete-pair subset may depend on cost and is not an unbiased substitute
    for the full requested bank. The screen below requires a complete bank.
    """
    tick, cpu = perf_counter(), process_time()
    rows = {"alight": [], "continue": []}
    completed = 0
    phase = "not_started"
    preparation = None
    allowance = max(0.0, float(max_process_cpu))
    expired = allowance <= 0.0
    if not expired:
        previous_handler = signal.getsignal(signal.SIGPROF)
        signal.signal(signal.SIGPROF, _expire)
        signal.setitimer(signal.ITIMER_PROF, allowance)
        try:
            phase = "preparation"
            snapshot = freeze_snapshot(history, row["clock"])
            paths = [*node.paths(), *contract.paths()]
            trips = {model._positions[cid, "departure"][0] for path in paths for cid in path}
            for trip in sorted(trips):
                snapshot.law(trip)
            bank = StableTripWorlds(
                snapshot.as_belief(), parameters, paths, count, seed, root_index, model._trips
            )
            public = snapshot.public_state()
            if rollout_planner_kind == "recursive":
                planner = PairRecursivePlanner(
                    graph, model, parameters, deadline,
                    exact_combined_max_width=exact_combined_max_width,
                )
            else:
                planner = PairMatchedPlanner(graph, model, parameters, deadline)
            engine = PairDecisionEngine(planner, variant="frontier")
            preparation = dict(process_cpu=process_time() - cpu, wall=perf_counter() - tick)
            phase = "paired_continuations"
            for index in range(count):
                pending = {}
                for arm in rows:
                    executor = PairSnapshotExecutor(
                        graph, planner, deadline, engine=engine, method="rolling_fixed",
                        waiting="combined", observation_contract=contract, trace=False,
                    )
                    outcome = executor.forced(
                        node, row, arm, snapshot.clone(), CheckingWorld(bank, index, public)
                    )
                    pending[arm] = dict(
                        world_index=index, loss=outcome["loss"],
                        status=outcome["status"], decisions=outcome["decisions"],
                    )
                    if "requested_root_action" in outcome:
                        pending[arm].update(
                            requested_root_action=outcome["requested_root_action"],
                            executed_root_action=outcome["executed_root_action"],
                        )
                # No expensive operation between the two list appends.
                rows["alight"].append(pending["alight"])
                rows["continue"].append(pending["continue"])
                completed += 1
            phase = "complete"
        except QueryCPUExpired:
            expired = True
        finally:
            signal.setitimer(signal.ITIMER_PROF, 0.0)
            signal.signal(signal.SIGPROF, previous_handler)
    # An interrupt between append and count commit must not retain half a pair.
    for arm in rows:
        del rows[arm][completed:]
        rows[arm].extend(
            dict(world_index=index, loss=None, status="cpu_unresolved", decisions={})
            for index in range(completed, count)
        )
    paired = empirical_pair_summary(
        [r["loss"] for r in rows["alight"]],
        [r["loss"] for r in rows["continue"]], deadline - row["clock"],
    )
    actual = process_time() - cpu
    return dict(
        mode=f"{rollout_planner_kind}_{count}", used_kernel="pair_common_observation_pi",
        rows=rows, paired=paired, count=count, completed_pairs=completed,
        seed=seed, root_index=root_index, future_messages=0,
        budget=dict(requested_process_cpu=allowance, interrupted=expired,
                    stopped_phase=phase, overshoot_seconds=max(0.0, actual - allowance),
                    guarantee="CPU interruption; C-kernel delivery and reporting can overrun"),
        cost=dict(process_cpu=actual, wall=perf_counter() - tick, preparation=preparation),
    )


def choose_paired_screen(gate_result, base_action, *, z_alpha=1.96, min_effect_seconds=5.0):
    """Heuristic full-bank paired normal screen, with no finite-sample guarantee."""
    paired = gate_result["paired"]
    lo, hi = paired["full_bank_mean_gain_interval"]
    stats = paired.get("complete_pair_statistics")
    if (paired.get("unresolved_pairs", 0) or stats is None
            or stats.get("mean_gain") is None or stats.get("se") is None):
        return base_action, dict(
            override=False, reason="unresolved_or_insufficient_pairs_retain_base",
            missing_outcome_interval=[float(lo), float(hi)],
        )
    mean, se = float(stats["mean_gain"]), float(stats["se"])
    low, high = min(float(lo), mean - z_alpha * se), max(float(hi), mean + z_alpha * se)
    chosen = ("alight" if high < -min_effect_seconds else
              "continue" if low > min_effect_seconds else base_action)
    return chosen, dict(
        override=chosen != base_action, selected_action=chosen, base_action=base_action,
        mean_gain=mean, se=se, normal_screen_interval=[low, high],
        guarantee="none: normal approximation with fixed full bank",
    )
