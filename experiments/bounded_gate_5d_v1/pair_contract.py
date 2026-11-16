"""G1 Platform-Pair MCT Contract + Async Dwell Execution + Bounded Selective Controllers.

When mct_contract == 'station_max' and clock_contract == 'ideal', this module
reproduces legacy V51 MatchedContinuationPlanner, MeanSnapshotCSA, PolicyExecutor,
and full128 rollout bit-for-bit. When mct_contract == 'platform_pair', every
component (outer executor, analytical planner, FrozenCombinedReference,
MeanSnapshotCSA, and inner rollout worlds) enforces the exact same per-pair
readiness ready_by_cid[dep_cid] = entry + tau(arr_cid, dep_cid).
"""

from dataclasses import replace
from itertools import groupby
from math import inf, isfinite
from time import perf_counter, process_time

import numpy as np

from experiments.decision_calibration_v1.platform_pairs import (
    endpoint,
    static_sources,
    transfer_pair,
)
from experiments.g2_reset_v1.calibration_split import classify_platform_pair_contract
from experiments.g2_reset_v1.certified_gate import choose_certified, should_invoke_rollout
from experiments.reconstruction_v33.execute import Executor as WaitingControls
from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v34.conditional_reference import (
    SnapshotExecutor as OriginalSnapshotExecutor,
)
from experiments.reconstruction_v34.execute import Executor as PublicExecutor, MissingEndpoint
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v37.controller import has_legal_continue, node_structure
from experiments.reconstruction_v39.state import raw_history_key
from experiments.reconstruction_v41.action_publisher import ActionPublisher, ActionToken
from experiments.reconstruction_v41.continuation import empirical_pair_summary
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import (
    CheckingWorld,
    PhysicalDecisionEngine,
    PhysicalPlanner,
)
from experiments.reconstruction_v41.run import normalized
from experiments.reconstruction_v44.matched import choose as choose_legacy_v44
from experiments.reconstruction_v51.baselines import MenuLayout
from experiments.reconstruction_v51.rollout import StableTripWorlds
from src.full_menu_opportunity import FullMenuOpportunity


class PairPlatform(Platform):
    """Platform node annotated with per-departure MCTs from its parent arrival leg."""

    def __init__(self, stop, rides, *, incoming_cid=None, transfer_times=None, default_mct=0.0):
        super().__init__(stop, rides)
        self.incoming_cid = incoming_cid
        self.default_mct = float(default_mct)
        self.transfer_times = (
            {int(cid): float(val) for cid, val in transfer_times.items()}
            if transfer_times is not None
            else {int(cid): float(default_mct) for cid in self.rides}
        )

    def mct_for(self, cid):
        return float(self.transfer_times.get(int(cid), self.default_mct))

    def ready_by_cid(self, entry):
        entry_f = float(entry)
        return {cid: entry_f + self.mct_for(cid) for cid in self.rides}

    def min_ready(self, entry, deadline):
        if not self.rides:
            return float(entry) + self.default_mct
        vals = [float(entry) + self.mct_for(cid) for cid in self.rides]
        finite_vals = [v for v in vals if isfinite(v)]
        return min(finite_vals) if finite_vals else float(deadline)

    def paths(self):
        return [path for ride in self.rides.values() for path in ride.paths()]


class DayPlatformIndex:
    """Fast per-day static GTFS platform-pair contract resolver."""

    def __init__(self, connections_or_metadata, regional_dir, static_dir, stop_mcts):
        if isinstance(connections_or_metadata, dict):
            self.by_cid = {int(cid): row for cid, row in connections_or_metadata.items()}
        else:
            self.by_cid = {int(row["id"]): row for row in connections_or_metadata}
        self.stop_mcts = {str(k): int(v) for k, v in stop_mcts.items()}
        self.index, self.stops, self.transfers = static_sources(
            list(self.by_cid.values()), regional=regional_dir, static=static_dir
        )
        self._pair_cache = {}

    def resolve_pair(self, arr_cid, dep_cid, *, tau_same_platform=90.0):
        key = (int(arr_cid), int(dep_cid), float(tau_same_platform))
        if key in self._pair_cache:
            return self._pair_cache[key]
        inc = self.by_cid[int(arr_cid)]
        out = self.by_cid[int(dep_cid)]
        stop_key = str(inc["arr_stop"])
        fallback_mct = self.stop_mcts[stop_key]
        raw = transfer_pair(inc, out, fallback_mct, self.index, self.stops, self.transfers)
        if raw.get("status") == "planned_pair_prohibited":
            contract = {
                "rule": "planned_pair_prohibited",
                "contract_mct_seconds": float("inf"),
                "legacy_mct_seconds": fallback_mct,
                "mct_delta_seconds": None,
                "status": raw["status"],
                "arr_stop_id": raw["arrival"].get("stop_id"),
                "dep_stop_id": raw["departure"].get("stop_id"),
            }
        else:
            classified = classify_platform_pair_contract(
                raw, tau_same_platform=tau_same_platform
            )
            # Preserve exact float sensitivity (e.g. 60.0 / 90.0 / 120.0)
            mct_val = (
                float(tau_same_platform)
                if classified["rule"] == "same_physical_platform"
                else float(classified["contract_mct_seconds"])
            )
            contract = {
                "rule": classified["rule"],
                "contract_mct_seconds": mct_val,
                "legacy_mct_seconds": fallback_mct,
                "mct_delta_seconds": mct_val - float(fallback_mct),
                "status": raw["status"],
                "arr_stop_id": raw["arrival"].get("stop_id"),
                "dep_stop_id": raw["departure"].get("stop_id"),
            }
        self._pair_cache[key] = contract
        return contract


def decorate_menu_with_mct(
    node,
    graph,
    *,
    mct_contract="platform_pair",
    platform_index=None,
    tau_same_platform=90.0,
    incoming_cid=None,
):
    """Return an acyclic menu tree whose platforms carry explicit per-offer MCTs."""
    if node is None:
        return None
    if isinstance(node, Ride):
        exits = {
            cid: None
            if child is None
            else decorate_menu_with_mct(
                child,
                graph,
                mct_contract=mct_contract,
                platform_index=platform_index,
                tau_same_platform=tau_same_platform,
                incoming_cid=cid,
            )
            for cid, child in node.exits.items()
        }
        return Ride(node.first_cid, tuple(node.path), exits)

    default_mct = float(graph.stops[node.stop].min_transfer_time)
    rides = {
        cid: decorate_menu_with_mct(
            ride,
            graph,
            mct_contract=mct_contract,
            platform_index=platform_index,
            tau_same_platform=tau_same_platform,
            incoming_cid=None,
        )
        for cid, ride in node.rides.items()
    }
    if mct_contract == "station_max" or incoming_cid is None:
        transfer_times = {cid: default_mct for cid in rides}
    elif mct_contract == "platform_pair":
        if platform_index is None:
            raise ValueError("platform_index is required for mct_contract='platform_pair'")
        transfer_times = {
            cid: platform_index.resolve_pair(
                incoming_cid, cid, tau_same_platform=tau_same_platform
            )["contract_mct_seconds"]
            for cid in rides
        }
    else:
        raise ValueError(f"Unknown mct_contract: {mct_contract}")

    return PairPlatform(
        node.stop,
        rides,
        incoming_cid=incoming_cid,
        transfer_times=transfer_times,
        default_mct=default_mct,
    )


def active_pair_platform(model, platform, history, entry, deadline):
    """Prune rides that have zero operating departure probability in [ready_cid, deadline)."""
    ready_map = (
        platform.ready_by_cid(entry)
        if isinstance(platform, PairPlatform)
        else {cid: float(entry) + 0.0 for cid in platform.rides}
    )
    retained = {}
    for cid, ride in platform.rides.items():
        r_cid = ready_map[cid]
        if not isfinite(r_cid) or r_cid >= deadline:
            continue
        trip, slot = model._positions[cid, "departure"]
        law = history.law(trip)
        operating = ~law.canceled[slot]
        times = law.times[slot][operating]
        mass = law.marginal(slot)[operating]
        # Keep any ride with positive mass from entry onward so local absence conditioning
        # and ready_cid filtering remain consistent on the active candidate set.
        if float(mass[(times >= r_cid) & (times < deadline)].sum()) > 1e-12:
            retained[cid] = ride
    if isinstance(platform, PairPlatform):
        return PairPlatform(
            platform.stop,
            retained,
            incoming_cid=platform.incoming_cid,
            transfer_times={cid: platform.transfer_times[cid] for cid in retained},
            default_mct=platform.default_mct,
        )
    return Platform(platform.stop, retained)


class PairMenuOpportunity(FullMenuOpportunity):
    """FullMenuOpportunity generalized to heterogeneous per-offer readiness ready_by_cid."""

    def __init__(self, compiler, ranks, entry, ready_by_cid, deadline):
        self.compiler, self.ranks = compiler, dict(ranks)
        self.entry, self.deadline = float(entry), float(deadline)
        self.ready_by_cid = {cid: float(ready_by_cid[cid]) for cid in self.ranks}
        finite_ready = [v for v in self.ready_by_cid.values() if isfinite(v)]
        self.ready = min(finite_ready) if finite_ready else self.deadline
        self.cids = tuple(self.ranks)
        self.distributions, self.cdfs, self.absent, self.marks = {}, {}, {}, {}
        self.mass_tails = {}
        for cid in self.cids:
            times, mass, cancel = compiler.departure_distribution(cid)
            times, mass = np.asarray(times), np.asarray(mass)
            self.distributions[cid] = times, mass, cancel
            self.cdfs[cid] = np.r_[0.0, np.cumsum(mass)]
            self.mass_tails[cid] = np.r_[np.cumsum(mass[::-1])[::-1], 0.0]
            self.absent[cid] = float(cancel + mass[times < self.entry].sum())
            r_cid = self.ready_by_cid[cid]
            legal = (times >= r_cid) & (times < self.deadline) & (mass > 0)
            clocks = times[legal]
            scores = compiler.clock_nominal_scores(cid, clocks.astype(np.int64))
            self.marks[cid] = clocks, mass[legal], scores

    def wait_score(self, clock, seen):
        """Expected nominal value of the first future unseen LEGAL offer after clock."""
        unseen = tuple(cid for cid in self.cids if cid not in seen)
        survival = {}
        for cid in unseen:
            times, _, _ = self.distributions[cid]
            after = np.searchsorted(times, clock, side="right")
            survival[cid] = self.absent[cid] + self.mass_tails[cid][after]
        active = tuple(cid for cid in unseen if survival[cid] > 0)
        gain = 0.0
        for cid in active:
            clocks, mass, scores = self.marks[cid]
            future = clocks > clock
            if not np.any(future):
                continue
            times = clocks[future]
            selected = mass[future] / survival[cid]
            for other in active:
                if other == cid:
                    continue
                other_times, _, _ = self.distributions[other]
                r_other = max(float(clock), self.ready_by_cid[other])
                side = "right" if self.ranks[other] < self.ranks[cid] else "left"
                # Within surviving mass (canceled or T < entry or T > clock),
                # other fails to make an earlier legal offer at t (> clock) iff:
                #   - when t < ready_other: always (probability 1.0 of survival[other])
                #   - when t >= ready_other: T not in (max(clock, ready_other_open), t],
                #     i.e. absent[other] + P(clock < T < ready_other) + P(T > t).
                idx_after_clock = np.searchsorted(other_times, clock, side="right")
                idx_before_ready = np.searchsorted(other_times, self.ready_by_cid[other], side="left")
                unboardable_after_clock = (
                    self.cdfs[other][idx_before_ready] - self.cdfs[other][idx_after_clock]
                    if idx_before_ready > idx_after_clock
                    else 0.0
                )
                later = self.mass_tails[other][np.searchsorted(other_times, times, side=side)]
                no_earlier_offer = np.where(
                    times < self.ready_by_cid[other],
                    survival[other],
                    self.absent[other] + unboardable_after_clock + later,
                )
                selected = selected * (no_earlier_offer / survival[other])
            gain += float(selected @ (self.deadline - scores[future]))
        return float(self.deadline - gain)


class PairFrozenCombinedReference:
    """Exact policy evaluation of fixed-Combined under per-offer readiness ready_by_cid."""

    def __init__(self, reference):
        self.reference = reference
        self.cids = tuple(reference.cids)
        self.entry, self.ready, self.deadline = reference.entry, reference.ready, reference.deadline
        self.ready_by_cid = getattr(
            reference, "ready_by_cid", {cid: float(reference.ready) for cid in self.cids}
        )
        self.times = np.array(
            sorted(
                {
                    float(t)
                    for cid in self.cids
                    for t, p in zip(*reference.distributions[cid][:2], strict=True)
                    if p > 0 and self.entry <= t < self.deadline
                }
            )
        )
        nstates, ntime = 1 << len(self.cids), len(self.times)
        self.full_mask = nstates - 1
        self.bit_by_cid = {cid: 1 << i for i, cid in enumerate(self.cids)}
        self.values = np.full((ntime + 1, nstates), np.nan)
        self.first_values = np.full_like(self.values, np.nan)
        survival = np.zeros((ntime + 1, len(self.cids)))
        hazards = np.zeros((ntime, len(self.cids)))
        single_wait = np.full_like(hazards, self.deadline)
        scores = np.full_like(hazards, self.deadline)
        boundaries = np.r_[self.times, self.deadline]
        supported = np.ones_like(self.values, dtype=bool)
        ready_vec = np.array([self.ready_by_cid[cid] for cid in self.cids], dtype=float)

        for i, cid in enumerate(self.cids):
            times, mass, cancel = reference.distributions[cid]
            times, mass = np.asarray(times), np.asarray(mass)
            tail = np.r_[np.cumsum(mass[::-1])[::-1], 0.0]
            absent = cancel + mass[times < self.entry].sum()
            survival[:, i] = absent + tail[np.searchsorted(times, boundaries, side="left")]
            r_cid = ready_vec[i]
            for t, p in zip(times, mass, strict=True):
                if p > 0 and self.entry <= t < self.deadline:
                    j = int(np.searchsorted(self.times, t))
                    hazards[j, i] += p
                    if t >= r_cid:
                        scores[j, i] = reference.score(cid, t)
            np.divide(hazards[:, i], survival[:-1, i], out=hazards[:, i], where=survival[:-1, i] > 0)
            clocks, probability, marks = reference.marks[cid]
            gain_tail = np.r_[np.cumsum((probability * (self.deadline - marks))[::-1])[::-1], 0.0]
            gain = gain_tail[np.searchsorted(clocks, self.times, side="right")]
            after = absent + tail[np.searchsorted(times, self.times, side="right")]
            single_wait[:, i] -= np.divide(gain, after, out=np.zeros_like(gain), where=after > 0)
            included = np.arange(nstates) & (1 << i) != 0
            supported[:, included] &= survival[:, i, None] > 0

        self.values[-1, supported[-1]] = self.deadline
        self.first_values[-1, supported[-1]] = self.deadline
        bits, subsets = [], []
        for mask in range(nstates):
            active = [i for i in range(len(self.cids)) if mask & (1 << i)]
            outcomes = np.array([0], dtype=int)
            for i in active:
                outcomes = np.r_[outcomes, outcomes | (1 << i)]
            bits.append(active)
            subsets.append(outcomes)
        rank_order = sorted(range(len(self.cids)), key=lambda i: reference.ranks[self.cids[i]])
        transitions = 0
        for j in range(ntime - 1, -1, -1):
            t_j = self.times[j]
            legal_mask_j = 0
            for i in range(len(self.cids)):
                if t_j >= ready_vec[i]:
                    legal_mask_j |= 1 << i
            best_board = np.full(nstates, self.deadline)
            ranked_board = np.full(nstates, self.deadline)
            wait = np.full(nstates, self.deadline)
            for mask in range(1, nstates):
                bit = mask & -mask
                i = bit.bit_length() - 1
                best_board[mask] = min(best_board[mask ^ bit], scores[j, i])
                ranked_board[mask] = scores[j, next(k for k in rank_order if mask & (1 << k))]
                target = min(single_wait[j, k] for k in bits[mask])
                wait[mask] = min(target, self.first_values[j + 1, mask])
            for mask in range(nstates):
                if not supported[j, mask]:
                    continue
                probability = np.ones(1)
                for i in bits[mask]:
                    hazard = hazards[j, i]
                    probability = np.r_[probability * (1 - hazard), probability * hazard]
                positive = probability > 0
                departed = subsets[mask][positive]
                probability = probability[positive]
                remaining = mask ^ departed
                future = self.values[j + 1, remaining]
                first_future = self.first_values[j + 1, remaining]
                if np.any(~np.isfinite(future)):
                    raise ValueError("Positive event mass reached an unsupported waiting history")
                departed_legal = departed & legal_mask_j
                board = (departed_legal != 0) & (
                    best_board[departed_legal] <= wait[remaining] + 1e-9
                )
                self.values[j, mask] = probability @ np.where(
                    board, best_board[departed_legal], future
                )
                self.first_values[j, mask] = probability @ np.where(
                    departed_legal != 0, ranked_board[departed_legal], first_future
                )
                transitions += len(probability)
        self.supported = supported
        self.value = float(self.values[0, self.full_mask])
        self.first_offer_value = float(self.first_values[0, self.full_mask])
        self.counters = dict(events=ntime, masks=nstates, positive_event_outcomes=transitions)


def _pair_structure_mixin_structure(planner, node):
    if node not in planner._structures:
        if hasattr(node, "first_cid"):
            planner._structures[node] = (
                "ride",
                node.first_cid,
                tuple(node.path),
                tuple(
                    (cid, None if node.exits[cid] is None else planner.structure(node.exits[cid]))
                    for cid in node.path
                    if cid in node.exits
                ),
            )
        elif isinstance(node, PairPlatform):
            planner._structures[node] = (
                "pair_platform",
                node.stop,
                tuple(sorted(node.transfer_times.items())),
                tuple((cid, planner.structure(ride)) for cid, ride in sorted(node.rides.items())),
            )
        else:
            planner._structures[node] = (
                "platform",
                node.stop,
                tuple((cid, planner.structure(ride)) for cid, ride in sorted(node.rides.items())),
            )
    return planner._structures[node]


class PairPhysicalPlanner(PhysicalPlanner):
    """PhysicalPlanner supporting per-offer transfer_times on PairPlatform nodes."""

    def __init__(self, graph, model, parameters, deadline):
        super().__init__(graph, model, parameters, deadline)
        self._pair_first_offer = {}

    def structure(self, node):
        return _pair_structure_mixin_structure(self, node)

    def scheduled_arrival(self, ride):
        key = self.structure(ride)
        if key not in self._ranks:
            values = []
            for cid, platform in ride.exits.items():
                arrival = self.graph.connections[cid].arr_time
                if platform is None:
                    values.append(arrival)
                elif isinstance(platform, PairPlatform):
                    values.extend(
                        self.scheduled_arrival(child)
                        for first, child in platform.rides.items()
                        if self.graph.connections[first].dep_time >= arrival + platform.mct_for(first)
                    )
                else:
                    ready = arrival + self.graph.stops[platform.stop].min_transfer_time
                    values.extend(
                        self.scheduled_arrival(child)
                        for first, child in platform.rides.items()
                        if self.graph.connections[first].dep_time >= ready
                    )
            self._ranks[key] = min(values, default=self.deadline)
        return self._ranks[key]

    def platform_reference(self, platform, history, entry, ready, method="optimal"):
        if isinstance(platform, PairPlatform):
            return PairMenuOpportunity(
                self.platform_compiler(platform, history, method),
                self.ranks(platform),
                entry,
                platform.ready_by_cid(entry),
                self.deadline,
            )
        return super().platform_reference(platform, history, entry, ready, method)

    def first_offer_entry_values(self, platform, history, entry_clocks, method="optimal"):
        """Integrate first legal offer given vector of arrival clocks entry_clocks."""
        if not isinstance(platform, PairPlatform):
            ready = np.asarray(entry_clocks, dtype=float) + self.graph.stops[platform.stop].min_transfer_time
            return self.first_offer_values(platform, history, ready, method)

        self.stats["first_offer_queries"] += 1
        entry_arr = np.asarray(entry_clocks, dtype=float)
        shape, flat = entry_arr.shape, entry_arr.reshape(-1)
        key = (*self._node_key(platform, history, method), "entry", tuple(flat))
        if key in self._pair_first_offer:
            self.stats["first_offer_hits"] += 1
            return self._pair_first_offer[key].reshape(shape)

        compiler = self.platform_compiler(platform, history, method)
        ranks, marks, prefix, suffix = self.ranks(platform), {}, {}, {}
        for cid in platform.rides:
            clocks, mass, cancellation = compiler.departure_distribution(cid)
            positive = mass > 0
            clocks, mass = clocks[positive], mass[positive]
            scores = compiler.clock_nominal_scores(cid, clocks)
            marks[cid] = clocks, mass, cancellation, scores
            prefix[cid] = np.r_[0.0, np.cumsum(mass)]
            suffix[cid] = np.r_[np.cumsum(mass[::-1])[::-1], 0.0]

        gain = np.zeros(len(flat))
        for cid, (clocks, mass, _, scores) in marks.items():
            tau_cid = platform.mct_for(cid)
            if not isfinite(tau_cid):
                continue
            ready_cid = flat + tau_cid
            legal = (clocks[None, :] >= ready_cid[:, None]) & (clocks[None, :] < self.deadline)
            if not np.any(legal):
                continue
            selected = legal * mass[None, :]
            for other, (other_clocks, _, cancellation, _) in marks.items():
                if other == cid:
                    continue
                tau_other = platform.mct_for(other)
                if not isfinite(tau_other):
                    continue
                ready_other = flat + tau_other
                lower = prefix[other][np.searchsorted(other_clocks, ready_other, side="left")]
                side = "right" if ranks[other] < ranks[cid] else "left"
                upper = suffix[other][np.searchsorted(other_clocks, clocks, side=side)]
                # When clocks[k] < ready_other[i], other cannot offer a legal ride at or before clocks[k].
                factor = np.where(
                    clocks[None, :] < ready_other[:, None],
                    1.0,
                    cancellation + lower[:, None] + upper[None, :],
                )
                selected *= factor
            gain += selected @ (self.deadline - scores)

        result = self.deadline - gain
        self._pair_first_offer[key] = result
        self.stats["first_offer_compiles"] += 1
        return result.reshape(shape)

    def exit_cost(self, ride, cid, history, clocks, method):
        if history.announced_unavailable(cid, "arrival"):
            return np.full_like(clocks, self.deadline, dtype=float)
        platform = ride.exits[cid]
        if platform is None:
            return np.minimum(clocks, self.deadline).astype(float)
        if isinstance(platform, PairPlatform):
            return self.first_offer_entry_values(platform, history, clocks, method)
        ready = clocks + self.graph.stops[platform.stop].min_transfer_time
        return self.first_offer_values(platform, history, ready, method)


class PairMatchedPlanner(MatchedContinuationPlanner, PairPhysicalPlanner):
    """Depth-1 selective continuation planner ('cheap') with PairPlatform support."""

    def __init__(self, graph, model, parameters, deadline):
        PairPhysicalPlanner.__init__(self, graph, model, parameters, deadline)
        self.tail_base = PairPhysicalPlanner(graph, model, parameters, deadline)
        self._selective_tails = {}
        self.stats.update(
            selective_tail_queries=0,
            selective_tail_hits=0,
            selective_tail_compiles=0,
            selective_tail_points=0,
        )

    def selective_entry_components(self, platform, history, entry_clocks, method="rolling_fixed"):
        if not isinstance(platform, PairPlatform):
            ready = np.asarray(entry_clocks, dtype=float) + self.graph.stops[platform.stop].min_transfer_time
            return self.selective_waiting_components(platform, history, ready, method)

        self.stats["selective_tail_queries"] += 1
        entry_arr = np.asarray(entry_clocks, dtype=float)
        shape, flat = entry_arr.shape, entry_arr.reshape(-1)
        key = (*self._node_key(platform, history, method), "entry", tuple(flat), shape)
        if key in self._selective_tails:
            self.stats["selective_tail_hits"] += 1
            return self._selective_tails[key]

        compiler = self.tail_base.platform_compiler(platform, history, method)
        first = self.tail_base.first_offer_entry_values(platform, history, flat, method).reshape(shape)
        by_target = {}
        for cid in platform.rides:
            tau_cid = platform.mct_for(cid)
            if not isfinite(tau_cid):
                by_target[cid] = np.full(shape, self.deadline)
                continue
            ready_cid = flat + tau_cid
            clocks, mass, _ = compiler.departure_distribution(cid)
            positive = mass > 0
            clocks, mass = clocks[positive], mass[positive]
            scores = compiler.clock_nominal_scores(cid, clocks)
            legal = (clocks[None, :] >= ready_cid[:, None]) & (clocks[None, :] < self.deadline)
            by_target[cid] = (self.deadline - legal @ (mass * (self.deadline - scores))).reshape(shape)
        single = (
            np.minimum.reduce(list(by_target.values()))
            if by_target
            else np.full(shape, self.deadline)
        )
        result = dict(
            value=np.minimum(first, single),
            first_offer=first,
            single_target=single,
            by_target=by_target,
        )
        self._selective_tails[key] = result
        self.stats["selective_tail_compiles"] += 1
        self.stats["selective_tail_points"] += flat.size
        self._tail_stats()
        return result

    def exit_cost(self, ride, cid, history, clocks, method):
        if history.announced_unavailable(cid, "arrival"):
            return np.full_like(clocks, self.deadline, dtype=float)
        platform = ride.exits[cid]
        if platform is None:
            return np.minimum(clocks, self.deadline).astype(float)
        return self.selective_entry_components(platform, history, clocks, method)["value"]


class PairRecursivePlanner(PairMatchedPlanner):
    """Multi-transfer recursive selective planner + active-pruned PairFrozenCombinedReference."""

    value_scope = (
        "Recursive multi-transfer selective continuation with active-pruned "
        "PairFrozenCombinedReference under PairPlatform MCT"
    )

    def __init__(self, graph, model, parameters, deadline, *, exact_combined_max_width=6):
        super().__init__(graph, model, parameters, deadline)
        self.exact_combined_max_width = int(exact_combined_max_width)
        self.stats.update(exact_combined_evaluations=0, active_pruned_rides=0)

    def selective_entry_components(self, platform, history, entry_clocks, method="rolling_fixed"):
        self.stats["selective_tail_queries"] += 1
        entry_arr = np.asarray(entry_clocks, dtype=float)
        shape, flat = entry_arr.shape, entry_arr.reshape(-1)
        key = (
            *self._node_key(platform, history, method),
            "recursive_entry",
            tuple(flat),
            shape,
            self.exact_combined_max_width,
        )
        if key in self._selective_tails:
            self.stats["selective_tail_hits"] += 1
            return self._selective_tails[key]

        # Recursive: use self (not self.tail_base) so downstream transfers also get selective waiting
        compiler = self.platform_compiler(platform, history, method)
        first = self.first_offer_entry_values(platform, history, flat, method).reshape(shape)
        by_target = {}
        for cid in platform.rides:
            tau_cid = (
                platform.mct_for(cid)
                if isinstance(platform, PairPlatform)
                else float(self.graph.stops[platform.stop].min_transfer_time)
            )
            if not isfinite(tau_cid):
                by_target[cid] = np.full(shape, self.deadline)
                continue
            ready_cid = flat + tau_cid
            clocks, mass, _ = compiler.departure_distribution(cid)
            positive = mass > 0
            clocks, mass = clocks[positive], mass[positive]
            scores = compiler.clock_nominal_scores(cid, clocks)
            legal = (clocks[None, :] >= ready_cid[:, None]) & (clocks[None, :] < self.deadline)
            by_target[cid] = (self.deadline - legal @ (mass * (self.deadline - scores))).reshape(shape)
        single = (
            np.minimum.reduce(list(by_target.values()))
            if by_target
            else np.full(shape, self.deadline)
        )
        envelope = np.minimum(first, single)

        if self.exact_combined_max_width > 0 and flat.size > 0:
            refined = envelope.reshape(-1).copy()
            for idx, entry_val in enumerate(flat):
                active = active_pair_platform(self.model, platform, history, float(entry_val), self.deadline)
                self.stats["active_pruned_rides"] += len(platform.rides) - len(active.rides)
                if 1 < len(active.rides) <= self.exact_combined_max_width:
                    trips = [self.model._positions[c, "departure"][0] for c in active.rides]
                    if len(trips) == len(set(trips)):
                        min_r = (
                            active.min_ready(entry_val, self.deadline)
                            if isinstance(active, PairPlatform)
                            else float(entry_val) + self.graph.stops[active.stop].min_transfer_time
                        )
                        ref = self.platform_reference(active, history, float(entry_val), min_r, method)
                        refined[idx] = PairFrozenCombinedReference(ref).value
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


class PairDecisionEngine(PhysicalDecisionEngine):
    """PhysicalDecisionEngine preserving PairPlatform transfer_times across platform views."""

    def _platform_view(self, platform, visible, seen):
        if self.variant == "fullmemo":
            return platform, seen
        removed = set(seen) - set(visible)
        retained = tuple(cid for cid in platform.rides if cid not in removed)
        self.stats["platform_dropped_roots"] += len(platform.rides) - len(retained)
        if len(retained) == len(platform.rides):
            return platform, seen
        key = self.planner.structure(platform), retained
        if key not in self._platform_views:
            rides_sub = {cid: platform.rides[cid] for cid in retained}
            if isinstance(platform, PairPlatform):
                self._platform_views[key] = PairPlatform(
                    platform.stop,
                    rides_sub,
                    incoming_cid=platform.incoming_cid,
                    transfer_times={cid: platform.transfer_times[cid] for cid in retained},
                    default_mct=platform.default_mct,
                )
            else:
                self._platform_views[key] = Platform(platform.stop, rides_sub)
            self.stats["platform_views"] += 1
        return self._platform_views[key], tuple(cid for cid in seen if cid in retained)


class PairMeanSnapshotCSA:
    """MeanSnapshotCSA generalized to PairPlatform per-offer transfer_times."""

    def __init__(self, graph, planner, deadline):
        self.graph, self.planner, self.deadline = graph, planner, float(deadline)
        self._layouts, self._snapshots = {}, {}
        self.stats = dict(
            csa_snapshot_builds=0,
            csa_snapshot_hits=0,
            csa_endpoint_means=0,
            csa_unavailable_connections=0,
            csa_inconsistent_mean_connections=0,
            csa_scans=0,
            csa_connection_visits=0,
            csa_equal_time_passes=0,
        )

    def _mct(self, platform, cid):
        if isinstance(platform, PairPlatform):
            return platform.mct_for(cid)
        return float(self.graph.stops[platform.stop].min_transfer_time)

    def _snapshot(self, node, history, clock):
        key = node, float(clock), raw_history_key(history, node.trips(self.planner.model))[0]
        if key in self._snapshots:
            self.stats["csa_snapshot_hits"] += 1
            return self._snapshots[key]
        if node not in self._layouts:
            self._layouts[node] = MenuLayout(node)
        layout = self._layouts[node]
        means = {}
        for record in layout.records:
            cid = record["cid"]
            for kind in ("departure", "arrival"):
                if (cid, kind) in means:
                    continue
                trip, slot = self.planner.model._positions[cid, kind]
                law = history.law(trip)
                mass = law.marginal(slot)
                active = (~law.canceled[slot]) & (mass > 0)
                times, probability = law.times[slot][active], mass[active]
                value = (
                    None
                    if not len(times)
                    else float(times[0])
                    if len(times) == 1
                    else float(times @ probability / probability.sum())
                )
                means[cid, kind] = value
                self.stats["csa_endpoint_means"] += 1
        records = []
        inconsistent = unavailable = 0
        for sequence, record in enumerate(layout.records):
            departure, arrival = (means[record["cid"], kind] for kind in ("departure", "arrival"))
            if departure is None or arrival is None:
                unavailable += 1
                continue
            if arrival < departure:
                inconsistent += 1
                continue
            records.append(dict(record, departure=departure, arrival=arrival, sequence=sequence))
        records.sort(key=lambda row: (row["departure"], row["sequence"]))
        result = dict(
            layout=layout,
            records=records,
            means=means,
            inconsistent=inconsistent,
            unavailable=unavailable,
        )
        self._snapshots[key] = result
        self.stats["csa_snapshot_builds"] += 1
        self.stats["csa_unavailable_connections"] += unavailable
        self.stats["csa_inconsistent_mean_connections"] += inconsistent
        return result

    def _scan(self, snapshot, clock, *, platform=None, entry=None, ride=None, position=None):
        self.stats["csa_scans"] += 1
        layout = snapshot["layout"]
        platform_entries, riding = {}, {}
        if platform is not None:
            platform_entries[id(platform)] = float(entry)
        else:
            riding[id(ride), position] = float(clock)
        best = self.deadline

        def lower(table, key, value):
            if value < table.get(key, inf):
                table[key] = value
                return True
            return False

        for departure, group in groupby(snapshot["records"], key=lambda row: row["departure"]):
            if departure < clock:
                continue
            if departure >= best:
                break
            block = tuple(group)
            changed = True
            while changed:
                changed = False
                self.stats["csa_equal_time_passes"] += 1
                for row in block:
                    self.stats["csa_connection_visits"] += 1
                    rid, index, cid = row["rid"], row["position"], row["cid"]
                    accessible = riding.get((rid, index), inf) <= departure
                    if index == 0:
                        accessible |= any(
                            platform_entries.get(parent, inf)
                            + self._mct(layout.platforms[parent], cid)
                            <= departure
                            for parent in layout.rides[rid]["parents"]
                        )
                    if not accessible:
                        continue
                    arrival = row["arrival"]
                    if row["destination"]:
                        best = min(best, arrival)
                        continue
                    updated = lower(riding, (rid, index + 1), arrival)
                    changed |= updated and arrival <= departure
                    if row["tail"] is not None:
                        updated = lower(platform_entries, row["tail"], arrival)
                        min_ready = arrival + layout.platforms[row["tail"]].min_ready(0.0, self.deadline) if isinstance(layout.platforms[row["tail"]], PairPlatform) else arrival + self.graph.stops[layout.platforms[row["tail"]].stop].min_transfer_time
                        changed |= updated and min_ready <= departure
        return float(best)

    def onboard(self, ride, cid, clock, history):
        snapshot = self._snapshot(ride, history, clock)
        tail = ride.exits[cid]
        alight = (
            float(clock)
            if tail is None
            else self._scan(snapshot, clock, platform=tail, entry=clock)
        )
        next_position = ride.path.index(cid) + 1
        continuation = (
            self.deadline
            if next_position == len(ride.path)
            else self._scan(snapshot, clock, ride=ride, position=next_position)
        )
        choice = alight < self.deadline and alight <= continuation + 1e-9
        return choice, dict(
            method="public_mean_snapshot_csa",
            alight=alight,
            continue_cost=continuation,
            mean_timetable_inconsistent=snapshot["inconsistent"],
            scope="Deterministic current-W operating means with PairPlatform MCTs",
        )

    def platform(self, platform, entry, ready, history, visible, observed, clock):
        snapshot = self._snapshot(platform, history, clock)
        visible_set, removed = set(visible), set(observed) - set(visible)
        ranks = self.planner.ranks(platform)
        values = {}
        for cid, ride in platform.rides.items():
            if cid in removed:
                continue
            r_cid = float(entry) + self._mct(platform, cid)
            departure = snapshot["means"].get((cid, "departure"))
            if departure is None or not isfinite(r_cid) or departure < r_cid or departure < clock:
                continue
            if cid not in visible_set and departure <= clock:
                continue
            values[cid] = self._scan(snapshot, departure, ride=ride, position=0)
        target = min(values, key=lambda cid: (values[cid], ranks[cid]), default=None)
        if target is not None and values[target] >= self.deadline:
            target = None
        return (target if target in visible_set else None), dict(
            method="public_mean_snapshot_csa",
            target=target,
            values=values,
            entry=float(entry),
            ready=float(ready),
            mean_timetable_inconsistent=snapshot["inconsistent"],
            scope="Deterministic current-W operating means with PairPlatform MCTs",
        )


class PairObservationMixin:
    """Common local observation contract with per-offer PairPlatform readiness."""

    def __init__(self, *args, observation_contract=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_contract = observation_contract
        if observation_contract is not None:
            self.counts.update(local_observation_epochs=0, observation_only_epochs=0)

    def observation_evidence(self, history, clock):
        return self.decision_history(history, clock)

    def platform(self, platform, entry, history, world, *, after=float("-inf")):
        ready_map = (
            platform.ready_by_cid(entry)
            if isinstance(platform, PairPlatform)
            else {
                cid: float(entry) + self.graph.stops[platform.stop].min_transfer_time
                for cid in platform.rides
            }
        )
        min_ready = (
            platform.min_ready(entry, self.deadline)
            if isinstance(platform, PairPlatform)
            else float(entry) + self.graph.stops[platform.stop].min_transfer_time
        )

        if self.observation_contract is None:
            departures = {cid: world.event(cid, "departure") for cid in platform.rides}
            clocks = sorted(
                {
                    clock
                    for cid, (clock, canceled, _) in departures.items()
                    if not canceled and ready_map[cid] <= clock < self.deadline and clock > after
                }
            )
            for clock in clocks:
                observed = {
                    cid: time
                    for cid, (time, canceled, _) in departures.items()
                    if not canceled and entry <= time <= clock
                }
                visible = tuple(
                    cid
                    for cid, (time, canceled, _) in departures.items()
                    if not canceled and time == clock and clock >= ready_map[cid]
                )
                if not visible:
                    continue
                tick = perf_counter()
                history.observe_platform(tuple(platform.rides), entry, clock, observed)
                evidence = self.observation_evidence(history, clock)
                choice, receipt = self.choose_platform(
                    platform, entry, min_ready, history, visible, observed, clock
                )
                self.compute_wall += perf_counter() - tick
                self.counts["platform"] += 1
                self.record(
                    dict(
                        kind="platform",
                        stop=platform.stop,
                        entry=entry,
                        ready=min_ready,
                        clock=clock,
                        observed=observed,
                        visible=visible,
                        choice=choice,
                        receipt=receipt,
                        **evidence,
                    )
                )
                if choice is not None:
                    return self.ride(platform.rides[choice], clock, history, world)
            return self.deadline, "no_legal_offer"

        cids = self.observation_contract.cids_at(platform.stop)
        if not set(platform.rides).issubset(cids):
            raise ValueError("Every action-menu departure must be visible in the observation contract")
        departures = {cid: world.event(cid, "departure") for cid in cids}
        clocks = sorted(
            {
                clock
                for clock, canceled, _ in departures.values()
                if not canceled and entry <= clock < self.deadline and clock > after
            }
        )
        for clock in clocks:
            observed = {
                cid: time
                for cid, (time, canceled, _) in departures.items()
                if not canceled and entry <= time <= clock
            }
            observed_visible = tuple(
                cid
                for cid, (time, canceled, _) in departures.items()
                if not canceled and time == clock
            )
            visible = tuple(
                cid
                for cid in platform.rides
                if cid in observed_visible and clock >= ready_map[cid]
            )
            tick = perf_counter()
            local_receipts = history.observe_platform(cids, entry, clock, observed)
            evidence = self.observation_evidence(history, clock)
            self.counts["local_observation_epochs"] += 1
            public = dict(
                stop=platform.stop,
                entry=entry,
                ready=min_ready,
                clock=clock,
                observation_cids=cids,
                observed=observed,
                observed_visible=observed_visible,
                visible=visible,
                local_receipts=local_receipts,
                **evidence,
            )
            if not visible:
                self.counts["observation_only_epochs"] += 1
                self.compute_wall += perf_counter() - tick
                self.record(dict(kind="observation", **public))
                continue
            action_observed = {cid: time for cid, time in observed.items() if cid in platform.rides}
            choice, receipt = self.choose_platform(
                platform, entry, min_ready, history, visible, action_observed, clock
            )
            if choice is not None and choice not in visible:
                raise ValueError("A controller selected a vehicle that is not a current legal offer")
            self.compute_wall += perf_counter() - tick
            self.counts["platform"] += 1
            self.record(
                dict(
                    kind="platform",
                    choice=choice,
                    receipt=receipt,
                    action_observed=action_observed,
                    **public,
                )
            )
            if choice is not None:
                return self.ride(platform.rides[choice], clock, history, world)
        return self.deadline, "no_legal_offer"


class PairSnapshotExecutor(PairObservationMixin, OriginalSnapshotExecutor):
    """SnapshotExecutor for rollout worlds under PairPlatform MCTs."""

    def __init__(
        self,
        graph,
        planner,
        deadline,
        *,
        engine,
        method="rolling_fixed",
        waiting="combined",
        observation_contract=None,
        trace=False,
    ):
        super().__init__(
            graph,
            planner,
            deadline,
            method=method,
            waiting=waiting,
            observation_contract=observation_contract,
            trace=trace,
        )
        self.engine = engine

    def observation_evidence(self, history, clock):
        if not self.keep_trace:
            return {}
        return dict(
            public_history=history.public_state(),
            message_state=dict(frame_index=history.frame_index, clock=history.message_clock),
        )

    def choose_platform(self, platform, entry, ready, history, visible, observed, clock):
        return self.engine.platform(
            platform, entry, ready, history, visible, observed, clock, choose=self.choose, method=self.method
        )

    def ride(self, ride, ready, history, world):
        from experiments.reconstruction_v34.conditional_reference import ProjectionIncompatible

        previous, vehicle = ready, None
        for cid in ride.path:
            departure, canceled, identity = world.event(cid, "departure")
            if canceled or departure >= self.deadline:
                return self.deadline, "service_or_deadline_failure"
            if departure < previous:
                raise ProjectionIncompatible(f"{cid}:departure={departure}<known_previous={previous}")
            if vehicle is not None and identity != vehicle:
                raise ValueError("A sampled ride changed its physical vehicle")
            vehicle = identity
            arrival, canceled_arr, arrival_identity = world.event(cid, "arrival")
            if canceled_arr:
                return self.deadline, "service_failure"
            if arrival_identity != vehicle:
                raise ValueError("Sampled arrival changed its physical vehicle")
            if arrival < departure:
                raise ProjectionIncompatible(f"{cid}:arrival={arrival}<departure={departure}")
            if arrival >= self.deadline:
                return self.deadline, "deadline"
            history.observe(cid, "departure", departure)
            history.observe(cid, "arrival", arrival)
            previous = arrival
            if cid not in ride.exits:
                continue
            platform = ride.exits[cid]
            if platform is None:
                return arrival, "arrived"
            tick = perf_counter()
            alight, _ = self.engine.onboard(ride, cid, arrival, history, method=self.method)
            self.compute_wall += perf_counter() - tick
            self.counts["onboard"] += 1
            self.record(dict(kind="onboard", cid=cid, clock=arrival, choice="alight" if alight else "continue"))
            if alight:
                return self.platform(platform, arrival, history, world)
        return self.deadline, "exhausted_ride"


def evaluate_pair_rollout(
    graph,
    model,
    parameters,
    deadline,
    node,
    row,
    history,
    contract,
    *,
    root_index,
    count=64,
    max_process_cpu=1.8,
    seed=510015,
    rollout_planner_kind="matched",
    exact_combined_max_width=6,
):
    """Evaluate paired onboard rollout worlds under PairPlatform MCTs and CPU cap."""
    tick, cpu = perf_counter(), process_time()
    snapshot = freeze_snapshot(history, row["clock"])
    paths = [*node.paths(), *contract.paths()]
    trips = {model._positions[cid, "departure"][0] for path in paths for cid in path}
    for trip in sorted(trips):
        snapshot.law(trip)
    bank = StableTripWorlds(
        snapshot.as_belief(), parameters, paths, count, seed, root_index, model._trips
    )
    preparation = dict(process_cpu=process_time() - cpu, wall=perf_counter() - tick)
    public = snapshot.public_state()
    if rollout_planner_kind == "recursive":
        planner = PairRecursivePlanner(
            graph, model, parameters, deadline, exact_combined_max_width=exact_combined_max_width
        )
    else:
        planner = PairMatchedPlanner(graph, model, parameters, deadline)
    engine = PairDecisionEngine(planner, variant="frontier")
    rows = {"alight": [], "continue": []}
    completed = 0
    for index in range(count):
        if process_time() - cpu >= max_process_cpu:
            break
        for arm in rows:
            world = CheckingWorld(bank, index, public)
            executor = PairSnapshotExecutor(
                graph,
                planner,
                deadline,
                engine=engine,
                method="rolling_fixed",
                waiting="combined",
                observation_contract=contract,
                trace=False,
            )
            outcome = executor.forced(node, row, arm, snapshot.clone(), world)
            rows[arm].append(
                dict(
                    world_index=index,
                    loss=outcome["loss"],
                    status=outcome["status"],
                    decisions=outcome["decisions"],
                )
            )
        completed += 1
    for index in range(completed, count):
        for arm in rows:
            rows[arm].append(
                dict(world_index=index, loss=None, status="cpu_unresolved", decisions={})
            )
    paired = empirical_pair_summary(
        [r["loss"] for r in rows["alight"]],
        [r["loss"] for r in rows["continue"]],
        deadline - row["clock"],
    )
    return dict(
        mode=f"{rollout_planner_kind}_{count}",
        used_kernel="pair_common_observation_pi",
        rows=rows,
        paired=paired,
        count=count,
        completed_pairs=completed,
        seed=seed,
        root_index=root_index,
        future_messages=0,
        cost=dict(process_cpu=process_time() - cpu, wall=perf_counter() - tick, preparation=preparation),
    )


class BoundedArmExecutor(PairObservationMixin, PublicExecutor):
    """Unified executor supporting ideal vs physical-async dwell clocks and all 5 controllers."""

    def __init__(
        self,
        graph,
        planner,
        deadline,
        *,
        arm_config,
        hyperparams,
        root_index,
        cpu_start,
        observation_contract=None,
        clock_rules=None,
    ):
        super().__init__(
            graph,
            planner,
            deadline,
            method="rolling_fixed",
            waiting="combined",
            observation_contract=observation_contract,
        )
        self.arm_config = dict(arm_config)
        self.controller = arm_config["controller"]
        self.clock_contract = arm_config["clock_contract"]
        self.hyperparams = hyperparams
        self.root_index = int(root_index)
        self.clock_rules = clock_rules or {
            "min_dwell_floor_seconds": 1.0,
            "terminal_dwell_window_seconds": 60.0,
            "platform_boarding_window_seconds": 15.0,
        }
        self.engine = PairDecisionEngine(self.planner, variant="frontier")
        self.waiting_control = WaitingControls(
            graph, self.planner, deadline, method="rolling_fixed", waiting="combined", trace=False
        )
        self.csa = (
            PairMeanSnapshotCSA(graph, self.planner, deadline)
            if self.controller == "snapshot_csa"
            else None
        )
        self.states, self.gates, self.publications, self.decision_costs = [], [], [], []
        self.pending_cpu = cpu_start
        self._current_dwell_info = None

    @property
    def control_stats(self):
        out = dict(self.engine.stats)
        if self.csa is not None:
            out.update(self.csa.stats)
        return out

    def _record_state(self, node, history, **fields):
        state = dict(
            node=node_structure(node),
            public_history=normalized(history.public_state()),
            deadline=self.deadline,
            root_index=self.root_index,
            **fields,
        )
        self.states.append(state)
        return state

    def _done(self, kind, clock, **extra):
        end = process_time()
        self.decision_costs.append(
            dict(kind=kind, clock=float(clock), process_cpu=end - self.pending_cpu, **extra)
        )
        self.pending_cpu = end

    def ride(self, ride, ready, history, world):
        """Identical to PublicExecutor.ride, plus physical dwell-window tracking for ActionPublisher."""
        previous, vehicle = ready, None
        for idx, cid in enumerate(ride.path):
            departure, canceled, identity = world.event(cid, "departure")
            if canceled or departure < previous or departure >= self.deadline:
                return self.deadline, "service_or_clock_failure"
            if identity is None or vehicle is not None and identity != vehicle:
                raise MissingEndpoint(f"{cid}:inconsistent_vehicle_identity")
            vehicle = identity
            arrival, canceled_arrival, arrival_identity = world.event(cid, "arrival")
            if canceled_arrival:
                return self.deadline, "service_failure"
            if arrival_identity != vehicle or arrival < departure:
                raise MissingEndpoint(f"{cid}:inconsistent_arrival_identity_or_clock")
            if arrival >= self.deadline:
                return self.deadline, "deadline"
            history.observe(cid, "departure", departure)
            history.observe(cid, "arrival", arrival)
            previous = arrival
            if cid not in ride.exits:
                continue
            platform = ride.exits[cid]
            if platform is None:
                self.record(dict(kind="destination", cid=cid, clock=arrival))
                return arrival, "arrived"

            # Measure physical train departure from this station ONLY for ActionToken enforcement;
            # never exposed to planner/controller/history!
            if idx + 1 < len(ride.path):
                next_cid = ride.path[idx + 1]
                next_dep, next_canceled, _ = world.event(next_cid, "departure")
                if next_canceled:
                    raw_dwell = None
                    closes_at = float(arrival) + float(self.clock_rules["terminal_dwell_window_seconds"])
                else:
                    raw_dwell = float(next_dep - arrival)
                    closes_at = max(
                        float(next_dep),
                        float(arrival) + float(self.clock_rules["min_dwell_floor_seconds"]),
                    )
            else:
                raw_dwell = None
                closes_at = float(arrival) + float(self.clock_rules["terminal_dwell_window_seconds"])

            self._current_dwell_info = dict(
                raw_dwell_seconds=raw_dwell,
                closes_at=closes_at,
                opens_at=float(arrival),
            )
            tick = perf_counter()
            evidence = self.decision_history(history, arrival)
            alight, receipt = self.choose_onboard(ride, cid, arrival, history)
            self._current_dwell_info = None
            self.compute_wall += perf_counter() - tick
            self.counts["onboard"] += 1
            self.record(
                dict(
                    kind="onboard",
                    cid=cid,
                    clock=arrival,
                    choice="alight" if alight else "continue",
                    receipt=receipt,
                    **evidence,
                )
            )
            if alight:
                return self.platform(platform, arrival, history, world)
        return self.deadline, "exhausted_ride"

    def _publish_action(
        self,
        *,
        opens_at,
        closes_at,
        legal,
        fallback,
        base_cpu,
        compute_fn=None,
        forced_on_total_expiry=None,
    ):
        """Enforce either ideal zero-latency clock or deterministic async process_cpu ActionToken."""
        if self.clock_contract == "ideal":
            publisher = ActionPublisher(wall_clock=lambda: float(opens_at))
            token = ActionToken(
                str(len(self.publications)),
                tuple(legal),
                float(opens_at),
                float(opens_at) + 1.0,
                "ideal logical clock",
            )
            req = publisher.request(token, fallback)
            bg_cpu = 0.0
            bg_row = None
            if compute_fn is not None:
                t0 = process_time()
                candidate = compute_fn()
                bg_cpu = process_time() - t0
                bg_row = req.submit(candidate)
            consumed = req.take()
            finished = req.finish()
            finished.update(
                clock_contract="ideal",
                base_cpu_seconds=base_cpu,
                background_cpu_seconds=bg_cpu,
                background_accepted=None if bg_row is None else bg_row["accepted"],
                background_rejection_reason=None if bg_row is None else bg_row["reason"],
            )
            self.publications.append(finished)
            return consumed["action"], finished

        # Async physical dwell contract: logical clock advances deterministically with solver process_cpu
        sim_now = [float(opens_at) + float(base_cpu)]
        publisher = ActionPublisher(wall_clock=lambda: sim_now[0])
        token = ActionToken(
            str(len(self.publications)),
            tuple(legal),
            float(opens_at),
            float(closes_at),
            "physical ist-daten dwell window",
        )
        req = publisher.request(token, fallback)
        bg_cpu = 0.0
        bg_row = None
        if compute_fn is not None:
            t0 = process_time()
            candidate = compute_fn()
            bg_cpu = process_time() - t0
            sim_now[0] = float(opens_at) + float(base_cpu) + float(bg_cpu)
            bg_row = req.submit(candidate)
        # Physical action is consumed at min(sim_now, closes_at - eps if fallback was valid!)
        # Wait: in ActionPublisher, fallback was ALREADY published at opens_at + base_cpu < closes_at!
        # When train doors close at closes_at (if sim_now >= closes_at), the door-close event
        # executes the currently published instruction just before closes_at; if sim_now < closes_at,
        # the controller finishes and acts at sim_now!
        consume_clock = (
            min(sim_now[0], float(closes_at) - 1e-6)
            if float(opens_at) + float(base_cpu) < float(closes_at)
            else sim_now[0]
        )
        sim_now[0] = consume_clock
        consumed = req.take()
        finished = req.finish()
        if consumed is None:
            chosen = forced_on_total_expiry
            fallback_expired = True
        else:
            chosen = consumed["action"]
            fallback_expired = False
        finished.update(
            clock_contract="async",
            opens_at=float(opens_at),
            closes_at=float(closes_at),
            window_seconds=float(closes_at - opens_at),
            base_cpu_seconds=float(base_cpu),
            background_cpu_seconds=float(bg_cpu),
            total_cpu_seconds=float(base_cpu + bg_cpu),
            fallback_expired=fallback_expired,
            background_submitted=bg_row is not None,
            background_accepted=None if bg_row is None else bool(bg_row["accepted"]),
            background_rejection_reason=None if bg_row is None else bg_row["reason"],
        )
        self.publications.append(finished)
        return chosen, finished

    def choose_onboard(self, node, cid, clock, history):
        has_cont = has_legal_continue(node, cid)
        dwell_info = self._current_dwell_info or dict(
            raw_dwell_seconds=None, closes_at=float(clock) + 60.0, opens_at=float(clock)
        )
        state = self._record_state(
            node,
            history,
            kind="onboard",
            cid=cid,
            clock=float(clock),
            legal_actions=["alight", "continue"],
            has_continuation=has_cont,
            raw_dwell_seconds=dwell_info["raw_dwell_seconds"],
            dwell_closes_at=dwell_info["closes_at"],
        )

        t0 = process_time()
        if self.csa is not None:
            alight, receipt = self.csa.onboard(node, cid, clock, history)
        else:
            alight, receipt = self.engine.onboard(node, cid, clock, history, method="rolling_fixed")
        base_cpu = process_time() - t0
        base = "alight" if alight else "continue"

        compute_fn = None
        gate_screen = None
        if has_cont and self.controller == "full128":
            hp = self.hyperparams["full128"]

            def compute_fn():
                gate = evaluate_pair_rollout(
                    self.graph,
                    self.planner.model,
                    self.planner.parameters,
                    self.deadline,
                    node,
                    dict(kind="onboard", cid=cid, clock=clock),
                    history,
                    self.observation_contract,
                    root_index=self.root_index,
                    count=hp["worlds_per_query"],
                    max_process_cpu=hp["max_process_cpu_seconds"],
                    seed=hp["seed"],
                    rollout_planner_kind="matched",
                )
                sel = choose_legacy_v44(gate, base)
                gate.update(
                    cid=cid,
                    clock=float(clock),
                    state_index=len(self.states) - 1,
                    base_action=base,
                    base_receipt=receipt,
                    proposed_action=sel,
                )
                self.gates.append(gate)
                state["gate_index"] = len(self.gates) - 1
                return sel

        elif has_cont and self.controller == "bounded_cert":
            hp = self.hyperparams["bounded_cert"]
            invoke, gate_screen = should_invoke_rollout(
                receipt,
                deadline=self.deadline,
                ambiguity_margin_seconds=hp["ambiguity_margin_seconds"],
            )
            state["gate_screen"] = gate_screen
            if invoke:
                # Reserve base_cpu out of the strict sub-dwell budget
                rem_budget = max(0.05, float(hp["max_process_cpu_seconds"]) - base_cpu)

                def compute_fn():
                    gate = evaluate_pair_rollout(
                        self.graph,
                        self.planner.model,
                        self.planner.parameters,
                        self.deadline,
                        node,
                        dict(kind="onboard", cid=cid, clock=clock),
                        history,
                        self.observation_contract,
                        root_index=self.root_index,
                        count=hp["worlds_per_query"],
                        max_process_cpu=rem_budget,
                        seed=hp["seed"],
                        rollout_planner_kind="recursive",
                        exact_combined_max_width=hp["exact_combined_max_width"],
                    )
                    sel, cert_info = choose_certified(
                        gate,
                        base,
                        z_alpha=hp["z_alpha"],
                        min_effect_seconds=hp["min_effect_seconds"],
                    )
                    gate.update(
                        cid=cid,
                        clock=float(clock),
                        state_index=len(self.states) - 1,
                        base_action=base,
                        base_receipt=receipt,
                        proposed_action=sel,
                        certification=cert_info,
                        gate_screen=gate_screen,
                    )
                    self.gates.append(gate)
                    state["gate_index"] = len(self.gates) - 1
                    return sel

        selected, pub = self._publish_action(
            opens_at=float(clock),
            closes_at=dwell_info["closes_at"],
            legal=("alight", "continue"),
            fallback=base,
            base_cpu=base_cpu,
            compute_fn=compute_fn,
            forced_on_total_expiry="continue" if has_cont else "alight",
        )
        if "gate_index" in state:
            self.gates[state["gate_index"]]["selected_action"] = selected
            self.gates[state["gate_index"]]["publication_accepted"] = pub.get("background_accepted")
        self._done("onboard", clock, background_accepted=pub.get("background_accepted"))
        state.update(
            selected_action=selected,
            base_action=base,
            background_accepted=pub.get("background_accepted"),
            background_rejection_reason=pub.get("background_rejection_reason"),
        )
        return selected == "alight", dict(
            arm=self.arm_config["label"],
            base=base,
            selected=selected,
            scoring=receipt,
            gate_screen=gate_screen,
        )

    def choose_platform(self, node, entry, ready, history, visible, observed, clock):
        state = self._record_state(
            node,
            history,
            kind="platform",
            entry=float(entry),
            ready=float(ready),
            clock=float(clock),
            visible=list(visible),
            observed=dict(observed),
            stop=node.stop,
            legal_actions=[*visible, None],
        )
        t0 = process_time()
        if self.csa is not None:
            choice, receipt = self.csa.platform(node, entry, ready, history, visible, observed, clock)
        else:
            choice, receipt = self.engine.platform(
                node,
                entry,
                ready,
                history,
                visible,
                observed,
                clock,
                choose=self.waiting_control.choose,
                method="rolling_fixed",
            )
        base_cpu = process_time() - t0
        closes_at = float(clock) + float(self.clock_rules["platform_boarding_window_seconds"])
        selected, pub = self._publish_action(
            opens_at=float(clock),
            closes_at=closes_at,
            legal=(*visible, None),
            fallback=choice,
            base_cpu=base_cpu,
            compute_fn=None,
            forced_on_total_expiry=None,
        )
        self._done("platform", clock)
        state.update(selected_action=selected, base_action=choice)
        return selected, dict(arm=self.arm_config["label"], scoring=receipt)
