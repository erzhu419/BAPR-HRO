"""Unit tests for experiments.g2_reset_v1 (protocol v2, calibration split, certified gating)."""

import json
from pathlib import Path
import numpy as np
import pytest

from experiments.g2_reset_v1.calibration_split import (
    PARTITIONS,
    classify_platform_pair_contract,
    extract_conditional_from_compact_summary,
    split_event_calibration,
    summarize_platform_contract,
)
from experiments.g2_reset_v1.certified_gate import (
    RecursiveSelectivePlanner,
    active_platform_rides,
    choose_certified,
    should_invoke_rollout,
)
from experiments.decision_calibration_v2.calibration_summary import summarize as legacy_summarize
from types import SimpleNamespace
from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from src.evolving_trip_belief import _ChainLaw
from src.transit_graph import Connection, Stop, TransitGraph


def test_protocol_v2_json_is_complete():
    path = Path(__file__).resolve().parents[1] / "experiments/g2_reset_v1/protocol_v2.json"
    proto = json.loads(path.read_text(encoding="utf-8"))
    assert proto["frozen_history_policy"]["historical_receipts_immutable"] is True
    assert proto["frozen_history_policy"]["original_72_additional_tuning_budget"] == 0
    assert proto["gate_g1_contract"]["execution_latency_contract"]["online_dwell_budget_cpu_seconds"] <= 2.0
    assert "center_A_selective_certified_recourse" in proto["gate_g2_method_and_value"]["allowable_research_centers"]
    assert "center_B_empirical_attribution_and_contract_design" in proto["gate_g2_method_and_value"]["allowable_research_centers"]


def test_algebraic_compact_extraction_matches_event_level_split_exactly():
    rng = np.random.default_rng(42)
    events = []
    manifest_days = []
    for d_idx in range(14):
        day = f"2023-05-{d_idx + 1:02d}"
        n_opp = 12
        for e_idx in range(n_opp):
            canceled_init = d_idx in (12, 13) and e_idx >= 9
            if canceled_init:
                events.append(
                    {
                        "day": day,
                        "root_index": d_idx * 2 + (1 if e_idx >= 9 else 0),
                        "event_index": e_idx,
                        "incoming_trip": f"in_{d_idx}_{1 if e_idx >= 9 else 0}",
                        "outgoing_trip": f"out_{d_idx}_{1 if e_idx >= 9 else 0}",
                        "initial_state": {"status": "service_or_clock_failure"},
                        "model": {"status": "probability_unresolved", "reason": "initial_history_unavailable"},
                        "actual": {
                            "status": "label_unresolved",
                            "possible_partitions": list(PARTITIONS),
                            "reason": "initial_history_unavailable",
                        },
                        "groups": {"forecast_age": "missing_initial_history", "mct_difference": "over_180s", "downstream_options": "4"},
                    }
                )
            else:
                raw = rng.dirichlet([1.0, 1.0, 1.0, 7.0])
                part = rng.choice(PARTITIONS, p=[0.03, 0.02, 0.12, 0.83])
                events.append(
                    {
                        "day": day,
                        "root_index": d_idx * 2,
                        "event_index": e_idx,
                        "incoming_trip": f"in_{d_idx}_0",
                        "outgoing_trip": f"out_{d_idx}_0",
                        "initial_state": {"status": "observed"},
                        "model": {"status": "resolved", "partitions": {k: float(raw[i]) for i, k in enumerate(PARTITIONS)}},
                        "actual": {"status": "resolved", "actual_partition": str(part), "possible_partitions": [str(part)]},
                        "groups": {"forecast_age": "at_most_300s", "mct_difference": "over_180s", "downstream_options": "4"},
                    }
                )
        manifest_days.append(
            {"day": day, "available": True, "denominator_known": True, "expected_opportunities": n_opp}
        )

    compact = legacy_summarize(events, {"selected_dates": [m["day"] for m in manifest_days], "days": manifest_days})
    direct = split_event_calibration(events, seed=20261115, replicates=2000)
    extracted = extract_conditional_from_compact_summary(compact, seed=20261115, replicates=2000)

    assert direct["service_availability"]["unreached_opportunities"] == 6
    assert extracted["unreached_initial_cancellation_opportunities"] == 6
    assert extracted["conditional_reached_opportunities"] == direct["conditional_onboard"]["reached_resolved_opportunities"]
    assert np.isclose(
        extracted["conditional_mean_brier_4class"],
        direct["conditional_onboard"]["mean_brier_4class"],
        atol=1e-12,
    )
    for k in PARTITIONS:
        assert np.isclose(
            extracted["date_cluster_inference"]["mean_residual"][k],
            direct["conditional_onboard"]["date_cluster_inference"]["mean_residual"][k],
            atol=1e-12,
        )
        assert np.allclose(
            extracted["date_cluster_inference"]["ci95_bonferroni_4class"][k],
            direct["conditional_onboard"]["date_cluster_inference"]["ci95_bonferroni_4class"][k],
            atol=1e-12,
        )


def test_platform_pair_contract_precedence():
    same_plat = {
        "role": "downstream",
        "fixed_mct_seconds": 420,
        "stated_pair_seconds": None,
        "arrival": {"stop_id": "8503000:0:41/42", "platform_specific": True},
        "departure": {"stop_id": "8503000:0:41/42", "platform_specific": True},
    }
    explicit_pair = {
        "role": "upstream",
        "fixed_mct_seconds": 240,
        "stated_pair_seconds": 180,
        "arrival": {"stop_id": "8503006:0:3", "platform_specific": True},
        "departure": {"stop_id": "8503006:0:6", "platform_specific": True},
    }
    unknown_pair = {
        "role": "downstream",
        "fixed_mct_seconds": 420,
        "stated_pair_seconds": None,
        "arrival": {"stop_id": "8503000", "platform_specific": False},
        "departure": {"stop_id": "8503000:0:5", "platform_specific": True},
    }
    r1 = classify_platform_pair_contract(same_plat, tau_same_platform=90)
    assert r1["rule"] == "same_physical_platform"
    assert r1["contract_mct_seconds"] == 90
    assert r1["mct_delta_seconds"] == -330

    r2 = classify_platform_pair_contract(explicit_pair, tau_same_platform=90)
    assert r2["rule"] == "explicit_gtfs_transfers_txt"
    assert r2["contract_mct_seconds"] == 180

    r3 = classify_platform_pair_contract(unknown_pair, tau_same_platform=90)
    assert r3["rule"] == "parent_station_fallback"
    assert r3["contract_mct_seconds"] == 420

    summary = summarize_platform_contract([same_plat, explicit_pair, unknown_pair], tau_same_platform=90)
    assert summary["total_pairs"] == 3
    assert np.isclose(summary["platform_resolved_fraction"], 2.0 / 3.0)


def test_ambiguity_margin_and_certified_selector_reject_mc_noise():
    # Clear 436s dominance should not invoke rollout
    invoke, info = should_invoke_rollout(
        {"alight": 29693.75, "continuation": 30130.33},
        deadline=32000.0,
        ambiguity_margin_seconds=120.0,
    )
    assert invoke is False
    assert info["reason"] == "analytical_margin_exceeds_threshold"

    # Close 4.0s competition should invoke rollout
    invoke2, _ = should_invoke_rollout(
        {"alight": 30384.88, "continuation": 30388.94},
        deadline=32000.0,
        ambiguity_margin_seconds=120.0,
    )
    assert invoke2 is True

    # Uncertified MC coin-flip (modeled on CI1 Root 4: mean=-2.34, se=16.89, base='continue')
    noisy_gate = {
        "paired": {
            "unresolved_pairs": 0,
            "full_bank_mean_gain_interval": [-2.34375, -2.34375],
            "complete_pair_statistics": {"mean_gain": -2.34375, "se": 16.8942},
        }
    }
    chosen, meta = choose_certified(noisy_gate, "continue", z_alpha=1.96, min_effect_seconds=5.0)
    assert chosen == "continue"
    assert meta["certified_override"] is False

    # Certified improvement (modeled on CI1 Root 62: mean=-28.59, se=7.47, base='continue')
    strong_gate = {
        "paired": {
            "unresolved_pairs": 0,
            "full_bank_mean_gain_interval": [-28.59375, -28.59375],
            "complete_pair_statistics": {"mean_gain": -28.59375, "se": 7.4694},
        }
    }
    chosen2, meta2 = choose_certified(strong_gate, "continue", z_alpha=1.96, min_effect_seconds=5.0)
    assert chosen2 == "alight"
    assert meta2["certified_override"] is True


class _HistoryStub:
    def __init__(self, laws, blocked=(), revision=0):
        self.laws, self.blocked, self.revision = laws, set(blocked), revision
        self._physical_bases = laws

    def law(self, trip):
        return self.laws[trip]

    def signature(self, relevant):
        return tuple((trip, self.revision, tuple(sorted(self.blocked))) for trip in sorted(relevant))

    def announced_unavailable(self, cid, kind):
        return (cid, kind) in self.blocked


def _make_point_law(departures, arrivals, probs):
    times = [np.array(departures, dtype=float), np.array(arrivals, dtype=float)]
    order = np.argsort(times[1])
    times[1] = times[1][order]
    matrix = np.eye(len(departures))[:, order]
    with np.errstate(divide="ignore"):
        transition = np.log(matrix)
    return _ChainLaw(
        times,
        np.log(np.array(probs, dtype=float)),
        lambda slot, m=transition: m,
        [np.zeros(len(t)) for t in times],
        [np.zeros(len(t), bool) for t in times],
        (True,),
    )


def test_recursive_selective_planner_and_active_pruning_on_two_transfer_menu():
    # Construct a 4-stop synthetic 2-transfer network:
    # Stop 0 -> Stop 1 (root ride c0: 0 -> 100)
    # Stop 1 -> Stop 2 (feeder ride c1: 180 -> 220) or slow direct to Goal Stop 3 (c2: 200 -> 700)
    # At Stop 2 (second transfer!):
    #   - c3 departs at 300, arrives Goal at 900 (first offer, poor!)
    #   - c4 departs at 330, arrives Goal at 420 (selective wait saves 480s!)
    #   - c5 departed at 50, before entry=220 (already gone prefix!)
    graph = TransitGraph()
    for s_id, mct in ((0, 0), (1, 60), (2, 60), (3, 0)):
        graph.add_stop(Stop(s_id, f"S{s_id}", min_transfer_time=mct))
    specs = [
        (0, "R0", "T0", 0, 1, [0.0], [100.0], [1.0]),
        (1, "R1", "T1", 1, 2, [180.0], [220.0], [1.0]),
        (2, "R2", "T_slow_direct", 1, 3, [200.0], [700.0], [1.0]),
        (3, "R3", "T2_early_slow", 2, 3, [300.0], [900.0], [1.0]),
        (4, "R4", "T2_fast", 2, 3, [330.0], [420.0], [1.0]),
        (5, "R5", "T2_already_gone", 2, 3, [50.0], [150.0], [1.0]),
    ]
    positions, laws = {}, {}
    for cid, route, trip, dep_s, arr_s, deps, arrs, probs in specs:
        graph.add_connection(Connection(cid, route, trip, dep_s, arr_s, int(deps[0]), int(arrs[0])))
        laws[trip] = _make_point_law(deps, arrs, probs)
        positions[cid, "departure"], positions[cid, "arrival"] = (trip, 0), (trip, 1)

    model = SimpleNamespace(_positions=positions)
    history = _HistoryStub(laws)
    deadline = 1200.0

    plat2 = Platform(
        2,
        {
            3: Ride(3, (3,), {3: None}),
            4: Ride(4, (4,), {4: None}),
            5: Ride(5, (5,), {5: None}),
        },
    )
    pruned = active_platform_rides(model, plat2, history, entry=220.0, deadline=deadline)
    assert set(pruned.rides) == {3, 4}

    plat1 = Platform(
        1,
        {
            1: Ride(1, (1,), {1: plat2}),
            2: Ride(2, (2,), {2: None}),
        },
    )
    root = Ride(0, (0,), {0: plat1})

    shallow_planner = MatchedContinuationPlanner(graph, model, {}, deadline)
    recursive_planner = RecursiveSelectivePlanner(
        graph, model, {}, deadline, exact_combined_max_width=4
    )

    shallow_val = shallow_planner.compile_ride(root, history, "rolling_fixed").policy.root_value
    recursive_val = recursive_planner.compile_ride(root, history, "rolling_fixed").policy.root_value

    # MatchedContinuationPlanner's depth-1 PhysicalPlanner tail_base takes first_offer at plat2 (900),
    # so plat1 picks c2 (700). RecursiveSelectivePlanner prices selective waiting at plat2 (420)!
    assert shallow_val == pytest.approx(700.0)
    assert recursive_val == pytest.approx(420.0)
    assert recursive_planner.stats["active_pruned_rides"] > 0
