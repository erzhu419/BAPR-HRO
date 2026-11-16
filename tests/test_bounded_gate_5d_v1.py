"""Unit tests for experiments/bounded_gate_5d_v1 (G1 PairPlatform MCT + Async Dwell + G2 Controllers)."""

import json
from pathlib import Path
from time import sleep

import numpy as np

from experiments.bounded_gate_5d_v1.pair_contract import (
    BoundedArmExecutor,
    PairDecisionEngine,
    PairFrozenCombinedReference,
    PairMatchedPlanner,
    PairMeanSnapshotCSA,
    PairPlatform,
    PairRecursivePlanner,
    PairSnapshotExecutor,
    decorate_menu_with_mct,
    evaluate_pair_rollout,
)
from experiments.reconstruction_v41.matched_continuation import (
    FrozenCombinedReference,
    MatchedContinuationPlanner,
)
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v51.baselines import MeanSnapshotCSA
from experiments.reconstruction_v51.rollout import StableTripWorlds, evaluate as evaluate_v51
from test_v46_shared import shared_case
from test_v48_frontier import forced_case

ROOT = Path(__file__).resolve().parents[1]


def test_station_max_exact_equivalence_with_legacy_v51(monkeypatch):
    """When mct_contract=='station_max', all Pair* classes match V41/V51 bit-for-bit."""
    graph, model, params, history, node, row = forced_case(monkeypatch)
    deadline = 100.0
    decorated = decorate_menu_with_mct(node, graph, mct_contract="station_max")

    old_planner = MatchedContinuationPlanner(graph, model, params, deadline)
    new_planner = PairMatchedPlanner(graph, model, params, deadline)

    old_comp = old_planner.compile_ride(node, history, "rolling_fixed")
    new_comp = new_planner.compile_ride(decorated, history, "rolling_fixed")
    assert old_comp.decision(row["cid"], row["clock"])[0] == new_comp.decision(row["cid"], row["clock"])[0]
    assert np.isclose(
        old_comp.decision(row["cid"], row["clock"])[1]["alight"],
        new_comp.decision(row["cid"], row["clock"])[1]["alight"],
        atol=1e-12,
    )
    assert np.isclose(
        old_comp.decision(row["cid"], row["clock"])[1]["continuation"],
        new_comp.decision(row["cid"], row["clock"])[1]["continuation"],
        atol=1e-12,
    )

    old_csa = MeanSnapshotCSA(graph, old_planner, deadline)
    new_csa = PairMeanSnapshotCSA(graph, new_planner, deadline)
    assert old_csa.onboard(node, row["cid"], row["clock"], history)[0] == new_csa.onboard(
        decorated, row["cid"], row["clock"], history
    )[0]
    assert np.isclose(
        old_csa.onboard(node, row["cid"], row["clock"], history)[1]["alight"],
        new_csa.onboard(decorated, row["cid"], row["clock"], history)[1]["alight"],
        atol=1e-12,
    )

    sight = ObservationContract.from_menu(node)
    old_roll = evaluate_v51(
        graph, model, params, deadline, node, row, history, sight, root_index=0, count=8, seed=12345
    )
    new_roll = evaluate_pair_rollout(
        graph,
        model,
        params,
        deadline,
        decorated,
        row,
        history,
        sight,
        root_index=0,
        count=8,
        seed=12345,
        max_process_cpu=60.0,
        rollout_planner_kind="matched",
    )
    assert [r["loss"] for r in old_roll["rows"]["alight"]] == [
        r["loss"] for r in new_roll["rows"]["alight"]
    ]
    assert [r["loss"] for r in old_roll["rows"]["continue"]] == [
        r["loss"] for r in new_roll["rows"]["continue"]
    ]
    assert np.isclose(
        old_roll["paired"]["complete_pair_statistics"]["mean_gain"],
        new_roll["paired"]["complete_pair_statistics"]["mean_gain"],
        atol=1e-12,
    )


def test_pair_frozen_combined_matches_legacy_on_uniform_mct(monkeypatch):
    graph, model, params, history, platform, entry = shared_case(monkeypatch, forecasts=True, cancel=0.05)
    deadline = 60.0
    mct = float(graph.stops[platform.stop].min_transfer_time)
    pair_plat = PairPlatform(
        platform.stop,
        platform.rides,
        incoming_cid=999,
        transfer_times={cid: mct for cid in platform.rides},
        default_mct=mct,
    )
    old_p = MatchedContinuationPlanner(graph, model, params, deadline)
    new_p = PairMatchedPlanner(graph, model, params, deadline)

    old_ref = old_p.platform_reference(platform, history, entry, entry + mct, "rolling_fixed")
    new_ref = new_p.platform_reference(pair_plat, history, entry, entry + mct, "rolling_fixed")

    old_fc = FrozenCombinedReference(old_ref)
    new_fc = PairFrozenCombinedReference(new_ref)
    assert np.isclose(old_fc.value, new_fc.value, atol=1e-12)
    assert np.isclose(old_fc.first_offer_value, new_fc.first_offer_value, atol=1e-12)


def test_heterogeneous_pair_mct_enforces_per_offer_readiness(monkeypatch):
    """Same-platform ride (mct=0) remains boardable while cross-platform ride (mct=50) is blocked."""
    graph, model, params, history, platform, entry = shared_case(monkeypatch, forecasts=False, cancel=0.0)
    deadline = 60.0
    cids = sorted(platform.rides)
    assert len(cids) >= 2
    same_cid, cross_cid = cids[0], cids[1]

    # Set cross_cid MCT high enough (50s) that it cannot be legally boarded before deadline
    transfer_times = {cid: (0.0 if cid == same_cid else 50.0) for cid in cids}
    pair_plat = PairPlatform(
        platform.stop,
        platform.rides,
        incoming_cid=777,
        transfer_times=transfer_times,
        default_mct=50.0,
    )
    planner = PairRecursivePlanner(graph, model, params, deadline, exact_combined_max_width=6)
    engine = PairDecisionEngine(planner, variant="frontier")
    sight = ObservationContract.from_menu(pair_plat)

    paths = [*pair_plat.paths(), *sight.paths()]
    bank = StableTripWorlds(history.as_belief(), params, paths, 32, 42, 0, model._trips)
    for w in range(32):
        from experiments.reconstruction_v41.physical_posterior import CheckingWorld

        world = CheckingWorld(bank, w, history.public_state())
        ex = PairSnapshotExecutor(
            graph,
            planner,
            deadline,
            engine=engine,
            method="rolling_fixed",
            waiting="combined",
            observation_contract=sight,
            trace=True,
        )
        arr, status = ex.platform(pair_plat, entry, history.clone(), world)
        for event in ex.trace:
            if event["kind"] == "platform" and event["choice"] is not None:
                chosen = event["choice"]
                assert event["clock"] >= entry + transfer_times[chosen]
                assert chosen == same_cid  # cross_cid requires entry + 50 >= 60 = deadline!


def test_async_dwell_clock_accepts_fast_and_rejects_expired_rollout(monkeypatch):
    """ActionPublisher accepts sub-dwell rollouts and rejects rollouts exceeding dwell window."""
    graph, model, params, history, node, row = forced_case(monkeypatch)
    deadline = 100.0
    decorated = decorate_menu_with_mct(node, graph, mct_contract="station_max")
    planner = PairRecursivePlanner(graph, model, params, deadline, exact_combined_max_width=6)

    arm_cfg = {
        "label": "bounded_cert_plat_async",
        "controller": "bounded_cert",
        "mct_contract": "platform_pair",
        "clock_contract": "async",
    }
    hp = {
        "bounded_cert": {
            "exact_combined_max_width": 6,
            "ambiguity_margin_seconds": 90.0,
            "worlds_per_query": 4,
            "max_process_cpu_seconds": 1.8,
            "z_alpha": 1.96,
            "min_effect_seconds": 5.0,
            "seed": 510015,
        }
    }
    ex = BoundedArmExecutor(
        graph,
        planner,
        deadline,
        arm_config=arm_cfg,
        hyperparams=hp,
        root_index=0,
        cpu_start=0.0,
        observation_contract=ObservationContract.from_menu(decorated),
    )

    # Case 1: Generous 30s dwell -> background compute finishes well before closes_at
    act_fast, pub_fast = ex._publish_action(
        opens_at=10.0,
        closes_at=40.0,
        legal=("alight", "continue"),
        fallback="continue",
        base_cpu=0.01,
        compute_fn=lambda: "alight",
        forced_on_total_expiry="continue",
    )
    assert act_fast == "alight"
    assert pub_fast["background_accepted"] is True

    # Case 2: Tight dwell (0.05s window) where background compute burns >0.06s CPU -> expired!
    def burn_cpu():
        x = np.arange(500000, dtype=np.float64)
        for _ in range(40):
            x = np.sin(x) + np.cos(x)
        assert x.sum() != 123456789.0
        return "alight"

    act_slow, pub_slow = ex._publish_action(
        opens_at=10.0,
        closes_at=10.02,
        legal=("alight", "continue"),
        fallback="continue",
        base_cpu=0.005,
        compute_fn=burn_cpu,
        forced_on_total_expiry="continue",
    )
    assert act_slow == "continue"
    assert pub_slow["background_accepted"] is False
    assert pub_slow["background_rejection_reason"] == "deadline_expired"


def test_bounded_gate_5d_protocol_schema():
    protocol = json.loads(
        (ROOT / "experiments/bounded_gate_5d_v1/protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["frozen_principles"]["original_72_additional_budget"] == 0
    assert protocol["frozen_principles"]["prior_exposed_dates_excluded_count"] == 72
    assert protocol["date_selection"]["target_days"] == 5
    assert protocol["date_selection"]["total_roots"] == 60
    assert len(protocol["arms"]) == 8
    assert protocol["verdict_rules"]["online_cpu_budget_seconds"] == 2.0
