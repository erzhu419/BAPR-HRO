"""Regression tests for the observed preparation and single-world budget failures."""

from time import process_time

from experiments.trc_revision_v1 import rollout
from experiments.bounded_gate_5d_v1.pair_contract import (
    decorate_menu_with_mct, evaluate_pair_rollout as historical_rollout,
    PairSnapshotExecutor as HistoricalSnapshotExecutor,
)
from experiments.reconstruction_v50.execute import ObservationContract
from test_v48_frontier import forced_case


def inputs(monkeypatch):
    graph, model, params, history, node, row = forced_case(monkeypatch)
    node = decorate_menu_with_mct(node, graph, mct_contract="station_max")
    return (graph, model, params, 100.0, node, row, history, ObservationContract.from_menu(node))


def test_preparation_is_inside_allowance(monkeypatch):
    args = inputs(monkeypatch)

    def stalled_preparation(*unused):
        while True:
            pass

    monkeypatch.setattr(rollout, "freeze_snapshot", stalled_preparation)
    start = process_time()
    result = rollout.evaluate_pair_rollout(*args, root_index=0, count=8, max_process_cpu=.01)
    assert process_time() - start < .5
    assert result["budget"]["stopped_phase"] == "preparation"
    assert result["completed_pairs"] == 0
    assert all(r["status"] == "cpu_unresolved" for rows in result["rows"].values() for r in rows)


def test_interrupted_pair_does_not_commit_one_arm(monkeypatch):
    args = inputs(monkeypatch)

    def interrupted(self, node, row, arm, history, world):
        if arm == "continue":
            raise rollout.QueryCPUExpired
        return dict(loss=5.0, status="arrived", decisions={})

    monkeypatch.setattr(rollout.PairSnapshotExecutor, "forced", interrupted)
    result = rollout.evaluate_pair_rollout(*args, root_index=0, count=8, max_process_cpu=2)
    assert result["completed_pairs"] == 0
    assert result["paired"]["unresolved_pairs"] == 8
    assert all(r["loss"] is None for rows in result["rows"].values() for r in rows)
    choice, screen = rollout.choose_paired_screen(result, "continue")
    assert choice == "continue" and not screen["override"]


def test_uninterrupted_bank_preserves_historical_values(monkeypatch):
    # Isolate the budget change. The development command policy is separately
    # tested and intentionally differs from the historical waiting policy.
    monkeypatch.setattr(rollout, "PairSnapshotExecutor", HistoricalSnapshotExecutor)
    args = inputs(monkeypatch)
    old = historical_rollout(*args, root_index=0, count=8, max_process_cpu=60, seed=12345)
    new = rollout.evaluate_pair_rollout(*args, root_index=0, count=8, max_process_cpu=60, seed=12345)
    assert new["completed_pairs"] == 8
    for arm in ("alight", "continue"):
        assert new["rows"][arm] == old["rows"][arm]
    assert new["paired"] == old["paired"]


def test_zero_allowance_does_not_prepare(monkeypatch):
    args = inputs(monkeypatch)
    def forbidden(*unused):
        raise AssertionError("No residual budget must not prepare a query")
    monkeypatch.setattr(rollout, "freeze_snapshot", forbidden)
    result = rollout.evaluate_pair_rollout(*args, root_index=0, count=8, max_process_cpu=0)
    assert result["completed_pairs"] == 0
    assert result["budget"]["stopped_phase"] == "not_started"
