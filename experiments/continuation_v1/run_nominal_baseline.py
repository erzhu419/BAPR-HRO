"""Compare the causal nominal policy directly with the repaired rollout."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.continuation_v1.run_candidate_cross import PANELS, paired, write_json
from experiments.continuation_v1.run_matched import run_one
from src.committed_transit import WAIT
from src.continuation_router import ContinuationRouter, nominal_action
from src.transit_graph import StopLabel

VERSION = "continuation_v2_nominal_baseline"


class NominalBaseline(ContinuationRouter):
    """Use the rollout's nominal tail at every real decision."""

    def select_connection(self, current_stop, current_time, rng, top_k=5):
        if self._valued_version != self._version:
            self._value()
        started = perf_counter()
        state = self.information_state
        menu = self._fresh_menu
        labels = {label.connection_id: label
                  for label in reversed(self.cached_result.stop_labels.get(state.stop, []))}
        parameters = self._parameters()
        board = self._boarding_candidates(state, menu, labels, parameters)
        action = WAIT
        if board:
            nominal = self.trip_belief.mean_events(parameters)
            action = nominal_action(self.graph, state, self.destination, nominal, menu,
                                    self.journey_deadline, self.lookahead,
                                    root_boardings=set(board))
        cid = None if action == WAIT else action[1]
        self.predictions.append({"connection": cid,
                                 "nominal_ms": (perf_counter() - started) * 1000})
        if cid is None:
            return None
        label = labels.get(cid)
        if label is None:
            connection = self.graph.connections[cid]
            label = StopLabel(cid, connection.arr_distribution, connection.arr_time, 0.0)
        return label, nominal[cid].arrival


def run_case(reference):
    panel, index = reference["panel"], reference["index"]
    archive, _ = PANELS[panel]
    row = run_one(index, "nominal", reference["samples"], archive, router_cls=NominalBaseline,
                  router_options={"candidate_mode": "refresh", "conditioning": True},
                  implementation=VERSION)
    if (Path(row["archive_case"]).resolve() != (ROOT / reference["archive_case"]).resolve()
            or row["spec"] != reference["spec"]):
        raise ValueError("Nominal world does not match its rollout reference")
    return {"loss": row["loss"], "arrived": row["arrived"], "total_ms": row["total_ms"],
            "nominal_ms": sum(p["nominal_ms"] for p in row["predictions"]),
            "decisions": len(row["predictions"]),
            "wait_decisions": sum(p["connection"] is None for p in row["predictions"]),
            "scans": row["scans"], "scan_ms": sum(row["query_ms"]),
            "rollout_steps": row["rollout_steps"]}


def summarize_group(rows):
    complete = [row for row in rows if row["status"] == "complete"]
    differences = {}
    for key in ("repaired_refresh_rescue", "archived_v1_conditioned_rescue", "archived_legacy"):
        differences[key] = paired([row["loss"] - row["reference_losses"][key]
                                   for row in complete if row["reference_losses"][key] is not None])
    timing_pairs = [row for row in complete if row["repaired_total_ms"] is not None]
    return {"attempted": len(rows), "complete": len(complete), "errors": len(rows) - len(complete),
            "mean_capped_loss": mean(row["loss"] for row in complete) if complete else None,
            "arrival_rate": mean(row["arrived"] for row in complete) if complete else None,
            "median_total_ms": median(row["total_ms"] for row in complete) if complete else None,
            "median_nominal_ms": median(row["nominal_ms"] for row in complete) if complete else None,
            "paired_minus_reference": differences,
            "median_repaired_to_nominal_runtime_ratio": median(
                row["repaired_total_ms"] / row["total_ms"] for row in timing_pairs) if timing_pairs else None}


def summarize(rows):
    summary = {"all": summarize_group(rows), "panels": {}}
    for panel in PANELS:
        group = [row for row in rows if row["panel"] == panel]
        summary["panels"][panel] = {
            "all": summarize_group(group),
            "by_stress": {name: summarize_group([row for row in group if row["spec"][1] == stress])
                          for name, stress in (("no_cancel", False), ("stress", True))},
            "by_date": {date: summarize_group([row for row in group if row["spec"][0] == date])
                        for date in sorted({row["spec"][0] for row in group})},
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross", type=Path, default=HERE / "candidate_cross_results.json")
    parser.add_argument("--output", type=Path, default=HERE / "nominal_baseline_final_results.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.output.exists():
        parser.error("choose a new output path for a new baseline run")
    cross = json.loads(args.cross.read_text())
    references = [row for row in cross["cases"] if row["variant"] == "refresh_rescue"]
    expected = {("primary", index) for index in range(24)} | {
        ("independent_generated", index) for index in range(6)}
    if len(references) != len(expected) or {(row["panel"], row["index"]) for row in references} != expected:
        raise ValueError("The comparison must cover the same 30 frozen worlds")

    started = perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_case, reference): reference for reference in references}
        for future in as_completed(jobs):
            reference = jobs[future]
            archived = reference["references"]
            row = {"panel": reference["panel"], "index": reference["index"], "spec": reference["spec"],
                   "archive_case": reference["archive_case"],
                   "reference_losses": {
                       "repaired_refresh_rescue": reference.get("loss"),
                       "archived_v1_conditioned_rescue": archived.get("conditioned_rescue", {}).get("loss"),
                       "archived_legacy": archived.get("legacy", {}).get("loss"),
                   }, "repaired_total_ms": reference.get("total_ms")}
            try:
                row.update(future.result(), status="complete")
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
            rows.append(row)
            print(json.dumps({"complete": len(rows), "total": len(references), "panel": row["panel"],
                              "case": row["index"], "loss": row.get("loss"), "error": row.get("error")}),
                  flush=True)
    rows.sort(key=lambda row: (row["panel"], row["index"]))
    errors = sum(row["status"] == "error" for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"implementation": VERSION, "cross_receipt": str(args.cross.resolve()),
                             "nominal_policy": "Refresh menu, shared first-boarding screen and top-five limit, "
                                               "causal nominal tail at every decision, zero sampled rollouts",
                             "runtime": {"workers": args.workers, "wall_seconds": perf_counter() - started,
                                         "repaired_reference_workers": cross["config"]["workers"]},
                             "execution_errors": errors, "summary": summarize(rows), "cases": rows}, indent=2)
    print(json.dumps({"output": str(args.output), "cases": len(rows), "execution_errors": errors}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
