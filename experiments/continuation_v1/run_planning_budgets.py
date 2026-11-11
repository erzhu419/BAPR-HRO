"""Paired work-budget curve for full and fixed-menu causal planning."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.continuation_v1.run_candidate_cross import PANELS, paired, write_json
from experiments.continuation_v1.run_matched import run_one
from src.causal_planning_router import CausalPlanningRouter

VERSION = "committed_causal_planning_work_budget_v2"
PANEL = {"primary": (2, 8, 14, 20), "independent_generated": (0, 3)}
METHODS = {"full_rollout": {"menu": "full", "algorithm": "rollout"},
           "fixed_rollout": {"menu": "fixed", "algorithm": "rollout"},
           "material_h_rollout": {"menu": "fixed_top5", "algorithm": "rollout"},
           "full_uct": {"menu": "full", "algorithm": "uct"}}


def run_case(panel, index, method, budget, samples):
    archive, _ = PANELS[panel]
    row = run_one(index, method, samples, archive, router_cls=CausalPlanningRouter,
                  router_options=METHODS[method] | {"work_budget": budget}, implementation=VERSION)
    queries = row["predictions"]
    used = [q["work"] for q in queries]
    if any(work > budget for work in used):
        raise ValueError("A query exceeded the committed-transition budget")
    return {"spec": row["spec"], "archive_case": row["archive_case"], "loss": row["loss"],
            "arrived": row["arrived"], "total_ms": row["total_ms"], "planning_ms": row["rollout_ms"],
            "initial_distribution_and_scan_ms": row["estimate_ms"] + sum(row["query_ms"]),
            "peak_process_kib_cumulative": row["peak_process_kib_cumulative"],
            "queries": len(queries), "transitions": sum(used), "max_query_transitions": max(used, default=0),
            "p50_query_ms": float(np.median([q["planning_ms"] for q in queries])) if queries else None,
            "p95_query_ms": float(np.quantile([q["planning_ms"] for q in queries], .95)) if queries else None,
            "max_root_actions": max((q["root_actions"] for q in queries), default=0),
            "max_scored_root_actions": max((q["scored_root_actions"] for q in queries), default=0),
            "queries_without_complete_root_score": sum(q["value"] is None for q in queries),
            "common_rounds": sum(q["complete_common_rounds"] for q in queries),
            "completed_rollouts": sum(q["completed_rollouts"] for q in queries),
            "tree_nodes": sum(q["tree_nodes"] for q in queries),
            "menu_size": queries[0]["menu_size"] if queries else None}


def summarize(rows, budgets, method_names):
    result = {}
    for budget in budgets:
        methods = {}
        by_case = {}
        for method in method_names:
            group = [row for row in rows if row["work_budget"] == budget and row["method"] == method]
            complete = [row for row in group if row["status"] == "complete"]
            for row in complete:
                by_case.setdefault((row["panel"], row["index"]), {})[method] = row["loss"]
            methods[method] = {"attempted": len(group), "complete": len(complete),
                               "mean_loss": mean(row["loss"] for row in complete) if complete else None,
                               "arrival_rate": mean(row["arrived"] for row in complete) if complete else None,
                               "median_total_ms": median(row["total_ms"] for row in complete) if complete else None,
                               "mean_transitions": mean(row["transitions"] for row in complete) if complete else None,
                               "paired_minus_nominal": paired([row["loss"] - row["nominal_loss"] for row in complete]),
                               "paired_minus_v2_rollout": paired([row["loss"] - row["v2_rollout_loss"] for row in complete])}
        complete = [losses for losses in by_case.values() if losses.keys() == set(method_names)]
        comparisons = {label: paired([r[method] - r["full_rollout"] for r in complete])
                       for method, label in (("fixed_rollout", "fixed_minus_full"),
                                             ("material_h_rollout", "material_h_minus_full"),
                                             ("full_uct", "uct_minus_full_rollout"))
                       if method in method_names and "full_rollout" in method_names}
        result[str(budget)] = {"methods": methods, "complete_worlds": len(complete)} | comparisons
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS),
                        default=["full_rollout", "fixed_rollout", "material_h_rollout"])
    parser.add_argument("--output", type=Path, default=HERE / "planning_budget_material_h_results.json")
    args = parser.parse_args()
    if min(*args.budgets, args.samples, args.workers) < 1:
        parser.error("budgets, samples and workers must be positive")
    budgets = list(dict.fromkeys(args.budgets))
    method_names = list(dict.fromkeys(args.methods))
    baseline = json.loads((HERE / "nominal_baseline_final_results.json").read_text())
    reference = {(row["panel"], row["index"]): row for row in baseline["cases"]}
    directory = args.output.with_name(args.output.stem + "_cases")
    directory.mkdir(parents=True, exist_ok=True)
    rows, pending = [], []
    started = perf_counter()
    for panel, indices in PANEL.items():
        for index in indices:
            old = reference[panel, index]
            for method in method_names:
                for budget in budgets:
                    path = directory / f"{panel}_{index:03d}_{method}_{budget}.json"
                    metadata = {"implementation": VERSION, "panel": panel, "index": index, "method": method,
                                "work_budget": budget, "samples": args.samples, "spec": old["spec"],
                                "archive_case": old["archive_case"], "nominal_loss": old["loss"],
                                "v2_rollout_loss": old["reference_losses"]["repaired_refresh_rescue"]}
                    if path.exists():
                        row = json.loads(path.read_text())
                        if any(row.get(k) != v for k, v in metadata.items()):
                            raise ValueError(f"Use a new output path for this configuration: {path}")
                        rows.append(row)
                    else:
                        pending.append((path, metadata))
    total = len(rows) + len(pending)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_case, m["panel"], m["index"], m["method"], m["work_budget"], m["samples"]): (p, m)
                for p, m in pending}
        for future in as_completed(jobs):
            path, metadata = jobs[future]
            try:
                result = future.result()
                if (result["spec"] != metadata["spec"]
                        or Path(result["archive_case"]).resolve() != (ROOT / metadata["archive_case"]).resolve()):
                    raise ValueError("The frozen world differs from its archived comparison")
                row = result | metadata | {"status": "complete"}
            except Exception as error:
                row = metadata | {"status": "error", "error": f"{type(error).__name__}: {error}"}
            write_json(path, row)
            rows.append(row)
            print(json.dumps({"complete": len(rows), "total": total, "panel": row["panel"], "case": row["index"],
                              "method": row["method"], "budget": row["work_budget"], "loss": row.get("loss"),
                              "error": row.get("error")}), flush=True)
    rows.sort(key=lambda row: (row["panel"], row["index"], row["work_budget"], row["method"]))
    errors = sum(row["status"] != "complete" for row in rows)
    write_json(args.output, {"implementation": VERSION, "panel": PANEL,
                             "scope": "Selected archived generated worlds on paired dates, ODs and stress conditions",
                             "budget": "Maximum committed model transition calls per online decision, including partial rollouts. "
                                       "Model setup, nominal path search and full elapsed time are measured separately from this limit.",
                             "config": {"budgets": budgets, "samples": args.samples, "workers": args.workers,
                                        "methods": {name: METHODS[name] for name in method_names},
                                        "timing_build": "deterministic PMF reachability cache", "tuning": "none"},
                             "elapsed_seconds": perf_counter() - started, "execution_errors": errors,
                             "summary": {"all": summarize(rows, budgets, method_names),
                                         "by_panel": {p: summarize([r for r in rows if r["panel"] == p], budgets, method_names)
                                                      for p in PANEL},
                                         "by_stress": {str(s): summarize([r for r in rows if r["spec"][1] == s], budgets, method_names)
                                                       for s in (False, True)}},
                             "cases": rows}, indent=2)
    print(json.dumps({"output": str(args.output), "cases": len(rows), "execution_errors": errors}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
