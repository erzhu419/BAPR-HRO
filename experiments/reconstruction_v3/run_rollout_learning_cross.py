"""Post-hoc route-mean and cancellation learning cross in the final causal planner."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.reconstruction_v3 import run_chronological_panel as panel
from src.causal_planning_router import CausalPlanningRouter


VERSION = "causal_full_rollout_learning_cross_v1"
METHODS = {f"mean{m}_cancel{c}": (bool(m), bool(c)) for m in (0, 1) for c in (0, 1)}
SAMPLES, BUDGET = 8, 512


class LearningRouter(CausalPlanningRouter):
    """Change fitted route updates while retaining every public vehicle fact."""

    def __init__(self, graph, priors, *, online_mean, online_cancel, **kwargs):
        super().__init__(graph, priors, menu="full", algorithm="rollout", **kwargs)
        self.online_mean, self.online_cancel = online_mean, online_cancel
        self.baseline_parameters = {route: self._route_parameters(route) for route in self._routes}
        self.parameter_trace = []

    def _parameters(self):
        parameters = super()._parameters()
        mean_delta = max(abs(row["mean"] - self.baseline_parameters[route]["mean"])
                         for route, row in parameters.items())
        cancel_delta = max(abs(row["cancel_rate"] - self.baseline_parameters[route]["cancel_rate"])
                           for route, row in parameters.items())
        self.parameter_trace.append({"time": self.information_state.time,
                                     "max_abs_mean_update": mean_delta,
                                     "max_abs_cancel_update": cancel_delta})
        return parameters


def run_case(spec):
    routers = {}

    def factory(method, graph, priors, cap, seed):
        update_mean, update_cancel = METHODS[method]
        router = LearningRouter(graph, priors, online_mean=update_mean, online_cancel=update_cancel,
                                max_time=cap, samples=SAMPLES, work_budget=BUDGET, seed=seed + 41000)
        routers[method] = router
        return router

    case = panel.run_case(spec, router_factory=factory)
    case["implementation"], case["transition_budget"] = VERSION, BUDGET
    for row in case["rows"]:
        router = routers[row["method"]]
        row["parameter_trace"] = router.parameter_trace
        row["planning_queries"] = len(router.predictions)
        row["complete_common_rounds"] = sum(p["complete_common_rounds"] for p in router.predictions)
        row["zero_complete_round_queries"] = sum(not p["complete_common_rounds"] for p in router.predictions)
        row["max_abs_mean_update"] = max((p["max_abs_mean_update"] for p in router.parameter_trace), default=0.)
        row["max_abs_cancel_update"] = max((p["max_abs_cancel_update"] for p in router.parameter_trace), default=0.)
    return case


def summarize(cases):
    summaries = {}
    for stress in (False, True):
        group = [case for case in cases if case["spec"][2] == stress]
        metrics = {}
        for method in METHODS:
            rows = [r for case in group for r in case["rows"] if r["method"] == method]
            metrics[method] = {key: mean(r[key] for r in rows) for key in (
                "loss", "arrived", "total_ms", "planning_transitions", "planning_queries",
                "complete_common_rounds", "zero_complete_round_queries")}
            metrics[method]["max_query_transitions"] = max(r["max_query_transitions"] for r in rows)
            for factor in ("mean", "cancel"):
                metrics[method][f"journeys_with_{factor}_update"] = sum(
                    r[f"max_abs_{factor}_update"] > 0 for r in rows)
        effects = {}
        contrasts = {"mean_on_minus_off": {m: .5 if flags[0] else -.5 for m, flags in METHODS.items()},
                     "cancel_on_minus_off": {m: .5 if flags[1] else -.5 for m, flags in METHODS.items()},
                     "mean_by_cancel_interaction": {m: 1 if flags[0] == flags[1] else -1
                                                    for m, flags in METHODS.items()}}
        for name, weights in contrasts.items():
            differences, dates = [], {}
            for case in group:
                loss = {r["method"]: r["loss"] for r in case["rows"]}
                delta = sum(weights[m] * loss[m] for m in METHODS)
                differences.append(delta)
                dates.setdefault(case["spec"][0], []).append(delta)
            effects[name] = {"mean_difference": mean(differences),
                             "negative": sum(d < 0 for d in differences),
                             "positive": sum(d > 0 for d in differences),
                             "ties": sum(d == 0 for d in differences),
                             "by_date": {date: mean(d) for date, d in dates.items()}}
        path_changes = {}
        for factor, pairs in {"mean": [("mean0_cancel0", "mean1_cancel0"), ("mean0_cancel1", "mean1_cancel1")],
                              "cancel": [("mean0_cancel0", "mean0_cancel1"), ("mean1_cancel0", "mean1_cancel1")]}.items():
            differing = 0
            for case in group:
                paths = {r["method"]: [(d["time"], d["connection_id"]) for d in r["decisions"]]
                         for r in case["rows"]}
                differing += sum(paths[left] != paths[right] for left, right in pairs)
            path_changes[factor] = {"different_paths": differing, "paired_paths": len(group) * 2}
        summaries[str(stress)] = {"od_conditions": len(group), "metrics": metrics,
                                  "factorial_effects": effects, "policy_path_changes": path_changes}
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path, default=HERE / "route_id_factorial_results_protocol.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=HERE / "causal_rollout_learning_factorial_results.json")
    args = parser.parse_args()
    source = json.loads(args.source_protocol.read_text())
    graph, _ = panel.load_network()
    actual = [[c.id, c.route, c.trip_id, c.dep_stop, c.arr_stop, c.dep_time, c.arr_time] for c in graph.connections]
    stops = [[sid, stop.name, stop.min_transfer_time] for sid, stop in graph.stops.items()]
    if source["connections"] != actual or source["stops"] != stops:
        parser.error("the network differs from the frozen source protocol")
    protocol = source | {"implementation": VERSION, "protocol": "post_hoc_final_planner_learning_cross",
                         "source_protocol": str(args.source_protocol.resolve()), "methods": list(METHODS),
                         "factorial_methods": {m: {"online_mean": flags[0], "online_cancel": flags[1]}
                                               for m, flags in METHODS.items()},
                         "workers": args.workers, "transition_budget": BUDGET, "samples": SAMPLES,
                         "menu": "full", "algorithm": "rollout", "conditioning": True,
                         "variance_penalty": 0, "predictive_std": "frozen chronological prior std",
                         "inference": "post-hoc diagnosis on previously inspected generated conditions"}
    protocol_path = panel.freeze_protocol(args.output, protocol)
    receipt_dir = args.output.with_name(args.output.stem + "_cases")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    cases, pending = [], []
    for i, spec in enumerate(source["specs"]):
        receipt = receipt_dir / f"case_{i:04}.json.gz"
        if receipt.exists():
            with gzip.open(receipt, "rt", encoding="utf-8") as handle:
                saved = json.load(handle)
            if (saved["spec"] != spec or saved["implementation"] != VERSION
                    or [r["method"] for r in saved["rows"]] != list(METHODS)):
                raise ValueError(f"Receipt differs from the frozen cross: {receipt}")
            cases.append(saved)
        else:
            pending.append((spec, receipt))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=panel.initialize,
                             initargs=(graph, source["ods"], source["world_inputs"], list(METHODS), BUDGET,
                                       source["data"])) as pool:
        jobs = {pool.submit(run_case, spec): receipt for spec, receipt in pending}
        for job in as_completed(jobs):
            case, receipt = job.result(), jobs[job]
            partial = receipt.with_suffix(".tmp")
            with gzip.open(partial, "wt", encoding="utf-8", compresslevel=3) as handle:
                json.dump(case, handle, ensure_ascii=False, allow_nan=False)
            partial.replace(receipt)
            cases.append(case)
            print(f"{len(cases)}/{len(source['specs'])} conditions complete", flush=True)
    cases.sort(key=lambda c: c["spec"])
    compact = [{**case, "rows": [{k: v for k, v in row.items() if k not in (
        "public_messages", "events", "queries_ms", "polls", "decisions", "attempts", "parameter_trace")}
        for row in case["rows"]]} for case in cases]
    panel.write_json(args.output, {"implementation": VERSION, "frozen_protocol": str(protocol_path),
        "inference": protocol["inference"], "od_conditions": len(cases), "journeys": len(cases) * len(METHODS),
        "samples": SAMPLES, "transition_budget": BUDGET, "summary": summarize(cases), "cases": compact}, indent=2)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
