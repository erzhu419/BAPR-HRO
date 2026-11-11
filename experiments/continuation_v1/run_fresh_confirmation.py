"""Evaluate the frozen policies on two new parameter dates and OD pairs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import pickle
from statistics import mean
import sys
from tempfile import TemporaryDirectory

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.continuation_v1.run_candidate_cross import paired, write_json
from experiments.continuation_v1.run_matched import run_one
from experiments.continuation_v1.run_nominal_baseline import NominalBaseline
from experiments.reconstruction_v2.run_matched_panel import historical_priors, world_distributions
from src.continuation_router import ContinuationRouter
from src.simulator import JourneyScenario, RegimeSchedule, set_regime_dist_fn


DATES = (("2023-10-12", 1327), ("2023-10-25", 4219))
ODS = (("Zürich, Bellevue", "Zürich, Triemli"),
       ("Zürich, Central", "Zürich Wiedikon, Bahnhof"))
VERSION = "continuation_v2_fresh_generated_branch_complete"


def freeze_cases(directory):
    with (ROOT / "data/zurich_wide.pkl").open("rb") as handle:
        graph = pickle.load(handle)
    with (ROOT / "data/per_day_distributions.pkl").open("rb") as handle:
        days = pickle.load(handle)
    routes = {c.route for c in graph.connections}
    index = 0
    for date, seed in DATES:
        priors = historical_priors(days, date, routes)
        for stress in (False, True):
            distributions = world_distributions(days[date], routes, stress)
            set_regime_dist_fn(lambda regime: distributions)
            scene = JourneyScenario.sample(graph, RegimeSchedule([(0, "confirmation")]),
                                           np.random.default_rng(seed))
            world = {cid: asdict(event) for cid, event in scene.events.items()}
            for origin, destination in ODS:
                case = {"spec": [date, stress, origin, destination, seed, 120, 5],
                        "world": world, "priors": priors, "records": []}
                write_json(directory / f"case_{index:03d}.json", case)
                index += 1
    return index


def run_task(task):
    index, method, archive = task
    policies = {"nominal": NominalBaseline, "rollout": ContinuationRouter}
    options = ({"router_cls": policies[method], "router_options": {}, "implementation": VERSION}
               if method in policies else {})
    row = run_one(index, method, 8, archive, **options)
    return {key: row[key] for key in ("index", "variant", "spec", "loss", "arrived", "total_ms",
                                       "rollout_ms", "scans", "rollout_steps")}


def summarize(rows):
    by_case = {}
    for row in rows:
        by_case.setdefault(row["index"], {})[row["variant"]] = row
    methods = {method: {"mean_loss": mean(row["loss"] for row in rows if row["variant"] == method),
                        "arrival_rate": mean(row["arrived"] for row in rows if row["variant"] == method),
                        "mean_total_ms": mean(row["total_ms"] for row in rows if row["variant"] == method)}
               for method in ("legacy", "nominal", "rollout")}
    return {"methods": methods, "rollout_minus_nominal": paired(
        [case["rollout"]["loss"] - case["nominal"]["loss"] for case in by_case.values()])}


def main():
    with TemporaryDirectory(prefix="bapr-continuation-confirm-") as temporary:
        archive = Path(temporary)
        count = freeze_cases(archive)
        tasks = [(i, method, archive) for i in range(count)
                 for method in ("legacy", "nominal", "rollout")]
        with ProcessPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(run_task, tasks))
    summary = {"all": summarize(rows)}
    for stress, group in ((False, "no_cancel"), (True, "stress")):
        summary[group] = summarize([row for row in rows if row["spec"][1] == stress])
    path = HERE / "fresh_confirmation_final_results.json"
    write_json(path, {"implementation": VERSION, "dates_and_seeds": DATES, "ods": ODS,
                      "samples": 8, "tuning": "none", "od_cases": count,
                      "scope": "New generated dates, worlds and OD pairs with frozen policies",
                      "summary": summary, "rows": rows}, indent=2)
    print(json.dumps({"output": str(path), "journeys": len(rows), "summary": summary}))


if __name__ == "__main__":
    main()
