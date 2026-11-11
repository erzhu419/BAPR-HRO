"""Small chronological, paired Swiss-topology simulation screening panel.

Historical daily delay moments calibrate a generated world; their invalid
delay-threshold cancellation proxy is not used. A separate, explicitly
synthetic cancellation stress case is included. No real-event replay claim.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter
import resource
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.matched_router import MatchedRouter
from src.native_router import NativeDurnerRouter
from src.simulator import JourneyScenario, RegimeSchedule, set_regime_dist_fn
from src.simulate_bandit import simulate_bandit_journey

ODS = [("Zürich, Paradeplatz", "Zürich, Goldbrunnenplatz"),
       ("Zürich, Stauffacher", "Zürich, Triemli"),
       ("Zürich Wiedikon, Bahnhof", "Zürich, Triemli")]
VARIANTS = {"offline": dict(), "mean": dict(online_mean=True),
            "mean_cancel": dict(online_mean=True, online_cancel=True),
            "mean_cancel_uncertainty": dict(online_mean=True, online_cancel=True, beta=1.5)}
IMPLEMENTATION = "causal_matched_v3_once_per_connection"


def historical_priors(days, date, routes):
    priors = {}
    for route in routes:
        past = [d[route] for key, d in days.items() if key < date and route in d]
        if not past:
            priors[route] = {"mean": 1., "std": 2., "cancel_rate": 0.}
            continue
        n = sum(d.get("n_obs", 1) for d in past)
        mean = sum(d["mean"] * d.get("n_obs", 1) for d in past) / n
        second = sum((d["std"] ** 2 + d["mean"] ** 2) * d.get("n_obs", 1) for d in past) / n
        priors[route] = {"mean": mean, "std": max(second - mean ** 2, .25) ** .5,
                         "cancel_rate": 0.}
    return priors


def world_distributions(day, routes, stress):
    result = {}
    for route in routes:
        d = day.get(route, {"mean": 1., "std": 2.})
        offsets = np.arange(-5, 65)
        mean, std = d["mean"] + (8 if stress else 0), max(d["std"], 4 if stress else .5)
        probs = np.exp(-.5 * ((offsets - mean) / std) ** 2)
        probs /= probs.sum()
        result[route] = {"delay_probs": probs, "delay_offset": -5,
                         "cancel_prob": .2 if stress else 0.}
    return result


def run_case(spec):
    date, stress, origin, destination, seed, cap, limit = spec
    with (ROOT / "data/zurich_wide.pkl").open("rb") as f:
        base = pickle.load(f)
    with (ROOT / "data/per_day_distributions.pkl").open("rb") as f:
        days = pickle.load(f)
    routes = {c.route for c in base.connections}
    priors = historical_priors(days, date, routes)
    by_name = {stop.name: sid for sid, stop in base.stops.items()}
    src, dst = by_name[origin], by_name[destination]
    distributions = world_distributions(days[date], routes, stress)
    set_regime_dist_fn(lambda regime: distributions)
    schedule = RegimeSchedule([(0, "generated")])
    # One immutable exogenous bank member is shared by every method in a pair.
    scenario = JourneyScenario.sample(base, schedule, np.random.default_rng(seed))
    records = []
    configurations = [(menu, score, settings) for menu in ("retain", "refresh")
                      for score, settings in VARIANTS.items()]
    configurations.append(("native_strict", "mean_cancel", {}))
    for menu, score, settings in configurations:
            graph = copy.deepcopy(base)
            router = (NativeDurnerRouter(graph, priors, max_time=cap, candidate_limit=limit)
                      if menu == "native_strict" else
                      MatchedRouter(graph, priors, candidate_mode=menu, max_time=cap,
                                    candidate_limit=limit, **settings))
            start = perf_counter()
            result = simulate_bandit_journey(graph, router, src, dst, 490, schedule,
                                            np.random.default_rng(9000 + seed),
                                            max_time=cap, scenario=scenario)
            elapsed = (perf_counter() - start) * 1000
            loss = min(max(result.arrival_time - 490, 0), cap)
            records.append({"date": date, "stress": stress, "origin": origin,
                            "destination": destination, "world_seed": seed,
                            "timeout": cap, "candidate_limit": limit,
                            "method": menu + "/" + score, "loss": loss,
                            "arrived": result.events[-1].details == "arrived",
                            "total_ms": elapsed, "initial_query_and_estimate_ms": router.preprocess_ms,
                            "estimate_ms": router.estimate_ms,
                            "query_ms": router.query_ms, "scans": router.scans,
                            "label_updates": router.label_updates,
                            "observations": router.total_observations,
                            "decisions": router.decisions,
                            "published_messages": [asdict(obs) for obs in result.observations],
                            "events": [asdict(event) for event in result.events]})
    return {"implementation": IMPLEMENTATION, "spec": spec, "records": records,
            "priors": priors,
            "world": {str(cid): asdict(event) for cid, event in scenario.events.items()},
            "peak_process_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def summarize(cases):
    rows = [row for case in cases for row in case["records"]]
    methods = {}
    for method in sorted({r["method"] for r in rows}):
        for stress in (False, True):
            group = [r for r in rows if r["method"] == method and r["stress"] == stress]
            if not group:
                continue
            queries = [q for r in group for q in r["query_ms"]]
            methods[f'{"synthetic_stress" if stress else "delay_calibrated_no_cancel"}/{method}'] = {
                "n_generated_journeys": len(group), "n_dates": len({r["date"] for r in group}),
                "mean_capped_loss": float(np.mean([r["loss"] for r in group])),
                "arrival_rate": float(np.mean([r["arrived"] for r in group])),
                "mean_total_ms": float(np.mean([r["total_ms"] for r in group])),
                "query_p50_ms": float(np.quantile(queries, .5)),
                "query_p95_ms": float(np.quantile(queries, .95))}
    menu_pairs = []
    for case in cases:
        by_method = {r["method"]: r for r in case["records"]}
        for score in VARIANTS:
            a, b = (by_method[menu + "/" + score] for menu in ("retain", "refresh"))
            menu_pairs.append({"date": a["date"], "stress": a["stress"],
                               "origin": a["origin"], "score": score,
                               "retain_minus_refresh": a["loss"] - b["loss"],
                               "same_actions": [d["selected"] for d in a["decisions"]]
                               == [d["selected"] for d in b["decisions"]]})
    return {"methods": methods, "menu_pairs": menu_pairs,
            "causal_optimality_gaps": "not computed for this marginal approximation panel",
            "statistical_inference": "screening only; generated repetitions are not independent real dates"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", default=["2023-10-07", "2023-10-29"])
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("matched_results.json"))
    args = parser.parse_args()
    specs = [(date, stress, src, dst, 711 + i, args.timeout, args.candidate_limit)
             for date in args.dates for stress in (False, True)
             for src, dst in ODS for i in range(args.replicates)]
    cases = []
    # Each completed case is a research asset. Save it before waiting for other
    # workers, so a failed case cannot discard already completed comparisons.
    checkpoint_dir = args.output.with_name(args.output.stem + "_cases")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for i, spec in enumerate(specs):
        checkpoint = checkpoint_dir / f"case_{i:03d}.json"
        if checkpoint.exists():
            case = json.loads(checkpoint.read_text())
            if tuple(case["spec"]) != spec or case["implementation"] != IMPLEMENTATION:
                raise ValueError("Saved case belongs to a different experiment; choose a new output path")
            cases.append(case)
        else:
            pending.append((spec, checkpoint))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, spec): checkpoint for spec, checkpoint in pending}
        for future in as_completed(futures):
            case = future.result()
            checkpoint = futures[future]
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(case, ensure_ascii=False) + "\n")
            temporary.replace(checkpoint)
            cases.append(case)
            print(json.dumps({"completed": len(cases), "total": len(specs), "case": case["spec"]}), flush=True)
    cases.sort(key=lambda case: case["spec"])
    args.output.write_text(json.dumps({"scope": "generated worlds on archived Swiss topology; not real event replay",
                                       "config": vars(args) | {"output": str(args.output), "tuning": "none",
                                                               "implementation": IMPLEMENTATION,
                                                               "information": "completed local events; last 10 minutes; 5 messages per poll; trip-deduplicated learning"},
                                       "summary": summarize(cases), "cases": cases},
                                      ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "journeys": sum(len(case["records"]) for case in cases)}))


if __name__ == "__main__":
    main()
