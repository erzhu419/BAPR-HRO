"""Matched, chronological routing panel using exact GTFS route identities."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import pickle
import resource
from statistics import mean
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.continuation_v1.run_candidate_cross import write_json
from experiments.continuation_v1.run_nominal_baseline import NominalBaseline
from src.causal_planning_router import CausalPlanningRouter
from src.continuation_router import ContinuationRouter
from src.matched_router import MatchedRouter
from src.native_router import NativeDurnerRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import JourneyScenario, RegimeSchedule, set_regime_dist_fn


VERSION = "route_id_chronological_cached_transfer"
DEFAULT_DATA = ROOT / "data/per_day_distributions_corrected.pkl"
IDENTITY = ROOT / "experiments/reconstruction_data_audit/graph_identity.json"
OD_SOURCE = ROOT / "experiments/swiss_full/results/swiss_multi_od_v3.json"
METHODS = ("marginal", "native_strict", "nominal", "rollout")
PLANNERS = {"full_rollout": ("full", "rollout"), "fixed_h_rollout": ("fixed", "rollout"),
            "material_h_rollout": ("fixed_top5", "rollout"), "belief_uct": ("full", "uct")}
DEPARTURE = 490
DELAYS = np.arange(-5, 65)


def load_network():
    with (ROOT / "data/zurich_wide.pkl").open("rb") as handle:
        graph = pickle.load(handle)
    by_trip = json.loads(IDENTITY.read_text())["by_trip"]
    for connection in graph.connections:
        connection.route = by_trip[connection.trip_id]["route_id"]
    ods = json.loads(OD_SOURCE.read_text())["viable_ods"]
    stops = {}
    for sid, stop in graph.stops.items():
        stops.setdefault(stop.name, sid)
    return graph, [(od["s1_name"], stops[od["s1_name"]], od["s2_name"], stops[od["s2_name"]]) for od in ods]


def chronological_prior(days, date, route):
    previous = [day[route] for key, day in days.items() if key < date and route in day]
    operating = [row for row in previous if row["n_obs"]]
    n = sum(row["n_obs"] for row in operating)
    n_status = sum(row["n_status_trips"] for row in previous)
    canceled = sum(row["n_canceled"] for row in previous)
    loc, scale = 1.0, 2.0
    if n:
        loc = sum(row["n_obs"] * row["mean"] for row in operating) / n
        second = sum(row["n_obs"] * (row["std"] ** 2 + row["mean"] ** 2) for row in operating) / n
        scale = max(second - loc ** 2, .25) ** .5
    return {"mean": loc, "std": scale, "cancel_rate": canceled / n_status if n_status else .01,
            "past_forecasts": n, "past_statuses": n_status}


def daily_world_parameters(day, priors, stress):
    params, coverage = {}, {"delay_routes": 0, "status_routes": 0}
    for route, prior in priors.items():
        row = day.get(route, {})
        has_delay = row.get("n_obs", 0) > 0
        has_status = row.get("n_status_trips", 0) > 0
        coverage["delay_routes"] += has_delay
        coverage["status_routes"] += has_status
        loc = (row["mean"] if has_delay else prior["mean"]) + (8 if stress else 0)
        scale = max(row["std"] if has_delay else prior["std"], 4 if stress else .5)
        log_mass = -.5 * ((DELAYS - loc) / scale) ** 2
        mass = np.exp(log_mass - log_mass.max())
        cancel = row["cancel_rate"] if has_status else prior["cancel_rate"]
        params[route] = {"delay_probs": mass / mass.sum(), "delay_offset": int(DELAYS[0]),
                         "cancel_prob": max(cancel, .2) if stress else cancel}
    return params, coverage


def initialize(graph, ods, days, methods, budget=128):
    global _graph, _ods, _days, _methods, _scenarios, _routes, _budget
    _graph, _ods, _days, _methods = graph, ods, days, methods
    _budget = budget
    _scenarios = {}
    _routes = {c.route for c in _graph.connections}


def scene_for(date, stress):
    key = date, stress
    if key not in _scenarios:
        priors = {route: chronological_prior(_days, date, route) for route in _routes}
        distributions, coverage = daily_world_parameters(_days[date], priors, stress)
        schedule = RegimeSchedule([(0, "generated")])
        seed = int(date.replace("-", "")) + 185
        set_regime_dist_fn(lambda regime: distributions)
        scene = JourneyScenario.sample(_graph, schedule, np.random.default_rng(seed))
        _scenarios[key] = scene, priors, schedule, seed, coverage
    return _scenarios[key]


def make_router(method, graph, priors, cap, seed):
    if method == "marginal":
        return MatchedRouter(graph, priors, online_mean=True, online_cancel=True, max_time=cap)
    if method == "native_strict":
        return NativeDurnerRouter(graph, priors, max_time=cap)
    if method in PLANNERS:
        menu, algorithm = PLANNERS[method]
        return CausalPlanningRouter(graph, priors, menu=menu, algorithm=algorithm, work_budget=_budget,
                                    max_time=cap, seed=seed + 41000, samples=8)
    cls = NominalBaseline if method == "nominal" else ContinuationRouter
    return cls(graph, priors, max_time=cap, seed=seed + 41000, samples=8)


def run_case(spec):
    date, od, stress, cap = spec
    source_name, source, target_name, target = _ods[od]
    scene, priors, schedule, seed, coverage = scene_for(date, stress)
    rows = []
    for method in _methods:
        graph = deepcopy(_graph)
        router = make_router(method, graph, priors, cap, seed)
        elapsed_queries = []
        selector = router.select_connection

        def timed_selector(*args, **kwargs):
            start = perf_counter()
            try:
                return selector(*args, **kwargs)
            finally:
                elapsed_queries.append((perf_counter() - start) * 1000)

        router.select_connection = timed_selector
        start = perf_counter()
        journey = simulate_bandit_journey(graph, router, source, target, DEPARTURE, schedule,
                                         np.random.default_rng(seed + 9000), max_time=cap, scenario=scene)
        elapsed = (perf_counter() - start) * 1000
        planning_work = [p["work"] for p in router.predictions] if method in PLANNERS else []
        if any(work > _budget for work in planning_work):
            raise AssertionError(f"Planning transition budget exceeded for {spec}/{method}")
        rows.append({"method": method, "loss": journey.arrival_time - DEPARTURE,
                     "arrived": journey.events[-1].details == "arrived", "total_ms": elapsed,
                     "initial_ms": router.preprocess_ms, "estimate_ms": router.estimate_ms,
                     "scan_ms": sum(router.query_ms), "scans": router.scans,
                     "label_updates": router.label_updates, "rollout_ms": getattr(router, "rollout_ms", 0.),
                     "planning_transitions": sum(planning_work),
                     "max_query_transitions": max(planning_work, default=0),
                     "menu_size": len(router.menu) if method in PLANNERS else None,
                     "queries_ms": elapsed_queries, "worker_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     "public_messages": [asdict(m) for m in journey.observations],
                     "events": [asdict(e) for e in journey.events]})
    return {"implementation": VERSION, "spec": spec, "od": [source_name, target_name],
            "world_seed": seed, "route_coverage": coverage,
            "transition_budget": _budget if any(m in PLANNERS for m in _methods) else None, "rows": rows}


def summarize(cases, methods):
    summaries = {}
    for stress, cap in sorted({tuple(c["spec"][2:]) for c in cases}):
        group = [c for c in cases if tuple(c["spec"][2:]) == (stress, cap)]
        metric = {}
        for method in methods:
            rows = [r for c in group for r in c["rows"] if r["method"] == method]
            query = [t for r in rows for t in r["queries_ms"]]
            metric[method] = {key: mean(r[key] for r in rows) for key in (
                "loss", "arrived", "total_ms", "initial_ms", "estimate_ms", "scan_ms",
                "scans", "label_updates", "rollout_ms")}
            metric[method]["decision_p50_p95_ms"] = np.quantile(query, [.5, .95]).tolist() if query else None
            metric[method]["largest_worker_peak_kib"] = max(r["worker_peak_kib"] for r in rows)
            if method in PLANNERS:
                metric[method]["planning_transitions"] = mean(r["planning_transitions"] for r in rows)
                metric[method]["max_query_transitions"] = max(r["max_query_transitions"] for r in rows)
                metric[method]["menu_size_range"] = [min(r["menu_size"] for r in rows),
                                                     max(r["menu_size"] for r in rows)]
        paired = {}
        reference = "nominal" if "nominal" in methods else "full_rollout" if "full_rollout" in methods else None
        if reference is not None:
            rng = np.random.default_rng(836)
            for method in methods:
                if method == reference:
                    continue
                by_date = {}
                for c in group:
                    losses = {r["method"]: r["loss"] for r in c["rows"]}
                    by_date.setdefault(c["spec"][0], []).append(losses[method] - losses[reference])
                differences = np.array([mean(values) for values in by_date.values()])
                boot = rng.choice(differences, size=(4096, len(differences)), replace=True).mean(axis=1)
                paired[method] = {"mean_date_difference": float(differences.mean()),
                                  "date_resampling_interval": np.quantile(boot, [.025, .975]).tolist(),
                                  "by_date": {date: mean(values) for date, values in by_date.items()}}
        summaries[f"stress={stress}/cap={cap}"] = {"od_cases": len(group),
            "generated_worlds": len({c["spec"][0] for c in group}), "metrics": metric,
            "paired_reference": reference, "paired_differences": paired}
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--ods", nargs="+", type=int)
    parser.add_argument("--caps", nargs="+", type=int, default=[120])
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=(*METHODS, *PLANNERS), default=list(METHODS))
    parser.add_argument("--transition-budget", type=int, default=128)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--output", type=Path, default=HERE / "chronological_results.json")
    args = parser.parse_args()
    budget = args.transition_budget if any(m in PLANNERS for m in args.methods) else None
    graph, ods = load_network()
    with args.data.open("rb") as handle:
        days = pickle.load(handle)
    dates = args.dates or sorted(days)
    if args.dates is None and len(dates) != 35:
        parser.error("the full panel requires all 35 daily calibration records")
    if any(date not in days for date in dates):
        parser.error("requested dates are missing calibration records")
    if len(ods) != 18:
        raise ValueError("Original 18 OD pairs could not be recovered")
    indices = args.ods or range(len(ods))
    specs = [(date, od, args.stress, cap) for date in dates for od in indices for cap in args.caps]
    receipt_dir = args.output.with_name(args.output.stem + "_cases")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    cases, pending = [], []
    for i, spec in enumerate(specs):
        receipt = receipt_dir / f"case_{i:04}.json.gz"
        if receipt.exists():
            with gzip.open(receipt, "rt", encoding="utf-8") as handle:
                saved = json.load(handle)
            if (tuple(saved["spec"]) != spec or saved["implementation"] != VERSION
                    or saved.get("transition_budget") != budget
                    or [r["method"] for r in saved["rows"]] != args.methods):
                raise ValueError(f"Use a different output for changed protocol: {receipt}")
            cases.append(saved)
        else:
            pending.append((spec, receipt))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(graph, ods, days, args.methods, args.transition_budget)) as pool:
        jobs = {pool.submit(run_case, spec): receipt for spec, receipt in pending}
        for job in as_completed(jobs):
            case = job.result()
            receipt = jobs[job]
            partial = receipt.with_suffix(".tmp")
            with gzip.open(partial, "wt", encoding="utf-8", compresslevel=3) as handle:
                json.dump(case, handle, ensure_ascii=False, allow_nan=False)
            partial.replace(receipt)
            cases.append(case)
            if len(cases) == len(specs) or len(cases) % 15 == 0:
                print(f"{len(cases)}/{len(specs)} OD conditions complete", flush=True)
    cases.sort(key=lambda c: c["spec"])
    compact = [{**c, "rows": [{k: v for k, v in r.items()
                              if k not in ("public_messages", "events", "queries_ms")} for r in c["rows"]]}
               for c in cases]
    write_json(args.output, {"implementation": VERSION, "data": str(args.data),
        "scope": "one generated world per date and condition; target-day latest forecast statistics and observed service declarations calibrate worlds on the same static topology; policy priors use strictly earlier dates",
        "inference": "paired date resampling describes these generated conditions, not independent observed service events",
        "od_conditions": len(cases), "journeys": sum(len(c["rows"]) for c in cases),
        "predeclared_departure": DEPARTURE, "methods": args.methods, "transition_budget": budget,
        "summary": summarize(cases, args.methods), "cases": compact}, indent=2)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
