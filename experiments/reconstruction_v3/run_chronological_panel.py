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


VERSION = "route_id_chronological_provenance"
DEFAULT_DATA = ROOT / "data/per_day_distributions_corrected.pkl"
IDENTITY = ROOT / "experiments/reconstruction_data_audit/graph_identity.json"
OD_SOURCE = ROOT / "experiments/swiss_full/results/swiss_multi_od_v3.json"
METHODS = ("marginal", "native_strict", "nominal", "rollout")
PLANNERS = {"full_rollout": ("full", "rollout"), "full_nominal": ("full", "nominal"),
            "fixed_h_rollout": ("fixed", "rollout"),
            "material_h_rollout": ("fixed_top5", "rollout"), "belief_uct": ("full", "uct")}
RESCUE = {f"{menu}_{score}": {"candidate_mode": menu, "rescue": score == "rescue"}
          for menu in ("retain", "refresh") for score in ("cap", "rescue")}
FITTERS = {f"mean{m}_cancel{c}_penalty{p}": {"online_mean": bool(m), "online_cancel": bool(c),
                                               "beta": 1.5 if p else 0.0}
           for m in (0, 1) for c in (0, 1) for p in (0, 1)}
FACTORIAL_DATES = ("2023-10-12", "2023-10-25", "2023-10-29")
FACTORIAL_ODS = (1, 5)
HOLDOUT_DATES = ("2023-11-09", "2023-11-10", "2023-11-11")
FINAL_HOLDOUT_DATES = ("2023-11-12", "2023-11-13", "2023-11-14")
# These are platform IDs. Several disconnected Stettbach platforms share a name.
HOLDOUT_ODS = (("Zürich, Seebach", 743038111, "Zürich, Sternen Oerlikon", 67001060),
               ("Stettbach, Bahnhof", 505872492, "Zürich, Roswiesen", 408934162))
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


def daily_world_settings(day, priors, stress):
    params, coverage = {}, {"delay_routes": 0, "status_routes": 0}
    for route, prior in priors.items():
        row = day.get(route, {})
        has_delay = row.get("n_obs", 0) > 0
        has_status = row.get("n_status_trips", 0) > 0
        coverage["delay_routes"] += has_delay
        coverage["status_routes"] += has_status
        loc = (row["mean"] if has_delay else prior["mean"]) + (8 if stress else 0)
        scale = max(row["std"] if has_delay else prior["std"], 4 if stress else .5)
        cancel = row["cancel_rate"] if has_status else prior["cancel_rate"]
        params[route] = {"mean": loc, "std": scale, "cancel_rate": max(cancel, .2) if stress else cancel}
    return params, coverage


def world_inputs(days, specs, graph, holdout):
    inputs = {}
    routes = sorted({c.route for c in graph.connections})
    for date, _, stress, _ in specs:
        key = f"{date}/{int(stress)}"
        if key in inputs:
            continue
        priors = {route: chronological_prior(days, date, route) for route in routes}
        params, coverage = daily_world_settings({} if holdout else days[date], priors, stress)
        seed = int(date.replace("-", "")) + (77500 + 1000 * stress if holdout else 185)
        inputs[key] = {"priors": priors, "generator": params, "seed": seed, "coverage": coverage,
                       "prior_dates": sorted(day for day in days if day < date)}
    return inputs


def freeze_protocol(output, protocol):
    """Save actual inputs and reject a resume if any of them change."""
    path = output.with_name(output.stem + "_protocol.json")
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError(f"Inputs or protocol changed. Use a new output path: {path}")
    else:
        receipts = output.with_name(output.stem + "_cases")
        if output.exists() or any(receipts.glob("case_*.json.gz")):
            raise ValueError(f"Existing results lack a frozen input protocol. Use a new output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, protocol)
    return path


def initialize(graph, ods, inputs, methods, budget, data_source):
    global _graph, _ods, _inputs, _methods, _scenarios, _budget, _data_source
    _graph, _ods, _inputs, _methods = graph, ods, inputs, methods
    _budget, _data_source = budget, data_source
    _scenarios = {}


def scene_for(date, stress):
    key = date, stress
    if key not in _scenarios:
        frozen = _inputs[f"{date}/{int(stress)}"]
        distributions = {}
        for route, params in frozen["generator"].items():
            log_mass = -.5 * ((DELAYS - params["mean"]) / params["std"]) ** 2
            mass = np.exp(log_mass - log_mass.max())
            distributions[route] = {"delay_probs": mass / mass.sum(), "delay_offset": int(DELAYS[0]),
                                    "cancel_prob": params["cancel_rate"]}
        schedule = RegimeSchedule([(0, "generated")])
        seed = frozen["seed"]
        set_regime_dist_fn(lambda regime: distributions)
        scene = JourneyScenario.sample(_graph, schedule, np.random.default_rng(seed))
        _scenarios[key] = scene, frozen["priors"], schedule, seed, frozen["coverage"]
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
    if method in RESCUE:
        return ContinuationRouter(graph, priors, max_time=cap, seed=seed + 41000, samples=8,
                                  **RESCUE[method])
    if method in FITTERS:
        return MatchedRouter(graph, priors, max_time=cap, candidate_mode="retain", **FITTERS[method])
    cls = NominalBaseline if method == "nominal" else ContinuationRouter
    return cls(graph, priors, max_time=cap, seed=seed + 41000, samples=8)


def record_online_decisions(router):
    selector = router.select_connection
    poll_callback = getattr(router, "observe_public_messages", None)
    attempt_callback = getattr(router, "observe_attempt_outcome", None)
    decisions, polls, attempts = [], [], []

    def select(stop, time, *args, **kwargs):
        start = perf_counter()
        result = selector(stop, time, *args, **kwargs)
        decisions.append({"stop": stop, "time": time,
                          "action": "wait" if result is None else "select",
                          "connection_id": None if result is None else result[0].connection_id,
                          "selector_ms": (perf_counter() - start) * 1000})
        return result

    def poll(stop, time, messages, seen_before):
        polls.append({"stop": stop, "received_at": time,
                      "event_ids": [message.event_id for message in messages],
                      "next_decision_index": len(decisions)})
        if poll_callback is not None:
            poll_callback(stop, time, messages, seen_before)

    def attempt(cid, kind, started, time):
        attempts.append({"connection_id": cid, "outcome": kind, "started": started, "time": time,
                         "decision_index": len(decisions) - 1})
        if attempt_callback is not None:
            attempt_callback(cid, kind, started, time)

    router.select_connection = select
    router.observe_public_messages = poll
    router.observe_attempt_outcome = attempt
    return decisions, polls, attempts


def run_case(spec, router_factory=None):
    date, od, stress, cap = spec
    source_name, source, target_name, target = _ods[od]
    scene, priors, schedule, seed, coverage = scene_for(date, stress)
    rows = []
    for method in _methods:
        graph = deepcopy(_graph)
        router = (router_factory or make_router)(method, graph, priors, cap, seed)
        decisions, polls, attempts = record_online_decisions(router)
        start = perf_counter()
        journey = simulate_bandit_journey(graph, router, source, target, DEPARTURE, schedule,
                                         np.random.default_rng(seed + 9000), max_time=cap, scenario=scene)
        elapsed = (perf_counter() - start) * 1000
        planning_work = [p["work"] for p in router.predictions] if hasattr(router, "work_budget") else []
        if any(work > _budget for work in planning_work):
            raise AssertionError(f"Planning transition budget exceeded for {spec}/{method}")
        rows.append({"method": method, "loss": journey.arrival_time - DEPARTURE,
                     "arrived": journey.events[-1].details == "arrived", "total_ms": elapsed,
                     "initial_ms": router.preprocess_ms, "estimate_ms": router.estimate_ms,
                     "scan_ms": sum(router.query_ms), "scans": router.scans,
                     "label_updates": router.label_updates, "rollout_ms": getattr(router, "rollout_ms", 0.),
                     "planning_transitions": sum(planning_work),
                     "max_query_transitions": max(planning_work, default=0),
                     "menu_size": len(router.menu) if hasattr(router, "menu") else None,
                     "queries_ms": [decision["selector_ms"] for decision in decisions],
                     "worker_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     "polls": polls, "decisions": decisions, "attempts": attempts,
                     "public_messages": [asdict(m) for m in journey.observations],
                     "events": [asdict(e) for e in journey.events]})
    return {"implementation": VERSION, "data_source": _data_source, "spec": spec, "od": [source_name, target_name],
            "world_seed": seed, "route_coverage": coverage,
            "transition_budget": _budget if any(m in PLANNERS for m in _methods) else None, "rows": rows}


def factorial_effects(cases, methods):
    contrasts = {}
    if set(RESCUE) <= set(methods):
        contrasts.update({
            "retain_minus_refresh_cap": {"retain_cap": 1, "refresh_cap": -1},
            "retain_minus_refresh_rescue": {"retain_rescue": 1, "refresh_rescue": -1},
            "rescue_minus_cap_retain": {"retain_rescue": 1, "retain_cap": -1},
            "rescue_minus_cap_refresh": {"refresh_rescue": 1, "refresh_cap": -1},
            "retention_by_rescue_interaction": {"retain_rescue": 1, "retain_cap": -1,
                                                "refresh_rescue": -1, "refresh_cap": 1},
        })
    if set(FITTERS) <= set(methods):
        for factor, on in (("online_mean", True), ("online_cancel", True), ("beta", 1.5)):
            contrasts[f"{factor}_on_minus_off"] = {
                method: .25 if options[factor] == on else -.25 for method, options in FITTERS.items()
            }
    effects = {}
    for name, contrast in contrasts.items():
        by_date = {}
        values = []
        for case in cases:
            losses = {row["method"]: row["loss"] for row in case["rows"]}
            effect = sum(weight * losses[method] for method, weight in contrast.items())
            values.append(effect)
            by_date.setdefault(case["spec"][0], []).append(effect)
        effects[name] = {"n": len(values), "mean_difference": mean(values),
                         "negative": sum(value < 0 for value in values),
                         "positive": sum(value > 0 for value in values),
                         "ties": sum(value == 0 for value in values),
                         "by_date": {date: mean(parts) for date, parts in by_date.items()}}
    return effects


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
        reference = ("nominal" if "nominal" in methods else "full_nominal" if "full_nominal" in methods
                     else "full_rollout" if "full_rollout" in methods else None)
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
            "paired_reference": reference, "paired_differences": paired,
            "factorial_effects": factorial_effects(group, methods)}
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--ods", nargs="+", type=int)
    parser.add_argument("--caps", nargs="+", type=int, default=[120])
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=(*METHODS, *PLANNERS, *RESCUE, *FITTERS))
    parser.add_argument("--transition-budget", type=int, default=128)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--protocol", choices=("chronological", "factorial", "holdout", "final_holdout"),
                        default="chronological")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    outputs = {"chronological": "chronological_provenance_results.json",
               "factorial": "route_id_factorial_results.json",
               "holdout": "final_platform_resolved_confirmation.json",
               "final_holdout": "final_causal_generated_confirmation.json"}
    output = args.output or HERE / outputs[args.protocol]
    data_source = args.data.resolve()
    graph, ods = load_network()
    with data_source.open("rb") as handle:
        days = pickle.load(handle)
    if len(ods) != 18:
        raise ValueError("Original 18 OD pairs could not be recovered")
    if args.protocol != "chronological" and (args.dates or args.ods or args.methods or args.stress or args.caps != [120]):
        parser.error("the factorial and holdout protocols fix dates, ODs, methods, stress conditions and cap")
    holdout = args.protocol in ("holdout", "final_holdout")
    if holdout:
        dates = FINAL_HOLDOUT_DATES if args.protocol == "final_holdout" else HOLDOUT_DATES
        if max(days) >= min(dates):
            parser.error("holdout pseudo-dates must follow every source calibration date")
        ods = list(HOLDOUT_ODS)
        if any(graph.stops[source].name != source_name or graph.stops[target].name != target_name
               for source_name, source, target_name, target in ods):
            parser.error("the frozen holdout platform IDs do not match this network")
        indices, stresses = range(len(ods)), (False, True)
        methods = ["full_nominal", "full_rollout"] if args.protocol == "final_holdout" else list(METHODS)
    elif args.protocol == "factorial":
        dates, indices, methods, stresses = FACTORIAL_DATES, FACTORIAL_ODS, [*RESCUE, *FITTERS], (False, True)
    else:
        dates, indices = args.dates or sorted(days), args.ods or range(len(ods))
        methods, stresses = args.methods or list(METHODS), (args.stress,)
        if args.dates is None and len(dates) != 35:
            parser.error("the full panel requires all 35 daily calibration records")
    if not holdout and any(date not in days for date in dates):
        parser.error("requested dates are missing calibration records")
    if len(set(methods)) != len(methods):
        parser.error("methods must be distinct")
    if any(not 0 <= od < len(ods) for od in indices):
        parser.error("OD index is outside this panel")
    budget = args.transition_budget if any(m in PLANNERS for m in methods) else None
    specs = [(date, od, stress, cap) for date in dates for od in indices for stress in stresses for cap in args.caps]
    if len(set(specs)) != len(specs):
        parser.error("case specifications must be distinct")
    inputs = world_inputs(days, specs, graph, holdout)
    scope = ("untouched prior-predictive generated pseudo-dates; no target-day observations"
             if holdout else "target-day forecast and service declarations calibrate generated worlds")
    protocol = {"implementation": VERSION, "protocol": args.protocol, "data": str(data_source),
                "scope": scope, "workers": args.workers, "methods": methods, "transition_budget": budget,
                "departure": DEPARTURE, "specs": [list(spec) for spec in specs],
                "ods": [list(od) for od in ods], "world_inputs": inputs,
                "connections": [[c.id, c.route, c.trip_id, c.dep_stop, c.arr_stop, c.dep_time, c.arr_time]
                                for c in graph.connections],
                "stops": [[sid, stop.name, stop.min_transfer_time] for sid, stop in graph.stops.items()],
                "factorial_methods": {method: options for method, options in (RESCUE | FITTERS).items()
                                      if method in methods}, "tuning": "none"}
    protocol_path = freeze_protocol(output, protocol)
    receipt_dir = output.with_name(output.stem + "_cases")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    cases, pending = [], []
    for i, spec in enumerate(specs):
        receipt = receipt_dir / f"case_{i:04}.json.gz"
        if receipt.exists():
            with gzip.open(receipt, "rt", encoding="utf-8") as handle:
                saved = json.load(handle)
            if (tuple(saved["spec"]) != spec or saved["implementation"] != VERSION
                    or saved.get("transition_budget") != budget
                    or saved.get("data_source") != str(data_source)
                    or [r["method"] for r in saved["rows"]] != methods):
                raise ValueError(f"Use a different output for changed protocol: {receipt}")
            cases.append(saved)
        else:
            pending.append((spec, receipt))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(graph, ods, inputs, methods, args.transition_budget, str(data_source))) as pool:
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
                              if k not in ("public_messages", "events", "queries_ms", "polls", "decisions", "attempts")}
                             for r in c["rows"]]}
               for c in cases]
    write_json(output, {"implementation": VERSION, "data": str(data_source),
        "frozen_protocol": str(protocol_path), "workers": args.workers, "scope": scope,
        "inference": "paired date resampling describes these generated conditions, not independent observed service events",
        "od_conditions": len(cases), "journeys": sum(len(c["rows"]) for c in cases),
        "predeclared_departure": DEPARTURE, "methods": methods, "transition_budget": budget,
        "summary": summarize(cases, methods), "cases": compact}, indent=2)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
