"""Continuation estimation: fixed archived worlds and crossed value rules."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import resource
import sys
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.continuation_v1.historical_router import HistoricalContinuationRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import JourneyScenario, VehicleEvent, RegimeSchedule

VERSION = "continuation_v1_public_history_rollout"
VARIANTS = {"legacy": {"legacy": True},
            "marginal_cap": {"conditioning": False, "rescue": False},
            "marginal_rescue": {"conditioning": False, "rescue": True},
            "conditioned_cap": {"conditioning": True, "rescue": False},
            "conditioned_rescue": {"conditioning": True, "rescue": True}}
ARCHIVE = ROOT / "experiments/reconstruction_v2/matched_corrected_results_cases"


def run_one(index, variant, samples, archive=ARCHIVE, *, router_cls=HistoricalContinuationRouter,
            router_options=None, implementation=VERSION):
    source = Path(archive) / f"case_{index:03d}.json"
    case = json.loads(source.read_text())
    date, stress, origin, destination, seed, cap, limit = case["spec"]
    with (ROOT / "data/zurich_wide.pkl").open("rb") as handle:
        graph = pickle.load(handle)
    by_name = {stop.name: sid for sid, stop in graph.stops.items()}
    scene = JourneyScenario({int(cid): VehicleEvent(**event) for cid, event in case["world"].items()})
    options = VARIANTS[variant] if router_options is None else router_options
    router = router_cls(graph, case["priors"], samples=samples, seed=41000 + seed,
                        candidate_limit=limit, max_time=cap, **options)
    started = perf_counter()
    result = simulate_bandit_journey(graph, router, by_name[origin], by_name[destination],
                                    490, RegimeSchedule([(0, "fixed_archive")]),
                                    np.random.default_rng(9000 + seed), max_time=cap, scenario=scene)
    total_ms = (perf_counter() - started) * 1000
    loss = min(cap, result.arrival_time - 490)
    previous = next((r for r in case["records"] if r["method"] == "refresh/mean_cancel"), None)
    return {"implementation": implementation, "archive_case": str(source.resolve()),
            "nominal_summary_implementation": "trip_cached",
            "index": index, "variant": variant, "samples": samples,
            "private_seed": 41000 + seed, "spec": case["spec"],
            "loss": loss, "arrived": result.events[-1].details == "arrived",
            "legacy_archived_loss": previous["loss"] if previous else None,
            "total_ms": total_ms, "rollout_ms": router.rollout_ms,
            "query_ms": router.query_ms, "estimate_ms": router.estimate_ms,
            "scans": router.scans, "rollout_steps": router.rollout_steps,
            "peak_process_kib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "predictions": router.predictions, "polls": router.polls,
            "decisions": router.decisions,
            "messages": [asdict(message) for message in result.observations],
            "events": [asdict(event) for event in result.events]}


def summarize(rows):
    summary = {}
    baselines = {r["index"]: r["loss"] for r in rows if r["variant"] == "legacy"}
    for variant in sorted({r["variant"] for r in rows}):
        for stress in (False, True):
            group = [r for r in rows if r["variant"] == variant and r["spec"][1] == stress]
            if group:
                delta = [r["loss"] - baseline for r in group
                         if (baseline := baselines.get(r["index"], r["legacy_archived_loss"])) is not None]
                summary[f'{variant}/{"stress" if stress else "no_cancel"}'] = {
                    "n": len(group), "mean_loss": float(np.mean([r["loss"] for r in group])),
                    "arrival_rate": float(np.mean([r["arrived"] for r in group])),
                    "paired_n": len(delta),
                    "mean_paired_minus_legacy": float(np.mean(delta)) if delta else None,
                    "better": sum(d < 0 for d in delta), "worse": sum(d > 0 for d in delta),
                    "timing_by_implementation": {
                        version: {"n": len(part),
                                  "mean_total_ms": float(np.mean([r["total_ms"] for r in part])),
                                  "mean_rollout_ms": float(np.mean([r["rollout_ms"] for r in part]))}
                        for version in {r.get("nominal_summary_implementation", "recomputed") for r in group}
                        if (part := [r for r in group if r.get("nominal_summary_implementation", "recomputed") == version])}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", nargs="+", type=int, default=list(range(24)))
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("matched_results.json"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    directory = args.output.with_name(args.output.stem + "_methods")
    directory.mkdir(parents=True, exist_ok=True)
    rows, pending = [], []
    for index in args.indices:
        for variant in args.variants:
            output = directory / f"case_{index:03d}_{variant}.json"
            if output.exists():
                row = json.loads(output.read_text())
                recorded_source = Path(row["archive_case"])
                if not recorded_source.is_absolute():
                    recorded_source = ROOT / recorded_source
                expected_source = (args.archive / f"case_{index:03d}.json").resolve()
                if (row["implementation"] != VERSION or row["samples"] != args.samples
                        or recorded_source.resolve() != expected_source):
                    raise ValueError("Use a new output path for a changed experiment")
                rows.append(row)
            else:
                pending.append((index, variant, output))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, index, variant, args.samples, args.archive): output
                   for index, variant, output in pending}
        for future in as_completed(futures):
            row = future.result()
            path = futures[future]
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, ensure_ascii=False) + "\n")
            temporary.replace(path)
            rows.append(row)
            print(json.dumps({"completed": len(rows), "total": len(args.indices) * len(args.variants),
                              "case": row["index"], "method": row["variant"], "loss": row["loss"]}), flush=True)
    rows.sort(key=lambda row: (row["index"], row["variant"]))
    args.output.write_text(json.dumps({"scope": "generated archived worlds; paired descriptive screening",
                                       "implementation": VERSION,
                                       "config": vars(args) | {"output": str(args.output), "archive": str(args.archive), "tuning": "none",
                                                               "numpy": np.__version__},
                                       "summary": summarize(rows), "methods": rows},
                                      ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "journeys": len(rows)}))


if __name__ == "__main__":
    main()
