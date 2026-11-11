"""Profile one static city decision using the retained or deferred belief runtime."""
from __future__ import annotations

import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import resource
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import ReplayRouter, dated_graph, fit_training
from experiments.reconstruction_v5.runtime_reference_evolving import EvolvingTripBelief as ReferenceBelief
from src.evolving_trip_belief import EvolvingTripBelief


def process_memory():
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmHWM:", "VmRSS:")):
            name, value = line.split(":", 1)
            fields[name] = int(value.split()[0])
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("reference", "lazy"), default="lazy")
    parser.add_argument("--date", default="2023-07-26")
    parser.add_argument("--query", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    source = ROOT / "experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz"
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    graph, _, stop_ids, source_map, _, _ = dated_graph(data, args.date, "gtfs_station_max", True)
    query = data["od_queries"][args.query]
    pooled = fit["route_priors"]["pooled"]
    priors = {c.route: fit["route_priors"].get(c.route, pooled) for c in graph.connections}
    belief_class = EvolvingTripBelief if args.implementation == "lazy" else ReferenceBelief
    belief = belief_class(graph, minute=60, public_occurrences=False, **fit["increments"])
    router = ReplayRouter(
        graph, priors, event_belief=belief, menu="full", algorithm="rollout", samples=4,
        work_budget=512, max_time=query["deadline_seconds"] - query["ready_seconds"],
        seed=int(args.date.replace("-", "")) + args.query * 1000, frames=(), source_to_local=source_map,
        forecast_sd={c.id: priors[c.route]["std"] * 60 for c in graph.connections})
    router.online_mean = router.online_cancel = False
    router._initial_cancels = {route: row["cancel_rate"] for route, row in priors.items()}
    origin, destination = stop_ids[query["origin"]], stop_ids[query["destination"]]
    clock = query["ready_seconds"]
    router.route(origin, destination, clock)
    router.set_execution_context(None, clock, set())
    router.set_information_context(origin, clock, None, clock, set(), set(), set())

    profiler = cProfile.Profile()
    before = resource.getrusage(resource.RUSAGE_SELF)
    memory_before = process_memory()
    cpu_start, wall_start = process_time(), perf_counter()
    profiler.enable()
    selected = router.select_connection(origin, clock, np.random.default_rng(8000 + args.query))
    profiler.disable()
    wall, cpu = perf_counter() - wall_start, process_time() - cpu_start
    after = resource.getrusage(resource.RUSAGE_SELF)
    memory_after = process_memory()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(15)
    entries = [{"file": path, "line": line, "function": name, "calls": record[1],
                "own_seconds": record[2], "cumulative_seconds": record[3]}
               for (path, line, name), record in stats.stats.items()]
    entries.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
    focus = {"sample", "mean_events", "_law", "marginals", "draw_uniforms", "trip_events",
             "_mean_event", "nominal_action", "transition", "logsumexp", "build_committed_model"}
    result = {"scope": "One initial static public-history decision; held-out actual world unused.",
              "implementation": args.implementation, "input": str(source.relative_to(ROOT)),
              "date": args.date, "query": args.query, "static_query": query,
              "graph_connections": len(graph.connections), "trips": len(belief._trips),
              "samples": 4, "transition_budget": 512, "wall_seconds": wall, "cpu_seconds": cpu,
              "peak_rss_kib": memory_after["VmHWM"], "peak_before_query_kib": memory_before["VmHWM"],
              "ru_maxrss_kib": after.ru_maxrss, "ru_maxrss_before_query_kib": before.ru_maxrss,
              "proc_memory_before_kib": memory_before, "proc_memory_after_kib": memory_after,
              "selected": None if selected is None else selected[0].connection_id,
              "planning": router.predictions[-1], "top15_cumulative": entries[:15],
              "focused_functions": [row for row in entries if row["function"] in focus]}
    if args.implementation == "lazy":
        result["trip_laws_cached"] = sum(memo.law is not None for memo in belief._cache.values())
        result["nominal_events_cached"] = sum(len(memo.events) for memo in belief._cache.values())
    label = "boundary" if args.implementation == "lazy" else "reference"
    output = Path(__file__).with_name(f"profile_{label}_{args.date}_q{args.query}.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    output.with_suffix(".txt").write_text(stream.getvalue())
    print(stream.getvalue())
    print(json.dumps({key: result[key] for key in ("wall_seconds", "cpu_seconds", "peak_rss_kib", "selected", "planning")}))
    print("Saved", output)


if __name__ == "__main__":
    main()
