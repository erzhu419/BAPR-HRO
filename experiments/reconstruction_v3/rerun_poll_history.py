"""Correctness reruns on frozen synthetic inputs after public-poll state repair."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.reconstruction_v3 import run_chronological_panel as panel
from experiments.reconstruction_v3 import run_rollout_learning_cross as learning


VERSION = "synthetic_poll_history_correctness_rerun_v1"
PANELS = {
    "confirmation": ("final_causal_generated_confirmation", "final_causal_poll_history_confirmation"),
    "learning": ("causal_rollout_learning_factorial_results", "causal_poll_history_learning_factorial_results"),
    "budget128": ("budget_128_results", "budget_128_poll_history_results"),
    "budget512": ("budget_512_results", "budget_512_poll_history_results"),
}


def inputs_for(name):
    old_stem, new_stem = PANELS[name]
    old_path = HERE / (old_stem + ".json")
    previous = json.loads(old_path.read_text())
    source_path = HERE / (old_stem + "_protocol.json")
    provenance = "Original frozen scalar inputs, network and OD identities reused."
    if not source_path.exists():
        source_path = HERE / "route_id_factorial_results_protocol.json"
        provenance = ("Legacy budget receipts have no scalar-input protocol. Reuse the later frozen "
                      "factorial inputs for the same dates, data source, OD identities and world seeds.")
    source = json.loads(source_path.read_text())
    graph, _ = panel.load_network()
    connections = [[c.id, c.route, c.trip_id, c.dep_stop, c.arr_stop, c.dep_time, c.arr_time]
                   for c in graph.connections]
    stops = [[sid, stop.name, stop.min_transfer_time] for sid, stop in graph.stops.items()]
    if source["connections"] != connections or source["stops"] != stops:
        raise ValueError("The network differs from the archived frozen inputs")
    specs = [case["spec"] for case in previous["cases"]]
    keys = {f"{date}/{int(stress)}" for date, _, stress, _ in specs}
    inputs = {key: value for key, value in source["world_inputs"].items() if key in keys}
    if set(inputs) != keys:
        raise ValueError("The archived scalar inputs do not cover every previous case")
    methods = previous.get("methods", list(learning.METHODS))
    for case in previous["cases"]:
        date, od, stress, _ = case["spec"]
        assert case["world_seed"] == inputs[f"{date}/{int(stress)}"]["seed"]
        assert case["od"] == [source["ods"][od][0], source["ods"][od][2]]
        assert [row["method"] for row in case["rows"]] == methods
    budget = previous["transition_budget"]
    if name == "learning":
        assert budget == learning.BUDGET and previous["samples"] == learning.SAMPLES
    return graph, source, inputs, specs, methods, budget, old_path, new_stem, source_path, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANELS, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    graph, source, inputs, specs, methods, budget, old_path, stem, source_path, provenance = inputs_for(args.panel)
    output = HERE / (stem + ".json")
    protocol = source | {"implementation": VERSION, "protocol": "correctness_rerun",
        "corrected_from": str(old_path), "source_protocol": str(source_path),
        "correction": "Include intermediate synthetic public polls in committed information state.",
        "scope": "Correctness rerun on previously inspected generated dates and seeds.",
        "input_provenance": provenance, "workers": args.workers, "specs": specs,
        "methods": methods, "transition_budget": budget, "samples": 8, "world_inputs": inputs}
    protocol_path = panel.freeze_protocol(output, protocol)
    receipt_dir = output.with_name(output.stem + "_cases")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    cases, pending = [], []
    for index, spec in enumerate(specs):
        path = receipt_dir / f"case_{index:04}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                saved = json.load(handle)
            assert saved["spec"] == spec and saved["implementation"] == VERSION
            assert [row["method"] for row in saved["rows"]] == methods
            cases.append(saved)
        else:
            pending.append((spec, path))
    run_case = learning.run_case if args.panel == "learning" else panel.run_case
    with ProcessPoolExecutor(max_workers=args.workers, initializer=panel.initialize,
                             initargs=(graph, source["ods"], inputs, methods, budget, source["data"])) as pool:
        jobs = {pool.submit(run_case, spec): path for spec, path in pending}
        for job in as_completed(jobs):
            case, path = job.result(), jobs[job]
            case["implementation"], case["correction_of"] = VERSION, str(old_path)
            partial = path.with_suffix(".tmp")
            with gzip.open(partial, "wt", encoding="utf-8", compresslevel=3) as handle:
                json.dump(case, handle, ensure_ascii=False, allow_nan=False)
            partial.replace(path)
            cases.append(case)
            print(f"{args.panel} {len(cases)}/{len(specs)} conditions complete", flush=True)
    cases.sort(key=lambda c: c["spec"])
    compact = [{**case, "rows": [{k: v for k, v in row.items() if k not in (
        "public_messages", "events", "queries_ms", "polls", "decisions", "attempts", "parameter_trace")}
        for row in case["rows"]]} for case in cases]
    summary = learning.summarize(cases) if args.panel == "learning" else panel.summarize(cases, methods)
    panel.write_json(output, {"implementation": VERSION, "frozen_protocol": str(protocol_path),
        "corrected_from": str(old_path), "scope": protocol["scope"], "input_provenance": provenance,
        "workers": args.workers, "od_conditions": len(cases), "journeys": sum(len(c["rows"]) for c in cases),
        "methods": methods, "samples": 8, "transition_budget": budget, "summary": summary, "cases": compact}, indent=2)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
