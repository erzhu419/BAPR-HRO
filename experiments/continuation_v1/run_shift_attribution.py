"""Same-shift-per-trip version of the exact probability/rescue factorial.

Unlike the preserved incoming-duration toy, every operating trip here shifts
its departure and arrival by one common D. Incoming departs after the root
choice in every world, so the initial local poll does not reveal its shift.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.continuation_v1.run_exact_attribution import run_case, summarize
from src.committed_transit import build_committed_model
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def make_shift_model(relationship, cancel_probability, cap, backup_arrival):
    pairs = {"same": ((0, 0), (6, 6)),
             "independent": ((0, 0), (0, 6), (6, 0), (6, 6)),
             "opposite": ((0, 6), (6, 0))}[relationship]
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=2))
    for connection in [Connection(0, "incoming", "incoming", 0, 1, 1, 7),
                       Connection(1, "target", "target", 1, 2, 10, 18),
                       Connection(2, "backup", "backup", 1, 2, 16, backup_arrival),
                       Connection(3, "direct", "direct", 0, 2, 0, 28)]:
        graph.add_connection(connection)
    bank, specifications = [], []
    for incoming_shift, target_shift in pairs:
        for canceled, cancel_weight in ((False, 1 - cancel_probability), (True, cancel_probability)):
            if not cancel_weight:
                continue
            probability = cancel_weight / len(pairs)
            events = {
                0: VehicleEvent(incoming_shift, False, 1 + incoming_shift, 7 + incoming_shift),
                1: VehicleEvent(999 if canceled else target_shift, canceled,
                                10 if canceled else 10 + target_shift,
                                18 if canceled else 18 + target_shift),
                2: VehicleEvent(0, False, 16, backup_arrival),
                3: VehicleEvent(0, False, 0, 28),
            }
            # This assertion detects precisely the earlier model mismatch:
            # all operating segments must shift departure and arrival together.
            for connection in graph.connections:
                event = events[connection.id]
                if not event.canceled:
                    if not (event.departure - connection.dep_time
                            == event.arrival - connection.arr_time == event.delay):
                        raise AssertionError("Operating trip violates its single-shift model")
            bank.append((probability, JourneyScenario(events)))
            specifications.append({"probability": probability, "incoming_shift": incoming_shift,
                                   "target_shift": target_shift, "target_canceled": canceled,
                                   "events": {cid: asdict(event) for cid, event in events.items()}})
    return build_committed_model(graph, bank, 0, 2, 0, cap), specifications


def run_shift_case(specification):
    return run_case(*specification, model_builder=make_shift_model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("shift_attribution_results.json"))
    args = parser.parse_args()
    specifications = [(relation, cancel, cap, backup) for relation in ("same", "independent", "opposite")
                      for cap in (40, 60, 90) for backup in (36, 48) for cancel in (0, 0.2)]
    indexed = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_shift_case, spec): i for i, spec in enumerate(specifications)}
        for future in as_completed(jobs):
            index = jobs[future]
            case = future.result()
            indexed[index] = case
            print(json.dumps({"completed": len(indexed), "index": index,
                              "relationship": case["relationship"], "cap": case["deadline_cap"],
                              "backup_arrival": case["backup_arrival"],
                              "cancel_probability": case["target_cancel_probability"],
                              "incoming_Q": case["true_incoming_Q"], "p_true": case["p_true"],
                              "optimum": case["full_same_information_root_restricted_optimum"]}), flush=True)
    cases = [indexed[i] for i in range(len(specifications))]
    result = {
        "scope": "exact known-model same-shift-per-trip constructed bank; no observed event or estimated model benefit",
        "preserved_duration_toy": "exact_attribution_results.json is retained separately; this common-shift model is the primary project-consistent mechanism check",
        "timetable": {"incoming": [1, 7], "target": [10, 18], "backup_departure": 16,
                      "backup_arrivals": [36, 48], "direct": [0, 28], "minimum_transfer": 2},
        "world_model": "D_incoming,D_target in {0,6}; each operating trip shifts both departure and arrival by its D; incoming and target joint relation same/independent/opposite; target cancellation independent with probability 0 or .2",
        "root_information": "decision at zero; incoming departs at 1 or 7, so initial local poll contains only direct departure and cannot reveal incoming D",
        "root_choice": "forced initial OBSERVE then incoming connection 0 or direct 3; root WAIT excluded identically from policies and reference",
        "suffix": "same-information exact causal optimum, never a realized-world rescue minimum",
        "information": "committed_transit local last10/max5 publication, zero buffer on same trip, two-minute transfer, twelve-minute commitment",
        "estimation_design": "2x2 independent/true catch probability by cap/causal rescue; successful conditional total journey cost fixed true",
        "conditional_endpoint": "six same/no-cancel cells have p=1: R_true null and independent/causal unidentifiable; no invented impossible-branch cost",
        "runtime_contract_preflight": "tests/test_committed_transit.py::test_same_shift_attribution_fixture_matches_runtime_before_information: two opposite-relation worlds passed full message, decision-time, learning-order and cost comparison (24/36)",
        "memoization": "reuse the public-state/posterior/stage exact specialization verified against full-history reference; no repeated generic solve in this grid",
        "workers": args.workers,
        "tuning": "none; all 36 parameter cells specified before execution",
        "summary": summarize(cases), "cases": cases,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases),
                      "harmful_single_repairs": len(result["summary"]["harmful_single_repairs"])}), flush=True)


if __name__ == "__main__":
    main()
