"""Exact full/H screening on two tiny archived Zurich timetable subgraphs.

The four worlds are deliberately constructed, not observed vehicle events.
All archived segments of each selected trip are retained. The source archive
itself is a 2000-connection slice, so this is not full-city or full-GTFS planning.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import asdict
import json
from math import isclose
from pathlib import Path
import pickle
import resource
import sys
from time import perf_counter
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.finite_transit import OBSERVE, WAIT, build_transit_model
from src.information_planner import ExactPlanner
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import TransitGraph

ODS = [("Zürich, Paradeplatz", "Zürich, Goldbrunnenplatz"),
       ("Zürich, Stauffacher", "Zürich, Triemli")]
DEPARTURE = 490


def select_subgraph(base, origin, destination):
    """Select first two direct-through archived trips by departure/arrival/ID."""
    by_name = {stop.name: sid for sid, stop in base.stops.items()}
    source, target = by_name[origin], by_name[destination]
    trips = defaultdict(list)
    for connection in base.connections:
        trips[connection.trip_id].append(connection)
    eligible = []
    for trip_id, connections in trips.items():
        if not trip_id:
            continue
        for first in sorted(connections, key=lambda c: (c.dep_time, c.id)):
            if first.dep_stop != source or not DEPARTURE <= first.dep_time < DEPARTURE + 20:
                continue
            arrivals = [c.arr_time for c in connections if c.arr_stop == target
                        and c.arr_time >= first.arr_time]
            if arrivals:
                eligible.append((first.dep_time, min(arrivals), trip_id))
                break
    chosen = sorted(eligible)[:2]
    if len(chosen) != 2:
        raise ValueError(f"Two archived through-trips unavailable for {origin} -> {destination}")
    selected_ids = {trip_id for _, _, trip_id in chosen}
    selected = [c for c in base.connections if c.trip_id in selected_ids]
    subgraph = TransitGraph()
    stop_ids = {sid for c in selected for sid in (c.dep_stop, c.arr_stop)}
    for sid in sorted(stop_ids):
        subgraph.add_stop(copy.copy(base.stops[sid]))
    old_to_new = {}
    for connection in sorted(selected, key=lambda c: c.id):
        old_id = connection.id
        subgraph.add_connection(copy.copy(connection))
        old_to_new[old_id] = subgraph.connections[-1].id
    provenance = {"origin": origin, "destination": destination,
                  "origin_id": source, "destination_id": target,
                  "selection_rule": "first two archived through-trips by origin departure, destination arrival, trip ID; [490,510)",
                  "trips": [{"trip_id": trip_id, "origin_departure": dep,
                             "destination_arrival": arr,
                             "archived_segment_count": len(trips[trip_id])}
                            for dep, arr, trip_id in chosen],
                  "old_connection_id_to_subgraph_id": old_to_new,
                  "subgraph_stops": len(subgraph.stops),
                  "subgraph_connections": len(subgraph.connections),
                  "coverage": "all segments of selected trips present in the archived 2000-connection graph; full GTFS coverage not asserted"}
    return subgraph, source, target, provenance


def constructed_bank(graph, trip_ids):
    specifications = [
        {"name": "all_operate_on_schedule", "canceled_trip": None, "shared_delay": 0},
        {"name": "primary_canceled", "canceled_trip": trip_ids[0], "shared_delay": 0},
        {"name": "backup_canceled", "canceled_trip": trip_ids[1], "shared_delay": 0},
        {"name": "shared_six_minute_delay", "canceled_trip": None, "shared_delay": 6},
    ]
    bank, metadata = [], []
    for specification in specifications:
        events = {}
        for connection in graph.connections:
            canceled = connection.trip_id == specification["canceled_trip"]
            delay = specification["shared_delay"]
            events[connection.id] = VehicleEvent(999 if canceled else delay, canceled,
                                                 connection.dep_time if canceled else connection.dep_time + delay,
                                                 connection.arr_time if canceled else connection.arr_time + delay)
        bank.append((0.25, JourneyScenario(events)))
        metadata.append(specification | {"probability": 0.25,
                                         "events_by_subgraph_connection_id": {
                                             cid: asdict(event) for cid, event in events.items()}})
    return bank, metadata


def profile(model, candidate_rule=None, clairvoyant=False):
    tracemalloc.start()
    start = perf_counter()
    planner = ExactPlanner(model, candidate_rule)
    if clairvoyant:
        value = planner.clairvoyant_lower_bound()
        policy_value, decision_loss = None, None
        solve_ms = (perf_counter() - start) * 1000
    else:
        value = planner.solve().value
        solve_ms = (perf_counter() - start) * 1000
        policy_value = planner.evaluate(planner.optimal_action)
        decision_loss = planner.decision_loss(planner.optimal_action)
        if not isclose(policy_value, value, abs_tol=1e-9) or abs(decision_loss) > 1e-9:
            raise AssertionError("Optimal policy execution must reproduce the causal Bellman value")
    total_ms = (perf_counter() - start) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"value": value, "executed_policy_cost": policy_value,
            "within_candidate_loss": decision_loss, "solve_ms": solve_ms,
            "solve_plus_policy_verification_ms": total_ms,
            "peak_traced_python_bytes": peak, "stats": dict(planner.stats)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("exact_transit_results.json"))
    args = parser.parse_args()
    source = ROOT / "data/zurich_wide.pkl"
    start = perf_counter()
    with source.open("rb") as handle:
        base = pickle.load(handle)
    load_ms = (perf_counter() - start) * 1000
    cases = []
    for origin, destination in ODS:
        start = perf_counter()
        graph, src, dst, provenance = select_subgraph(base, origin, destination)
        trip_ids = [trip["trip_id"] for trip in provenance["trips"]]
        bank, worlds = constructed_bank(graph, trip_ids)
        selection_and_bank_ms = (perf_counter() - start) * 1000
        excluded_ids = {c.id for c in graph.connections if c.trip_id == trip_ids[1]}
        for cap in (30, 60):
            start = perf_counter()
            model = build_transit_model(graph, bank, src, dst, DEPARTURE, DEPARTURE + cap)
            model_build_ms = (perf_counter() - start) * 1000
            rule = lambda node: tuple(action for action in model.actions(node.state)
                                      if action in (OBSERVE, WAIT) or action[1] not in excluded_ids)
            full = profile(model)
            restricted = profile(model, rule)
            clairvoyant = profile(model, clairvoyant=True)
            support_loss = restricted["value"] - full["value"]
            if support_loss < -1e-9 or clairvoyant["value"] > full["value"] + 1e-9:
                raise AssertionError("Candidate inclusion / information relaxation ordering failed")
            cases.append({"provenance": provenance, "timeout_cap": cap,
                          "worlds": worlds, "excluded_whole_trip": trip_ids[1],
                          "excluded_connection_ids": sorted(excluded_ids),
                          "selection_and_bank_ms": selection_and_bank_ms,
                          "model_build_ms": model_build_ms,
                          "full_causal_optimum": full, "restricted_causal_optimum": restricted,
                          "clairvoyant_information_relaxation": clairvoyant,
                          "candidate_support_loss": support_loss,
                          "within_candidate_loss_of_restricted_optimal_policy": restricted["within_candidate_loss"],
                          "same_information_policy_gap": restricted["executed_policy_cost"] - full["value"]})
            print(json.dumps({"origin": origin, "cap": cap, "full": full["value"],
                              "restricted": restricted["value"], "support_loss": support_loss,
                              "full_solve_ms": full["solve_ms"]}), flush=True)
    result = {"scope": "exact known-model diagnostics on two tiny archived Zurich subgraphs and four constructed worlds; not observed event replay or full-city computation",
              "source": str(source.relative_to(ROOT)),
              "source_stops": len(base.stops), "source_connections": len(base.connections),
              "source_load_ms": load_ms, "departure": DEPARTURE,
              "information": "global cumulative occurred departure/arrival announcements; explicit cancellations at scheduled departure; no future actual times",
              "timing_note": "tracemalloc enabled during solve; measurements describe this reference implementation, not an optimized routing baseline",
              "cost": "actual elapsed duration when arrived, otherwise common timeout cap",
              "tuning": "none; selection rule, four worlds and caps fixed before solving",
              "inference": "deterministic finite-bank enumeration; no statistical population or real disruption benefit inference",
              "research_gate": "not passed: small-subgraph support loss is diagnosed, full-network bottleneck and new algorithm value remain unestablished; no phase-four expansion",
              "process_peak_rss_kib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "cases": cases}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}), flush=True)


if __name__ == "__main__":
    main()
