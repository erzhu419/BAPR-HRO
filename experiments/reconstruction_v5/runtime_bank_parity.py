"""Whole-city fixed-bank and nominal-action parity against the retained v6 model."""
from pathlib import Path
import json
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.profile_city_runtime import process_memory
from experiments.reconstruction_v5.runtime_reference_evolving import EvolvingTripBelief as ReferenceBelief
from src.committed_transit import CommittedState
from src.continuation_router import nominal_action
from src.evolving_trip_belief import EvolvingTripBelief


def main():
    source = ROOT / "experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz"
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    graph, _, stops, _, _, _ = dated_graph(data, "2023-07-26", "gtfs_station_max", True)
    pooled = fit["route_priors"]["pooled"]
    priors = {c.route: {key: fit["route_priors"].get(c.route, pooled)[key]
                        for key in ("mean", "std", "cancel_rate")} for c in graph.connections}
    models = [cls(graph, minute=60, public_occurrences=False, **fit["increments"])
              for cls in (ReferenceBelief, EvolvingTripBelief)]
    timings, banks, nominals, states = [], [], [], []
    for model in models:
        rng = np.random.default_rng(np.random.SeedSequence([20230726, 0]))
        cpu_start, wall_start = process_time(), perf_counter()
        scenes = model.sample(priors, rng, 4)
        sample_seconds = perf_counter() - wall_start
        # The fixed bank must be independent of which world/connection is asked first.
        worlds = [None] * 4
        for world in range(3, -1, -1):
            worlds[world] = {cid: scenes[world].events[cid] for cid in reversed(tuple(scenes[world].events))}
        materialized_seconds = perf_counter() - wall_start
        nominal_start = perf_counter()
        nominal = dict(model.mean_events(priors))
        nominal_seconds = perf_counter() - nominal_start
        timings.append({"sample_call_seconds": sample_seconds,
                        "sample_and_full_materialization_seconds": materialized_seconds,
                        "all_nominal_seconds": nominal_seconds,
                        "total_cpu_seconds": process_time() - cpu_start})
        banks.append(worlds)
        nominals.append(nominal)
        states.append(rng.bit_generator.state)
    assert banks[0] == banks[1]
    assert states[0] == states[1]
    fields = ("mean_delay", "departure", "arrival", "cancel_probability")
    max_error = {field: max(abs(getattr(nominals[0][cid], field) - getattr(nominals[1][cid], field))
                           for cid in nominals[0]) for field in fields}
    assert max(max_error.values()) < 1e-7
    actions = []
    menu = set(range(len(graph.connections)))
    for query in data["od_queries"]:
        state = CommittedState(stops[query["origin"]], query["ready_seconds"], None,
                               query["ready_seconds"], initialized=True)
        pair = [nominal_action(graph, state, stops[query["destination"]], nominal, menu,
                               query["deadline_seconds"], 25 * 60, minute=60) for nominal in nominals]
        assert pair[0] == pair[1]
        actions.append(pair[0])
    result = {"scope": "Initial held-out static model only; actual held-out outcomes unused.",
              "date": "2023-07-26", "graph_connections": len(graph.connections),
              "trips": len(models[1]._trips), "worlds": 4, "events_checked": 4 * len(graph.connections),
              "all_vehicle_events_equal": True, "rng_states_equal": True,
              "max_abs_nominal_error": max_error, "initial_nominal_actions": actions,
              "reference_timing": timings[0], "lazy_timing": timings[1],
              "proc_memory_kib": process_memory()}
    output = Path(__file__).with_name("runtime_bank_parity_boundary_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
