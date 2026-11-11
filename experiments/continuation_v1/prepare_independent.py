"""Freeze a new parameter date and exogenous seed before evaluating all rules."""
import json
from pathlib import Path
import pickle
import sys
from dataclasses import asdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v2.run_matched_panel import historical_priors, world_distributions, ODS
from src.simulator import JourneyScenario, RegimeSchedule, set_regime_dist_fn


def main():
    output = Path(__file__).with_name("independent_worlds")
    output.mkdir(exist_ok=True)
    protocol = {"parameter_date": "2023-10-15", "world_seed": 913,
                "origins": ODS, "stress": [False, True], "samples": 8,
                "methods": ["legacy", "marginal_cap", "marginal_rescue", "conditioned_cap", "conditioned_rescue"],
                "tuning": "none; all methods frozen before reading primary results",
                "scope": "new generated worlds and parameter date; not independent observed transit events"}
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    with (ROOT / "data/zurich_wide.pkl").open("rb") as handle:
        graph = pickle.load(handle)
    with (ROOT / "data/per_day_distributions.pkl").open("rb") as handle:
        days = pickle.load(handle)
    date, seed = protocol["parameter_date"], protocol["world_seed"]
    routes = {c.route for c in graph.connections}
    priors = historical_priors(days, date, routes)
    schedule = RegimeSchedule([(0, "frozen_independent")])
    index = 0
    for stress in (False, True):
        distributions = world_distributions(days[date], routes, stress)
        set_regime_dist_fn(lambda name: distributions)
        scene = JourneyScenario.sample(graph, schedule, np.random.default_rng(seed))
        for origin, destination in ODS:
            case = {"spec": [date, stress, origin, destination, seed, 120, 5],
                    "records": [], "priors": priors,
                    "world": {str(cid): asdict(event) for cid, event in scene.events.items()}}
            (output / f"case_{index:03d}.json").write_text(json.dumps(case, ensure_ascii=False) + "\n")
            index += 1
    print(json.dumps({"output": str(output), "cases": index, "date": date, "seed": seed}))


if __name__ == "__main__":
    main()
