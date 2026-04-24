"""Reproduce historical synthetic results: V1 / V2 / DRO / Adaptive-β on disrupted_402.

Target: V2-LCB -18.6%, V1-LCB -12.7% from bdf5851
         Adaptive-β -12.2% from 259fbcd/cced236

Output: results/synthetic_reproduction.json
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from collections import defaultdict

from src.synthetic_network import create_bus_story_network
from src.router import StaticRouter
from src.bandit_router import BanditRouter
from src.bandit_router_v2 import BanditRouterV2
from src.dro_router import DRORouter
from src.adaptive_bandit_router import AdaptiveBetaBanditRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import RegimeSchedule


def run_scenario(scenario_name, schedule, n_journeys=100, seed=42):
    rng = np.random.default_rng(seed)
    methods = {
        "Static":      lambda g: StaticRouter(g),
        "V1-LCB":      lambda g: BanditRouter(g),
        "V2-LCB":      lambda g: BanditRouterV2(g, n_estimators=5,
                                                beta_base=1.0, beta_ood=1.0, seed=seed),
        "DRO":         lambda g: DRORouter(g, beta=1.5, gamma=60.0),
        "Adaptive-β":  lambda g: AdaptiveBetaBanditRouter(g),
    }

    out = {}
    for name, make_router in methods.items():
        tts, timeouts = [], 0
        for i in range(n_journeys):
            graph = create_bus_story_network()
            router = make_router(graph)
            if isinstance(router, AdaptiveBetaBanditRouter):
                router.route(0, 9, 490)  # warm up before journey
            t_dep = 480 + rng.integers(0, 20)
            jrng = np.random.default_rng(seed + i)
            res = simulate_bandit_journey(
                graph, router, 0, 9, int(t_dep), schedule, jrng, 180)
            tt = res.arrival_time - res.departure_time
            tts.append(tt)
            if tt >= 180:
                timeouts += 1
        arr = np.array(tts)
        out[name] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "std": float(arr.std()),
            "timeouts": int(timeouts),
            "n": n_journeys,
        }
        print(f"  {name:12s} mean={arr.mean():.1f} p95={np.percentile(arr,95):.1f} "
              f"timeouts={timeouts}/{n_journeys}")

    # improvements over Static
    base = out["Static"]["mean"]
    for name in out:
        if name == "Static":
            out[name]["improvement_pct"] = 0.0
        else:
            out[name]["improvement_pct"] = (base - out[name]["mean"]) / base * 100
    return out


if __name__ == "__main__":
    print("=" * 70)
    print("Synthetic Reproduction: V1 / V2 / DRO / Adaptive-β")
    print("=" * 70)

    scenarios = {
        "no_disruption": RegimeSchedule(shifts=[(0, "normal")]),
        "disrupted_402": RegimeSchedule(shifts=[
            (0, "normal"), (490, "disrupted_402"), (540, "normal")]),
        "rush_hour": RegimeSchedule(shifts=[
            (0, "normal"), (480, "rush_hour"), (570, "normal")]),
        "multi_shift": RegimeSchedule(shifts=[
            (0, "normal"), (485, "rush_hour"),
            (510, "disrupted_402"), (540, "normal")]),
    }

    results = {}
    for scen_name, sched in scenarios.items():
        print(f"\n--- {scen_name} ---")
        t0 = time.time()
        results[scen_name] = run_scenario(scen_name, sched, n_journeys=100, seed=42)
        print(f"  [{time.time()-t0:.1f}s]")
        # summary
        base = results[scen_name]["Static"]["mean"]
        print(f"  Improvements over Static (mean={base:.1f}):")
        for m, s in results[scen_name].items():
            if m != "Static":
                print(f"    {m:12s} {s['improvement_pct']:+.1f}%")

    os.makedirs("experiments/swiss_full/results", exist_ok=True)
    out_path = "experiments/swiss_full/results/synthetic_reproduction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, "item") else str(o))
    print(f"\nSaved to {out_path}")
