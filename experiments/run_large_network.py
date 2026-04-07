"""Compare routing methods on the large grid network with corridor disruptions."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time

from src.large_network import create_grid_network, create_grid_regime_distributions
from src.router import StaticRouter, AdaptiveRouter, PeriodicRouter
from src.simulator import (simulate_journey, RegimeSchedule,
                           set_regime_dist_fn, _regime_dist_cache)


def run_large_experiment(
    n_journeys: int = 30,
    scenario: str = "central_disruption",
    seed: int = 42,
):
    # Use grid network distributions
    set_regime_dist_fn(create_grid_regime_distributions)

    if scenario == "no_disruption":
        schedule = RegimeSchedule(shifts=[(0, "normal")])
    elif scenario == "central_disruption":
        # Normal → central corridor collapses → recovers
        schedule = RegimeSchedule(shifts=[
            (0, "normal"),
            (490, "central_disruption"),
            (560, "normal"),
        ])
    elif scenario == "full_chaos":
        schedule = RegimeSchedule(shifts=[
            (0, "normal"),
            (485, "central_disruption"),
            (520, "full_chaos"),
            (560, "normal"),
        ])
    else:
        raise ValueError(scenario)

    regime_names = ["normal", "south_weather", "central_disruption", "full_chaos"]

    methods = {
        "Static": lambda: StaticRouter(create_grid_network()),
        "Periodic-5m": lambda: PeriodicRouter(create_grid_network(), 5),
        "Adaptive": lambda: AdaptiveRouter(
            create_grid_network(),
            regime_names=regime_names,
            recompute_threshold=0.5,
            hazard_rate=0.05,
            regime_dist_fn=create_grid_regime_distributions,
        ),
    }

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario} | {n_journeys} journeys | Grid 7x7 (49 stops, 12 lines)")
    print(f"{'='*70}")

    for method_name, make_router in methods.items():
        _regime_dist_cache.clear()
        travel_times = []
        replans = []
        comp_times = []
        t_start = time.time()

        for i in range(n_journeys):
            router = make_router()
            rng = np.random.default_rng(seed + i)
            t_dep = 480 + rng.integers(0, 15)

            result = simulate_journey(
                graph=router.graph if hasattr(router, 'graph') else create_grid_network(),
                router=router,
                s_source=0,
                s_dest=48,
                t_depart=t_dep,
                regime_schedule=schedule,
                rng=rng,
                max_time=180,
            )
            travel_times.append(result.arrival_time - result.departure_time)
            replans.append(result.n_replans)
            comp_times.append(result.total_computation_ms)

        wall_time = time.time() - t_start
        arr = np.array(travel_times)
        print(f"  {method_name:<15s}: mean={arr.mean():6.1f}  med={np.median(arr):6.1f}  "
              f"p95={np.percentile(arr,95):6.1f}  "
              f"replans={np.mean(replans):4.1f}  "
              f"comp={np.mean(comp_times):7.0f}ms  "
              f"wall={wall_time:5.1f}s")

    return


if __name__ == "__main__":
    for scenario in ["no_disruption", "central_disruption", "full_chaos"]:
        run_large_experiment(n_journeys=20, scenario=scenario)
