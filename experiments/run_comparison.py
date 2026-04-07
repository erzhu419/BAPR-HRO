"""Compare static vs adaptive vs periodic routing under regime shifts.

This is the main evaluation script. Simulates N journeys under different
regime shift scenarios and compares the three routing strategies.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from collections import defaultdict

from src.synthetic_network import create_bus_story_network, create_regime_distributions
from src.router import StaticRouter, AdaptiveRouter, PeriodicRouter
from src.simulator import simulate_journey, RegimeSchedule, JourneyResult


def run_experiment(
    n_journeys: int = 100,
    scenario: str = "disrupted_402",
    seed: int = 42,
):
    """Run comparison experiment.

    Args:
        n_journeys: Number of journeys to simulate per method.
        scenario: Regime shift scenario.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)

    # Define regime schedule based on scenario
    if scenario == "disrupted_402":
        # Normal until 8:10, then 402 disrupted, normal again at 9:00
        schedule = RegimeSchedule(shifts=[
            (0, "normal"),
            (490, "disrupted_402"),
            (540, "normal"),
        ])
    elif scenario == "rush_hour":
        schedule = RegimeSchedule(shifts=[
            (0, "normal"),
            (480, "rush_hour"),
            (570, "normal"),
        ])
    elif scenario == "multi_shift":
        # Multiple regime changes
        schedule = RegimeSchedule(shifts=[
            (0, "normal"),
            (485, "rush_hour"),
            (510, "disrupted_402"),
            (540, "normal"),
        ])
    elif scenario == "no_disruption":
        schedule = RegimeSchedule(shifts=[(0, "normal")])
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    methods = {
        "Static": lambda g: StaticRouter(g),
        "Periodic-5min": lambda g: PeriodicRouter(g, recompute_interval=5),
        "Adaptive-BOCD": lambda g: AdaptiveRouter(
            g,
            regime_names=["normal", "rush_hour", "disrupted_402", "weather"],
            recompute_threshold=0.5,
            hazard_rate=0.05,
        ),
    }

    results: dict[str, list[JourneyResult]] = defaultdict(list)

    for method_name, make_router in methods.items():
        for i in range(n_journeys):
            # Fresh graph for each journey (distributions may change)
            graph = create_bus_story_network()
            router = make_router(graph)

            # Random departure time between 8:00 and 8:20
            t_depart = 480 + rng.integers(0, 20)
            journey_rng = np.random.default_rng(seed + i)  # reproducible per journey

            result = simulate_journey(
                graph=graph,
                router=router,
                s_source=0,
                s_dest=9,
                t_depart=t_depart,
                regime_schedule=schedule,
                rng=journey_rng,
            )
            results[method_name].append(result)

    return results


def print_results(results: dict[str, list], scenario: str):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario}")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Mean Arr':>10} {'Med Arr':>10} {'95th':>10} "
          f"{'Transfers':>10} {'Replans':>10} {'Comp(ms)':>10}")
    print("-" * 70)

    for method, journeys in results.items():
        arrivals = [j.arrival_time for j in journeys]
        departures = [j.departure_time for j in journeys]
        travel_times = [a - d for a, d in zip(arrivals, departures)]

        mean_arr = np.mean(travel_times)
        med_arr = np.median(travel_times)
        p95_arr = np.percentile(travel_times, 95)
        mean_transfers = np.mean([j.n_transfers for j in journeys])
        mean_replans = np.mean([j.n_replans for j in journeys])
        mean_comp = np.mean([j.total_computation_ms for j in journeys])

        print(f"{method:<20} {mean_arr:>10.1f} {med_arr:>10.1f} {p95_arr:>10.1f} "
              f"{mean_transfers:>10.1f} {mean_replans:>10.1f} {mean_comp:>10.1f}")

    # Improvement over static
    static_times = [j.arrival_time - j.departure_time
                    for j in results["Static"]]
    static_mean = np.mean(static_times)

    print(f"\nImprovement over Static:")
    for method, journeys in results.items():
        if method == "Static":
            continue
        times = [j.arrival_time - j.departure_time for j in journeys]
        improvement = static_mean - np.mean(times)
        pct = (improvement / static_mean) * 100 if static_mean > 0 else 0
        print(f"  {method}: {improvement:+.1f} min ({pct:+.1f}%)")


def main():
    scenarios = ["no_disruption", "disrupted_402", "rush_hour", "multi_shift"]

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario}...")
        results = run_experiment(n_journeys=50, scenario=scenario, seed=42)
        print_results(results, scenario)


if __name__ == "__main__":
    main()
