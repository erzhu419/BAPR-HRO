"""Crew Scheduling Experiment: SPP vs BAPR-HRO LCB.

Compares three approaches across multiple disruption scenarios:
  1. Static SPP: solve once with scheduled durations, never update
  2. Recompute SPP: re-solve from scratch when disruptions occur
  3. BAPR-HRO LCB: keep structure, update ranking via Bayesian posteriors

Scenarios:
  - normal: small random delays (~1 min mean)
  - rush_hour: peak-hour congestion (5 min delays during peak)
  - disrupted: major route disruption (route 0 has 10 min delays)
  - driver_shortage: 2x absence probability
"""

from __future__ import annotations

import numpy as np
import time
import json
from pathlib import Path
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from crew_env import CrewSchedulingEnv, WorkRules
from baselines.spp_solver import SPPSolver
from lcb_scheduler import LCBCrewScheduler


def run_single_day(
    scenario: str,
    method: str,
    seed: int,
    n_drivers: int = 20,
    beta: float = 1.5,
    gamma: float = 30.0,
    n_prior_days: int = 5,
) -> dict:
    """Simulate one day of crew scheduling.

    Phase 1: Prior learning (n_prior_days of observations)
    Phase 2: Assignment (create schedule)
    Phase 3: Execution (simulate actual trip durations, handle disruptions)
    """
    rng = np.random.default_rng(seed)

    # --- Phase 1: Build environment ---
    env = CrewSchedulingEnv(
        n_drivers=n_drivers,
        seed=seed,
        disruption_level=scenario,
    )

    result = {
        "scenario": scenario,
        "method": method,
        "seed": seed,
        "n_trips": len(env.trips),
        "n_pieces": len(env.pieces),
        "n_drivers": n_drivers,
    }

    if method == "static_spp":
        # --- Static SPP: solve once, never update ---
        t0 = time.perf_counter()
        spp = SPPSolver(env)
        spp.enumerate_feasible_duties()
        assignments = spp.solve_greedy()
        plan_time = time.perf_counter() - t0

        # Simulate execution with actual durations
        actual_durations = {}
        total_delay = 0.0
        for trip in env.trips:
            actual = env.sample_trip_duration(trip)
            actual_durations[trip.trip_id] = actual
            trip.delay = actual - (trip.scheduled_arr - trip.scheduled_dep)
            total_delay += trip.delay

        # Check driver availability
        absent_drivers = set()
        for driver in env.drivers:
            if driver.driver_id in assignments:
                if not env.sample_driver_availability(driver):
                    absent_drivers.add(driver.driver_id)

        # Pieces assigned to absent drivers become uncovered
        uncovered_from_absence = set()
        for did in absent_drivers:
            if did in assignments:
                uncovered_from_absence.update(assignments[did])
                del assignments[did]

        metrics = env.get_schedule_metrics(assignments)
        metrics["uncovered_from_absence"] = len(uncovered_from_absence)
        metrics["plan_time"] = plan_time
        metrics["replan_time"] = 0.0
        metrics["total_delay_minutes"] = total_delay / 60
        metrics["n_absent_drivers"] = len(absent_drivers)

        result.update(metrics)

    elif method == "recompute_spp":
        # --- Recompute SPP: re-solve with actual durations ---
        t0 = time.perf_counter()
        spp = SPPSolver(env)
        spp.enumerate_feasible_duties()
        initial_assignments = spp.solve_greedy()
        plan_time = time.perf_counter() - t0

        # Simulate execution
        actual_durations = {}
        for trip in env.trips:
            actual = env.sample_trip_duration(trip)
            actual_durations[trip.trip_id] = actual
            trip.delay = actual - (trip.scheduled_arr - trip.scheduled_dep)

        # Check driver availability
        absent_drivers = set()
        for driver in env.drivers:
            if driver.driver_id in initial_assignments:
                if not env.sample_driver_availability(driver):
                    absent_drivers.add(driver.driver_id)

        # Full recompute with actual durations (excluding absent drivers)
        t1 = time.perf_counter()
        assignments, replan_time_inner = spp.solve_with_recompute(actual_durations)
        # Remove absent drivers
        for did in absent_drivers:
            if did in assignments:
                del assignments[did]
        replan_time = time.perf_counter() - t1

        total_delay = sum(t.delay for t in env.trips)
        metrics = env.get_schedule_metrics(assignments)
        metrics["plan_time"] = plan_time
        metrics["replan_time"] = replan_time
        metrics["total_delay_minutes"] = total_delay / 60
        metrics["n_absent_drivers"] = len(absent_drivers)

        result.update(metrics)

    elif method == "lcb":
        # --- BAPR-HRO LCB: structure once, update ranking ---
        lcb = LCBCrewScheduler(env, beta=beta, gamma=gamma)

        # Phase 1: Prior learning from historical days
        t0 = time.perf_counter()
        for day in range(n_prior_days):
            day_rng = np.random.default_rng(seed * 1000 + day)
            for trip in env.trips:
                duration = env.sample_trip_duration(trip)
                delay_min = (duration - (trip.scheduled_arr - trip.scheduled_dep)) / 60
                lcb.observe_trip_duration(trip.route_id, delay_min)

            for driver in env.drivers:
                if day_rng.random() > driver.true_absence_prob:
                    lcb.observe_driver_present(driver.driver_id)
                else:
                    lcb.observe_driver_absent(driver.driver_id)

        # Compute structure once
        lcb.compute_structure()
        plan_time = time.perf_counter() - t0

        # Phase 2: Initial assignment via LCB
        assignments, assign_time = lcb.assign()

        # Phase 3: Simulate execution
        actual_durations = {}
        for trip in env.trips:
            actual = env.sample_trip_duration(trip)
            actual_durations[trip.trip_id] = actual
            trip.delay = actual - (trip.scheduled_arr - trip.scheduled_dep)
            # Online update of beliefs
            delay_min = trip.delay / 60
            lcb.observe_trip_duration(trip.route_id, delay_min)

        # Check driver availability
        absent_drivers = set()
        for driver in env.drivers:
            if driver.driver_id in assignments:
                if not env.sample_driver_availability(driver):
                    absent_drivers.add(driver.driver_id)
                    lcb.observe_driver_absent(driver.driver_id)
                else:
                    lcb.observe_driver_present(driver.driver_id)

        # Adaptive re-assignment (structure preserved, only re-rank)
        t1 = time.perf_counter()
        if absent_drivers:
            disrupted = set()
            for did in absent_drivers:
                if did in assignments:
                    disrupted.update(assignments[did])
            available = [d.driver_id for d in env.drivers
                         if d.driver_id not in absent_drivers]
            assignments, _ = lcb.adaptive_reassign(
                assignments, disrupted, available
            )
        replan_time = time.perf_counter() - t1

        total_delay = sum(t.delay for t in env.trips)
        metrics = env.get_schedule_metrics(assignments)
        metrics["plan_time"] = plan_time
        metrics["replan_time"] = replan_time
        metrics["total_delay_minutes"] = total_delay / 60
        metrics["n_absent_drivers"] = len(absent_drivers)

        # Add LCB-specific metrics
        belief_summary = lcb.get_belief_summary()
        metrics["sigma_max"] = belief_summary["sigma_max_route"]
        metrics["theoretical_bound"] = lcb.theoretical_bound(len(env.pieces))
        metrics["n_observations"] = belief_summary["total_observations"]

        result.update(metrics)

    return result


def run_experiment(
    scenarios: list[str] = None,
    methods: list[str] = None,
    n_seeds: int = 20,
    n_drivers: int = 20,
) -> list[dict]:
    """Run full comparison experiment."""
    if scenarios is None:
        scenarios = ["normal", "rush_hour", "disrupted", "driver_shortage"]
    if methods is None:
        methods = ["static_spp", "recompute_spp", "lcb"]

    all_results = []

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"{'='*60}")

        for method in methods:
            costs = []
            plan_times = []
            replan_times = []
            coverages = []
            overtimes = []

            for seed in range(n_seeds):
                result = run_single_day(
                    scenario=scenario,
                    method=method,
                    seed=seed,
                    n_drivers=n_drivers,
                )
                all_results.append(result)
                costs.append(result["total_cost"])
                plan_times.append(result["plan_time"])
                replan_times.append(result["replan_time"])
                coverages.append(result["coverage_rate"])
                overtimes.append(result["overtime_hours"])

            print(f"\n  Method: {method}")
            print(f"    Cost:      {np.mean(costs):.2f} +/- {np.std(costs):.2f}")
            print(f"    Coverage:  {np.mean(coverages):.1%} +/- {np.std(coverages):.1%}")
            print(f"    Overtime:  {np.mean(overtimes):.2f}h +/- {np.std(overtimes):.2f}h")
            print(f"    Plan time: {np.mean(plan_times)*1000:.1f}ms")
            print(f"    Replan:    {np.mean(replan_times)*1000:.1f}ms")

            if method == "lcb" and all_results:
                last = all_results[-1]
                if "sigma_max" in last:
                    print(f"    sigma_max: {last['sigma_max']:.3f}")
                    print(f"    Bound:     {last['theoretical_bound']:.2f}")

    return all_results


def save_results(results: list[dict], output_dir: str | Path = None):
    """Save experiment results to JSON."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / "crew_comparison.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    print("Bus Crew Scheduling: SPP vs BAPR-HRO LCB Comparison")
    print("=" * 60)

    results = run_experiment(n_seeds=20, n_drivers=40)
    save_results(results)

    # Summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    import pandas as pd
    df = pd.DataFrame(results)

    summary = df.groupby(["scenario", "method"]).agg({
        "total_cost": ["mean", "std"],
        "coverage_rate": "mean",
        "overtime_hours": "mean",
        "plan_time": "mean",
        "replan_time": "mean",
    }).round(3)

    print(summary.to_string())
