"""Crew Scheduling Experiment: SPP vs TS vs BAPR-HRO LCB.

Compares four approaches across multiple disruption scenarios:
  1. Static SPP: solve once with scheduled durations, never update
  2. Recompute SPP: re-solve from scratch when disruptions occur
  3. Thompson Sampling (TS): same structure as LCB, but samples from posterior
  4. BAPR-HRO LCB: keep structure, update ranking via pessimistic posteriors

Key metric: Effective Cost = crew_cost + PENALTY * n_uncovered_pieces
  (an uncovered piece = a bus trip with no driver = unacceptable in operations)

Also includes multi-day convergence experiment showing sigma_max shrinkage.
"""

from __future__ import annotations

import numpy as np
import time
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from crew_env import CrewSchedulingEnv, WorkRules
from baselines.spp_solver import SPPSolver
from baselines.ts_scheduler import TSCrewScheduler
from lcb_scheduler import LCBCrewScheduler

# Penalty per uncovered piece (hours): a trip with no driver is catastrophic
UNCOVERED_PENALTY = 10.0


def compute_effective_cost(metrics: dict) -> float:
    """Total cost including penalty for uncovered pieces."""
    return metrics["total_cost"] + UNCOVERED_PENALTY * metrics["n_uncovered"]


def run_single_day(
    scenario: str,
    method: str,
    seed: int,
    n_drivers: int = 40,
    beta: float = 1.5,
    gamma: float = 30.0,
    n_prior_days: int = 5,
) -> dict:
    """Simulate one day of crew scheduling."""
    env = CrewSchedulingEnv(
        n_drivers=n_drivers,
        seed=seed,
        disruption_level=scenario,
    )

    result = {
        "scenario": scenario, "method": method, "seed": seed,
        "n_trips": len(env.trips), "n_pieces": len(env.pieces),
    }

    # --- Shared simulation helpers ---
    def simulate_trips():
        total_delay = 0.0
        for trip in env.trips:
            actual = env.sample_trip_duration(trip)
            trip.delay = actual - (trip.scheduled_arr - trip.scheduled_dep)
            total_delay += trip.delay
        return total_delay

    def check_absences(assignments):
        absent = set()
        for driver in env.drivers:
            if driver.driver_id in assignments:
                if not env.sample_driver_availability(driver):
                    absent.add(driver.driver_id)
        uncovered = set()
        for did in absent:
            if did in assignments:
                uncovered.update(assignments[did])
                del assignments[did]
        return absent, uncovered

    # --- Method-specific logic ---
    if method == "static_spp":
        t0 = time.perf_counter()
        spp = SPPSolver(env)
        spp.enumerate_feasible_duties()
        assignments = spp.solve_greedy()
        plan_time = time.perf_counter() - t0

        total_delay = simulate_trips()
        absent, _ = check_absences(assignments)

        metrics = env.get_schedule_metrics(assignments)
        metrics.update(plan_time=plan_time, replan_time=0.0,
                       total_delay_min=total_delay / 60,
                       n_absent=len(absent))

    elif method == "recompute_spp":
        t0 = time.perf_counter()
        spp = SPPSolver(env)
        spp.enumerate_feasible_duties()
        spp.solve_greedy()
        plan_time = time.perf_counter() - t0

        total_delay = simulate_trips()
        actual_durations = {t.trip_id: (t.scheduled_arr - t.scheduled_dep + t.delay)
                           for t in env.trips}

        t1 = time.perf_counter()
        assignments, _ = spp.solve_with_recompute(actual_durations)
        replan_time = time.perf_counter() - t1

        absent, _ = check_absences(assignments)

        metrics = env.get_schedule_metrics(assignments)
        metrics.update(plan_time=plan_time, replan_time=replan_time,
                       total_delay_min=total_delay / 60,
                       n_absent=len(absent))

    elif method in ("lcb", "ts"):
        if method == "lcb":
            scheduler = LCBCrewScheduler(env, beta=beta, gamma=gamma)
        else:
            scheduler = TSCrewScheduler(env, seed=seed)

        # Prior learning
        t0 = time.perf_counter()
        for day in range(n_prior_days):
            day_rng = np.random.default_rng(seed * 1000 + day)
            for trip in env.trips:
                duration = env.sample_trip_duration(trip)
                delay_min = (duration - (trip.scheduled_arr - trip.scheduled_dep)) / 60
                scheduler.observe_trip_duration(trip.route_id, delay_min)
            for driver in env.drivers:
                if day_rng.random() > driver.true_absence_prob:
                    scheduler.observe_driver_present(driver.driver_id)
                else:
                    scheduler.observe_driver_absent(driver.driver_id)

        scheduler.compute_structure()
        plan_time = time.perf_counter() - t0

        assignments, _ = scheduler.assign()

        # Simulate execution + online updates
        total_delay = 0.0
        for trip in env.trips:
            actual = env.sample_trip_duration(trip)
            trip.delay = actual - (trip.scheduled_arr - trip.scheduled_dep)
            total_delay += trip.delay
            scheduler.observe_trip_duration(trip.route_id, trip.delay / 60)

        absent = set()
        for driver in env.drivers:
            if driver.driver_id in assignments:
                if not env.sample_driver_availability(driver):
                    absent.add(driver.driver_id)
                    scheduler.observe_driver_absent(driver.driver_id)
                else:
                    scheduler.observe_driver_present(driver.driver_id)

        # Adaptive re-assignment
        t1 = time.perf_counter()
        if absent:
            disrupted = set()
            for did in absent:
                if did in assignments:
                    disrupted.update(assignments[did])
            available = [d.driver_id for d in env.drivers
                         if d.driver_id not in absent]
            assignments, _ = scheduler.adaptive_reassign(
                assignments, disrupted, available)
        replan_time = time.perf_counter() - t1

        metrics = env.get_schedule_metrics(assignments)
        metrics.update(plan_time=plan_time, replan_time=replan_time,
                       total_delay_min=total_delay / 60,
                       n_absent=len(absent))

        if method == "lcb":
            bs = scheduler.get_belief_summary()
            metrics["sigma_max"] = bs["sigma_max_route"]
            metrics["theoretical_bound"] = scheduler.theoretical_bound(len(env.pieces))

    metrics["effective_cost"] = compute_effective_cost(metrics)
    result.update(metrics)
    return result


# -----------------------------------------------------------------------
# Multi-day convergence experiment
# -----------------------------------------------------------------------

def run_convergence(
    scenario: str = "disrupted",
    n_days: int = 30,
    seed: int = 0,
    n_drivers: int = 40,
) -> list[dict]:
    """Run LCB over multiple days, tracking sigma_max convergence.

    Shows that posterior std shrinks as O(1/sqrt(n_days)),
    and effective cost converges to near-optimal.
    """
    env = CrewSchedulingEnv(n_drivers=n_drivers, seed=seed,
                            disruption_level=scenario)

    lcb = LCBCrewScheduler(env, beta=1.5, gamma=30.0)
    lcb.compute_structure()

    daily_results = []

    for day in range(n_days):
        day_rng = np.random.default_rng(seed * 10000 + day)

        # Reset env randomness for this day
        env.rng = day_rng

        # Assign using current beliefs
        assignments, assign_time = lcb.assign()

        # Simulate day
        for trip in env.trips:
            actual = env.sample_trip_duration(trip)
            trip.delay = actual - (trip.scheduled_arr - trip.scheduled_dep)
            lcb.observe_trip_duration(trip.route_id, trip.delay / 60)

        # Driver availability
        absent = set()
        for driver in env.drivers:
            if driver.driver_id in assignments:
                if not env.sample_driver_availability(driver):
                    absent.add(driver.driver_id)
                    lcb.observe_driver_absent(driver.driver_id)
                else:
                    lcb.observe_driver_present(driver.driver_id)

        if absent:
            disrupted = set()
            for did in absent:
                if did in assignments:
                    disrupted.update(assignments[did])
            available = [d.driver_id for d in env.drivers
                         if d.driver_id not in absent]
            assignments, _ = lcb.adaptive_reassign(
                assignments, disrupted, available)

        metrics = env.get_schedule_metrics(assignments)
        bs = lcb.get_belief_summary()

        daily_results.append({
            "day": day + 1,
            "effective_cost": compute_effective_cost(metrics),
            "total_cost": metrics["total_cost"],
            "coverage_rate": metrics["coverage_rate"],
            "n_uncovered": metrics["n_uncovered"],
            "sigma_max": bs["sigma_max_route"],
            "n_observations": bs["total_observations"],
            "theoretical_bound": lcb.theoretical_bound(len(env.pieces)),
        })

    return daily_results


# -----------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------

def run_experiment(
    scenarios: list[str] = None,
    methods: list[str] = None,
    n_seeds: int = 20,
    n_drivers: int = 40,
) -> list[dict]:
    if scenarios is None:
        scenarios = ["normal", "rush_hour", "disrupted", "driver_shortage"]
    if methods is None:
        methods = ["static_spp", "recompute_spp", "ts", "lcb"]

    all_results = []

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"{'='*60}")

        for method in methods:
            eff_costs, coverages = [], []

            for seed in range(n_seeds):
                result = run_single_day(scenario=scenario, method=method,
                                        seed=seed, n_drivers=n_drivers)
                all_results.append(result)
                eff_costs.append(result["effective_cost"])
                coverages.append(result["coverage_rate"])

            print(f"\n  {method:18s}  eff_cost={np.mean(eff_costs):7.1f}±{np.std(eff_costs):5.1f}"
                  f"  coverage={np.mean(coverages):.0%}")

    return all_results


def save_results(results, filename, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    Path(output_dir).mkdir(exist_ok=True)
    filepath = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 1: Cross-method comparison (4 scenarios)")
    print("=" * 60)
    results = run_experiment(n_seeds=20, n_drivers=40)
    save_results(results, "crew_comparison_v2.json")

    # Summary
    import pandas as pd
    df = pd.DataFrame(results)
    print("\n\nSUMMARY (effective cost = crew_cost + 10h × uncovered_pieces)")
    print("-" * 80)
    summary = df.groupby(["scenario", "method"]).agg(
        eff_cost_mean=("effective_cost", "mean"),
        eff_cost_std=("effective_cost", "std"),
        coverage=("coverage_rate", "mean"),
        overtime=("overtime_hours", "mean"),
    ).round(2)
    print(summary.to_string())

    print("\n\n" + "=" * 60)
    print("EXPERIMENT 2: Multi-day convergence (disrupted scenario)")
    print("=" * 60)
    conv_results = run_convergence(scenario="disrupted", n_days=30, seed=0)
    save_results(conv_results, "convergence.json")

    print("\nDay  sigma_max  eff_cost  coverage  bound")
    print("-" * 50)
    for r in conv_results:
        if r["day"] <= 5 or r["day"] % 5 == 0:
            print(f"{r['day']:3d}  {r['sigma_max']:9.4f}  {r['effective_cost']:8.1f}"
                  f"  {r['coverage_rate']:.0%}     {r['theoretical_bound']:6.2f}")
