"""V1 vs V2 comparison on Durner transit routing + crew scheduling.

V1: Normal-Gamma parametric posterior, fixed beta=1.5
V2: Ensemble disagreement, dynamic beta(s) = beta_base + beta_ood * OOD(s)

Tests both on:
1. Transit routing (Durner environment) — BAPR-HRO's native domain
2. Crew scheduling — cross-domain test
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from collections import defaultdict

from src.synthetic_network import create_bus_story_network, create_regime_distributions
from src.router import StaticRouter
from src.bandit_router import BanditRouter
from src.bandit_router_v2 import BanditRouterV2
from src.ssp_mdp import PosteriorSamplingRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import RegimeSchedule, JourneyResult


# -----------------------------------------------------------------------
# Part 1: Transit routing (Durner environment)
# -----------------------------------------------------------------------

def run_transit_experiment(
    n_journeys: int = 100,
    scenario: str = "disrupted_402",
    seed: int = 42,
):
    """Compare V1 vs V2 on transit routing."""
    rng = np.random.default_rng(seed)

    schedules = {
        "no_disruption": RegimeSchedule(shifts=[(0, "normal")]),
        "disrupted_402": RegimeSchedule(shifts=[
            (0, "normal"), (490, "disrupted_402"), (540, "normal")]),
        "rush_hour": RegimeSchedule(shifts=[
            (0, "normal"), (480, "rush_hour"), (570, "normal")]),
        "multi_shift": RegimeSchedule(shifts=[
            (0, "normal"), (485, "rush_hour"),
            (510, "disrupted_402"), (540, "normal")]),
    }
    schedule = schedules[scenario]

    methods = {
        "Static": lambda g: StaticRouter(g),
        "V1-LCB(β=1.5)": lambda g: BanditRouter(g),
        "V1-TS": lambda g: PosteriorSamplingRouter(g),
        "V2-LCB(dynamic)": lambda g: BanditRouterV2(
            g, n_estimators=5, beta_base=1.0, beta_ood=1.0, seed=seed),
        "V2-LCB(conserv)": lambda g: BanditRouterV2(
            g, n_estimators=5, beta_base=1.5, beta_ood=1.5, seed=seed),
    }

    results: dict[str, list[JourneyResult]] = defaultdict(list)

    for method_name, make_router in methods.items():
        for i in range(n_journeys):
            graph = create_bus_story_network()
            router = make_router(graph)

            t_depart = 480 + rng.integers(0, 20)
            journey_rng = np.random.default_rng(seed + i)

            result = simulate_bandit_journey(
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


def print_transit_results(results: dict[str, list], scenario: str):
    print(f"\n{'='*80}")
    print(f"TRANSIT ROUTING — Scenario: {scenario}")
    print(f"{'='*80}")
    print(f"{'Method':<22} {'Mean TT':>8} {'Med TT':>8} {'P95 TT':>8} "
          f"{'Timeout%':>8} {'Observations':>12}")
    print("-" * 80)

    static_mean = None
    for method, journeys in results.items():
        travel_times = [j.arrival_time - j.departure_time for j in journeys]
        timeouts = sum(1 for tt in travel_times if tt >= 180) / len(travel_times)
        mean_tt = np.mean(travel_times)
        med_tt = np.median(travel_times)
        p95_tt = np.percentile(travel_times, 95)
        mean_obs = np.mean([j.n_replans for j in journeys])

        if method == "Static":
            static_mean = mean_tt

        print(f"{method:<22} {mean_tt:>8.1f} {med_tt:>8.1f} {p95_tt:>8.1f} "
              f"{timeouts:>8.0%} {mean_obs:>12.0f}")

    if static_mean:
        print(f"\nImprovement over Static (mean travel time):")
        for method, journeys in results.items():
            if method == "Static":
                continue
            tt = np.mean([j.arrival_time - j.departure_time for j in journeys])
            imp = static_mean - tt
            pct = imp / static_mean * 100
            print(f"  {method:<22} {imp:>+6.1f} min ({pct:>+5.1f}%)")


# -----------------------------------------------------------------------
# Part 2: Crew scheduling
# -----------------------------------------------------------------------

def run_crew_experiment(n_seeds: int = 20, n_drivers: int = 40):
    """Compare V1-LCB vs V2-LCB vs TS on crew scheduling."""
    crew_path = os.path.join(os.path.dirname(__file__), "..", "crew_scheduling")
    sys.path.insert(0, crew_path)

    from crew_env import CrewSchedulingEnv
    from lcb_scheduler import LCBCrewScheduler
    from baselines.ts_scheduler import TSCrewScheduler
    from lcb_scheduler_v2 import LCBCrewSchedulerV2

    UNCOVERED_PENALTY = 10.0
    scenarios = ["normal", "disrupted", "driver_shortage"]
    methods_list = ["V1-LCB", "V2-LCB", "TS"]

    print(f"\n{'='*80}")
    print("CREW SCHEDULING — V1 vs V2 vs TS")
    print(f"{'='*80}")

    for scenario in scenarios:
        print(f"\n--- {scenario} ---")
        print(f"{'Method':<12} {'Eff.Cost':>10} {'Coverage':>10} {'Overtime':>10}")
        print("-" * 45)

        for method in methods_list:
            eff_costs = []
            coverages = []
            overtimes = []

            for seed in range(n_seeds):
                env = CrewSchedulingEnv(
                    n_drivers=n_drivers, seed=seed,
                    disruption_level=scenario)

                if method == "V1-LCB":
                    sched = LCBCrewScheduler(env, beta=1.5, gamma=30.0)
                elif method == "V2-LCB":
                    sched = LCBCrewSchedulerV2(
                        env, n_estimators=5,
                        beta_base=1.0, beta_ood=1.0, gamma=30.0, seed=seed)
                else:
                    sched = TSCrewScheduler(env, seed=seed)

                # Prior learning
                for day in range(5):
                    day_rng = np.random.default_rng(seed * 1000 + day)
                    for trip in env.trips:
                        dur = env.sample_trip_duration(trip)
                        delay_min = (dur - (trip.scheduled_arr - trip.scheduled_dep)) / 60
                        sched.observe_trip_duration(trip.route_id, delay_min)
                    for driver in env.drivers:
                        if day_rng.random() > driver.true_absence_prob:
                            sched.observe_driver_present(driver.driver_id)
                        else:
                            sched.observe_driver_absent(driver.driver_id)

                sched.compute_structure()
                assignments, _ = sched.assign()

                # Simulate execution
                for trip in env.trips:
                    actual = env.sample_trip_duration(trip)
                    trip.delay = actual - (trip.scheduled_arr - trip.scheduled_dep)
                    sched.observe_trip_duration(trip.route_id, trip.delay / 60)

                absent = set()
                for driver in env.drivers:
                    if driver.driver_id in assignments:
                        if not env.sample_driver_availability(driver):
                            absent.add(driver.driver_id)
                            sched.observe_driver_absent(driver.driver_id)
                        else:
                            sched.observe_driver_present(driver.driver_id)

                if absent:
                    disrupted = set()
                    for did in absent:
                        if did in assignments:
                            disrupted.update(assignments[did])
                    available = [d.driver_id for d in env.drivers
                                 if d.driver_id not in absent]
                    assignments, _ = sched.adaptive_reassign(
                        assignments, disrupted, available)

                metrics = env.get_schedule_metrics(assignments)
                eff = metrics["total_cost"] + UNCOVERED_PENALTY * metrics["n_uncovered"]
                eff_costs.append(eff)
                coverages.append(metrics["coverage_rate"])
                overtimes.append(metrics["overtime_hours"])

            print(f"{method:<12} {np.mean(eff_costs):>10.1f} "
                  f"{np.mean(coverages):>10.0%} {np.mean(overtimes):>10.2f}h")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("BAPR-HRO V1 vs V2 Comparison")
    print("=" * 80)

    # Part 1: Transit routing
    for scenario in ["no_disruption", "disrupted_402", "rush_hour", "multi_shift"]:
        results = run_transit_experiment(
            n_journeys=100, scenario=scenario, seed=42)
        print_transit_results(results, scenario)

    # Part 2: Crew scheduling
    run_crew_experiment(n_seeds=20, n_drivers=40)
