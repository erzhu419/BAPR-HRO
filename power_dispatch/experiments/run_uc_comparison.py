"""BAPR-HRO vs baselines on Unit Commitment — with wind regime shifts."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time as _t
from uc_env import generate_candidate_schedules, execute_schedule, _load_rl4uc_env
from lcb_uc import StaticRouter, LCBRouter, LCBRouterV2, TSRouter, HybridRouter, AdaptiveBetaRouter


def run_experiment(
    num_gen: int = 10, n_candidates: int = 6, n_days: int = 40, voll: float = 500,
    methods: list[str] | None = None,
):
    if methods is None:
        methods = ["Static", "TS", "V1-LCB", "V2-LCB", "Hybrid", "Adapt-β"]

    env = _load_rl4uc_env(num_gen=num_gen, voll=voll)
    gen_max = env.max_output
    profiles = env.profiles_df
    day0 = profiles[profiles["date"] == profiles["date"].unique()[0]]
    fd, fw = day0["demand"].values[:48], day0["wind"].values[:48]

    scheds = generate_candidate_schedules(
        num_gen=num_gen, n_candidates=n_candidates,
        forecast_demand=fd, forecast_wind=fw, gen_max=gen_max,
    )

    # Wind regime schedule: normal, then low-wind regime, then back
    # Simulates: "summer (high wind)" → "winter storm (low wind)" → "recovery"
    wind_regimes = []
    for d in range(n_days):
        if d < 15:
            wind_regimes.append("normal")
        elif d < 25:
            wind_regimes.append("low_wind")  # regime shift!
        else:
            wind_regimes.append("normal")

    # Warm-start from normal conditions
    warm_costs = []
    for i, s in enumerate(scheds):
        r = execute_schedule(s, num_gen=num_gen, seed=9999, voll=voll, wind_regime="normal")
        warm_costs.append(r.total_cost)

    print(f"\n{'='*80}")
    print(f"  UC: {num_gen}gen, {n_candidates}sched, {n_days}days, VOLL=${voll}")
    print(f"  Wind regimes: normal(d1-15) → LOW_WIND(d16-25) → normal(d26-40)")
    print(f"  Warm-start: S0=${warm_costs[0]:,.0f} ... S5=${warm_costs[-1]:,.0f}")
    print(f"{'='*80}")

    print(f"\n{'Method':<12} {'Normal':>8} {'LowWind':>8} {'Recov':>8}  {'Mean':>8} {'Δ%':>6}")
    print("-" * 55)

    static_mean = None
    for method in methods:
        if method == "Static":
            router = StaticRouter(n_candidates)
        elif method == "TS":
            router = TSRouter(n_candidates, seed=0, warm_costs=warm_costs)
        elif method == "V1-LCB":
            router = LCBRouter(n_candidates, beta0=2.0, warm_costs=warm_costs)
        elif method == "V2-LCB":
            router = LCBRouterV2(n_candidates, beta_base=1.5, beta_ood=0.5,
                                 seed=0, warm_costs=warm_costs)
        elif method == "Hybrid":
            router = HybridRouter(n_candidates, beta0=2.0, switch_ep=10,
                                  warm_costs=warm_costs)
        elif method == "Adapt-β":
            router = AdaptiveBetaRouter(n_candidates, seed=0,
                                         warm_costs=warm_costs)

        day_costs = []
        picks = []
        for day in range(n_days):
            idx = router.select_schedule()
            picks.append(idx)
            result = execute_schedule(scheds[idx], num_gen=num_gen, seed=day,
                                      voll=voll, wind_regime=wind_regimes[day])
            router.observe(idx, result)
            day_costs.append(result.total_cost)

        mean_all = np.mean(day_costs)
        if static_mean is None:
            static_mean = mean_all
        delta = (mean_all - static_mean) / static_mean * 100

        normal = np.mean(day_costs[:15])
        low_wind = np.mean(day_costs[15:25])
        recovery = np.mean(day_costs[25:])

        print(f"{method:<12} {normal:>8.0f} {low_wind:>8.0f} {recovery:>8.0f}"
              f"  {mean_all:>8.0f} {delta:>+5.1f}%")

    # Show picks
    print(f"\nSchedule picks (S0=safe ... S5=aggressive):")
    for method in methods:
        if method == "Static":
            router = StaticRouter(n_candidates)
        elif method == "TS":
            router = TSRouter(n_candidates, seed=0, warm_costs=warm_costs)
        elif method == "V1-LCB":
            router = LCBRouter(n_candidates, beta0=2.0, warm_costs=warm_costs)
        elif method == "V2-LCB":
            router = LCBRouterV2(n_candidates, beta_base=1.5, beta_ood=0.5,
                                 seed=0, warm_costs=warm_costs)
        elif method == "Hybrid":
            router = HybridRouter(n_candidates, beta0=2.0, switch_ep=10,
                                  warm_costs=warm_costs)
        elif method == "Adapt-β":
            router = AdaptiveBetaRouter(n_candidates, seed=0,
                                         warm_costs=warm_costs)
        picks = []
        for day in range(n_days):
            idx = router.select_schedule()
            picks.append(idx)
            r = execute_schedule(scheds[idx], num_gen=num_gen, seed=day,
                                 voll=voll, wind_regime=wind_regimes[day])
            router.observe(idx, r)
        # Compress picks display
        phases = [picks[:15], picks[15:25], picks[25:]]
        labels = ["normal", "LOW", "recov"]
        print(f"  {method:<10}: ", end="")
        for label, phase in zip(labels, phases):
            from collections import Counter
            c = Counter(phase)
            most = c.most_common(1)[0]
            print(f"{label}=S{most[0]}({most[1]}/{len(phase)}) ", end="")
        print()


if __name__ == "__main__":
    run_experiment()
