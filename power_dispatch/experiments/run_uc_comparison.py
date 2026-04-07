"""BAPR-HRO vs baselines on Unit Commitment (rl4uc Kazarlis 10-gen)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time as _t

from uc_env import generate_candidate_schedules, execute_schedule, _load_rl4uc_env
from lcb_uc import StaticRouter, LCBRouter, LCBRouterV2, TSRouter


def run_experiment(
    num_gen: int = 10,
    n_candidates: int = 6,
    n_days: int = 30,
    voll: float = 500,
    methods: list[str] | None = None,
):
    if methods is None:
        methods = ["Static", "TS", "V1-LCB", "V2-LCB"]

    env = _load_rl4uc_env(num_gen=num_gen, voll=voll)
    gen_max = env.max_output
    profiles = env.profiles_df
    day0 = profiles[profiles["date"] == profiles["date"].unique()[0]]
    fd = day0["demand"].values[:48]
    fw = day0["wind"].values[:48]

    scheds = generate_candidate_schedules(
        num_gen=num_gen, n_candidates=n_candidates,
        forecast_demand=fd, forecast_wind=fw, gen_max=gen_max,
    )

    print(f"\n{'='*75}")
    print(f"  UC: {num_gen} gen, {n_candidates} schedules, {n_days} days, VOLL=${voll}")
    for i, s in enumerate(scheds):
        print(f"    S{i}: on={s.mean():.0%}", end="")
    print(f"\n{'='*75}")

    print(f"\n{'Method':<12} {'D1':>8} {'D10':>8} {'D30':>8}  {'Mean':>8} {'Δ%':>6} {'Post4':>8}")
    print("-" * 65)

    static_mean = None

    for method in methods:
        t0 = _t.time()

        if method == "Static":
            router = StaticRouter(n_candidates)
        elif method == "TS":
            router = TSRouter(n_candidates, seed=0)
        elif method == "V1-LCB":
            router = LCBRouter(n_candidates, beta=1.0)
        elif method == "V2-LCB":
            router = LCBRouterV2(n_candidates, beta_base=0.8, beta_ood=0.8, seed=0)

        day_costs = []
        picks = []
        for day in range(n_days):
            idx = router.select_schedule()
            picks.append(idx)
            result = execute_schedule(scheds[idx], num_gen=num_gen,
                                      seed=day, voll=voll)
            router.observe(idx, result)
            day_costs.append(result.total_cost)

        mean_all = np.mean(day_costs)
        if static_mean is None:
            static_mean = mean_all
        delta = (mean_all - static_mean) / static_mean * 100
        post = np.mean(day_costs[4:])
        elapsed = _t.time() - t0

        print(f"{method:<12} {day_costs[0]:>8.0f} {day_costs[9]:>8.0f} {day_costs[-1]:>8.0f}"
              f"  {mean_all:>8.0f} {delta:>+5.1f}% {post:>8.0f}  ({elapsed:.1f}s)")

    # Show picks
    print(f"\nSchedule picks:")
    for method in methods:
        if method == "Static":
            router = StaticRouter(n_candidates)
        elif method == "TS":
            router = TSRouter(n_candidates, seed=0)
        elif method == "V1-LCB":
            router = LCBRouter(n_candidates, beta=1.0)
        elif method == "V2-LCB":
            router = LCBRouterV2(n_candidates, beta_base=0.8, beta_ood=0.8, seed=0)
        picks = []
        for day in range(n_days):
            idx = router.select_schedule()
            picks.append(idx)
            result = execute_schedule(scheds[idx], num_gen=num_gen,
                                      seed=day, voll=voll)
            router.observe(idx, result)
        print(f"  {method:<10}: {picks}")


if __name__ == "__main__":
    run_experiment()
