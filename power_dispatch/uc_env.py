"""Unit Commitment environment for BAPR-HRO, wrapping rl4uc.

Mapping to BAPR-HRO transit routing:
  Hyperpath alternatives → K candidate commitment schedules
  Delay per route → wind/demand forecast error (ARMA)
  LCB re-ranking → pick schedule with lowest pessimistic operating cost
  Regime shift → sudden wind drop or demand spike
  Irrecoverable → thermal unit started → must stay on min_up hours

A commitment schedule is a (T, N_gen) binary matrix specifying which
generators are on at each half-hour. The environment executes a schedule,
observing actual demand/wind, and returns the total operating cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass, field

# Add rl4uc to path
_rl4uc_dir = os.path.join(os.path.dirname(__file__), "rl4uc")
sys.path.insert(0, _rl4uc_dir)


def _load_rl4uc_env(num_gen: int = 10, mode: str = "test", voll: float = 500):
    """Load rl4uc environment with specified generator count.

    voll: Value of Lost Load ($/MWh). Default 500 gives meaningful
    trade-off between fuel cost (running extra generators) vs ENS risk.
    """
    from rl4uc.environment import Env

    data_dir = os.path.join(_rl4uc_dir, "rl4uc", "data")
    gen_info = pd.read_csv(os.path.join(data_dir, "kazarlis_units_10.csv"))
    gen_info = gen_info[:num_gen]

    profiles = pd.read_csv(os.path.join(data_dir, "test_data_10gen.csv"))

    env = Env(gen_info=gen_info, profiles_df=profiles, mode=mode, voll=voll)
    return env


@dataclass
class ScheduleResult:
    """Result of executing one commitment schedule for one day."""
    total_cost: float          # fuel + startup + lost load
    fuel_cost: float
    startup_cost: float
    lost_load_cost: float
    wind_errors: list[float]   # observed wind forecast errors per period
    demand_errors: list[float] # observed demand forecast errors per period
    net_demands: list[float]   # actual net demand per period
    n_periods: int = 48


def generate_candidate_schedules(
    num_gen: int = 5,
    n_periods: int = 48,
    forecast_demand: np.ndarray | None = None,
    forecast_wind: np.ndarray | None = None,
    gen_max: np.ndarray | None = None,
    gen_min: np.ndarray | None = None,
    n_candidates: int = 8,
    seed: int = 0,
) -> list[np.ndarray]:
    """Generate K candidate commitment schedules.

    Strategy: vary the number of committed generators based on different
    assumptions about wind availability (conservative → aggressive).

    Schedule 0: ALL generators on (most expensive, safest)
    Schedule 1-K: progressively fewer generators, assuming more wind.

    Each schedule is (n_periods, num_gen) binary array.
    """
    rng = np.random.default_rng(seed)
    schedules = []

    # Schedule 0: all-on (baseline safe plan)
    schedules.append(np.ones((n_periods, num_gen), dtype=int))

    if gen_max is None:
        # Kazarlis 10-gen system
        _max = [455, 455, 130, 130, 162, 80, 85, 55, 55, 55]
        gen_max = np.array(_max[:num_gen])
    if gen_min is None:
        _min = [150, 150, 20, 20, 25, 20, 25, 10, 10, 10]
        gen_min = np.array(_min[:num_gen])

    total_cap = gen_max.sum()

    # Sort generators by cost efficiency (cheapest first = large coal)
    # For Kazarlis: gen 0,1 (coal, cheap), gen 2,3 (gas, medium), gen 4+ (oil, expensive)
    # We want to turn off expensive generators first
    cost_order = list(range(num_gen))  # already roughly sorted by efficiency

    for k in range(1, n_candidates):
        sched = np.ones((n_periods, num_gen), dtype=int)

        # Turn off the k most expensive generators during low-demand periods
        n_off = min(k, num_gen - 1)  # keep at least 1 generator on
        gens_to_maybe_off = cost_order[-n_off:]  # most expensive

        for t in range(n_periods):
            # Estimate how much capacity we need
            if forecast_demand is not None and forecast_wind is not None:
                net = forecast_demand[t] - forecast_wind[t]
            else:
                # Default: assume moderate demand profile
                hour = (t * 0.5) % 24
                # Typical demand curve: low at night, high during day
                base_demand = 800 + 200 * np.sin((hour - 6) * np.pi / 12)
                base_wind = 100 + rng.normal(0, 30)
                net = max(base_demand - base_wind, 0)

            # How much slack do we have?
            remaining_cap = sum(gen_max[g] for g in range(num_gen)
                                if g not in gens_to_maybe_off)

            # Add safety margin that varies by candidate
            safety = 1.0 + 0.1 * (n_candidates - k)  # more aggressive → less safety
            if remaining_cap >= net * safety:
                for g in gens_to_maybe_off:
                    sched[t, g] = 0
            else:
                # Keep some on based on need
                needed = net * safety - remaining_cap
                for g in gens_to_maybe_off:
                    if needed <= 0:
                        sched[t, g] = 0
                    else:
                        needed -= gen_max[g]

        # Enforce minimum up/down times (simplified)
        sched = _enforce_min_updown(sched, min_up=4, min_down=2)
        schedules.append(sched)

    return schedules


def _enforce_min_updown(schedule: np.ndarray, min_up: int = 4,
                        min_down: int = 2) -> np.ndarray:
    """Enforce minimum up/down times by smoothing commitment changes."""
    T, N = schedule.shape
    for g in range(N):
        t = 1
        while t < T:
            if schedule[t, g] != schedule[t - 1, g]:
                # State change: enforce minimum duration
                if schedule[t, g] == 1:  # turned on
                    for dt in range(min(min_up, T - t)):
                        schedule[t + dt, g] = 1
                    t += min_up
                else:  # turned off
                    for dt in range(min(min_down, T - t)):
                        schedule[t + dt, g] = 0
                    t += min_down
            else:
                t += 1
    return schedule


def execute_schedule(
    schedule: np.ndarray,
    num_gen: int = 10,
    day_idx: int = 0,
    seed: int = 0,
    voll: float = 500,
) -> ScheduleResult:
    """Execute a commitment schedule on the rl4uc environment.

    Args:
        schedule: (T, num_gen) binary commitment matrix
        num_gen: number of generators
        day_idx: which day in test data to use
        seed: random seed for demand/wind noise
        voll: value of lost load ($/MWh)

    Returns:
        ScheduleResult with cost breakdown and observed errors.
    """
    np.random.seed(seed)

    env = _load_rl4uc_env(num_gen=num_gen, mode="test", voll=voll)

    # Reset to specific day
    env.reset()

    T = schedule.shape[0]
    total_fuel = 0.0
    total_startup = 0.0
    total_ens = 0.0
    wind_errors = []
    demand_errors = []
    net_demands = []

    for t in range(min(T, env.episode_length)):
        action = schedule[t]
        obs, reward, done = env.step(action, deterministic=False)

        total_fuel += env.fuel_cost
        total_startup += env.start_cost
        total_ens += env.ens_cost
        wind_errors.append(float(env.arma_wind.xs[0]))
        demand_errors.append(float(env.arma_demand.xs[0]))
        net_demands.append(float(env.net_demand))

        if done:
            break

    return ScheduleResult(
        total_cost=total_fuel + total_startup + total_ens,
        fuel_cost=total_fuel,
        startup_cost=total_startup,
        lost_load_cost=total_ens,
        wind_errors=wind_errors,
        demand_errors=demand_errors,
        net_demands=net_demands,
        n_periods=t + 1,
    )
