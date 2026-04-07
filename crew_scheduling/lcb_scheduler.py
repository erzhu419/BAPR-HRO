"""BAPR-HRO LCB Crew Scheduler.

Core idea (adapted from BAPR-HRO transit routing):
  "Keep the STRUCTURE (feasible duty set), update the RANKING (via LCB)."

Instead of re-solving SPP when disruptions occur, we:
  1. Pre-compute feasible duties ONCE (the "hyperpath structure")
  2. Maintain Bayesian posteriors over uncertain parameters:
     - Trip duration per route: Normal-Gamma conjugate
     - Driver absence probability: Beta-Binomial conjugate
  3. At each assignment decision, score candidates via LCB:
     score(duty) = expected_cost + beta * sigma_cost + gamma * p_failure
  4. Greedily assign lowest-score duty (most pessimistically robust)

Suboptimality bound (from BAPR-HRO Lean proof):
  excess_cost <= |J| * 2(1+beta) * sigma_max
  where |J| = number of duty pieces, sigma_max = max posterior std.
  Since sigma_max = O(1/sqrt(n)), excess vanishes with observations.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import time

from crew_env import CrewSchedulingEnv, DutyPiece, Duty, WorkRules
from baselines.spp_solver import SPPSolver, FeasibleDuty


# ---------------------------------------------------------------------------
# Bayesian belief states
# ---------------------------------------------------------------------------

@dataclass
class TripDurationBelief:
    """Normal-Gamma posterior over trip duration for a route.

    Conjugate prior for N(mu, sigma^2) with both unknown:
      mu | tau ~ N(mu_0, (kappa_0 * tau)^{-1})
      tau ~ Gamma(alpha_0, beta_0)

    Posterior updates are closed-form.
    """
    # Hyperparameters
    mu_0: float = 0.0       # prior mean delay (minutes)
    kappa_0: float = 2.0    # prior precision weight
    alpha_0: float = 2.0    # shape (prior observations / 2)
    beta_0: float = 4.0     # rate (prior sum of squares / 2)

    # Sufficient statistics
    n_obs: int = 0
    obs_sum: float = 0.0
    obs_sq_sum: float = 0.0

    @property
    def kappa_n(self) -> float:
        return self.kappa_0 + self.n_obs

    @property
    def mu_n(self) -> float:
        return (self.kappa_0 * self.mu_0 + self.obs_sum) / self.kappa_n

    @property
    def alpha_n(self) -> float:
        return self.alpha_0 + self.n_obs / 2

    @property
    def beta_n(self) -> float:
        if self.n_obs == 0:
            return self.beta_0
        obs_mean = self.obs_sum / self.n_obs
        obs_var = self.obs_sq_sum / self.n_obs - obs_mean ** 2
        return (self.beta_0
                + 0.5 * self.n_obs * max(obs_var, 0)
                + 0.5 * self.kappa_0 * self.n_obs * (obs_mean - self.mu_0) ** 2
                / self.kappa_n)

    @property
    def posterior_mean(self) -> float:
        """E[delay] under posterior."""
        return self.mu_n

    @property
    def posterior_std(self) -> float:
        """Posterior predictive std of delay."""
        # Marginal is Student-t: variance = beta_n / (alpha_n * kappa_n) * (kappa_n+1)/kappa_n
        var = self.beta_n / (self.alpha_n * self.kappa_n) * (self.kappa_n + 1) / self.kappa_n
        return max(var, 1e-6) ** 0.5

    def update(self, delay_minutes: float):
        """Update posterior with observed delay."""
        self.n_obs += 1
        self.obs_sum += delay_minutes
        self.obs_sq_sum += delay_minutes ** 2


@dataclass
class DriverAvailabilityBelief:
    """Beta-Binomial posterior over driver absence probability."""
    alpha: float = 1.0     # prior successes (shows up)
    beta: float = 1.0      # prior failures (absent)

    @property
    def absence_prob(self) -> float:
        return self.beta / (self.alpha + self.beta)

    @property
    def absence_std(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b / ((a + b) ** 2 * (a + b + 1))) ** 0.5

    def update_present(self):
        self.alpha += 1

    def update_absent(self):
        self.beta += 1


# ---------------------------------------------------------------------------
# LCB Scheduler
# ---------------------------------------------------------------------------

class LCBCrewScheduler:
    """Crew scheduler using Lower Confidence Bound ranking.

    Architecture:
    1. Pre-compute feasible duty set Q (same as SPP, done ONCE)
    2. Maintain per-route trip duration beliefs (Normal-Gamma)
    3. Maintain per-driver availability beliefs (Beta-Binomial)
    4. Score each candidate duty via LCB at assignment time
    5. Greedy assignment: pick lowest LCB score for each uncovered piece

    The key insight: feasible duties (structure) don't change when trip
    durations shift. Only the RANKING of which duty to prefer changes.
    """

    def __init__(
        self,
        env: CrewSchedulingEnv,
        beta: float = 1.5,
        gamma: float = 30.0,
        max_pieces_per_duty: int = 4,
    ):
        self.env = env
        self.beta = beta        # uncertainty penalty weight
        self.gamma = gamma      # failure penalty weight (minutes)
        self.max_pieces_per_duty = max_pieces_per_duty

        # Bayesian beliefs
        self.route_beliefs: dict[int, TripDurationBelief] = {}
        self.driver_beliefs: dict[int, DriverAvailabilityBelief] = {}

        # Pre-computed feasible duty set (structure)
        self.spp = SPPSolver(env, max_pieces_per_duty)
        self.feasible_duties: list[FeasibleDuty] = []
        self._structure_computed = False

        # Metrics
        self.n_observations = 0
        self.assignment_times: list[float] = []

    def compute_structure(self):
        """One-time computation of feasible duty set."""
        self.feasible_duties = self.spp.enumerate_feasible_duties()
        self._structure_computed = True

    def _get_route_belief(self, route_id: int) -> TripDurationBelief:
        if route_id not in self.route_beliefs:
            self.route_beliefs[route_id] = TripDurationBelief()
        return self.route_beliefs[route_id]

    def _get_driver_belief(self, driver_id: int) -> DriverAvailabilityBelief:
        if driver_id not in self.driver_beliefs:
            self.driver_beliefs[driver_id] = DriverAvailabilityBelief()
        return self.driver_beliefs[driver_id]

    # ------------------------------------------------------------------
    # Observation interface
    # ------------------------------------------------------------------

    def observe_trip_duration(self, route_id: int, delay_minutes: float):
        """Feed observed trip delay to update route belief."""
        belief = self._get_route_belief(route_id)
        belief.update(delay_minutes)
        self.n_observations += 1

    def observe_driver_present(self, driver_id: int):
        """Driver showed up for their shift."""
        belief = self._get_driver_belief(driver_id)
        belief.update_present()

    def observe_driver_absent(self, driver_id: int):
        """Driver didn't show up."""
        belief = self._get_driver_belief(driver_id)
        belief.update_absent()

    # ------------------------------------------------------------------
    # LCB scoring
    # ------------------------------------------------------------------

    def score_duty(
        self, fd: FeasibleDuty, driver_id: int
    ) -> float:
        """LCB score for assigning a feasible duty to a driver.

        score = E[cost] + beta * sigma[cost] + gamma * p_failure

        Lower is better (pessimistic selection).
        """
        pieces = [self.env.pieces[pid] for pid in fd.piece_ids]

        # 1. Expected cost adjustment from delay beliefs
        total_delay_mean = 0.0
        total_delay_var = 0.0

        for piece in pieces:
            for tid in piece.trip_ids:
                trip = self.env.trips[tid]
                belief = self._get_route_belief(trip.route_id)
                total_delay_mean += belief.posterior_mean
                total_delay_var += belief.posterior_std ** 2

        # Adjusted cost: nominal + delay mean (in hours)
        cost_adj = fd.cost + total_delay_mean / 60

        # 2. Uncertainty penalty
        cost_std = max(total_delay_var, 1e-6) ** 0.5 / 60  # in hours
        uncertainty_penalty = self.beta * cost_std

        # 3. Driver failure penalty
        driver_belief = self._get_driver_belief(driver_id)
        failure_penalty = self.gamma * driver_belief.absence_prob

        return cost_adj + uncertainty_penalty + failure_penalty

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign(
        self,
        available_drivers: Optional[list[int]] = None,
    ) -> tuple[dict[int, list[int]], float]:
        """Assign drivers to duty pieces using LCB ranking.

        Greedy construction with LCB scoring:
        For each uncovered piece (chronological order):
          1. Find feasible duties containing this piece
          2. Score each (duty, driver) pair via LCB
          3. Pick the lowest-score pair

        Returns: (assignments, solve_time_seconds)
        """
        t0 = time.perf_counter()

        if not self._structure_computed:
            self.compute_structure()

        if available_drivers is None:
            available_drivers = [d.driver_id for d in self.env.drivers]

        uncovered = set(range(len(self.env.pieces)))
        assignments: dict[int, list[int]] = {d: [] for d in available_drivers}
        used_duties: set[int] = set()

        while uncovered:
            best_coverage = 0
            best_score = float('inf')
            best_duty = None
            best_driver = None

            for fd in self.feasible_duties:
                if fd.duty_id in used_duties:
                    continue
                covered_new = set(fd.piece_ids) & uncovered
                n_new = len(covered_new)
                if n_new == 0:
                    continue

                # Find best driver for this duty (pick first feasible, lowest score)
                # Only check unassigned drivers first, then assigned ones
                best_driver_for_duty = None
                best_driver_score = float('inf')

                for driver_id in available_drivers:
                    if assignments[driver_id]:
                        if not self.env.is_feasible_assignment(
                            self.env.drivers[driver_id],
                            self.env.pieces[fd.piece_ids[0]],
                            assignments,
                        ):
                            continue

                    score = self.score_duty(fd, driver_id)
                    if score < best_driver_score:
                        best_driver_score = score
                        best_driver_for_duty = driver_id

                if best_driver_for_duty is None:
                    continue

                # Primary: maximize coverage; secondary: minimize LCB score
                if (n_new > best_coverage or
                    (n_new == best_coverage and best_driver_score < best_score)):
                    best_coverage = n_new
                    best_score = best_driver_score
                    best_duty = fd
                    best_driver = best_driver_for_duty

            if best_duty is None:
                break

            used_duties.add(best_duty.duty_id)
            assignments[best_driver].extend(best_duty.piece_ids)
            uncovered -= set(best_duty.piece_ids)

        # Remove empty assignments
        assignments = {k: v for k, v in assignments.items() if v}

        solve_time = time.perf_counter() - t0
        self.assignment_times.append(solve_time)
        return assignments, solve_time

    # ------------------------------------------------------------------
    # Adaptive re-assignment (the key advantage over SPP)
    # ------------------------------------------------------------------

    def adaptive_reassign(
        self,
        current_assignments: dict[int, list[int]],
        disrupted_pieces: set[int],
        available_drivers: Optional[list[int]] = None,
    ) -> tuple[dict[int, list[int]], float]:
        """Re-assign only disrupted pieces using LCB, keeping structure.

        This is where BAPR-HRO shines: instead of re-solving the entire SPP,
        we only re-rank candidates for the affected pieces using updated
        posteriors.

        Returns: (updated_assignments, solve_time_seconds)
        """
        t0 = time.perf_counter()

        if available_drivers is None:
            available_drivers = list(current_assignments.keys())

        # Remove disrupted pieces from current assignments
        updated = {}
        for driver_id, piece_ids in current_assignments.items():
            remaining = [p for p in piece_ids if p not in disrupted_pieces]
            if remaining:
                updated[driver_id] = remaining

        uncovered = disrupted_pieces.copy()

        # Re-assign only uncovered pieces via LCB
        while uncovered:
            best_score = float('inf')
            best_duty = None
            best_driver = None

            for fd in self.feasible_duties:
                covered_new = set(fd.piece_ids) & uncovered
                if not covered_new:
                    continue
                # Also check no conflict with kept assignments
                kept_pieces = set()
                for pids in updated.values():
                    kept_pieces.update(pids)
                if set(fd.piece_ids) & kept_pieces:
                    continue

                for driver_id in available_drivers:
                    if not self.env.is_feasible_assignment(
                        self.env.drivers[driver_id],
                        self.env.pieces[fd.piece_ids[0]],
                        updated,
                    ):
                        continue

                    score = self.score_duty(fd, driver_id)
                    score_per_piece = score / len(covered_new)

                    if score_per_piece < best_score:
                        best_score = score_per_piece
                        best_duty = fd
                        best_driver = driver_id

            if best_duty is None:
                break

            if best_driver not in updated:
                updated[best_driver] = []
            updated[best_driver].extend(best_duty.piece_ids)
            uncovered -= set(best_duty.piece_ids)

        solve_time = time.perf_counter() - t0
        self.assignment_times.append(solve_time)
        return updated, solve_time

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_belief_summary(self) -> dict:
        """Return current Bayesian belief states for analysis."""
        route_summary = {}
        for rid, belief in self.route_beliefs.items():
            route_summary[rid] = {
                "mean_delay_min": belief.posterior_mean,
                "std_delay_min": belief.posterior_std,
                "n_obs": belief.n_obs,
            }

        driver_summary = {}
        for did, belief in self.driver_beliefs.items():
            driver_summary[did] = {
                "absence_prob": belief.absence_prob,
                "absence_std": belief.absence_std,
                "n_obs": int(belief.alpha + belief.beta - 2),
            }

        return {
            "routes": route_summary,
            "drivers": driver_summary,
            "total_observations": self.n_observations,
            "sigma_max_route": max(
                (b.posterior_std for b in self.route_beliefs.values()),
                default=0.0,
            ),
        }

    def theoretical_bound(self, n_pieces: int) -> float:
        """Compute theoretical suboptimality bound.

        From BAPR-HRO Lean proof:
          excess <= |J| * 2(1+beta) * sigma_max
        """
        summary = self.get_belief_summary()
        sigma_max = summary["sigma_max_route"]
        return n_pieces * 2 * (1 + self.beta) * sigma_max
