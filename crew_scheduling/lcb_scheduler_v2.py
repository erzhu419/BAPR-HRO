"""BAPR-HRO V2 LCB Crew Scheduler with Ensemble + Dynamic Beta.

V2 improvements over lcb_scheduler.py:
  1. Ensemble-based uncertainty (no Normal distribution assumption)
  2. Dynamic beta: beta(route) = beta_base + beta_ood * OOD(route)
  3. OOD score from ensemble disagreement → unfamiliar routes get more pessimism
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import time

from crew_env import CrewSchedulingEnv
from baselines.spp_solver import SPPSolver, FeasibleDuty
from lcb_scheduler import DriverAvailabilityBelief


@dataclass
class TripDurationEnsemble:
    """Ensemble-based belief over trip duration for a route."""
    n_estimators: int = 5
    _means: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _vars: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _counts: np.ndarray = field(default_factory=lambda: np.zeros(0))

    prior_mean: float = 0.0
    prior_var: float = 4.0
    prior_n: float = 2.0

    def __post_init__(self):
        if len(self._means) == 0:
            self._means = np.full(self.n_estimators, self.prior_mean)
            self._vars = np.full(self.n_estimators, self.prior_var)
            self._counts = np.full(self.n_estimators, self.prior_n)

    @property
    def ensemble_mean(self) -> float:
        return float(self._means.mean())

    @property
    def ensemble_std(self) -> float:
        return float(self._means.std())

    @property
    def ood_score(self) -> float:
        total_obs = int(self._counts.sum() - self.n_estimators * self.prior_n)
        if total_obs < 2:
            return 1.0
        avg_std = float(np.sqrt(max(self._vars.mean(), 1e-8)))
        if avg_std < 1e-6:
            return 0.0
        return min(self.ensemble_std / avg_std, 3.0)

    @property
    def n_obs(self) -> int:
        return int(self._counts.sum() - self.n_estimators * self.prior_n)

    def update(self, delay_minutes: float, rng: np.random.Generator):
        weights = rng.poisson(1, self.n_estimators)
        for k in range(self.n_estimators):
            w = weights[k]
            if w == 0:
                continue
            for _ in range(w):
                self._counts[k] += 1
                n = self._counts[k]
                old_mean = self._means[k]
                self._means[k] += (delay_minutes - old_mean) / n
                self._vars[k] += ((delay_minutes - old_mean) *
                                  (delay_minutes - self._means[k]) - self._vars[k]) / n


class LCBCrewSchedulerV2:
    """V2 crew scheduler with ensemble uncertainty + dynamic beta."""

    def __init__(
        self,
        env: CrewSchedulingEnv,
        n_estimators: int = 5,
        beta_base: float = 1.0,
        beta_ood: float = 1.0,
        gamma: float = 30.0,
        max_pieces_per_duty: int = 4,
        seed: int = 42,
    ):
        self.env = env
        self.n_estimators = n_estimators
        self.beta_base = beta_base
        self.beta_ood = beta_ood
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.route_beliefs: dict[int, TripDurationEnsemble] = {}
        self.driver_beliefs: dict[int, DriverAvailabilityBelief] = {}

        self.spp = SPPSolver(env, max_pieces_per_duty)
        self.feasible_duties: list[FeasibleDuty] = []
        self._structure_computed = False
        self.n_observations = 0
        self.assignment_times: list[float] = []

    def compute_structure(self):
        self.feasible_duties = self.spp.enumerate_feasible_duties()
        self._structure_computed = True

    def _get_route_belief(self, route_id: int) -> TripDurationEnsemble:
        if route_id not in self.route_beliefs:
            self.route_beliefs[route_id] = TripDurationEnsemble(
                n_estimators=self.n_estimators)
        return self.route_beliefs[route_id]

    def _get_driver_belief(self, driver_id: int) -> DriverAvailabilityBelief:
        if driver_id not in self.driver_beliefs:
            self.driver_beliefs[driver_id] = DriverAvailabilityBelief()
        return self.driver_beliefs[driver_id]

    def observe_trip_duration(self, route_id: int, delay_minutes: float):
        self._get_route_belief(route_id).update(delay_minutes, self.rng)
        self.n_observations += 1

    def observe_driver_present(self, driver_id: int):
        self._get_driver_belief(driver_id).update_present()

    def observe_driver_absent(self, driver_id: int):
        self._get_driver_belief(driver_id).update_absent()

    def _compute_dynamic_beta(self, route_ids: list[int]) -> float:
        if not route_ids:
            return self.beta_base + self.beta_ood
        ood_scores = [self._get_route_belief(r).ood_score for r in route_ids]
        return self.beta_base + self.beta_ood * max(ood_scores)

    def score_duty(self, fd: FeasibleDuty, driver_id: int) -> float:
        """V2 LCB score with ensemble std and dynamic beta."""
        pieces = [self.env.pieces[pid] for pid in fd.piece_ids]

        route_ids = set()
        total_delay_mean = 0.0
        total_ensemble_var = 0.0

        for piece in pieces:
            for tid in piece.trip_ids:
                trip = self.env.trips[tid]
                belief = self._get_route_belief(trip.route_id)
                route_ids.add(trip.route_id)
                total_delay_mean += belief.ensemble_mean
                total_ensemble_var += belief.ensemble_std ** 2

        # Dynamic beta based on OOD
        beta = self._compute_dynamic_beta(list(route_ids))

        cost_adj = fd.cost + total_delay_mean / 60
        uncertainty_penalty = beta * max(total_ensemble_var, 1e-6) ** 0.5 / 60
        driver_belief = self._get_driver_belief(driver_id)
        failure_penalty = self.gamma * driver_belief.absence_prob

        return cost_adj + uncertainty_penalty + failure_penalty

    def assign(
        self, available_drivers: Optional[list[int]] = None,
    ) -> tuple[dict[int, list[int]], float]:
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

                best_d_for_duty = None
                best_d_score = float('inf')
                for driver_id in available_drivers:
                    if assignments[driver_id]:
                        if not self.env.is_feasible_assignment(
                            self.env.drivers[driver_id],
                            self.env.pieces[fd.piece_ids[0]], assignments):
                            continue
                    score = self.score_duty(fd, driver_id)
                    if score < best_d_score:
                        best_d_score = score
                        best_d_for_duty = driver_id

                if best_d_for_duty is None:
                    continue
                if (n_new > best_coverage or
                    (n_new == best_coverage and best_d_score < best_score)):
                    best_coverage = n_new
                    best_score = best_d_score
                    best_duty = fd
                    best_driver = best_d_for_duty

            if best_duty is None:
                break
            used_duties.add(best_duty.duty_id)
            assignments[best_driver].extend(best_duty.piece_ids)
            uncovered -= set(best_duty.piece_ids)

        assignments = {k: v for k, v in assignments.items() if v}
        solve_time = time.perf_counter() - t0
        self.assignment_times.append(solve_time)
        return assignments, solve_time

    def adaptive_reassign(
        self,
        current_assignments: dict[int, list[int]],
        disrupted_pieces: set[int],
        available_drivers: Optional[list[int]] = None,
    ) -> tuple[dict[int, list[int]], float]:
        t0 = time.perf_counter()
        if available_drivers is None:
            available_drivers = list(current_assignments.keys())

        updated = {}
        for did, pids in current_assignments.items():
            remaining = [p for p in pids if p not in disrupted_pieces]
            if remaining:
                updated[did] = remaining

        uncovered = disrupted_pieces.copy()
        while uncovered:
            best_coverage = 0
            best_score = float('inf')
            best_duty = None
            best_driver = None

            for fd in self.feasible_duties:
                covered_new = set(fd.piece_ids) & uncovered
                n_new = len(covered_new)
                if n_new == 0:
                    continue
                kept = set()
                for pids in updated.values():
                    kept.update(pids)
                if set(fd.piece_ids) & kept:
                    continue

                for did in available_drivers:
                    if not self.env.is_feasible_assignment(
                        self.env.drivers[did],
                        self.env.pieces[fd.piece_ids[0]], updated):
                        continue
                    score = self.score_duty(fd, did)
                    if (n_new > best_coverage or
                        (n_new == best_coverage and score < best_score)):
                        best_coverage = n_new
                        best_score = score
                        best_duty = fd
                        best_driver = did

            if best_duty is None:
                break
            if best_driver not in updated:
                updated[best_driver] = []
            updated[best_driver].extend(best_duty.piece_ids)
            uncovered -= set(best_duty.piece_ids)

        solve_time = time.perf_counter() - t0
        self.assignment_times.append(solve_time)
        return updated, solve_time

    def get_belief_summary(self) -> dict:
        route_summary = {}
        for rid, belief in self.route_beliefs.items():
            route_summary[rid] = {
                "ensemble_mean": belief.ensemble_mean,
                "ensemble_std": belief.ensemble_std,
                "ood_score": belief.ood_score,
                "n_obs": belief.n_obs,
                "dynamic_beta": self._compute_dynamic_beta([rid]),
            }
        return {
            "routes": route_summary,
            "total_observations": self.n_observations,
            "sigma_max_route": max(
                (b.ensemble_std for b in self.route_beliefs.values()), default=0.0),
        }

    def theoretical_bound(self, n_pieces: int) -> float:
        s = self.get_belief_summary()
        sigma_max = s["sigma_max_route"]
        beta_max = self.beta_base + self.beta_ood  # worst case beta
        return n_pieces * 2 * (1 + beta_max) * sigma_max
