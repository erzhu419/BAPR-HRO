"""Thompson Sampling Crew Scheduler (PS-SSP baseline).

Instead of LCB's pessimistic ranking, Thompson Sampling draws a sample
from the posterior and picks the duty that looks best under that sample.

In multi-episode settings (e.g., standard RL), TS is near-optimal because
exploration costs amortize. But in single-shot crew scheduling, optimistic
draws can lead to assigning a driver to a duty that turns out much worse
than expected — and the cost is irrecoverable.

This baseline demonstrates why LCB (pessimism) outperforms TS (exploration)
in the single-day, irrecoverable crew scheduling setting.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crew_env import CrewSchedulingEnv
from baselines.spp_solver import SPPSolver, FeasibleDuty
from lcb_scheduler import TripDurationBelief, DriverAvailabilityBelief


class TSCrewScheduler:
    """Thompson Sampling crew scheduler.

    Same structure as LCB (pre-computed feasible duty set + Bayesian posteriors),
    but uses posterior SAMPLING instead of LCB scoring at assignment time.

    score_TS(duty) = sampled_cost (drawn from posterior)

    This occasionally produces optimistic estimates for bad duties,
    leading to worse single-day performance than LCB.
    """

    def __init__(
        self,
        env: CrewSchedulingEnv,
        max_pieces_per_duty: int = 4,
        seed: int = 42,
    ):
        self.env = env
        self.max_pieces_per_duty = max_pieces_per_duty
        self.rng = np.random.default_rng(seed)

        self.route_beliefs: dict[int, TripDurationBelief] = {}
        self.driver_beliefs: dict[int, DriverAvailabilityBelief] = {}

        self.spp = SPPSolver(env, max_pieces_per_duty)
        self.feasible_duties: list[FeasibleDuty] = []
        self._structure_computed = False
        self.n_observations = 0
        self.assignment_times: list[float] = []

    def compute_structure(self):
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

    def observe_trip_duration(self, route_id: int, delay_minutes: float):
        belief = self._get_route_belief(route_id)
        belief.update(delay_minutes)
        self.n_observations += 1

    def observe_driver_present(self, driver_id: int):
        self._get_driver_belief(driver_id).update_present()

    def observe_driver_absent(self, driver_id: int):
        self._get_driver_belief(driver_id).update_absent()

    def score_duty_ts(self, fd: FeasibleDuty, driver_id: int) -> float:
        """Thompson Sampling score: SAMPLE from posterior instead of LCB.

        Draws delay from Normal posterior, driver availability from Beta.
        Returns sampled total cost.
        """
        pieces = [self.env.pieces[pid] for pid in fd.piece_ids]

        total_sampled_delay = 0.0
        for piece in pieces:
            for tid in piece.trip_ids:
                trip = self.env.trips[tid]
                belief = self._get_route_belief(trip.route_id)
                # Sample delay from posterior (can be optimistic!)
                sampled_delay = self.rng.normal(
                    belief.posterior_mean, belief.posterior_std
                )
                total_sampled_delay += sampled_delay

        cost_sampled = fd.cost + total_sampled_delay / 60

        # Sample driver availability
        driver_belief = self._get_driver_belief(driver_id)
        if self.rng.random() < driver_belief.absence_prob:
            cost_sampled += 60.0  # huge penalty if sampled as absent

        return cost_sampled

    def assign(
        self, available_drivers: Optional[list[int]] = None,
    ) -> tuple[dict[int, list[int]], float]:
        """Assign using Thompson Sampling."""
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

                    score = self.score_duty_ts(fd, driver_id)
                    if score < best_driver_score:
                        best_driver_score = score
                        best_driver_for_duty = driver_id

                if best_driver_for_duty is None:
                    continue

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
        """Re-assign disrupted pieces using TS."""
        t0 = time.perf_counter()

        if available_drivers is None:
            available_drivers = list(current_assignments.keys())

        updated = {}
        for driver_id, piece_ids in current_assignments.items():
            remaining = [p for p in piece_ids if p not in disrupted_pieces]
            if remaining:
                updated[driver_id] = remaining

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
                    score = self.score_duty_ts(fd, driver_id)
                    if (n_new > best_coverage or
                        (n_new == best_coverage and score < best_score)):
                        best_coverage = n_new
                        best_score = score
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
