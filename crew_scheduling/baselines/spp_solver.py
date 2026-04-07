"""Set Partitioning Problem (SPP) baseline for crew scheduling.

Traditional approach (Ceder Ch.10, Section 10.3):
  1. Enumerate all feasible duties
  2. Solve SPP: min-cost set of duties covering all pieces exactly once
  3. Uses greedy heuristic + LP relaxation (no commercial ILP solver needed)

This is the "recompute from scratch" baseline that BAPR-HRO aims to beat.
When trip durations change, the entire SPP must be re-solved.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crew_env import CrewSchedulingEnv, DutyPiece, Duty, WorkRules


@dataclass
class FeasibleDuty:
    """A candidate duty for the SPP."""
    duty_id: int
    piece_ids: list[int]
    cost: float
    start_time: float
    end_time: float
    has_meal_break: bool


class SPPSolver:
    """Greedy Set Partitioning solver for crew scheduling.

    Follows Ceder's SPM (Shortest-Path and Matching) approach:
    1. Generate feasible duty pieces from vehicle blocks
    2. Build acyclic network per block (nodes=relief points, arcs=pieces)
    3. Shortest path gives minimum-cost piece combination per block
    4. Greedy matching combines pieces into legal duties
    """

    def __init__(self, env: CrewSchedulingEnv, max_pieces_per_duty: int = 4):
        self.env = env
        self.max_pieces_per_duty = max_pieces_per_duty
        self.feasible_duties: list[FeasibleDuty] = []

    def enumerate_feasible_duties(self) -> list[FeasibleDuty]:
        """Enumerate all feasible duties (combinations of pieces).

        For tractability, limit to 1-3 piece combinations.
        Check work-rule feasibility for each.
        """
        self.feasible_duties = []
        duty_id = 0
        wr = self.env.work_rules
        pieces = self.env.pieces

        # Single-piece duties
        for p in pieces:
            if p.duration <= wr.max_shift_hours * 3600:
                fd = self._make_duty(duty_id, [p])
                if fd is not None:
                    self.feasible_duties.append(fd)
                    duty_id += 1

        # Two-piece duties
        for i, p1 in enumerate(pieces):
            for j, p2 in enumerate(pieces):
                if j <= i:
                    continue
                if p2.start_time <= p1.end_time:
                    continue  # overlap
                gap = p2.start_time - p1.end_time
                if gap < wr.min_rest_between_pieces:
                    continue  # too close

                combined_duration = p2.end_time - p1.start_time
                if combined_duration > wr.max_shift_hours * 3600:
                    continue  # too long

                # Terminal compatibility
                if p1.end_terminal != p2.start_terminal and gap < 30 * 60:
                    continue  # need deadhead time

                fd = self._make_duty(duty_id, [p1, p2])
                if fd is not None:
                    self.feasible_duties.append(fd)
                    duty_id += 1

        # Three-piece duties (limited search)
        if len(pieces) <= 50:  # only for small instances
            for i, p1 in enumerate(pieces):
                for j, p2 in enumerate(pieces):
                    if j <= i or p2.start_time <= p1.end_time:
                        continue
                    gap1 = p2.start_time - p1.end_time
                    if gap1 < wr.min_rest_between_pieces:
                        continue

                    for k, p3 in enumerate(pieces):
                        if k <= j or p3.start_time <= p2.end_time:
                            continue
                        gap2 = p3.start_time - p2.end_time
                        if gap2 < wr.min_rest_between_pieces:
                            continue

                        total = p3.end_time - p1.start_time
                        if total > wr.max_shift_hours * 3600:
                            continue

                        fd = self._make_duty(duty_id, [p1, p2, p3])
                        if fd is not None:
                            self.feasible_duties.append(fd)
                            duty_id += 1

        return self.feasible_duties

    def _make_duty(
        self, duty_id: int, pieces: list[DutyPiece]
    ) -> Optional[FeasibleDuty]:
        """Create a feasible duty from pieces, checking constraints."""
        pieces = sorted(pieces, key=lambda p: p.start_time)
        wr = self.env.work_rules

        start = pieces[0].start_time
        end = pieces[-1].end_time

        # Check meal break feasibility
        has_meal = False
        for i in range(len(pieces) - 1):
            gap_start = max(pieces[i].end_time, wr.meal_window_start)
            gap_end = min(pieces[i + 1].start_time, wr.meal_window_end)
            if gap_end - gap_start >= wr.min_meal_break:
                has_meal = True
                break

        # If shift spans meal window and no break, check if that's ok
        if not has_meal:
            if start < wr.meal_window_end and end > wr.meal_window_start:
                # Shift spans meal window, check if short enough
                if end - start > wr.max_continuous_work * 3600:
                    return None  # violates continuous work limit

        # Compute cost
        working = sum(p.duration for p in pieces)
        idle = 0
        for i in range(len(pieces) - 1):
            gap = pieces[i + 1].start_time - pieces[i].end_time
            idle += min(gap, wr.max_paid_idle)

        paid_hours = (working + idle) / 3600
        cost = paid_hours
        if paid_hours > wr.max_shift_hours:
            cost += (paid_hours - wr.max_shift_hours) * (wr.overtime_rate - 1)

        # Split penalty
        n_splits = sum(1 for i in range(len(pieces) - 1)
                       if pieces[i + 1].start_time - pieces[i].end_time > wr.max_paid_idle)
        cost += n_splits * wr.split_duty_penalty * paid_hours

        return FeasibleDuty(
            duty_id=duty_id,
            piece_ids=[p.piece_id for p in pieces],
            cost=cost,
            start_time=start,
            end_time=end,
            has_meal_break=has_meal,
        )

    def solve_greedy(self) -> dict[int, list[int]]:
        """Greedy set covering: pick duty maximizing coverage efficiency.

        Follows Ceder Ch.10 SPM approach. Scoring prioritizes:
        1. Duties covering more uncovered pieces (saves drivers)
        2. Lower cost per piece among equal-coverage duties

        Returns: {driver_id: [piece_ids]}
        """
        if not self.feasible_duties:
            self.enumerate_feasible_duties()

        uncovered = set(range(len(self.env.pieces)))
        assignments: dict[int, list[int]] = {}
        used_duties: set[int] = set()
        driver_id = 0
        n_drivers = len(self.env.drivers)

        while uncovered and driver_id < n_drivers:
            best_duty = None
            best_coverage = 0
            best_cost = float('inf')

            for fd in self.feasible_duties:
                if fd.duty_id in used_duties:
                    continue
                covered = set(fd.piece_ids) & uncovered
                n_covered = len(covered)
                if n_covered == 0:
                    continue

                # Primary: maximize new coverage; secondary: minimize cost
                if (n_covered > best_coverage or
                    (n_covered == best_coverage and fd.cost < best_cost)):
                    best_coverage = n_covered
                    best_cost = fd.cost
                    best_duty = fd

            if best_duty is None:
                break

            used_duties.add(best_duty.duty_id)
            assignments[driver_id] = best_duty.piece_ids
            uncovered -= set(best_duty.piece_ids)
            driver_id += 1

        return assignments

    def solve_with_recompute(
        self, actual_durations: dict[int, float]
    ) -> tuple[dict[int, list[int]], float]:
        """Full recompute: re-enumerate and re-solve with updated durations.

        This simulates the traditional approach where any disruption
        triggers a complete re-solve of the SPP.

        Returns: (assignments, solve_time_seconds)
        """
        import copy
        t0 = time.perf_counter()

        # Save original piece timings
        original_end_times = {p.piece_id: p.end_time for p in self.env.pieces}

        # Update piece durations based on actual trip durations
        for piece in self.env.pieces:
            total_actual = 0
            for tid in piece.trip_ids:
                if tid in actual_durations:
                    total_actual += actual_durations[tid]
                else:
                    trip = self.env.trips[tid]
                    total_actual += trip.scheduled_arr - trip.scheduled_dep
            piece.end_time = piece.start_time + total_actual

        # Re-enumerate with updated durations
        self.feasible_duties = []
        self.enumerate_feasible_duties()

        # Re-solve
        assignments = self.solve_greedy()

        # Restore original timings so env is not permanently modified
        for piece in self.env.pieces:
            piece.end_time = original_end_times[piece.piece_id]

        solve_time = time.perf_counter() - t0
        return assignments, solve_time
