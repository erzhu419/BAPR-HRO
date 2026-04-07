"""Crew Scheduling Environment for Bus Transit.

Models the bus crew scheduling problem (Ceder Ch.10) as a sequential
decision problem under uncertainty. Drivers are assigned to duty pieces
(trip segments between relief points), subject to work-rule constraints.

Uncertain parameters (learned online via Bayesian posteriors):
  - Trip duration (traffic, weather, demand variability)
  - Driver availability (sick calls, no-shows)
  - Connection time between consecutive trips at relief points

Deterministic constraints (define feasible duty set):
  - Maximum shift duration (e.g., 8 hours)
  - Mandatory meal break (e.g., 30 min within 11:00-13:00)
  - Minimum rest between shifts (e.g., 30 min)
  - Relief points: terminals only (drivers can only swap at terminals)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DriverStatus(Enum):
    AVAILABLE = auto()
    ON_DUTY = auto()
    ON_BREAK = auto()
    OFF_DUTY = auto()
    ABSENT = auto()


@dataclass
class Trip:
    """A single bus trip (one direction, one route)."""
    trip_id: int
    route_id: int
    direction: int          # 0=downstream, 1=upstream
    dep_terminal: int       # departure terminal stop_id
    arr_terminal: int       # arrival terminal stop_id
    scheduled_dep: float    # seconds since midnight
    scheduled_arr: float    # seconds since midnight
    n_stops: int            # number of intermediate stops

    # Realized values (filled during simulation)
    actual_dep: float = 0.0
    actual_arr: float = 0.0
    assigned_driver: int = -1
    delay: float = 0.0      # actual - scheduled arrival


@dataclass
class DutyPiece:
    """A duty piece: contiguous sequence of trips assigned to one driver.

    Created by splitting vehicle blocks at relief points (terminals).
    This is the atomic unit in the SPP formulation.
    """
    piece_id: int
    trip_ids: list[int]
    start_time: float       # earliest departure
    end_time: float         # latest arrival (scheduled)
    start_terminal: int
    end_terminal: int

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class Duty:
    """A feasible driver duty (shift): sequence of duty pieces.

    Must satisfy all work-rule constraints.
    """
    duty_id: int
    piece_ids: list[int]
    driver_id: int = -1
    start_time: float = 0.0
    end_time: float = 0.0
    has_meal_break: bool = False
    total_paid_time: float = 0.0
    total_working_time: float = 0.0
    cost: float = 0.0


@dataclass
class Driver:
    """A bus driver with availability and fatigue state."""
    driver_id: int
    name: str
    status: DriverStatus = DriverStatus.AVAILABLE
    shift_start: float = 0.0
    accumulated_work: float = 0.0       # seconds worked today
    last_break_end: float = 0.0         # when last break ended
    had_meal_break: bool = False
    assigned_pieces: list[int] = field(default_factory=list)

    # Stochastic properties (unknown, learned via posterior)
    true_absence_prob: float = 0.05     # ground truth, hidden from scheduler
    true_speed_factor: float = 1.0      # >1 means slower than average


@dataclass
class WorkRules:
    """Labour agreement constraints (deterministic, known)."""
    max_shift_hours: float = 8.0
    max_continuous_work: float = 5.0    # hours before mandatory break
    min_meal_break: float = 30 * 60     # 30 min in seconds
    meal_window_start: float = 11 * 3600  # 11:00
    meal_window_end: float = 13 * 3600    # 13:00
    min_rest_between_pieces: float = 5 * 60  # 5 min changeover
    max_paid_idle: float = 40 * 60      # T_max from Ceder (swing time)
    overtime_rate: float = 1.5          # pay multiplier after 8h
    split_duty_penalty: float = 0.2     # cost penalty for split shifts


# ---------------------------------------------------------------------------
# Network and schedule generation
# ---------------------------------------------------------------------------

class CrewSchedulingEnv:
    """Bus crew scheduling environment.

    Builds on the RE-SAC transit network (copied to sim_core/).
    Generates trips from the timetable, creates duty pieces at terminals,
    and presents the crew scheduling problem.
    """

    def __init__(
        self,
        data_dir: str | Path = None,
        n_drivers: int = 20,
        seed: int = 42,
        work_rules: WorkRules = None,
        disruption_level: str = "normal",
    ):
        self.rng = np.random.default_rng(seed)
        self.work_rules = work_rules or WorkRules()
        self.disruption_level = disruption_level

        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)

        # Load transit network data
        self.trips: list[Trip] = []
        self.pieces: list[DutyPiece] = []
        self.drivers: list[Driver] = []
        self.duties: list[Duty] = []

        self._load_timetable()
        self._create_duty_pieces()
        self._create_drivers(n_drivers)

        # Simulation state
        self.current_time: float = 0.0
        self.sim_log: list[dict] = []

    def _load_timetable(self):
        """Load trips from the RE-SAC timetable Excel file."""
        tt_path = self.data_dir / "time_table.xlsx"
        stop_path = self.data_dir / "stop_news.xlsx"
        route_path = self.data_dir / "route_news.xlsx"

        # Always use synthetic multi-route timetable for crew scheduling
        # (RE-SAC data is single-route, not rich enough for crew experiments)
        self._generate_synthetic_trips()

        # Sort by departure time
        self.trips.sort(key=lambda t: t.scheduled_dep)

    def _find_terminals(self, df_stops: pd.DataFrame) -> list[int]:
        """Identify terminal stops (first and last)."""
        stop_ids = df_stops.iloc[:, 0].tolist()
        if len(stop_ids) >= 2:
            return [stop_ids[0], stop_ids[-1]]
        return [0, 1]

    def _generate_synthetic_trips(self):
        """Generate a realistic synthetic timetable.

        3 routes, 2 directions each, ~6:00 to 22:00 service.
        Matches the scale of a medium bus network.
        """
        routes = [
            {"id": 0, "name": "7", "terminals": (0, 1), "headway_peak": 6,
             "headway_offpeak": 12, "travel_time": 35},
            {"id": 1, "name": "102", "terminals": (0, 2), "headway_peak": 8,
             "headway_offpeak": 15, "travel_time": 45},
            {"id": 2, "name": "311", "terminals": (1, 2), "headway_peak": 10,
             "headway_offpeak": 20, "travel_time": 30},
        ]

        trip_id = 0
        for route in routes:
            for direction in [0, 1]:
                dep_term = route["terminals"][direction]
                arr_term = route["terminals"][1 - direction]
                t = 6 * 3600  # 6:00 AM

                while t < 22 * 3600:  # until 10:00 PM
                    is_peak = (7 * 3600 <= t <= 9 * 3600) or \
                              (17 * 3600 <= t <= 19 * 3600)
                    headway = route["headway_peak" if is_peak else "headway_offpeak"]
                    travel = route["travel_time"] * 60

                    trip = Trip(
                        trip_id=trip_id,
                        route_id=route["id"],
                        direction=direction,
                        dep_terminal=dep_term,
                        arr_terminal=arr_term,
                        scheduled_dep=t,
                        scheduled_arr=t + travel,
                        n_stops=15 + route["id"] * 5,
                    )
                    self.trips.append(trip)
                    trip_id += 1
                    t += headway * 60

    def _create_duty_pieces(self):
        """Create duty pieces by grouping trips into vehicle blocks.

        A vehicle block is a sequence of trips served by the same bus.
        Pieces are split at relief points when:
          - Idle time exceeds max_paid_idle, OR
          - Accumulated piece duration exceeds max_piece_duration (2h)

        The second criterion ensures pieces are manageable for crew
        scheduling even when bus headways are short.
        """
        from collections import defaultdict
        blocks = defaultdict(list)

        for trip in self.trips:
            key = (trip.route_id, trip.dep_terminal)
            blocks[key].append(trip)

        max_piece_duration = 2.0 * 3600  # 2 hours max per piece

        piece_id = 0
        for key, trip_list in blocks.items():
            trip_list.sort(key=lambda t: t.scheduled_dep)

            current_piece_trips = []
            for trip in trip_list:
                if current_piece_trips:
                    prev = current_piece_trips[-1]
                    gap = trip.scheduled_dep - prev.scheduled_arr
                    piece_duration = trip.scheduled_arr - current_piece_trips[0].scheduled_dep

                    # Split if gap too large OR piece too long
                    if gap > self.work_rules.max_paid_idle or piece_duration > max_piece_duration:
                        piece = DutyPiece(
                            piece_id=piece_id,
                            trip_ids=[t.trip_id for t in current_piece_trips],
                            start_time=current_piece_trips[0].scheduled_dep,
                            end_time=current_piece_trips[-1].scheduled_arr,
                            start_terminal=current_piece_trips[0].dep_terminal,
                            end_terminal=current_piece_trips[-1].arr_terminal,
                        )
                        self.pieces.append(piece)
                        piece_id += 1
                        current_piece_trips = []

                current_piece_trips.append(trip)

            # Last piece in block
            if current_piece_trips:
                piece = DutyPiece(
                    piece_id=piece_id,
                    trip_ids=[t.trip_id for t in current_piece_trips],
                    start_time=current_piece_trips[0].scheduled_dep,
                    end_time=current_piece_trips[-1].scheduled_arr,
                    start_terminal=current_piece_trips[0].dep_terminal,
                    end_terminal=current_piece_trips[-1].arr_terminal,
                )
                self.pieces.append(piece)
                piece_id += 1

        self.pieces.sort(key=lambda p: p.start_time)

    def _create_drivers(self, n_drivers: int):
        """Create driver pool with heterogeneous properties."""
        for i in range(n_drivers):
            absence_prob = self.rng.beta(2, 38)  # mean ~5%, most drivers reliable
            speed_factor = self.rng.lognormal(0, 0.1)  # slight variation

            driver = Driver(
                driver_id=i,
                name=f"Driver_{i:02d}",
                true_absence_prob=absence_prob,
                true_speed_factor=speed_factor,
            )
            self.drivers.append(driver)

    # ------------------------------------------------------------------
    # Feasibility checking (work rules)
    # ------------------------------------------------------------------

    def is_feasible_assignment(
        self, driver: Driver, piece: DutyPiece, current_assignments: dict[int, list[int]]
    ) -> bool:
        """Check if assigning piece to driver satisfies all work rules."""
        assigned = current_assignments.get(driver.driver_id, [])
        assigned_pieces = [self.pieces[pid] for pid in assigned]

        # 1. Check shift duration
        if assigned_pieces:
            shift_start = min(p.start_time for p in assigned_pieces)
            shift_end = max(p.end_time for p in assigned_pieces)
            new_start = min(shift_start, piece.start_time)
            new_end = max(shift_end, piece.end_time)
        else:
            new_start = piece.start_time
            new_end = piece.end_time

        total_shift = new_end - new_start
        if total_shift > self.work_rules.max_shift_hours * 3600:
            return False

        # 2. Check rest between pieces
        for ap in assigned_pieces:
            gap = piece.start_time - ap.end_time
            if 0 < gap < self.work_rules.min_rest_between_pieces:
                return False
            gap2 = ap.start_time - piece.end_time
            if 0 < gap2 < self.work_rules.min_rest_between_pieces:
                return False

        # 3. Check no overlap
        for ap in assigned_pieces:
            if piece.start_time < ap.end_time and piece.end_time > ap.start_time:
                return False

        # 4. Check terminal compatibility (driver must be at the right place)
        for ap in assigned_pieces:
            if ap.end_time <= piece.start_time:
                if ap.end_terminal != piece.start_terminal:
                    # Need deadhead time, simplified: disallow for now
                    gap = piece.start_time - ap.end_time
                    if gap < 30 * 60:  # need at least 30 min for deadhead
                        return False

        return True

    def check_meal_break(
        self, driver_id: int, assignments: dict[int, list[int]]
    ) -> bool:
        """Check if driver's schedule has a valid meal break window."""
        assigned = assignments.get(driver_id, [])
        if not assigned:
            return True

        pieces = sorted(
            [self.pieces[pid] for pid in assigned],
            key=lambda p: p.start_time
        )

        wr = self.work_rules
        # Check if there's a gap >= min_meal_break within meal window
        for i in range(len(pieces) - 1):
            gap_start = max(pieces[i].end_time, wr.meal_window_start)
            gap_end = min(pieces[i + 1].start_time, wr.meal_window_end)
            if gap_end - gap_start >= wr.min_meal_break:
                return True

        # Check if first piece starts after meal window (break before shift)
        if pieces[0].start_time >= wr.meal_window_end:
            return True
        # Check if last piece ends before meal window
        if pieces[-1].end_time <= wr.meal_window_start:
            return True

        # Single piece with gap in meal window
        if len(pieces) == 1:
            p = pieces[0]
            if p.end_time <= wr.meal_window_start or p.start_time >= wr.meal_window_end:
                return True
            # Piece spans meal window with no break → infeasible if > max_continuous_work
            if p.duration > wr.max_continuous_work * 3600:
                return False

        return True

    # ------------------------------------------------------------------
    # Simulation: sample actual trip durations
    # ------------------------------------------------------------------

    def sample_trip_duration(self, trip: Trip) -> float:
        """Sample actual trip duration based on disruption level.

        Returns the realized travel time in seconds.
        """
        scheduled_duration = trip.scheduled_arr - trip.scheduled_dep

        if self.disruption_level == "normal":
            # Normal: mean +1 min delay, std 2 min
            delay = self.rng.normal(1 * 60, 2 * 60)
        elif self.disruption_level == "rush_hour":
            # Rush hour: larger delays during peak
            is_peak = (7 * 3600 <= trip.scheduled_dep <= 9 * 3600) or \
                      (17 * 3600 <= trip.scheduled_dep <= 19 * 3600)
            if is_peak:
                delay = self.rng.normal(5 * 60, 4 * 60)
            else:
                delay = self.rng.normal(1 * 60, 2 * 60)
        elif self.disruption_level == "disrupted":
            # Major disruption: route 0 heavily affected
            if trip.route_id == 0:
                delay = self.rng.normal(10 * 60, 6 * 60)
            else:
                delay = self.rng.normal(2 * 60, 3 * 60)
        elif self.disruption_level == "driver_shortage":
            # Normal delays but some drivers call in sick
            delay = self.rng.normal(1 * 60, 2 * 60)
        else:
            delay = self.rng.normal(1 * 60, 2 * 60)

        return max(scheduled_duration + delay, scheduled_duration * 0.8)

    def sample_driver_availability(self, driver: Driver) -> bool:
        """Sample whether a driver shows up for their shift."""
        if self.disruption_level == "driver_shortage":
            # Double the absence probability
            prob = min(driver.true_absence_prob * 2, 0.5)
        else:
            prob = driver.true_absence_prob
        return self.rng.random() > prob

    # ------------------------------------------------------------------
    # Duty cost computation (Ceder's objective)
    # ------------------------------------------------------------------

    def compute_duty_cost(self, duty: Duty) -> float:
        """Compute cost of a duty following Ceder's model.

        Cost components:
        - Base pay: hours worked × rate
        - Overtime: hours beyond max_shift × overtime_rate
        - Idle time penalty: paid idle time (swing time)
        - Split duty penalty: if duty has unpaid gaps
        """
        wr = self.work_rules
        pieces = [self.pieces[pid] for pid in duty.piece_ids]
        pieces.sort(key=lambda p: p.start_time)

        shift_start = pieces[0].start_time
        shift_end = pieces[-1].end_time
        total_span = shift_end - shift_start

        # Working time (sum of piece durations)
        working_time = sum(p.duration for p in pieces)

        # Idle time (gaps between pieces, paid up to T_max each)
        idle_time = 0
        for i in range(len(pieces) - 1):
            gap = pieces[i + 1].start_time - pieces[i].end_time
            idle_time += min(gap, wr.max_paid_idle)

        # Total paid time
        paid_time = working_time + idle_time
        paid_hours = paid_time / 3600

        # Base cost
        cost = paid_hours

        # Overtime
        if paid_hours > wr.max_shift_hours:
            overtime_hours = paid_hours - wr.max_shift_hours
            cost += overtime_hours * (wr.overtime_rate - 1)

        # Split duty penalty
        n_gaps = sum(1 for i in range(len(pieces) - 1)
                     if pieces[i + 1].start_time - pieces[i].end_time > wr.max_paid_idle)
        cost += n_gaps * wr.split_duty_penalty * paid_hours

        duty.total_paid_time = paid_time
        duty.total_working_time = working_time
        duty.cost = cost
        return cost

    # ------------------------------------------------------------------
    # Summary / metrics
    # ------------------------------------------------------------------

    def get_schedule_metrics(
        self, assignments: dict[int, list[int]]
    ) -> dict:
        """Compute metrics for a complete crew schedule."""
        n_covered = set()
        total_cost = 0.0
        n_duties = 0
        overtime_hours = 0.0
        meal_violations = 0

        for driver_id, piece_ids in assignments.items():
            if not piece_ids:
                continue
            n_duties += 1
            n_covered.update(piece_ids)

            duty = Duty(
                duty_id=n_duties,
                piece_ids=piece_ids,
                driver_id=driver_id,
            )
            cost = self.compute_duty_cost(duty)
            total_cost += cost

            paid_hours = duty.total_paid_time / 3600
            if paid_hours > self.work_rules.max_shift_hours:
                overtime_hours += paid_hours - self.work_rules.max_shift_hours

            if not self.check_meal_break(driver_id, assignments):
                meal_violations += 1

        uncovered = set(range(len(self.pieces))) - n_covered

        return {
            "n_pieces_total": len(self.pieces),
            "n_pieces_covered": len(n_covered),
            "n_uncovered": len(uncovered),
            "n_duties": n_duties,
            "total_cost": total_cost,
            "avg_cost_per_duty": total_cost / max(n_duties, 1),
            "overtime_hours": overtime_hours,
            "meal_violations": meal_violations,
            "coverage_rate": len(n_covered) / max(len(self.pieces), 1),
        }

    def __repr__(self) -> str:
        return (
            f"CrewSchedulingEnv("
            f"trips={len(self.trips)}, "
            f"pieces={len(self.pieces)}, "
            f"drivers={len(self.drivers)}, "
            f"disruption={self.disruption_level})"
        )
