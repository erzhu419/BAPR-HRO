"""
sim_core/co_line_scheduler.py
==============================
Virtual Co-Line Bus Scheduler for BusSimEnv.

Provides analytic positions of 102X and 705X buses so that
BusSimEnv (single-line mode) can compute co-line headways
matching SUMO rl_bridge.py logic.

Algorithm
---------
Each co-line bus departs at a known time and progresses through
its route at a calibrated average speed. We track its fractional
stop index at any sim_time, then convert shared-stop distances
to temporal headways for the 7X bus.

Calibration (from SUMO log):
  - Average inter-stop travel time: 66.2 s
  - 102X: 38 stops → total route ≈ 2451 s (37 gaps × 66.2 s)
  - 705X: 48 stops → total route ≈ 3113 s (47 gaps × 66.2 s)

Shared stops (7X stop_name → {102X stop_idx, 705X stop_idx}):
  7X05_102X18  → 102X stop 17
  7X06_102X19  → 102X stop 18
  ...
  7X25_102X38  → 102X stop 37
  7X10_102X23_705X24 → 705X stop 23
  ...
  7X18_102X31_705X32 → 705X stop 31

Usage (inside BusSimEnv.step):
    co_bus_dict = self._co_line_scheduler.get_co_line_buses(self.current_time)
    state, reward, done = super().step(action_dict, co_line_buses=co_bus_dict)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------
# Calibration constants (from SUMO log analysis)
# ---------------------------------------------------------------------
_AVG_S_PER_STOP_GAP: float = 66.2     # seconds per inter-stop gap

# Timetable: depart times for each co-line (extracted from SUMO timetable)
_102X_DEPARTS = [
    2160, 2640, 3180, 3660, 4200, 4740, 5280, 5820, 6360, 6840,
    7380, 7920, 8460, 8940, 9480, 10020, 10500, 11040, 11580, 12060,
    12540, 13020, 13560, 14040, 14520, 15060, 15540, 16020, 16500, 17040, 17580,
]  # 31 buses

_705X_DEPARTS = [
    2760, 3480, 4320, 5040, 5760, 6480, 7200, 7920, 8640, 9360,
    10080, 10800, 11520, 12240, 12960, 13680, 14400, 15120, 15840, 16560, 17880,
]  # 21 buses

# Co-line route metrics
_102X_N_STOPS: int = 38
_705X_N_STOPS: int = 48
_102X_TOTAL_TIME: float = (_102X_N_STOPS - 1) * _AVG_S_PER_STOP_GAP   # 2451 s
_705X_TOTAL_TIME: float = (_705X_N_STOPS - 1) * _AVG_S_PER_STOP_GAP   # 3113 s

# Shared stop positions: 7X stop_name → (co_line, co_stop_idx_0based, 7x_stop_idx_0based)
# 102X shared stops: co_stop_idx 17..37 → 7X stop_idx 4..24
_SHARED_STOPS: Dict[str, List[Tuple[str, int, int]]] = {
    "7X05_102X18":           [("102X", 17,  4)],
    "7X06_102X19":           [("102X", 18,  5)],
    "7X07_102X20":           [("102X", 19,  6)],
    "7X08_102X21":           [("102X", 20,  7)],
    "7X10_102X23_705X24":    [("102X", 22,  9), ("705X", 23,  9)],
    "7X12_102X25_705X26":    [("102X", 24, 11), ("705X", 25, 11)],
    "7X13_102X26_705X27":    [("102X", 25, 12), ("705X", 26, 12)],
    "7X14_102X27_705X28":    [("102X", 26, 13), ("705X", 27, 13)],
    "7X15_102X28_705X29":    [("102X", 27, 14), ("705X", 28, 14)],
    "7X16_102X29_705X30":    [("102X", 28, 15), ("705X", 29, 15)],
    "7X17_102X30_705X31":    [("102X", 29, 16), ("705X", 30, 16)],
    "7X18_102X31_705X32":    [("102X", 30, 17), ("705X", 31, 17)],
    "7X19_102X32":           [("102X", 31, 18)],
    "7X20_102X33":           [("102X", 32, 19)],
    "7X21_102X34":           [("102X", 33, 20)],
    "7X22_102X35":           [("102X", 34, 21)],
    "7X23_102X36":           [("102X", 35, 22)],
    "7X24_102X37":           [("102X", 36, 23)],
    "7X25_102X38":           [("102X", 37, 24)],
}

# ── Derive: for each co-line, arrival time at each shared stop from depart ──
# time_to_reach_stop[line][stop_idx] = seconds after depart
_102X_STOP_TIMES = [i * _AVG_S_PER_STOP_GAP for i in range(_102X_N_STOPS)]
_705X_STOP_TIMES = [i * _AVG_S_PER_STOP_GAP for i in range(_705X_N_STOPS)]

_STOP_TIMES: Dict[str, List[float]] = {
    "102X": _102X_STOP_TIMES,
    "705X": _705X_STOP_TIMES,
}
_DEPARTS: Dict[str, List[float]] = {
    "102X": _102X_DEPARTS,
    "705X": _705X_DEPARTS,
}
_TOTAL_TIMES: Dict[str, float] = {
    "102X": _102X_TOTAL_TIME,
    "705X": _705X_TOTAL_TIME,
}


class VirtualCoLineScheduler:
    """
    Provides co-line bus positions for BusSimEnv at any sim_time.

    Returns a dict: {7X_stop_name: [(abs_distance_proxy, speed, line_id), ...]}
    where abs_distance_proxy is the 7X route absolute distance of the co-line
    bus's *current* position, allowing _compute_co_line_headways() to work
    using the standard distance-based formula.

    Because the co-line buses travel on different physical routes, we convert
    their fractional progress to an equivalent 7X route distance so that the
    headway formula (dist_diff / effective_speed) stays meaningful.

    Specifically: if a 102X bus is at fractional position f in its route,
    and the shared stop is at fractional position f_shared, the spatial gap
    along the 7X route between the 7X bus and the 102X bus is:
        gap_stops = (f_shared - f) * n_stops  (positive = co-line ahead)
        gap_time  = gap_stops * avg_s_per_stop
    We return abs_distance_proxy = 7X_pos ± gap_time * seg_speed so that
    dist_diff / seg_speed ≈ gap_time.
    """

    def __init__(self, x7_station_positions: Dict[str, float]):
        """
        Parameters
        ----------
        x7_station_positions : dict
            {stop_name → abs_distance_from_7X_origin (m)}, from BusSimEnv._station_linear_pos
        """
        self._x7_pos = x7_station_positions  # for reference (not critical)

    def get_co_line_buses(
        self,
        sim_time: float,
        seg_speed: float = 8.0,
        target_headway: float = 360.0,
    ) -> Dict[str, List[Tuple[float, float, str]]]:
        """
        Compute co-line bus positions at sim_time.

        For each active co-line bus, we estimate its absolute position along the
        7X route coordinate using its nearest shared stop:

            proxy_pos = x7_stop_abs_pos - time_diff * seg_speed

        where time_diff = time_to_reach_shared_stop - elapsed_since_depart.
          - time_diff > 0 → bus hasn't reached shared stop yet (bus is behind = smaller pos)
          - time_diff < 0 → bus has passed shared stop (bus is ahead = larger pos)

        _compute_co_line_headways then computes:
            dist_diff = ego_bus.absolute_distance - proxy_pos
           (positive → co-line is behind = backward, negative → ahead = forward)

        Only buses within ±2×target_headway (seconds) of the shared stop are included.
        """
        result: Dict[str, List[Tuple[float, float, str]]] = {}
        max_time_gap = target_headway * 2.0

        for stop_name, entries in _SHARED_STOPS.items():
            # bus.py looks up co_line_buses[self.next_station.station_name]
            # station_name is the simple 7X stop id (e.g. "7X05"), not compound
            x7_key = stop_name.split('_')[0]    # "7X05_102X18" → "7X05"

            for line_id, co_stop_idx, _x7_stop_idx in entries:
                departs    = _DEPARTS[line_id]
                stoptimes  = _STOP_TIMES[line_id]
                total_t    = _TOTAL_TIMES[line_id]
                time_to_shared = stoptimes[co_stop_idx]

                for depart in departs:
                    elapsed = sim_time - depart
                    if elapsed < 0 or elapsed > total_t:
                        continue

                    # Signed time gap to the shared stop:
                    #   positive → bus hasn't reached stop yet (bus is behind 7X bus → backward)
                    #   negative → bus has already passed stop (bus is ahead of 7X bus → forward)
                    time_diff = time_to_shared - elapsed

                    if abs(time_diff) > max_time_gap:
                        continue

                    # Encode as direct time (speed=0.0 flags "time mode" to _compute_co_line_headways)
                    # proxy_pos = time_diff (seconds), speed = 0.0
                    result.setdefault(x7_key, []).append(
                        (time_diff, 0.0, line_id)
                    )

        return result
