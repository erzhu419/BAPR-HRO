"""Generate a synthetic transit network for testing.

Creates a small network modeling the user's bus experience:
- Multiple lines with overlapping coverage
- Transfer points where lines intersect
- Realistic headways and travel times

No GTFS download needed. Network follows GTFS schema conventions
so real data can be swapped in later.
"""

from __future__ import annotations

import numpy as np
from .transit_graph import TransitGraph, Stop, Connection


def create_bus_story_network() -> TransitGraph:
    """Create a network matching the motivation scenario.

    Topology (20 stops, 5 lines):

        Line 402 (direct): A(0)→S1→S2→B(3)→S4→S5→C(6)→S7→S8→D(9)
        Line 102:          A(0)→S10→S11→B(3)
        Line 311:          B(3)→S12→S13→C(6)→S14→D(9)
        Line 317:          B(3)→S15→S16→C(6)
        Line 202:          C(6)→S17→S18→D(9)

    Transfer stops: A(0), B(3), C(6), D(9)
    """
    g = TransitGraph()

    # Stops
    stop_names = {
        0: "A (Origin)", 1: "S1", 2: "S2",
        3: "B (Transfer 1)", 4: "S4", 5: "S5",
        6: "C (Transfer 2)", 7: "S7", 8: "S8",
        9: "D (Destination)",
        10: "S10 (102)", 11: "S11 (102)",
        12: "S12 (311)", 13: "S13 (311)", 14: "S14 (311)",
        15: "S15 (317)", 16: "S16 (317)",
        17: "S17 (202)", 18: "S18 (202)",
    }
    for sid, name in stop_names.items():
        transfer_time = 3 if sid in (0, 3, 6, 9) else 2
        g.add_stop(Stop(id=sid, name=name, min_transfer_time=transfer_time))

    # Generate connections: each line runs every `headway` minutes
    # starting from `first_dep` for `n_trips` trips.
    line_defs = [
        {
            "route": "402",
            "stops": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "travel_times": [5, 5, 5, 5, 5, 5, 5, 5, 5],  # 45 min total
            "dwell_time": 1,
            "headway": 20,
            "first_dep": 360,   # 6:00
            "n_trips": 30,      # runs until ~16:00
        },
        {
            "route": "102",
            "stops": [0, 10, 11, 3],
            "travel_times": [4, 4, 4],  # 12 min A→B
            "dwell_time": 1,
            "headway": 10,
            "first_dep": 360,
            "n_trips": 60,
        },
        {
            "route": "311",
            "stops": [3, 12, 13, 6, 14, 9],
            "travel_times": [4, 4, 4, 5, 5],  # 22 min B→D
            "dwell_time": 1,
            "headway": 15,
            "first_dep": 360,
            "n_trips": 40,
        },
        {
            "route": "317",
            "stops": [3, 15, 16, 6],
            "travel_times": [5, 5, 5],  # 15 min B→C
            "dwell_time": 1,
            "headway": 12,
            "first_dep": 360,
            "n_trips": 50,
        },
        {
            "route": "202",
            "stops": [6, 17, 18, 9],
            "travel_times": [4, 4, 4],  # 12 min C→D
            "dwell_time": 1,
            "headway": 8,
            "first_dep": 360,
            "n_trips": 75,
        },
    ]

    for line in line_defs:
        route = line["route"]
        stops = line["stops"]
        ttimes = line["travel_times"]
        dwell = line["dwell_time"]
        headway = line["headway"]

        for trip_idx in range(line["n_trips"]):
            trip_id = f"{route}_trip{trip_idx}"
            dep = line["first_dep"] + trip_idx * headway

            for seg_idx in range(len(stops) - 1):
                s_dep = stops[seg_idx]
                s_arr = stops[seg_idx + 1]
                t_dep = dep
                t_arr = dep + ttimes[seg_idx]

                g.add_connection(Connection(
                    id=-1,  # assigned by add_connection
                    route=route,
                    trip_id=trip_id,
                    dep_stop=s_dep,
                    arr_stop=s_arr,
                    dep_time=t_dep,
                    arr_time=t_arr,
                ))
                dep = t_arr + dwell  # next segment departs after dwell

    # Assign default delay distributions (normal regime)
    g.assign_distributions({})
    return g


def create_regime_distributions(regime: str = "normal") -> dict[str, dict]:
    """Create delay distributions for different regimes.

    Args:
        regime: One of "normal", "disrupted_402", "rush_hour", "weather"

    Returns:
        Dict mapping route → delay distribution parameters.
    """
    if regime == "normal":
        # All routes: slight positive delay, centered around +1 min
        delays = np.arange(-2, 8)
        probs = np.exp(-0.5 * ((delays - 1) / 1.5) ** 2)
        probs /= probs.sum()
        return {r: {"delay_probs": probs, "delay_offset": -2}
                for r in ["402", "102", "311", "317", "202"]}

    elif regime == "disrupted_402":
        # 402 is massively delayed or canceled, 311 also affected (corridor disruption)
        base = create_regime_distributions("normal")
        # 402: mostly canceled
        delays_402 = np.arange(0, 35)
        probs_402 = np.exp(-0.5 * ((delays_402 - 20) / 5) ** 2)
        probs_402 /= probs_402.sum()
        probs_402 *= 0.15  # 85% cancel probability
        base["402"] = {"delay_probs": probs_402, "delay_offset": 0, "cancel_prob": 0.85}
        # 311: partially affected (shares some infrastructure with 402)
        delays_311 = np.arange(-1, 20)
        probs_311 = np.exp(-0.5 * ((delays_311 - 8) / 4) ** 2)
        probs_311 /= probs_311.sum()
        probs_311 *= 0.7  # 30% cancel
        base["311"] = {"delay_probs": probs_311, "delay_offset": -1, "cancel_prob": 0.3}
        return base

    elif regime == "rush_hour":
        # All routes: higher delays
        delays = np.arange(-1, 15)
        probs = np.exp(-0.5 * ((delays - 5) / 3) ** 2)
        probs /= probs.sum()
        return {r: {"delay_probs": probs, "delay_offset": -1}
                for r in ["402", "102", "311", "317", "202"]}

    elif regime == "weather":
        # All routes: much higher variance
        delays = np.arange(-2, 20)
        probs = np.exp(-0.5 * ((delays - 3) / 5) ** 2)
        probs /= probs.sum()
        return {r: {"delay_probs": probs, "delay_offset": -2}
                for r in ["402", "102", "311", "317", "202"]}

    else:
        raise ValueError(f"Unknown regime: {regime}")
