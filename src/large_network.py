"""Large synthetic transit network for realistic evaluation.

Grid-based city network: 7x7 grid of stops with 12 bus lines.
Origin at top-left, destination at bottom-right.
Multiple corridors: north, central, south.

Corridor disruption blocks the central corridor, forcing rerouting
through north or south. Static hyperpath may not contain these
alternatives if central corridor was dominant in normal regime.
"""

from __future__ import annotations

import numpy as np
from .transit_graph import TransitGraph, Stop, Connection


def create_grid_network(
    grid_rows: int = 7,
    grid_cols: int = 7,
    n_lines: int = 12,
    headway_range: tuple[int, int] = (8, 20),
    seed: int = 0,
) -> TransitGraph:
    """Create a grid-based transit network.

    Layout (7x7 = 49 stops):
        (0,0)  (0,1)  ...  (0,6)     ← North corridor
        (1,0)  (1,1)  ...  (1,6)
        ...    ...    ...  ...        ← Central corridor (rows 2-4)
        ...    ...    ...  ...
        (6,0)  (6,1)  ...  (6,6)     ← South corridor

    Origin: (0,0) = stop 0
    Destination: (6,6) = stop 48

    Lines:
    - 3 east-west express lines (north/central/south corridors)
    - 3 north-south feeder lines (west/middle/east)
    - 4 diagonal/local lines connecting corridors
    - 2 crosstown express lines
    """
    rng = np.random.default_rng(seed)
    g = TransitGraph()

    # Create stops on grid
    for r in range(grid_rows):
        for c in range(grid_cols):
            sid = r * grid_cols + c
            is_major = (r in (0, 3, 6) or c in (0, 3, 6))
            transfer_time = 2 if is_major else 3
            g.add_stop(Stop(
                id=sid,
                name=f"S({r},{c})",
                lat=r, lon=c,
                min_transfer_time=transfer_time,
            ))

    # Define lines as sequences of (row, col) pairs
    line_definitions = [
        # === ASYMMETRIC DESIGN ===
        # Central corridor is DOMINANT in normal regime (fast + frequent).
        # North/South corridors are SLOW alternatives that only make sense
        # when central is disrupted. This creates the conditions where
        # adaptive routing has a genuine advantage: the static hyperpath
        # will favor central corridor, and when it's disrupted, the
        # adaptive router can switch to alternatives not in the static hyperpath.

        # Central corridor: FAST (travel=3) and FREQUENT (headway=6)
        {"name": "EW-Central", "stops": [(3, c) for c in range(7)],
         "headway": 6,  "travel": 3, "first": 360, "trips": 70},

        # North corridor: SLOW (travel=6) and INFREQUENT (headway=15)
        {"name": "EW-North",  "stops": [(0, c) for c in range(7)],
         "headway": 15, "travel": 6, "first": 360, "trips": 35},

        # South corridor: SLOW (travel=6) and INFREQUENT (headway=15)
        {"name": "EW-South",  "stops": [(6, c) for c in range(7)],
         "headway": 15, "travel": 6, "first": 360, "trips": 35},

        # N-S feeders: NS-Mid is fast (feeds central), NS-West/East are slow
        {"name": "NS-West",   "stops": [(r, 0) for r in range(7)],
         "headway": 15, "travel": 6, "first": 360, "trips": 35},
        {"name": "NS-Mid",    "stops": [(r, 3) for r in range(7)],
         "headway": 8,  "travel": 4, "first": 360, "trips": 50},
        {"name": "NS-East",   "stops": [(r, 6) for r in range(7)],
         "headway": 15, "travel": 6, "first": 360, "trips": 35},

        # Diagonal: SLOW alternatives
        {"name": "Diag-NE",   "stops": [(i, i) for i in range(7)],
         "headway": 20, "travel": 7, "first": 365, "trips": 25},
        {"name": "Diag-SE",   "stops": [(6-i, i) for i in range(7)],
         "headway": 20, "travel": 7, "first": 365, "trips": 25},

        # Links to central corridor: fast (feeds the dominant route)
        {"name": "NC-Link1",  "stops": [(0,2),(1,2),(2,2),(3,2)],
         "headway": 10, "travel": 4, "first": 370, "trips": 45},
        {"name": "NC-Link2",  "stops": [(0,4),(1,4),(2,4),(3,4)],
         "headway": 10, "travel": 4, "first": 370, "trips": 45},
        {"name": "CS-Link1",  "stops": [(3,2),(4,2),(5,2),(6,2)],
         "headway": 10, "travel": 4, "first": 370, "trips": 45},
        {"name": "CS-Link2",  "stops": [(3,4),(4,4),(5,4),(6,4)],
         "headway": 10, "travel": 4, "first": 370, "trips": 45},
    ]

    for line_def in line_definitions:
        route = line_def["name"]
        stop_coords = line_def["stops"]
        stop_ids = [r * grid_cols + c for r, c in stop_coords]
        headway = line_def["headway"]
        travel = line_def["travel"]
        dwell = 1

        for trip_idx in range(line_def["trips"]):
            trip_id = f"{route}_t{trip_idx}"
            dep = line_def["first"] + trip_idx * headway

            for seg in range(len(stop_ids) - 1):
                g.add_connection(Connection(
                    id=-1,
                    route=route,
                    trip_id=trip_id,
                    dep_stop=stop_ids[seg],
                    arr_stop=stop_ids[seg + 1],
                    dep_time=dep,
                    arr_time=dep + travel,
                ))
                dep = dep + travel + dwell

    # Assign default distributions
    g.assign_distributions({})
    return g


def create_grid_regime_distributions(regime: str = "normal") -> dict[str, dict]:
    """Regime distributions for the grid network.

    Regimes:
    - normal: all lines run with small delays
    - central_disruption: EW-Central + NC/CS links through central corridor are
      heavily delayed or canceled. Forces rerouting through north or south.
    - south_weather: south corridor has weather delays
    - full_chaos: multiple corridors affected simultaneously
    """
    all_routes = [
        "EW-North", "EW-Central", "EW-South",
        "NS-West", "NS-Mid", "NS-East",
        "Diag-NE", "Diag-SE",
        "NC-Link1", "NC-Link2", "CS-Link1", "CS-Link2",
    ]

    if regime == "normal":
        delays = np.arange(-2, 8)
        probs = np.exp(-0.5 * ((delays - 1) / 1.5) ** 2)
        probs /= probs.sum()
        return {r: {"delay_probs": probs, "delay_offset": -2} for r in all_routes}

    elif regime == "central_disruption":
        base = create_grid_regime_distributions("normal")
        # Central corridor: mostly canceled
        for r in ["EW-Central"]:
            d = np.arange(0, 40)
            p = np.exp(-0.5 * ((d - 25) / 5) ** 2)
            p /= p.sum()
            p *= 0.1  # 90% cancel
            base[r] = {"delay_probs": p, "delay_offset": 0, "cancel_prob": 0.9}
        # N-S links through central: heavily delayed
        for r in ["NS-Mid", "NC-Link1", "NC-Link2", "CS-Link1", "CS-Link2"]:
            d = np.arange(-1, 25)
            p = np.exp(-0.5 * ((d - 10) / 4) ** 2)
            p /= p.sum()
            p *= 0.6  # 40% cancel
            base[r] = {"delay_probs": p, "delay_offset": -1, "cancel_prob": 0.4}
        return base

    elif regime == "south_weather":
        base = create_grid_regime_distributions("normal")
        for r in ["EW-South", "CS-Link1", "CS-Link2", "Diag-SE"]:
            d = np.arange(-2, 20)
            p = np.exp(-0.5 * ((d - 5) / 4) ** 2)
            p /= p.sum()
            base[r] = {"delay_probs": p, "delay_offset": -2}
        return base

    elif regime == "full_chaos":
        base = create_grid_regime_distributions("central_disruption")
        # Also hit south
        for r in ["EW-South", "Diag-SE"]:
            d = np.arange(0, 30)
            p = np.exp(-0.5 * ((d - 15) / 5) ** 2)
            p /= p.sum()
            p *= 0.5
            base[r] = {"delay_probs": p, "delay_offset": 0, "cancel_prob": 0.5}
        return base

    else:
        raise ValueError(f"Unknown regime: {regime}")
