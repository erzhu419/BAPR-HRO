"""Parse real GTFS data into TransitGraph.

Loads Swiss GTFS static timetable (stops.txt, routes.txt, trips.txt,
stop_times.txt, transfers.txt) into our TransitGraph structure.

The full Swiss dataset has 42K stops and 21M stop_times. For practical
routing experiments, we filter to a time window and geographic region.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .transit_graph import TransitGraph, Stop, Connection


def parse_time(time_str: str) -> int:
    """Parse HH:MM:SS to minutes from midnight. Handles hours >= 24."""
    parts = time_str.strip().strip('"').split(":")
    h, m = int(parts[0]), int(parts[1])
    return h * 60 + m


def load_stops(gtfs_dir: str) -> dict[str, Stop]:
    """Load stops.txt → dict of stop_id → Stop."""
    stops = {}
    path = os.path.join(gtfs_dir, "stops.txt")
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["stop_id"].strip('"')
            name = row["stop_name"].strip('"')
            lat = float(row["stop_lat"]) if row["stop_lat"] else 0.0
            lon = float(row["stop_lon"]) if row["stop_lon"] else 0.0
            stops[sid] = Stop(
                id=hash(sid) % (10**9),
                name=name,
                lat=lat,
                lon=lon,
                min_transfer_time=2,
            )
    return stops


def load_transfers(gtfs_dir: str) -> dict[tuple[str, str], int]:
    """Load transfers.txt → dict of (from_stop, to_stop) → min_transfer_time (minutes)."""
    transfers = {}
    path = os.path.join(gtfs_dir, "transfers.txt")
    if not os.path.exists(path):
        return transfers
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_stop = row["from_stop_id"].strip('"')
            to_stop = row["to_stop_id"].strip('"')
            min_time = int(row.get("min_transfer_time", "120"))
            transfers[(from_stop, to_stop)] = max(min_time // 60, 1)  # seconds → minutes
    return transfers


def load_routes(gtfs_dir: str) -> dict[str, str]:
    """Load routes.txt → dict of route_id → route_short_name."""
    routes = {}
    path = os.path.join(gtfs_dir, "routes.txt")
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["route_id"].strip('"')
            name = row.get("route_short_name", "").strip('"')
            if not name:
                name = row.get("route_long_name", rid).strip('"')
            routes[rid] = name
    return routes


def load_trip_routes(gtfs_dir: str) -> dict[str, str]:
    """Load trips.txt → dict of trip_id → route_id."""
    trip_routes = {}
    path = os.path.join(gtfs_dir, "trips.txt")
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["trip_id"].strip('"')
            rid = row["route_id"].strip('"')
            trip_routes[tid] = rid
    return trip_routes


def load_stop_times(
    gtfs_dir: str,
    time_start: int = 0,
    time_end: int = 1440,
    trip_filter: Optional[set[str]] = None,
) -> list[dict]:
    """Load stop_times.txt, filtered by time window.

    Returns list of dicts with: trip_id, stop_id, arrival, departure, sequence.
    Sorted by (trip_id, sequence).
    """
    records = []
    path = os.path.join(gtfs_dir, "stop_times.txt")
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["trip_id"].strip('"')
            if trip_filter and tid not in trip_filter:
                continue
            arr = parse_time(row["arrival_time"])
            dep = parse_time(row["departure_time"])
            if dep < time_start or dep > time_end:
                continue
            records.append({
                "trip_id": tid,
                "stop_id": row["stop_id"].strip('"'),
                "arrival": arr,
                "departure": dep,
                "sequence": int(row["stop_sequence"]),
            })
    records.sort(key=lambda x: (x["trip_id"], x["sequence"]))
    return records


def build_transit_graph(
    gtfs_dir: str,
    time_start: int = 420,    # 7:00
    time_end: int = 600,      # 10:00
    region_lat: tuple[float, float] = (46.9, 47.5),  # Zurich area
    region_lon: tuple[float, float] = (8.3, 8.8),
    max_connections: int = 50000,
    verbose: bool = True,
) -> TransitGraph:
    """Build a TransitGraph from real Swiss GTFS data.

    Filters to a geographic region and time window to keep the graph
    manageable for experiments.

    Args:
        gtfs_dir: Path to directory with stops.txt, routes.txt, etc.
        time_start, time_end: Time window in minutes from midnight.
        region_lat, region_lon: Lat/lon bounding box for geographic filter.
        max_connections: Cap on number of connections (for memory).
        verbose: Print progress.
    """
    if verbose:
        print("Loading Swiss GTFS data...")

    # Step 1: Load stops and filter by region
    all_stops = load_stops(gtfs_dir)
    region_stops = {}
    for sid, stop in all_stops.items():
        if (region_lat[0] <= stop.lat <= region_lat[1] and
                region_lon[0] <= stop.lon <= region_lon[1]):
            region_stops[sid] = stop
    if verbose:
        print(f"  Stops: {len(all_stops)} total, {len(region_stops)} in region")

    # Step 2: Load transfers for region stops
    transfers = load_transfers(gtfs_dir)
    for (from_s, to_s), min_time in transfers.items():
        if from_s in region_stops:
            region_stops[from_s] = Stop(
                id=region_stops[from_s].id,
                name=region_stops[from_s].name,
                lat=region_stops[from_s].lat,
                lon=region_stops[from_s].lon,
                min_transfer_time=max(min_time, 1),
            )

    # Step 3: Load routes and trip→route mapping
    route_names = load_routes(gtfs_dir)
    trip_routes = load_trip_routes(gtfs_dir)
    if verbose:
        print(f"  Routes: {len(route_names)}, Trips: {len(trip_routes)}")

    # Step 4: Two-pass loading for speed
    # Pass 1: scan stop_times to find trip_ids that serve region stops
    # (only check stop_id membership, skip time parsing)
    if verbose:
        print(f"  Pass 1: finding trips serving region stops...")
    region_trip_ids = set()
    region_stop_set = set(region_stops.keys())
    region_parent_set = set(s.split(":")[0] for s in region_stop_set)
    st_path = os.path.join(gtfs_dir, "stop_times.txt")
    with open(st_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["stop_id"].strip('"')
            if sid in region_stop_set or sid.split(":")[0] in region_parent_set:
                region_trip_ids.add(row["trip_id"].strip('"'))
    if verbose:
        print(f"  Found {len(region_trip_ids)} trips serving region")

    # Pass 2: load only those trips' stop_times within time window
    if verbose:
        print(f"  Pass 2: loading stop_times for region trips (window {time_start}-{time_end})...")
    stop_times = load_stop_times(gtfs_dir, time_start, time_end, trip_filter=region_trip_ids)
    if verbose:
        print(f"  Stop times loaded: {len(stop_times)} records")

    # Group by trip
    trip_stops = defaultdict(list)
    for rec in stop_times:
        if rec["stop_id"] in region_stop_set or rec["stop_id"].split(":")[0] in region_parent_set:
            trip_stops[rec["trip_id"]].append(rec)

    if verbose:
        print(f"  Trips in region: {len(trip_stops)}")

    # Step 5: Build connections
    g = TransitGraph()

    # Add stops
    stop_id_map = {}  # string stop_id → int id
    for sid, stop in region_stops.items():
        int_id = stop.id
        stop_id_map[sid] = int_id
        g.add_stop(stop)

    # Also add parent stops (some stop_ids have :0:1 suffixes)
    for sid in list(region_stops.keys()):
        parent = sid.split(":")[0]
        if parent not in stop_id_map and parent in all_stops:
            stop = all_stops[parent]
            stop_id_map[parent] = stop.id
            g.add_stop(stop)

    # Build connections from consecutive stop_times within each trip
    n_conn = 0
    for tid, recs in trip_stops.items():
        recs.sort(key=lambda x: x["sequence"])
        route_id = trip_routes.get(tid, "unknown")
        route_name = route_names.get(route_id, route_id)

        for i in range(len(recs) - 1):
            dep_sid = recs[i]["stop_id"]
            arr_sid = recs[i + 1]["stop_id"]

            # Resolve stop IDs (handle :0:1 suffixes)
            dep_parent = dep_sid.split(":")[0]
            arr_parent = arr_sid.split(":")[0]
            dep_int = stop_id_map.get(dep_sid, stop_id_map.get(dep_parent))
            arr_int = stop_id_map.get(arr_sid, stop_id_map.get(arr_parent))

            if dep_int is None or arr_int is None:
                continue

            g.add_connection(Connection(
                id=-1,
                route=route_name,
                trip_id=tid,
                dep_stop=dep_int,
                arr_stop=arr_int,
                dep_time=recs[i]["departure"],
                arr_time=recs[i + 1]["arrival"],
            ))
            n_conn += 1
            if n_conn >= max_connections:
                break
        if n_conn >= max_connections:
            break

    if verbose:
        print(f"  Built: {g.summary()}")

    # Assign default delay distributions
    g.assign_distributions({})

    return g


if __name__ == "__main__":
    import time
    t0 = time.time()
    g = build_transit_graph(
        "data/swiss_gtfs",
        time_start=420,
        time_end=600,
        verbose=True,
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    # Quick routing test
    from .durner.topocsa import topocsa
    transfers = g.get_transfer_stops()
    print(f"\n  Transfer stops: {len(transfers)}")

    # Pick two random stops with labels
    stops_with_deps = [s for s in g.stops if g.get_connections_from(s)]
    if len(stops_with_deps) >= 2:
        s1, s2 = stops_with_deps[0], stops_with_deps[-1]
        print(f"  Routing: {g.stops[s1].name} → {g.stops[s2].name}")
        t0 = time.time()
        result = topocsa(g, s1, s2, 480)
        elapsed = (time.time() - t0) * 1000
        print(f"  Result: mean_arrival={result.mean_arrival:.1f}, "
              f"hyperpath={len(result.hyperpath_connections)} conns, "
              f"runtime={elapsed:.0f}ms")
