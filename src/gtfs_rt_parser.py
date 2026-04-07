"""Parse GTFS-RT data to extract real delay distributions.

Processes Swiss GTFS-RT archives (tar.bz2, one per day, ~720 snapshots/day)
to build empirical delay distributions per route, per time-of-day.

Output: delay_distributions.pkl containing:
  {route_id: {hour: {"delays": [int], "cancel_count": int, "total": int}}}
"""

from __future__ import annotations

import os
import tarfile
import pickle
import numpy as np
from collections import defaultdict
from typing import Optional

from google.transit import gtfs_realtime_pb2


def parse_single_snapshot(data: bytes) -> list[dict]:
    """Parse a single GTFS-RT protobuf snapshot.

    Returns list of {route_id, stop_id, delay_seconds, hour}.
    """
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)

    records = []
    for entity in feed.entity:
        if not entity.HasField('trip_update'):
            continue
        tu = entity.trip_update
        route_id = tu.trip.route_id

        for stu in tu.stop_time_update:
            # Prefer arrival delay, fallback to departure delay
            delay = None
            if stu.HasField('arrival') and stu.arrival.delay != 0:
                delay = stu.arrival.delay
            elif stu.HasField('departure'):
                delay = stu.departure.delay

            if delay is None:
                continue

            # Estimate hour from scheduled time if available
            hour = -1
            if stu.HasField('arrival') and stu.arrival.time > 0:
                from datetime import datetime
                hour = datetime.fromtimestamp(stu.arrival.time).hour
            elif stu.HasField('departure') and stu.departure.time > 0:
                from datetime import datetime
                hour = datetime.fromtimestamp(stu.departure.time).hour

            records.append({
                'route_id': route_id,
                'stop_id': stu.stop_id,
                'delay_seconds': delay,
                'hour': hour,
            })

    return records


def process_day(tar_path: str, sample_interval: int = 10) -> list[dict]:
    """Process one day's tar.bz2 archive.

    Args:
        tar_path: Path to YYYY-MM-DD.tar.bz2
        sample_interval: Process every Nth snapshot (for speed).
                         1 = all ~720, 10 = ~72 snapshots.
    """
    records = []
    try:
        with tarfile.open(tar_path, 'r:bz2') as tar:
            members = [m for m in tar.getmembers() if m.name.endswith('.gtfsrt')]
            members.sort(key=lambda m: m.name)

            for idx, member in enumerate(members):
                if idx % sample_interval != 0:
                    continue
                try:
                    f = tar.extractfile(member)
                    if f:
                        data = f.read()
                        recs = parse_single_snapshot(data)
                        records.extend(recs)
                except Exception:
                    continue
    except Exception as e:
        print(f"  Error processing {tar_path}: {e}")
    return records


def build_delay_distributions(
    rt_dir: str,
    route_names: Optional[dict[str, str]] = None,
    n_days: int = 7,
    sample_interval: int = 20,
    verbose: bool = True,
) -> dict:
    """Build empirical delay distributions from GTFS-RT archives.

    Args:
        rt_dir: Directory containing YYYY-MM-DD.tar.bz2 files.
        route_names: {route_id → short_name} from routes.txt.
        n_days: Number of days to process (for speed).
        sample_interval: Process every Nth snapshot per day.
        verbose: Print progress.

    Returns:
        Dict: {route_id: {
            "delays_min": np.array (delays in minutes),
            "mean": float, "std": float,
            "cancel_rate": float,
            "n_obs": int,
            "by_hour": {hour: {"mean": float, "std": float, "n": int}}
        }}
    """
    tar_files = sorted([f for f in os.listdir(rt_dir) if f.endswith('.tar.bz2')])

    if n_days < len(tar_files):
        # Pick weekday samples spread across the period
        tar_files = tar_files[:n_days]

    if verbose:
        print(f"Processing {len(tar_files)} days of GTFS-RT data "
              f"(sample_interval={sample_interval})...")

    # Accumulate delays per route
    route_delays: dict[str, list[int]] = defaultdict(list)
    route_by_hour: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

    for i, tf in enumerate(tar_files):
        tar_path = os.path.join(rt_dir, tf)
        if verbose:
            print(f"  [{i+1}/{len(tar_files)}] {tf} ...", end=" ", flush=True)

        recs = process_day(tar_path, sample_interval)

        for rec in recs:
            rid = rec['route_id']
            delay_sec = rec['delay_seconds']
            delay_min = delay_sec / 60.0

            # Filter extreme values (likely data errors)
            if abs(delay_min) > 120:
                continue

            route_delays[rid].append(delay_min)
            if rec['hour'] >= 0:
                route_by_hour[rid][rec['hour']].append(delay_min)

        if verbose:
            print(f"{len(recs)} records")

    # Build distributions
    if verbose:
        print(f"\nBuilding distributions for {len(route_delays)} routes...")

    distributions = {}
    for rid, delays in route_delays.items():
        if len(delays) < 10:
            continue

        delays_arr = np.array(delays)
        name = route_names.get(rid, rid) if route_names else rid

        # Cancellation: delays > 30 min treated as effective cancellation
        n_cancel = int((np.abs(delays_arr) > 30).sum())

        # By-hour breakdown
        by_hour = {}
        for hour, hdelays in route_by_hour[rid].items():
            if len(hdelays) >= 5:
                harr = np.array(hdelays)
                by_hour[hour] = {
                    "mean": float(harr.mean()),
                    "std": float(harr.std()),
                    "n": len(hdelays),
                }

        distributions[rid] = {
            "name": name,
            "delays_min": delays_arr,
            "mean": float(delays_arr.mean()),
            "std": float(delays_arr.std()),
            "median": float(np.median(delays_arr)),
            "p95": float(np.percentile(delays_arr, 95)),
            "cancel_rate": n_cancel / len(delays),
            "n_obs": len(delays),
            "by_hour": by_hour,
        }

    if verbose:
        # Summary statistics
        all_means = [d["mean"] for d in distributions.values()]
        all_stds = [d["std"] for d in distributions.values()]
        print(f"\nSummary ({len(distributions)} routes):")
        print(f"  Mean delay: {np.mean(all_means):.1f} ± {np.std(all_means):.1f} min")
        print(f"  Mean std:   {np.mean(all_stds):.1f} min")
        print(f"  Total observations: {sum(d['n_obs'] for d in distributions.values()):,}")

    return distributions


def save_distributions(distributions: dict, path: str):
    """Save distributions to pickle."""
    # Convert numpy arrays to lists for serialization
    serializable = {}
    for rid, d in distributions.items():
        d_copy = dict(d)
        d_copy["delays_min"] = d_copy["delays_min"].tolist()
        serializable[rid] = d_copy
    with open(path, 'wb') as f:
        pickle.dump(serializable, f)


def load_distributions(path: str) -> dict:
    """Load distributions from pickle."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    for rid, d in data.items():
        d["delays_min"] = np.array(d["delays_min"])
    return data


if __name__ == "__main__":
    import time

    # Load route names from static GTFS
    from .gtfs_parser import load_routes
    route_names = load_routes("data/swiss_gtfs")

    t0 = time.time()
    dists = build_delay_distributions(
        "data/swiss_rt/gtfs-rt",
        route_names=route_names,
        n_days=7,
        sample_interval=20,
        verbose=True,
    )
    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # Save
    save_distributions(dists, "data/delay_distributions.pkl")
    print(f"Saved to data/delay_distributions.pkl")

    # Show top 10 most delayed routes
    sorted_routes = sorted(dists.values(), key=lambda d: d["mean"], reverse=True)
    print(f"\nTop 10 most delayed routes:")
    for d in sorted_routes[:10]:
        print(f"  {d['name']:>10s}: mean={d['mean']:.1f}min std={d['std']:.1f} "
              f"p95={d['p95']:.1f} cancel={d['cancel_rate']:.1%} n={d['n_obs']}")
