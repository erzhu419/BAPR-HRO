"""Build per-day delay distributions for the 35 days of GTFS-RT data.

Parallel version: 15 worker processes via multiprocessing.Pool.

For each day's tar.bz2 archive, extract delay records and aggregate
per route → save a dict of:

    per_day_distributions[YYYY-MM-DD][route_short_name] = {
        'mean': float, 'std': float, 'cancel_rate': float, 'n_obs': int
    }
"""

import sys, os, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from collections import defaultdict
from multiprocessing import Pool

# Worker imports (must be importable from worker subprocess)
from src.gtfs_rt_parser import process_day
from src.gtfs_parser import load_routes


# Global state set per worker
_ROUTE_NAMES = None


def _init_worker(route_names: dict):
    global _ROUTE_NAMES
    _ROUTE_NAMES = route_names


def _build_one_day(args):
    """Worker: process one day. Returns (date, dist_dict)."""
    date, tar_path, sample_interval = args
    try:
        recs = process_day(tar_path, sample_interval=sample_interval)
    except Exception as e:
        return date, {}, f'process_day error: {e}'

    delays_by_name: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        rid = r.get('route_id')
        if not rid:
            continue
        d = r.get('delay_seconds', 0) / 60.0
        if abs(d) > 120:
            continue
        name = _ROUTE_NAMES.get(rid, rid) if _ROUTE_NAMES else rid
        delays_by_name[name].append(d)

    out = {}
    for name, dlist in delays_by_name.items():
        if len(dlist) < 5:
            continue
        arr = np.asarray(dlist, dtype=float)
        n_cancel = int((np.abs(arr) > 30).sum())
        out[name] = {
            'mean': float(arr.mean()),
            'std': float(arr.std() if arr.size > 1 else 1.0),
            'cancel_rate': n_cancel / len(arr),
            'n_obs': len(arr),
        }
    return date, out, None


def main(rt_dir: str = 'data/swiss_rt/gtfs-rt',
         out_path: str = 'data/per_day_distributions.pkl',
         sample_interval: int = 30,
         n_workers: int = 15):
    print(f'Loading routes ...', flush=True)
    route_names = load_routes('data/swiss_gtfs')
    print(f'Loaded {len(route_names)} route names.', flush=True)

    tar_files = sorted([f for f in os.listdir(rt_dir) if f.endswith('.tar.bz2')])
    print(f'{len(tar_files)} day files. Pool size = {n_workers}.', flush=True)

    args_list = [
        (tf.replace('.tar.bz2', ''),
         os.path.join(rt_dir, tf),
         sample_interval)
        for tf in tar_files
    ]

    per_day = {}
    t0 = time.time()
    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(route_names,)) as pool:
        for i, (date, dist, err) in enumerate(
                pool.imap_unordered(_build_one_day, args_list)):
            if err:
                print(f'  [{i+1}/{len(args_list)}] {date} FAILED: {err}', flush=True)
                per_day[date] = {}
            else:
                print(f'  [{i+1}/{len(args_list)}] {date} -> {len(dist)} routes',
                      flush=True)
                per_day[date] = dist

    elapsed = time.time() - t0
    print(f'\nWall time: {elapsed:.1f}s', flush=True)

    with open(out_path, 'wb') as f:
        pickle.dump(per_day, f)
    print(f'Saved to {out_path}', flush=True)

    # Summary
    print('\n=== Per-day summary ===', flush=True)
    print(f"{'date':<12} {'routes':>7} {'mean_delay':>11} {'mean_cancel':>11}")
    for d in sorted(per_day):
        dist = per_day[d]
        if not dist:
            print(f'{d:<12} (empty)', flush=True)
            continue
        means = [v['mean'] for v in dist.values()]
        cancels = [v['cancel_rate'] for v in dist.values()]
        print(f'{d:<12} {len(dist):>7} {np.mean(means):>10.2f} '
              f'{np.mean(cancels):>10.3f}', flush=True)


if __name__ == '__main__':
    main()
