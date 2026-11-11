"""Build route-ID summaries of unique predictions and explicit trip statuses.

Outputs are forecast-calibrated simulation inputs, not realized event replay.
Historical proxy-based pickles are preserved; corrected files use a new name.
"""
import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gtfs_rt_parser import load_route_metadata, process_day, summarize_predictions

_ROUTE_METADATA = None
_ROUTE_IDS = None


def _init_worker(route_metadata, route_ids=None):
    global _ROUTE_METADATA, _ROUTE_IDS
    _ROUTE_METADATA = route_metadata
    _ROUTE_IDS = route_ids


def _build_one_day(args):
    date, tar_path, sample_interval = args
    records = process_day(tar_path, sample_interval=sample_interval, route_ids=_ROUTE_IDS,
                          service_date=date.replace('-', ''))
    summaries = summarize_predictions(records, _ROUTE_METADATA)
    return date, summaries


def main(rt_dir='data/swiss_rt/gtfs-rt',
         out_path='data/per_day_distributions_corrected.pkl',
         sample_interval=30, n_workers=4, graph_identity=None, dates=None):
    metadata = load_route_metadata('data/swiss_gtfs')
    route_ids = None
    if graph_identity is not None:
        with open(graph_identity, encoding='utf-8') as handle:
            by_trip = json.load(handle)['by_trip']
        route_ids = {trip['route_id'] for trip in by_trip.values()}
    inputs = [(f.removesuffix('.tar.bz2'), os.path.join(rt_dir, f), sample_interval)
              for f in sorted(os.listdir(rt_dir)) if f.endswith('.tar.bz2')
              and (dates is None or f.removesuffix('.tar.bz2') in dates)]
    per_day = {}
    temporary = f'{out_path}.tmp'
    with Pool(n_workers, initializer=_init_worker, initargs=(metadata, route_ids)) as pool:
        for date, summaries in pool.imap_unordered(_build_one_day, inputs):
            per_day[date] = summaries
            with open(temporary, 'wb') as handle:
                pickle.dump(per_day, handle)
            os.replace(temporary, out_path)
            print(f'{date}: {len(summaries)} route IDs', flush=True)
    print(f'Saved {out_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rt-dir', default='data/swiss_rt/gtfs-rt')
    parser.add_argument('--output', default='data/per_day_distributions_corrected.pkl')
    parser.add_argument('--sample-interval', type=int, default=30)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--graph-identity', help='Restrict to exact route IDs from this graph identity JSON')
    parser.add_argument('--dates', nargs='+', help='Service days in YYYY-MM-DD format')
    args = parser.parse_args()
    main(args.rt_dir, args.output, args.sample_interval, args.workers, args.graph_identity,
         set(args.dates) if args.dates else None)
