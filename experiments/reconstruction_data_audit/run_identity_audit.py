"""Recover the saved graph's route identities by exact GTFS trip lookup."""
from collections import defaultdict
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.gtfs_rt_parser import recover_graph_route_identity, scheduled_services_by_date

ROOT = Path(__file__).resolve().parents[2]


def main():
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as handle:
        graph = pickle.load(handle)
    gtfs = ROOT / 'data/swiss_gtfs'
    identity = recover_graph_route_identity(graph, str(gtfs))
    names = defaultdict(set)
    mismatches = []
    for connection in graph.connections:
        match = identity['by_trip'].get(connection.trip_id)
        if match is not None:
            names[connection.route].add((match['route_id'], match['agency_id']))
            if connection.route != match['name']:
                mismatches.append(connection.id)
    identity['graph_name_to_exact_routes'] = {
        name: [{'route_id': rid, 'agency_id': agency} for rid, agency in sorted(routes)]
        for name, routes in sorted(names.items())}
    identity['graph_name_mismatch_connection_ids'] = mismatches
    identity['scope'] = 'Exact static trip-ID lookup; recovers identity, not realized vehicle outcomes.'
    identity['graph'] = 'data/zurich_wide.pkl'
    identity['static_trips'] = 'data/swiss_gtfs/trips.txt'
    services = {trip: row['service_id'] for trip, row in identity['by_trip'].items()}
    dates = sorted(path.name.removesuffix('.tar.bz2') for path in (ROOT / 'data/swiss_rt/gtfs-rt').glob('*.tar.bz2'))
    calendars = scheduled_services_by_date(gtfs, dates, set(services.values()))
    identity['service_date_coverage'] = {
        day: {'active_service_ids': sorted(active),
              'active_trips': sum(service in active for service in services.values()),
              'active_connections': sum(services.get(c.trip_id) in active for c in graph.connections)}
        for day, active in calendars.items()
    }
    destination = Path(__file__).with_name('graph_identity.json')
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(identity, ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(destination)
    print(json.dumps({key: value for key, value in identity.items()
                      if key not in ('by_trip', 'graph_name_to_exact_routes', 'service_date_coverage')}, indent=2))
    print(f'matched_trips={len(identity["by_trip"])}, graph_names={len(names)}, '
          f'names_using_multiple_exact_routes={sum(len(routes)>1 for routes in names.values())}, '
          f'output_bytes={destination.stat().st_size}')


if __name__ == '__main__':
    main()
