"""Recover actual platform identities without copying the full static feed."""
import csv
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v5.calibration_run import data_inputs
from src.platform_transfers import PlatformTransfers


def read(path):
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as handle:
        return json.load(handle)


def main():
    index = PlatformTransfers.from_gtfs(ROOT / 'data/swiss_gtfs')
    rail = data_inputs()
    city = read(ROOT / 'experiments/reconstruction_v6/city_confirmation_source_completed.json.gz')
    city_development = read(ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
    city['dates'].update(city_development['dates'])
    sources = [
        ('rail', ROOT / 'experiments/reconstruction_v4/dated_replay_final_results.json'),
        ('city', ROOT / 'experiments/reconstruction_v6/city_itinerary_charged.json'),
        ('city', ROOT / 'experiments/reconstruction_v6/city_source_completed_paired_charged.json')]
    rows, by_trip = [], {}
    for domain, path in sources:
        data = rail if domain == 'rail' else city
        for journey in read(path)['rows']:
            source = {c['id']: c for c in data['dates'][journey['date']]['connections']}
            previous = None
            for decision in journey['decisions']:
                if decision.get('outcome') != 'board':
                    continue
                cid = decision.get('source_connection')
                if cid is None:
                    continue
                current = source[cid]
                if previous and previous['trip_id'] != current['trip_id']:
                    item = dict(domain=domain, result=str(path.relative_to(ROOT)), date=journey['date'],
                        query=journey['query'], method=journey['method'],
                        incoming=previous['id'], outgoing=current['id'], station=previous['arr_stop'],
                        incoming_trip=previous['trip_id'], outgoing_trip=current['trip_id'],
                        incoming_sequence=previous['arr_sequence'], outgoing_sequence=current['dep_sequence'],
                        arrival_platform=previous.get('arr_platform'), departure_platform=current.get('dep_platform'))
                    rows.append(item)
                    if domain == 'rail':
                        by_trip[previous['trip_id']] = True
                        by_trip[current['trip_id']] = True
                previous = current
    patterns = HERE / 'platform_selected_trips.txt'
    patterns.write_text(''.join('"' + t + '",\n' for t in sorted(by_trip)))
    static = ROOT / 'data/swiss_gtfs/stop_times.txt'
    result = subprocess.run(['rg', '-F', '-f', str(patterns), str(static)], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    with static.open(encoding='utf-8-sig') as handle:
        header = handle.readline()
    platforms = {(r['trip_id'], int(r['stop_sequence'])): r['stop_id']
                 for r in csv.DictReader(io.StringIO(header + result.stdout)) if r['trip_id'] in by_trip}
    for row in rows:
        if row['domain'] == 'rail':
            row['arrival_platform'] = platforms.get((row['incoming_trip'], row['incoming_sequence']))
            row['departure_platform'] = platforms.get((row['outgoing_trip'], row['outgoing_sequence']))
        declaration = index.resolve(row['arrival_platform'], row['departure_platform'])
        row['declaration'] = asdict(declaration)
    stations = {row['station'] for row in rows}
    with gzip.open(HERE / 'platform_pair_declarations.csv.gz', 'wt') as handle:
        writer = csv.writer(handle)
        writer.writerow(['from_stop_id', 'to_stop_id', 'transfer_type', 'declared_seconds', 'physical_lower_seconds'])
        for (a, b), rules in sorted(index.rules.items()):
            if a.split(':')[0] in stations and b.split(':')[0] in stations:
                for kind, seconds in sorted(rules):
                    writer.writerow([a, b, kind, seconds, 0])
    result = dict(scope='Existing V1–V6 development histories; static scheduled platforms, not actual platform assignment.',
        reference='https://gtfs.org/documentation/schedule/reference/#transferstxt',
        source_files=['data/swiss_gtfs/stops.txt', 'data/swiss_gtfs/transfers.txt', 'data/swiss_gtfs/stop_times.txt'],
        pathways_present=(ROOT / 'data/swiss_gtfs/pathways.txt').exists(),
        min_transfer_semantics='Operator requirement includes schedule-variance buffer; not identified physical shortest walking time.',
        n_selected_trips=len(by_trip), recovered_static_platform_rows=len(platforms),
        summary={domain: dict(n=sum(r['domain'] == domain for r in rows),
            specificity=dict(Counter(r['declaration']['specificity'] for r in rows if r['domain'] == domain)),
            declared_seconds=dict(Counter(str(r['declaration']['seconds']) for r in rows if r['domain'] == domain)))
                 for domain in ('rail', 'city')},
        actual_platform_assignment='Unidentified: actual source does not provide an observed platform field in these records.',
        physical_time_lower_bound_seconds=0, rows=rows)
    (HERE / 'platform_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
