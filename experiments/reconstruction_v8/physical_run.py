"""Restore existing municipal stopping points and audit conditional motion floors."""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import io
import json
from pathlib import Path
from statistics import median
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_data_audit.build_vbz_city_replay import STOP_NAME_ALIASES
from experiments.reconstruction_data_audit.build_vbz_replay_sample import stop_crosswalk
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from src.evolving_trip_belief import EvolvingTripBelief, logsumexp
from src.physical_bounds import elapsed_seconds_floor, wgs84_reference_chord

AUDIT = ROOT / 'experiments/reconstruction_data_audit'
V4 = ROOT / 'experiments/reconstruction_v4'
V6 = ROOT / 'experiments/reconstruction_v6'
V7 = ROOT / 'experiments/reconstruction_v7'
STATIC = ROOT / 'data/swiss_gtfs'
MUNICIPAL = 'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd_2023'
STOP_POINTS_URL = MUNICIPAL + '/download/Haltepunkt.csv'
FLEET = 'https://www.stadt-zuerich.ch/vbz/de/die-vbz/fahrzeuge/trams0.html'
RAIL = 'https://www.sbb.ch/en/travel-information/services-on-train/our-trains/regio-double-decker.html'
NATIONAL = 'https://opentransportdata.swiss/de/cookbook/historic-and-statistics-cookbook/actual-data/'
RADII = (0, 50, 150)


def read(path):
    with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as handle:
        return json.load(handle)


def city_day_index(data):
    return {(day, row['id']): row for day, record in data['dates'].items() for row in record['connections']}


def municipal_key(day, row):
    dep, arr = row['departure'], row['arrival']
    if row['outcome'] != 'MEASURED' or not dep['official_journey'].startswith('municipal:'):
        return None
    _, _, journey, vehicle = dep['official_journey'].split(':')
    return (day, journey, vehicle, row['route_name'], row['dep_stop'], row['arr_stop'],
            dep['provider_scheduled_seconds'], arr['provider_scheduled_seconds'],
            dep['time_seconds'], arr['time_seconds'])


def restore_points(rows, raw_paths):
    """Reverse the preserved municipal identity and clock match, keeping ambiguities."""
    keys = {key for day, row in rows if (key := municipal_key(day, row)) is not None}
    wanted = {key[:3] for key in keys}
    stop_file = AUDIT / 'vbz_stop_mapping.csv'
    crosswalk = stop_crosswalk(STATIC, stop_file.read_bytes())
    with stop_file.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['halt_lang'] in STOP_NAME_ALIASES:
                crosswalk[row['halt_id']] = STOP_NAME_ALIASES[row['halt_lang']]
    candidates = defaultdict(set)
    for path in raw_paths:
        with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                day = datetime.strptime(row['betriebsdatum'], '%d.%m.%y').date()
                identity = (day.isoformat(), row['fahrt_id'], row['fahrzeug'])
                if identity not in wanted:
                    continue
                values = tuple(row[k] for k in ('soll_ab_von', 'soll_an_nach', 'ist_ab_von', 'ist_an_nach1'))
                if not all(values):
                    continue
                dep_offset, arr_offset = (
                    (datetime.strptime(row[f'datum_{side}'], '%d.%m.%y').date() - day).days * 86400
                    for side in ('von', 'nach'))
                sd, sa, ad, aa = map(int, values)
                key = (*identity, row['linie'], crosswalk.get(row['halt_id_von']),
                       crosswalk.get(row['halt_id_nach']), sd + dep_offset, sa + arr_offset,
                       ad + dep_offset, aa + arr_offset)
                if key in keys:
                    candidates[key].add((row['halt_punkt_id_von'], row['halt_punkt_id_nach']))
    return candidates


def fleet_cap(key):
    if key is None:
        return None, None
    vehicle = int(key[2])
    if 3001 <= vehicle <= 3088:
        return 'Cobra', 70
    if 4001 <= vehicle <= 4070:
        return 'Flexity', 60
    return None, None


def row_time(row):
    if row['outcome'] != 'MEASURED':
        return None
    return row['arrival']['time_seconds'] - row['departure']['time_seconds']


def floors(distance, cap_kmh):
    return {str(radius): elapsed_seconds_floor(distance, cap_kmh / 3.6, radius, radius)
            for radius in RADII}


def summary(values):
    return dict(n=len(values), minimum=min(values), median=median(values), maximum=max(values)) if values else dict(n=0)


def select_rail_legs(replay, rail):
    keys = {(journey['date'], decision['source_connection'])
            for journey in replay['rows'] for decision in journey['decisions']
            if decision.get('outcome') == 'board' and decision.get('source_connection') is not None}
    return [(day, rail[day, cid]) for day, cid in sorted(keys)]


def rail_platforms(rail_legs):
    wanted = {(row['trip_id'], row[side + '_sequence']) for _, row in rail_legs for side in ('dep', 'arr')}
    trips = {trip for trip, _ in wanted}
    # Only retain matching static rows from the existing feed; no new full static copy.
    patterns = HERE / 'physical_selected_rail_trips.txt'
    patterns.write_text(''.join(f'"{trip}",\n' for trip in sorted(trips)))
    file = STATIC / 'stop_times.txt'
    result = subprocess.run(['rg', '-F', '-f', str(patterns), str(file)], capture_output=True, text=True, check=True)
    with file.open(encoding='utf-8-sig') as handle:
        header = handle.readline()
    return {(row['trip_id'], int(row['stop_sequence'])): row['stop_id']
            for row in csv.DictReader(io.StringIO(header + result.stdout))
            if (row['trip_id'], int(row['stop_sequence'])) in wanted}


def joint_operating_edge(law, slot):
    marginal = law.marginal(slot)
    future = law._backward_from(max(slot + 1, law._last_effect), slot + 1)
    edge = law._transition(slot) + law.emissions[slot + 1] + future
    norm = logsumexp(edge, axis=1, keepdims=True)
    conditional = np.exp(edge - np.where(np.isfinite(norm), norm, 0.))
    clock_delta = law.times[slot + 1][None, :] - law.times[slot][:, None]
    operating = ~law.canceled[slot][:, None] & ~law.canceled[slot + 1][None, :]
    mass = np.where(operating, marginal[:, None] * conditional, 0.)
    return clock_delta, mass


def main():
    point_file = Path(sys.argv[1]) if len(sys.argv) == 2 else HERE / 'physical_vbz_point_subset.csv'
    stop_points = {}
    with point_file.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            try:
                position = float(row['GPS_Latitude']), float(row['GPS_Longitude'])
            except ValueError:
                continue
            if 47 <= position[0] <= 48 and 8 <= position[1] <= 9:
                stop_points[row['halt_punkt_id']] = (row['halt_id'], position)
    with (STATIC / 'stops.txt').open(encoding='utf-8-sig') as handle:
        coordinates = {r['stop_id']: (float(r['stop_lat']), float(r['stop_lon'])) for r in csv.DictReader(handle)
                       if r['stop_lat'] and r['stop_lon']}
    city_input = AUDIT / 'city_vbz_dated_replay_v2.json.gz'
    city_data, _ = load_city(city_input)
    city = city_day_index(city_data)
    city.update(city_day_index(read(V6 / 'city_confirmation_source_completed.json.gz')))
    rail = city_day_index(read(AUDIT / 'dated_replay_v2.json.gz'))
    raw_paths = [AUDIT / 'vbz_peak_allkinds_rows.csv.gz', V6 / 'city_confirmation_peak_rows.csv.gz']
    fit = fit_training(city_data, retain_no_pickup=True)
    support = read(V7 / 'boundary_support_audit.json')
    day = support['date']
    graph, _, _, _, metadata, _ = dated_graph(city_data, day, 'gtfs_station_max', True)
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    transfers = read(V7 / 'platform_audit.json')['rows']
    points = restore_points([(day, metadata[row['connection']]) for row in support['rows']] +
                            [(row['date'], city[row['date'], row[side]]) for row in transfers
                             if row['domain'] == 'city' for side in ('incoming', 'outgoing')], raw_paths)
    used_points = set()

    def points_for(day, row):
        key = municipal_key(day, row)
        choices = points[key] if key is not None else ()
        if len(choices) != 1:
            return None
        ids = next(iter(choices))
        used_points.update(p for p in ids if p in stop_points)
        return ids

    def distance(ids, index):
        if not ids or any(p not in index for p in ids):
            return None
        return wgs84_reference_chord(index[ids[0]], index[ids[1]])

    locations = {point: row[1] for point, row in stop_points.items()}
    runtime = []
    for entry in support['rows']:
        cid = entry['connection']
        row = metadata[cid]
        ids = points_for(day, row)
        actual_chord = distance(ids, locations)
        model, cap = fleet_cap(municipal_key(day, row)) if row['mode'] == 'tram' else (None, None)
        conditional = floors(actual_chord, cap) if actual_chord is not None and cap is not None else {}
        mass = {}
        if conditional:
            trip, slot = belief._positions[cid, 'departure']
            clocks, weights = joint_operating_edge(belief._law(trip, parameters), slot)
            mass = {radius: float(weights[clocks < bound].sum()) for radius, bound in conditional.items()}
        runtime.append(dict(connection=cid, source_connection=row['id'], mode=row['mode'], route=row['route_id'],
            stops=[row['dep_stop'], row['arr_stop']], source_point_ids=ids, source_point_chord_m=actual_chord,
            scheduled_point_chord_m=distance((row['dep_platform'], row['arr_platform']), coordinates),
            fleet=model, cap_kmh=cap, realized_runtime_seconds=row_time(row), conditional_floors_seconds=conditional,
            old_model_below_floor_operating_mass=mass,
            old_model_zero_runtime_probability=entry['current_model_zero_operating_runtime_probability']))
    rail_legs = select_rail_legs(read(V4 / 'dated_replay_final_results.json'), rail)
    platforms = rail_platforms(rail_legs)
    railway = []
    for day, row in rail_legs:
        ids = tuple(platforms.get((row['trip_id'], row[side + '_sequence'])) for side in ('dep', 'arr'))
        chord = distance(ids, coordinates)
        # Reference RABe 511 specification for sensitivity only. The archive
        # records the service number, not the physical stock.
        conditional = floors(chord, 160) if chord is not None and row['operator_id'] == '85:11' else {}
        railway.append(dict(date=day, source_connection=row['id'], route=row['route_name'],
            scheduled_platforms=ids, scheduled_point_chord_m=chord, realized_runtime_seconds=row_time(row),
            conditional_rabe_511_floors_seconds=conditional))
    transfer_rows = []
    for row in transfers:
        ids = row['arrival_platform'], row['departure_platform']
        item = dict(row, scheduled_point_chord_m=distance(ids, coordinates))
        if row['domain'] == 'city':
            incoming = points_for(row['date'], city[row['date'], row['incoming']])
            outgoing = points_for(row['date'], city[row['date'], row['outgoing']])
            actual = (incoming[1], outgoing[0]) if incoming and outgoing else None
            item.update(source_point_ids=actual, source_point_chord_m=distance(actual, locations))
        transfer_rows.append(item)
    # Only preserve the joined point subset, not the original catalog.
    subset_path = HERE / 'physical_vbz_point_subset.csv'
    with subset_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('halt_punkt_id', 'halt_id', 'GPS_Latitude', 'GPS_Longitude'))
        for point in sorted(used_points):
            stop, (latitude, longitude) = stop_points[point]
            writer.writerow((point, stop, latitude, longitude))
    city_match = [r for r in runtime if r['source_point_chord_m'] is not None]
    city_fleet = [r for r in runtime if r['conditional_floors_seconds']]
    rail_fleet = [r for r in railway if r['conditional_rabe_511_floors_seconds']]
    city_transfers = [r for r in transfer_rows if r['domain'] == 'city']
    rail_transfers = [r for r in transfer_rows if r['domain'] == 'rail']

    def floor_summary(rows, field):
        return {str(radius): dict(
                    floor_seconds=summary([r[field][str(radius)] for r in rows]),
                    positive=sum(r[field][str(radius)] > 0 for r in rows),
                    observed_below=sum(r['realized_runtime_seconds'] is not None and
                                       r['realized_runtime_seconds'] < r[field][str(radius)] for r in rows))
                for radius in RADII}

    result = dict(
        scope='Existing development records, no new weekly or national actual data.',
        sources={'municipal_catalog': MUNICIPAL, 'municipal_stop_points': STOP_POINTS_URL,
                 'municipal_stop_point_input_bytes': point_file.stat().st_size, 'vbz_fleet': FLEET,
                 'rabe_511_example': RAIL, 'national_archive_platform_change': NATIONAL,
                 'raw_municipal_peak_inputs': [str(p.relative_to(ROOT)) for p in raw_paths]},
        geometry='WGS84 reference-point chord. Runtime floor assumes a speed cap and true projected endpoints within stated radii. No source identifies those radii.',
        radii_m_each=RADII,
        city={'n': len(runtime), 'matched_stop_points': len(city_match), 'identified_fleet_rows': len(city_fleet),
              'fleet_counts': dict(Counter(row['fleet'] for row in city_fleet)),
              'identified_fleet_realized_runtime_seconds': summary([r['realized_runtime_seconds'] for r in city_fleet]),
              'identified_fleet_realized_under_60_seconds': sum(r['realized_runtime_seconds'] < 60 for r in city_fleet),
              'point_chord_m': summary([r['source_point_chord_m'] for r in city_match]),
              'floors': floor_summary(city_fleet, 'conditional_floors_seconds'),
              'old_model_below_conditional_floors': {
                  str(radius): summary([r['old_model_below_floor_operating_mass'][str(radius)] for r in city_fleet
                                      if r['conditional_floors_seconds'][str(radius)] > 0]) for radius in RADII}},
        rail={'n': len(railway), 'rabe_511_conditional_rows': len(rail_fleet),
              'floors': floor_summary(rail_fleet, 'conditional_rabe_511_floors_seconds')},
        transfer={'city_n': len(city_transfers), 'rail_n': len(rail_transfers),
                  'city_positive_point_distances': sum((r['source_point_chord_m'] or 0) > 0 for r in city_transfers),
                  'rail_positive_point_distances': sum((r['scheduled_point_chord_m'] or 0) > 0 for r in rail_transfers),
                  'city_source_point_chord_m': summary([r['source_point_chord_m'] for r in city_transfers
                                                      if r['source_point_chord_m'] is not None]),
                  'rail_scheduled_point_chord_m': summary([r['scheduled_point_chord_m'] for r in rail_transfers
                                                          if r['scheduled_point_chord_m'] is not None]),
                  'unconditional_physical_lower_seconds': 0},
        city_runtime_rows=runtime, rail_runtime_rows=railway, transfer_rows=transfer_rows)
    (HERE / 'physical_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('city', 'rail', 'transfer')}, indent=2))


if __name__ == '__main__':
    main()
