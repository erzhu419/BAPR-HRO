"""Join the complete remotely filtered municipal peak files to every city candidate."""
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_data_audit.build_vbz_replay_sample import (
    CATALOGUE, SAMPLE, STOPS, DATES, measured_rows, join_day, stop_crosswalk,
)

STOP_NAME_ALIASES = {
    'Zürich, Brunnenhof': '8591307',
    'Zürich, Bahnhof Altstetten': '8591056',
    'Zürich, Bahnhof Wiedikon': '8573710',
    'Zürich, Bahnhof Hardbrücke': '8591060',
    'Zürich, Bahnhof Enge/Bederstr.': '8591059',
    'Zürich, Bahnhof Oerlikon Ost': '8591063',
    'Zürich, Bahnhof Oerlikon': '8580449',
    'Zürich, Bahnhof Oerlikon Nord': '8591062',
    'Zürich, Bahnhof Enge': '8591058',
    'Zürich, Bahnhof Stadelhofen': '8503059',
    'Zürich, Bahnhof Tiefenbrunnen': '8576182',
    'Zürich, Bahnhof Stettbach': '8591065',
    'Zürich, Bahnhof Wipkingen': '8591066',
    'Zürich, Bahnhof Affoltern': '8591054',
    'Zürich, Sihlpost / HB': '8591367',
    'Glattpark': '8591830',
    'Zch, Bhf.Wollishofen/Staubstr.': '8591081',
}


def main():
    with gzip.open(HERE / 'vbz_peak_allkinds_rows.csv.gz', 'rb') as handle:
        raw = handle.read()
    # This is the small original municipal matching table, never the weekly file.
    stop_file = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'vbz_stop_mapping.csv'
    crosswalk = stop_crosswalk(ROOT / 'data/swiss_gtfs', stop_file.read_bytes())
    with stop_file.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['halt_lang'] in STOP_NAME_ALIASES:
                crosswalk[row['halt_id']] = STOP_NAME_ALIASES[row['halt_lang']]
    measured, exclusions = measured_rows(raw, crosswalk, normal_only=False)
    results = {}
    for day in DATES:
        with gzip.open(HERE / f'city_schedule_{day}.json.gz', 'rt', encoding='utf-8') as handle:
            schedule = json.load(handle)
        connections = schedule['connections']
        index = defaultdict(list)
        for c in connections:
            key = (c['route_name'], c['dep_stop'], c['arr_stop'], c['scheduled_departure_seconds'] // 60,
                   c['scheduled_arrival_seconds'] // 60)
            index[key].append(c)
        counts, matches = join_day(measured[day], index, {c['service_id'] for c in connections})
        by_id = {match['id']: match for match in matches}
        endpoints = Counter()
        by_agency = defaultdict(Counter)
        for c in connections:
            actual = by_id.get(c['id'])
            if actual is None:
                c['departure'] = {'status': 'MISSING', 'time_seconds': None, 'official_journey': None}
                c['arrival'] = {'status': 'MISSING', 'time_seconds': None, 'official_journey': None}
                c['outcome'] = 'UNRESOLVED'
                endpoints['MISSING'] += 2
            else:
                provider_id = f"municipal:{day}:{actual['provider_trip_id']}:{actual['provider_vehicle']}"
                for kind in ('departure', 'arrival'):
                    c[kind] = {'status': 'REAL', 'time_seconds': actual[f'actual_{kind}_seconds'],
                               'official_journey': provider_id,
                               'provider_scheduled_seconds': actual[f'provider_scheduled_{kind}_seconds']}
                c['outcome'] = 'MEASURED'
                c['provider_service_kind'] = actual['provider_service_kind']
                endpoints['REAL'] += 2
            by_agency[c['agency_id']][c['outcome']] += 1
        results[day] = {**schedule, 'coverage': {
            'scheduled_connections': len(connections), 'endpoint_status_counts': dict(endpoints),
            'connection_outcome_counts': dict(Counter(c['outcome'] for c in connections)),
            'source_matching_counts': dict(counts),
            'source_exclusions': {reason: n for (date_, reason), n in exclusions.items() if date_ == day},
            'by_agency': {agency: dict(stats) for agency, stats in by_agency.items()}}}
        print(day, results[day]['coverage'], flush=True)
    audit = json.loads((HERE / 'vbz_peak_allkinds_rows.csv.audit.json').read_text())
    output = {'scope': 'Complete municipal weekly file scanned remotely; all calendar-active city bus/tram candidates '
              'in the declared rectangle and peak window retained. Municipal records measure performed services; '
              'unmatched candidates are unresolved and cancellation outcomes are unobserved.',
              'source': SAMPLE, 'catalogue': CATALOGUE, 'source_audit': audit,
              'explicit_municipal_stop_name_aliases_to_uic': STOP_NAME_ALIASES,
              'matching_table': STOPS, 'planned_time_join_precision_seconds': 60,
              'train_dates': ['2023-07-25'], 'holdout_dates': ['2023-07-26', '2023-07-27'],
              'dates': results}
    with gzip.open(HERE / 'city_vbz_dated_replay_v2.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(output, handle, ensure_ascii=False, separators=(',', ':'))


if __name__ == '__main__':
    main()
