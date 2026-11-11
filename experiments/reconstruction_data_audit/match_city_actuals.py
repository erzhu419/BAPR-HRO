"""Join a complete-day, remotely filtered national city actuals cohort to GTFS."""
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_data_audit.run_realized_subset import FIELDS, seconds_of_service
from experiments.reconstruction_data_audit.actual_event_parser import parse_actual_endpoint


def operator_agencies(operator):
    agency = operator.split(':', 1)[-1]
    # INFO+ assigns VBZ tram schedules agency 3849; VBZ operates as 849.
    return {'849', '3849'} if agency == '849' else {agency}


def read_observations(path, schedule):
    indices = defaultdict(set)
    for c in schedule['connections']:
        for kind, side in (('departure', 'dep'), ('arrival', 'arr')):
            key = c['route_name'], c[f'{side}_stop'], kind, c[f'scheduled_{kind}_seconds'] // 60
            indices[key].add((c['trip_id'], c['agency_id'], c[f'{side}_sequence']))
    observations = []
    scores = defaultdict(lambda: defaultdict(set))
    with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle, delimiter=';'):
            if row['ZUSATZFAHRT_TF'] == 'true':
                continue
            agencies = operator_agencies(row['BETREIBER_ID'])
            journey = row['FAHRT_BEZEICHNER']
            for kind, (planned, actual, status) in FIELDS.items():
                if not row[planned]:
                    continue
                minute = seconds_of_service(row[planned], schedule['date']) // 60
                key = row['LINIEN_TEXT'], row['BPUIC'], kind, minute
                candidates = {(trip, seq) for trip, agency, seq in indices[key] if agency in agencies}
                outcome, seconds, _, reported, canceled, raw_status = parse_actual_endpoint(row, kind, schedule['date'])
                observations.append((journey, kind, candidates, outcome, seconds, reported, canceled, raw_status))
                for trip, seq in candidates:
                    scores[journey][trip].add((seq, kind, minute))
    return observations, scores


def main():
    with gzip.open(HERE / 'city_schedule.json.gz', 'rt', encoding='utf-8') as handle:
        schedule = json.load(handle)
    path = HERE / 'city_actual_rows.csv.gz'
    observations, scores = read_observations(path, schedule)
    by_trip = defaultdict(set)
    journey_map = {}
    reasons = Counter()
    for journey, candidates in scores.items():
        high = max(map(len, candidates.values()))
        winners = [trip for trip, matches in candidates.items() if len(matches) == high]
        if len(winners) == 1 and high >= 2:
            trip = winners[0]
            journey_map[journey] = trip
            by_trip[trip].add(journey)
        else:
            reasons['ambiguous_journey' if len(winners) > 1 else 'insufficient_matched_endpoints'] += 1
    records = defaultdict(set)
    for journey, kind, candidates, outcome, seconds, reported, canceled, raw_status in observations:
        trip = journey_map.get(journey)
        if trip is None or len(by_trip[trip]) != 1:
            continue
        for candidate, sequence in candidates:
            if candidate == trip:
                records[trip, sequence, kind].add((outcome, seconds, journey, reported, canceled, raw_status))
    endpoints, outcomes = Counter(), Counter()
    by_agency = defaultdict(Counter)
    for c in schedule['connections']:
        for kind, side in (('departure', 'dep'), ('arrival', 'arr')):
            values = records[c['trip_id'], c[f'{side}_sequence'], kind]
            if len(values) != 1:
                status, seconds, journey = ('MISSING' if not values else 'AMBIGUOUS'), None, None
                reported, canceled, raw_status = None, None, None
            else:
                status, seconds, journey, reported, canceled, raw_status = next(iter(values))
            c[kind] = {'status': status, 'time_seconds': seconds, 'official_journey': journey,
                       'reported_time_seconds': reported, 'canceled_flag': canceled, 'reported_status': raw_status}
            endpoints[status] += 1
        departure, arrival = c['departure'], c['arrival']
        same = departure['official_journey'] is not None and departure['official_journey'] == arrival['official_journey']
        if (departure['status'] == arrival['status'] == 'REAL' and same
                and departure['time_seconds'] <= arrival['time_seconds']):
            c['outcome'] = 'MEASURED'
        elif departure['status'] == arrival['status'] == 'CANCELED' and same:
            c['outcome'] = 'CANCELED'
        else:
            c['outcome'] = 'UNRESOLVED'
        outcomes[c['outcome']] += 1
        by_agency[c['agency_id']][c['outcome']] += 1
    audit = {'scheduled_connections': len(schedule['connections']), 'endpoint_status_counts': dict(endpoints),
             'connection_outcome_counts': dict(outcomes), 'by_agency': {key: dict(counts) for key, counts in by_agency.items()},
             'journey_matching_exclusions': dict(reasons), 'matched_actual_journeys': len(journey_map),
             'conflicting_actual_journeys_per_static_trip': sum(len(value) > 1 for value in by_trip.values())}
    output = {**schedule, 'actual_source': str(path.relative_to(ROOT)), 'coverage': audit,
              'join_rule': 'A unique maximal overlap of at least two planned route/stop/time/event endpoints identifies '
              'each actual journey, with operator agency compatibility. Conflicting journey maps stay unresolved.'}
    with gzip.open(HERE / 'city_dated_replay.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(output, handle, ensure_ascii=False, separators=(',', ':'))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
