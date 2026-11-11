"""Inspect measured-event variability and chronology before policy replay."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def audit(data):
    by_agency = defaultdict(list)
    by_trip = defaultdict(list)
    endpoint_counts = Counter()
    for connection in data['connections']:
        by_trip[connection['trip_id']].append(connection)
        for kind in ('departure', 'arrival'):
            observation = connection[kind]
            endpoint_counts[observation['status']] += 1
            if observation['status'] == 'REAL':
                by_agency[connection['agency_id']].append(
                    observation['time_seconds'] - connection[f'scheduled_{kind}_seconds'])
    violations = []
    for trip, sequence in by_trip.items():
        sequence.sort(key=lambda connection: connection['dep_sequence'])
        for a, b in zip(sequence, sequence[1:]):
            if (a['arr_sequence'] == b['dep_sequence'] and a['outcome'] == b['outcome'] == 'MEASURED'
                    and a['arrival']['time_seconds'] > b['departure']['time_seconds']):
                violations.append({'trip_id': trip, 'incoming': a['id'], 'outgoing': b['id']})
    variability = {agency: {'n_real_endpoints': len(values), 'exactly_schedule_fraction': float(np.mean(np.asarray(values) == 0)),
                             'delay_std_seconds': float(np.std(values)),
                             'delay_min_p10_median_p90_max_seconds': np.quantile(values, [0, .1, .5, .9, 1]).tolist()}
                   for agency, values in by_agency.items()}
    return {'endpoint_status_counts': dict(endpoint_counts), 'variability_by_agency': variability,
            'same_trip_chronology_violations': violations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'city_dated_replay.json.gz')
    args = parser.parse_args()
    with gzip.open(args.input, 'rt', encoding='utf-8') as handle:
        data = json.load(handle)
    output = {day: audit(value) for day, value in data['dates'].items()} if 'dates' in data else audit(data)
    path = args.input.with_name(args.input.name.removesuffix('.json.gz') + '_audit.json')
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
